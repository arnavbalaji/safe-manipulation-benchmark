import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import json
import h5py
import random
import pickle
import torch as th
import numpy as np
from collections import defaultdict

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.controllers.controller_base import IsGraspingState
from omnigibson import object_states

from safety_benchmark.damageable_env import (
    DamageableEnvironment,
    DamageableDataCollectionWrapper,
    DamageableDataPlaybackWrapper,
)
from safety_benchmark.utils.misc_utils import (
    save_rgb_camera_video,
    save_rgb_temperature_video,
)


gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False

# Task objects configuration (mirrors teleop_firewood.py)
TASK_OBJECTS = {
    "fireplace": {
        "type": "DatasetObject",
        "name": "fireplace",
        "category": "wood_fireplace",
        "model": "gpnsij",
        "position": [-1.5, -2.0, 0.5],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 0.75, 0.75],
        "fixed_base": True,
        "abilities": {
            "heatSource": {
                "temperature": 100.0,
                "heating_rate": 0.1,
                "distance_threshold": 0.12,
                "requires_toggled_on": False,
            }
        },
        "initial_state": {
            "temperature": 100.0,
        },
    },
    "log_center": {
        "type": "DatasetObject",
        "name": "log_center",
        "category": "log",
        "model": "pepele",
        "position": [-1.65, -2.0, 0.15],
        "orientation": [0, 0, 0, 1],
        "scale": [0.8, 0.6, 0.6],
        "abilities": {
            "flammable": {},
        },
        "initial_state": {
            "onFire": True,
        },
        "damage_params": {
            "damage_evaluators": ["mechanical"],
            "health_thresholds": [90.0, 60.0, 30.0],
            "mechanical": {
                "damage_threshold": 200.0,
                "scale": 0.5,
                "instant_coefficient": 1.0,
                "creep_coefficient": 0.0,
                "object_type": "brittle",
            },
        },
    },
    "log_left": {
        "type": "DatasetObject",
        "name": "log_left",
        "category": "log",
        "model": "pepele",
        "position": [-1.65, -2.15, 0.17],
        "orientation": [0, 0, 0, 1],
        "scale": [0.8, 0.6, 0.6],
        "abilities": {
            "flammable": {},
        },
        "initial_state": {
            "onFire": True,
        },
        "damage_params": {
            "damage_evaluators": ["mechanical"],
            "health_thresholds": [90.0, 60.0, 30.0],
            "mechanical": {
                "damage_threshold": 200.0,
                "scale": 0.5,
                "instant_coefficient": 1.0,
                "creep_coefficient": 0.0,
                "object_type": "brittle",
            },
        },
    },
    "target_object": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "log",
        "model": "pepele",
        # Position moved further to the left of robot (robot at y=-2.0)
        "position": [-1.0, -2.25, 0.1],
        "orientation": [0, 0, 0, 1],
        "scale": [0.7, 0.5, 0.5],
        "abilities": {
            "flammable": {},
        },
        "initial_state": {
            "onFire": False,
        },
        "damage_params": {
            "damage_evaluators": ["mechanical"],
            "health_thresholds": [90.0, 60.0, 30.0],
            "mechanical": {
                "damage_threshold": 200.0,
                "scale": 0.5,
                "instant_coefficient": 1.0,
                "creep_coefficient": 0.0,
                "object_type": "brittle",
            },
        },
    },
}


def _ensure_firewood_states(env):
    """
    Make sure firewood-specific object states are active for health tracking.
    
    This ensures:
    - Robot has Temperature state (for thermal health tracking)
    - Fireplace has HeatSourceOrSink state (for heating the robot)
    """
    # Ensure robot damage evaluators are initialized (adds Temperature state)
    if env.robots:
        robot = env.robots[0]
        if hasattr(robot, "track_damage") and robot.track_damage:
            if not hasattr(robot, "params") or not robot.params:
                from safety_benchmark.params.test_params import PARAMS
                if "agent" in PARAMS:
                    robot.set_params(PARAMS["agent"])
            
            if not hasattr(robot, "damage_evaluators") or len(robot.damage_evaluators) == 0:
                if hasattr(robot, "_initialize_damage_evaluators"):
                    robot._initialize_damage_evaluators()
    
    # Ensure fireplace has HeatSourceOrSink state for health tracking
    fireplace = env.scene.object_registry("name", "fireplace")
    if fireplace is not None:
        fireplace_cfg = TASK_OBJECTS.get("fireplace", {})
        heat_source_cfg = fireplace_cfg.get("abilities", {}).get("heatSource", {})
        
        if not hasattr(fireplace, "_abilities"):
            fireplace._abilities = {}
        if "heatSource" not in fireplace._abilities:
            fireplace._abilities["heatSource"] = heat_source_cfg
        
        if object_states.HeatSourceOrSink not in fireplace.states:
            heat_source_state = object_states.HeatSourceOrSink(
                obj=fireplace,
                temperature=heat_source_cfg.get("temperature", 100.0),
                heating_rate=heat_source_cfg.get("heating_rate", 0.1),
                distance_threshold=heat_source_cfg.get("distance_threshold", 0.15),
                requires_toggled_on=heat_source_cfg.get("requires_toggled_on", False),
            )
            fireplace.add_state(heat_source_state)
            if fireplace._initialized:
                heat_source_state.initialize()
        
        if object_states.Temperature in fireplace.states:
            fireplace.states[object_states.Temperature].set_value(100.0)
        fireplace.fixed_base = True
        fireplace.keep_still()
    
    # Step a few times to let states settle
    for _ in range(5):
        og.sim.step()


def _reset_firewood_transforms(env):
    """
    Re-apply canonical poses / scales from TASK_OBJECTS after a state load.
    This keeps objects aligned even if the saved pkl was generated with
    different scales (e.g., after increasing log x-scale).
    """
    for name in ["fireplace", "log_center", "log_left", "target_object"]:
        obj = env.scene.object_registry("name", name)
        if obj is None:
            continue
        if name not in TASK_OBJECTS:
            print(f"Warning: {name} not found in TASK_OBJECTS, skipping transform reset")
            continue
        cfg = TASK_OBJECTS[name]
        # Restore pose from the config (check keys exist first)
        if "position" not in cfg or "orientation" not in cfg:
            print(f"Warning: {name} config missing position/orientation, skipping transform reset")
            continue
        try:
            obj.set_position_orientation(cfg["position"], cfg["orientation"])
        except Exception as e:
            print(f"Warning: Failed to set position/orientation for {name}: {e}")
            continue
        # Ensure scale matches the config in case the saved state had older values
        if "scale" in cfg:
            try:
                obj.set_scale(cfg["scale"])
            except Exception:
                pass  # Some objects may not expose set_scale; ignore silently


def reset_env(env):
    """
    Reset environment and load from saved state pkl file.
    Similar to pour_glass.py reset_env function.
    """
    env.reset()
    
    # CRITICAL: Initialize damage evaluators (which adds Temperature state to robot) BEFORE loading state
    # This ensures the robot structure matches what will be in the saved state
    # We need to step once to trigger damage evaluator initialization
    robot = env.robots[0]
    zero_action = th.zeros(robot.action_dim)
    env.step(zero_action)  # This triggers _initialize_damage_evaluators which adds Temperature state
    
    # Load state from pkl file
    state_path = "safe-manipulation-benchmark/resources/saved_states/firewood_init_state.pkl"
    try:
        with open(state_path, "rb") as f:
            state_flat_array = pickle.load(f)
        og.sim.load_state(state_flat_array, serialized=True)
    except AssertionError as e:
        if "Invalid state deserialization" in str(e):
            print("=" * 80)
            print("ERROR: Saved state file is incompatible with current code.")
            print(f"Error details: {str(e)}")
            print("=" * 80)
            print("SOLUTION: Re-save the state file:")
            print("  1. Run: python safe-manipulation-benchmark/teleop_scripts/teleop_firewood.py")
            print("  2. Wait for simulation to load")
            print("  3. Press 'S' key to save the state")
            print("  4. This will create a new state file compatible with the current code")
            print("=" * 80)
            raise
        else:
            raise

    # Re-apply transforms to match current config (covers saved states with old scales)
    _reset_firewood_transforms(env)
    
    # CRITICAL: Get all original positions IMMEDIATELY after loading state, before any sim steps
    # This prevents NaN quaternion issues from simulation instability
    robot = env.robots[0]
    robot_pos, robot_orn = robot.get_position_orientation()
    robot_joint_positions = robot.get_joint_positions()  # Save joint positions to restore later
    
    # Now sync robot and run sim steps
    robot.keep_still()
    for _ in range(10):
        robot.keep_still()
        og.sim.step()
    
    # Sync robot controller state after loading - prevents random movement
    robot.keep_still()
    og.sim.step()
    
    # Set gripper to closed by default (unless grasping something from saved state)
    # Create action to keep gripper closed and arm still
    keep_gripper_action = th.zeros(robot.action_dim)
    # Check if robot is grasping - if so, keep closed; otherwise also closed by default
    if robot.is_grasping().value == IsGraspingState.TRUE:
        keep_gripper_action[robot.gripper_action_idx[robot.default_arm]] = -1.0  # Close if grasping
        print("Gripper kept closed to maintain saved grasping state")
    else:
        keep_gripper_action[robot.gripper_action_idx[robot.default_arm]] = -1.0  # Closed by default
        print("Gripper set to closed (default)")
    
    for _ in range(10):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        env.step(keep_gripper_action)
        # Force restore after step to prevent any drift
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
    
    # Let simulation settle with env.step() for proper physics
    for _ in range(30):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        env.step(keep_gripper_action)
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
    
    # Restore robot position to ensure it stays fixed across episodes
    robot.set_position_orientation(robot_pos, robot_orn)
    robot.set_joint_positions(robot_joint_positions)
    # CRITICAL: Zero out all joint velocities to prevent residual movement
    robot.set_joint_velocities(th.zeros(robot.n_dof))
    
    # Reset the arm controller's internal state
    arm_controller = robot.controllers.get(f"arm_{robot.default_arm}")
    if arm_controller is not None:
        arm_controller.reset()
    
    # Final sync after position restoration - critical for IK controller
    robot.keep_still()
    for _ in range(10):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        og.sim.step()
    
    # Set fireplace fixed_base to True after settling
    fireplace = env.scene.object_registry("name", "fireplace")
    if fireplace is not None:
        fireplace.fixed_base = True
        # Keep fireplace still to ensure it stays fixed
        fireplace.keep_still()
        print("Fireplace fixed_base set to True")
    
    # One more keep_still to ensure controller is synced before teleoperation starts
    robot.keep_still()
    # Make sure fire / heat states are active after load
    _ensure_firewood_states(env)
    
    # Randomize robot pose (and log if holding it) by applying random delta noise
    target_object = env.scene.object_registry("name", "target_object")
    if target_object is not None and robot.is_grasping(candidate_obj=target_object).value == IsGraspingState.TRUE:
        print("Robot is holding target_object, randomizing pose...")
        # Get arm and gripper action indices
        arm_idx = robot.arm_control_idx[robot.default_arm]
        gripper_idx = robot.gripper_action_idx[robot.default_arm]
        
        # Apply random delta noise to arm for 10 steps while keeping gripper closed
        noise_scale = 0.05  # Small noise to avoid dropping the log
        for _ in range(5):
            # Sample random delta noise for arm action
            arm_noise = th.randn(len(arm_idx)) * noise_scale
            # Create action: arm noise + gripper closed
            random_action = th.zeros(robot.action_dim)
            random_action[arm_idx] = arm_noise
            random_action[gripper_idx] = -1.0  # Keep gripper closed
            
            env.step(random_action)
        
        # Let simulation settle after randomization
        robot.keep_still()
        for _ in range(10):
            # Keep gripper closed while settling
            settle_action = th.zeros(robot.action_dim)
            settle_action[gripper_idx] = -1.0
            env.step(settle_action)
            robot.keep_still()
        
        print("Pose randomization complete")
    else:
        print("Robot is not holding target_object, skipping pose randomization")


def get_visualization_config(robot_name: str):
    """
    Configuration for visualizing temperature / health for the firewood task.

    We track:
    - Robot end-effector and hand links
    - Fireplace
    - All logs (center, left, target)
    """
    target_objects_health_with_links = [
        f"{robot_name}@eef_link",
        f"{robot_name}@panda_hand",
        f"{robot_name}@panda_leftfinger",
        f"{robot_name}@panda_rightfinger",
        "fireplace@base_link",
        "log_center@base_link",
        "log_left@base_link",
        "target_object@base_link",
    ]
    target_objects_health = [
        robot_name,
        "fireplace",
        "log_center",
        "log_left",
        "target_object",
    ]
    # We will visualize temperature for these fully-qualified link names
    target_objects_temperature = target_objects_health_with_links
    return {
        "target_objects_health_with_links": target_objects_health_with_links,
        "target_objects_health": target_objects_health,
        "target_objects_temperature": target_objects_temperature,
    }


def __main__():

    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_hdf5_path', type=str, help='Target hdf5 path', default="resources/teleop_data/firewood.hdf5")
    parser.add_argument('--playback_hdf5_path', type=str, help='Output hdf5 path', default="resources/playback_data/firewood_playback.hdf5")
    parser.add_argument('--n_episodes', type=int, help='Number of episodes', default=1)
    parser.add_argument('--teleop', action='store_true', help='Teleoperate the robot')
    parser.add_argument('--playback', action='store_true', help='Playback the data')
    parser.add_argument('--visualize', action='store_true', help='Visualize the data')
    parser.add_argument('--compute_metrics', action='store_true', help='Compute metrics')
    parser.add_argument('--task_name', type=str, help='Task name', default="firewood")
    parser.add_argument('--live_feedback', action='store_true', help='Show live health graph window during teleop (use with --teleop)')
    args = parser.parse_args()

    seed = random.randint(0, 1000000)
    np.random.seed(seed)
    th.manual_seed(seed)

    # -------------------------
    # TELEOP DATA COLLECTION
    # -------------------------
    if args.teleop:
        if not os.path.exists(args.collect_hdf5_path):
            os.makedirs(os.path.dirname(args.collect_hdf5_path), exist_ok=True)

        # Load base config and adapt it to match teleop_firewood.py
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Scene: Rs_int, no robots included in the scene file
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_task_relevant_only": True,
        }

        # Franka robot (same as teleop_firewood)
        cfg["robots"][0] = {
            "type": "FrankaPanda",
            "name": "franka0",
            "position": [-0.85, -2.0, 0.0],
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth"],
            "action_normalize": False,
            "self_collisions": True,
            "controller_config": {
                "arm_0": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_0": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
            },
        }

        # External cameras – two cameras like shelve_item.py
        robot_name = cfg["robots"][0].get("name", "franka0")
        robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()
        # First camera: same as viewer camera position
        viewer_camera_pos = [-0.37351322174072266, -0.9105080366134644, 0.9984497427940369]
        viewer_camera_orn = [0.1866627037525177, 0.5293360948562622, 0.7805155515670776, 0.2752378284931183]
        # Second camera: new position
        second_camera_pos = [-0.5087745785713196, -3.052588701248169, 0.9984493851661682]
        second_camera_orn = [0.5276271104812622, 0.19144046306610107, 0.2822819948196411, 0.7779955267906189]
        
        EXTERNAL_CAMERA_CONFIGS = {
            "external_sensor_0": {
                "position": viewer_camera_pos,
                "orientation": viewer_camera_orn,
                "horizontal_aperture": 30.0,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
            },
            "external_sensor_1": {
                "position": second_camera_pos,
                "orientation": second_camera_orn,
                "horizontal_aperture": 30.0,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1",
            },
        }
        external_sensors_config = []
        for name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
            i = name.split("_")[-1]
            position = camera_cfg["position"]
            orientation = camera_cfg["orientation"]
            external_sensors_config.append(
                {
                    "sensor_type": "VisionSensor",
                    "name": f"external_sensor{i}",
                    "relative_prim_path": camera_cfg["relative_prim_path"],
                    "modalities": ["rgb", "seg_instance"],
                    "sensor_kwargs": {
                        "image_height": 720,
                        "image_width": 720,
                        "horizontal_aperture": camera_cfg["horizontal_aperture"],
                    },
                    "position": th.tensor(position, dtype=th.float32),
                    "orientation": th.tensor(orientation, dtype=th.float32),
                    "pose_frame": "world",
                }
            )
        cfg["env"]["external_sensors"] = external_sensors_config

        # Objects: use local TASK_OBJECTS (mirrors teleop_firewood.py)
        cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

        # Create environment and wrap with data collection
        env = DamageableEnvironment(configs=cfg)
        env = DamageableDataCollectionWrapper(
            env=env,
            output_path=args.collect_hdf5_path,
            only_successes=False,
            enable_dump_filters=False,
        )

        robot = env.robots[0]

        # Viewer camera (you can overwrite at runtime with TAB-print from teleop_firewood)
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([-0.37351322174072266, -0.9105080366134644, 0.9984497427940369]),
            orientation=th.tensor([0.1866627037525177, 0.5293360948562622, 0.7805155515670776, 0.2752378284931183]),
        )
        for _ in range(10):
            og.sim.step()

        # Keyboard Teleop
        action_generator = KeyboardRobotController(robot=robot)
        action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=lambda: env.reset(),
        )
        action_generator.print_keyboard_teleop_info()

        # ======================== Health visualization setup (if enabled) ========================
        enable_health_graph = args.live_feedback
        tracked_objects = {}
        
        if enable_health_graph:
            # Enable health visualization using the general environment method
            env.enable_health_visualization()
            
            # Get references to objects we want to track for health visualization
            for obj_name in ["log_center", "log_left", "target_object", "fireplace"]:
                obj = env.scene.object_registry("name", obj_name)
                if obj is not None:
                    tracked_objects[obj_name] = obj
                    print(f"Found object '{obj_name}' for health tracking")
                else:
                    print(f"Warning: Object '{obj_name}' not found in scene")
        # ====================================================================================

        # ======================== Data collection ========================
        n_episodes = args.n_episodes
        completed_episodes = 0
        while completed_episodes < n_episodes:
            print(f"Episode {completed_episodes} starts (target: {n_episodes})")
                        
            # Reset until robot health is 100
            reset_attempts = 0
            max_reset_attempts = 10
            while True:
                reset_env(env)
                reset_attempts += 1
                
                # Check robot health
                robot = env.robots[0]
                if hasattr(robot, "link_healths") and robot.link_healths:
                    min_health = min(robot.link_healths.values())
                    if min_health >= 100.0:
                        if reset_attempts > 1:
                            print(f"Robot health is 100 after {reset_attempts} reset attempts")
                        break
                    else:
                        print(f"Reset attempt {reset_attempts}: Robot health is {min_health:.2f} (expected 100.0), retrying reset...")
                        if reset_attempts >= max_reset_attempts:
                            print(f"Warning: After {max_reset_attempts} reset attempts, robot health is still {min_health:.2f}. Continuing anyway.")
                            break
                else:
                    # If link_healths not available, assume health is fine
                    break
            
            action_generator.current_keypress = None

            # If the robot is grasping, set the persistent gripper action to -1.0
            if robot.is_grasping().value == IsGraspingState.TRUE:
                action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
            else:
                # Default to closed if not grasping
                action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
                action_generator.gripper_direction[action_generator.binary_grippers[0]] = -1.0
            episode_starts = False

            print("Ready for teleoperation. Press TAB to end episode, BACKSPACE/DELETE to discard and reset.")
            breakpoint()
            # Default gripper action is 1.0
            discard_episode = False
            episode_step_count = 0
            init_skip_steps = 3
            while True:
                ret = action_generator.get_teleop_action()
                if isinstance(ret, tuple) and len(ret) == 2:
                    action, keypress_str = ret
                else:
                    action = ret
                    keypress_str = None
                
                if not episode_starts:
                    episode_starts = (action[:-1].sum() > 0).item() if len(action) > 1 else False
                
                # TAB: end episode and save
                if keypress_str and keypress_str.upper() == "TAB":
                    print("TAB pressed - ending episode")
                    breakpoint()
                    inp = input("Do you want to save the data? (y/n)")
                    if inp == "y":
                        print("Saving as task success being True")
                        env.task._success = True
                        break
                    else:
                        print("BACKSPACE/DELETE pressed - discarding current trajectory and resetting...")
                        breakpoint()
                        # Clear the current trajectory data without saving
                        steps_to_remove = len(env.current_traj_history)
                        env.current_traj_history = []
                        env.step_count -= steps_to_remove
                        print(f"Discarded {steps_to_remove} steps from current trajectory")
                        discard_episode = True
                        break
                
                # BACKSPACE/DELETE: discard current trajectory and reset
                if keypress_str and keypress_str.upper() in ("BACKSPACE", "DEL", "DELETE"):
                    print("BACKSPACE/DELETE pressed - discarding current trajectory and resetting...")
                    breakpoint()
                    # Clear the current trajectory data without saving
                    steps_to_remove = len(env.current_traj_history)
                    env.current_traj_history = []
                    env.step_count -= steps_to_remove
                    print(f"Discarded {steps_to_remove} steps from current trajectory")
                    discard_episode = True
                    break

                if episode_step_count == init_skip_steps:
                    # Update link positions and velocities for all damage evaluators
                    for obj in env.scene.objects:
                        if hasattr(obj, "track_damage") and obj.track_damage:
                            for evaluator in obj.damage_evaluators:
                                if evaluator.name == "mechanical":
                                    evaluator.update_link_positions_and_velocities()
                
                if episode_starts:
                    # print("action: ", action)
                    # print("telemoma_action: ", telemoma_action)
                    env.step(action.clone(), episode_step_count=episode_step_count, init_skip_steps=init_skip_steps)
                    episode_step_count += 1
                else:
                    # Step simulation even when episode hasn't started to allow keyboard input to be processed
                    og.sim.step()
                
                # Checking success: target_object log within xy tolerance of fireplace and gripper open
                if episode_starts:
                    target_object = env.scene.object_registry("name", "target_object")
                    fireplace = env.scene.object_registry("name", "fireplace")
                    
                    if target_object is not None and fireplace is not None:
                        # Get positions
                        target_pos, _ = target_object.get_position_orientation()
                        fireplace_pos, _ = fireplace.get_position_orientation()
                        
                        # Calculate xy distance only (ignore z)
                        distance_xy = th.norm((target_pos[:2] - fireplace_pos[:2])).item()
                        
                        # Tolerance: log should be close to fireplace horizontally
                        tolerance_xy = 0.25  # 25cm horizontal tolerance
                        
                        log_within_tolerance = distance_xy < tolerance_xy
                        gripper_open = robot.is_grasping(candidate_obj=target_object).value == IsGraspingState.FALSE
                        
                        if log_within_tolerance and gripper_open:
                            print("=" * 80)
                            print("SUCCESS: Target log is within xy tolerance of fireplace and gripper is open!")
                            print(f"  Distance (xy): {distance_xy:.3f} (tolerance: {tolerance_xy})")
                            print("=" * 80)
                            env.task._success = True
                            breakpoint()
                            inp = input("Do you want to save the data? (y/n)")
                            if inp == "y":
                                print("Saving as task success being True")
                                break
                            else:
                                print("Discarding current trajectory and resetting...")
                                # Clear the current trajectory data without saving
                                steps_to_remove = len(env.current_traj_history)
                                env.current_traj_history = []
                                env.step_count -= steps_to_remove
                                print(f"Discarded {steps_to_remove} steps from current trajectory")
                                discard_episode = True
                                break

            # Only count completed episodes (not discarded ones)
            if not discard_episode:
                completed_episodes += 1
                print(f"Episode completed ({completed_episodes}/{n_episodes})")
            else:
                print(f"Episode discarded, redoing...")
                # Use continue to restart the same episode (don't increment completed_episodes)
                continue

        env.save_data()
        print("Data saved")

        # Close live health visualization if it was enabled
        if enable_health_graph:
            env.disable_health_visualization()

        og.shutdown()

    # -------------------------
    # PLAYBACK
    # -------------------------
    if args.playback:
        # Load the config EXACTLY like teleop does
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Scene: Rs_int, no robots included in the scene file (EXACTLY like teleop)
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_task_relevant_only": True,
        }

        # Robot config (EXACTLY like teleop)
        cfg["robots"][0] = {
            "type": "FrankaPanda",
            "name": "franka0",
            "position": [-0.85, -2.0, 0.0],
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth"],
            "action_normalize": False,
            "self_collisions": True,
            "controller_config": {
                "arm_0": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_0": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
            },
        }

        robot_name = "franka0"
        robot_type = "FrankaPanda"
        image_height = 256
        image_width = 256

        # External cameras (EXACTLY like teleop)
        # First camera: same as viewer camera position
        viewer_camera_pos = [-0.37351322174072266, -0.9105080366134644, 0.9984497427940369]
        viewer_camera_orn = [0.1866627037525177, 0.5293360948562622, 0.7805155515670776, 0.2752378284931183]
        # Second camera: new position
        second_camera_pos = [-0.5087745785713196, -3.052588701248169, 0.9984493851661682]
        second_camera_orn = [0.5276271104812622, 0.19144046306610107, 0.2822819948196411, 0.7779955267906189]
        
        EXTERNAL_CAMERA_CONFIGS = {
            "external_sensor_0": {
                "position": viewer_camera_pos,
                "orientation": viewer_camera_orn,
                "horizontal_aperture": 30.0,
                "relative_prim_path": f"/controllable__damageable{robot_type.lower()}__{robot_name}/base_link/external_sensor0",
            },
            "external_sensor_1": {
                "position": second_camera_pos,
                "orientation": second_camera_orn,
                "horizontal_aperture": 30.0,
                "relative_prim_path": f"/controllable__damageable{robot_type.lower()}__{robot_name}/base_link/external_sensor1",
            },
        }
        external_sensors_config = []
        for name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
            i = name.split("_")[-1]
            position = camera_cfg["position"]
            orientation = camera_cfg["orientation"]
            external_sensors_config.append(
                {
                    "sensor_type": "VisionSensor",
                    "name": f"external_sensor{i}",
                    "relative_prim_path": camera_cfg["relative_prim_path"],
                    "modalities": ["rgb", "seg_instance"],
                    "sensor_kwargs": {
                        "image_height": image_height,
                        "image_width": image_width,
                        "horizontal_aperture": camera_cfg["horizontal_aperture"],
                    },
                    "position": th.tensor(position, dtype=th.float32),
                    "orientation": th.tensor(orientation, dtype=th.float32),
                    "pose_frame": "world",
                }
            )

        robot_sensor_config = {
            "VisionSensor": {
                "modalities": ["rgb", "seg_instance"],
                "sensor_kwargs": {
                    "image_height": image_height,
                    "image_width": image_width,
                },
            },
        }

        # Create a custom wrapper class that sets up firewood states during playback
        class FirewoodPlaybackWrapper(DamageableDataPlaybackWrapper):
            def _setup_playback_object_states(self):
                """Set up firewood-specific states after state is loaded"""
                _ensure_firewood_states(self.env)
        
        # Use DamageableDataPlaybackWrapper.create_from_hdf5 like shelve_item.py and pour_glass.py
        env = FirewoodPlaybackWrapper.create_from_hdf5(
            input_path=args.collect_hdf5_path,
            output_path=args.playback_hdf5_path,
            robot_obs_modalities=["proprio", "rgb", "seg_instance"],
            robot_sensor_config=robot_sensor_config,
            external_sensors_config=external_sensors_config,
            n_render_iterations=1,
            only_successes=False,
        )

        # Set viewer camera to match teleop position
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([-0.37351322174072266, -0.9105080366134644, 0.9984497427940369]),
            orientation=th.tensor([0.1866627037525177, 0.5293360948562622, 0.7805155515670776, 0.2752378284931183]),
        )
        for _ in range(10):
            og.sim.step()

        # Wrap playback_episode to print health after each episode
        original_playback_episode = env.playback_episode
        
        def playback_episode_with_health_printing(episode_id, record_data=True, video_writers=None, callback=None, replay_for_annotation=False, break_after_n_steps=100):
            """Wrapper that prints health after each episode"""
            result = original_playback_episode(
                episode_id=episode_id,
                record_data=record_data,
                video_writers=video_writers,
                callback=callback,
                replay_for_annotation=replay_for_annotation,
                break_after_n_steps=break_after_n_steps,
            )
            
            # Print health after episode completes
            print("\n" + "=" * 80)
            print(f"FINAL HEALTH AFTER EPISODE {episode_id}")
            print("=" * 80)
            
            # Print robot health
            if env.robots:
                robot = env.robots[0]
                if hasattr(robot, "link_healths"):
                    print(f"Robot ({robot.name}) health per link:")
                    for link_name, health_val in robot.link_healths.items():
                        print(f"  {link_name}: {health_val:.2f}")
                    min_health = min(robot.link_healths.values()) if robot.link_healths else None
                    if min_health is not None:
                        print(f"  Overall robot health (min): {min_health:.2f}")
                else:
                    print(f"Robot ({robot.name}) does not have link_healths attribute")
            
            # Print health for other damageable objects
            for obj in env.scene.objects:
                if hasattr(obj, "track_damage") and obj.track_damage:
                    robot_name = env.robots[0].name if env.robots else None
                    if obj.name != robot_name:
                        if hasattr(obj, "link_healths"):
                            print(f"\n{obj.name} health per link:")
                            for link_name, health_val in obj.link_healths.items():
                                print(f"  {link_name}: {health_val:.2f}")
                            min_health = min(obj.link_healths.values()) if obj.link_healths else None
                            if min_health is not None:
                                print(f"  Overall {obj.name} health (min): {min_health:.2f}")
            print("=" * 80 + "\n")
            
            return result
        
        # Override playback_episode method
        env.playback_episode = playback_episode_with_health_printing

        # Use the standard playback_dataset method
        env.playback_dataset(record_data=True)
        env.save_data()
        og.shutdown()

    # -------------------------
    # VISUALIZATION (temperature)
    # -------------------------
    if args.visualize:
        f = h5py.File(args.playback_hdf5_path, "r")
        robot_name = "franka0"
        camera_type = "external"
        # Use both external cameras
        camera_names = ["external_sensor0", "external_sensor1"]

        output_video_dir = "resources/videos/firewood"
        os.makedirs(output_video_dir, exist_ok=True)

        vis_cfg = get_visualization_config(robot_name)
        target_objects_health_with_links = vis_cfg["target_objects_health_with_links"]
        target_objects_health = vis_cfg["target_objects_health"]
        target_objects_temperature = vis_cfg["target_objects_temperature"]

        final_obj_healths = defaultdict(list)
        final_env_healths = []

        # Iterate over demos
        for demo_key in sorted(f["data"].keys()):
            if not demo_key.startswith("demo_"):
                continue
            demo_idx = int(demo_key.split("_")[-1])
            print(f"Episode: {demo_idx}")

            # Parse obs_info for seg_instance mapping (if needed later)
            obs_info_list = []
            for i in range(len(f[f"data/{demo_key}/info/obs_info"])):
                obs_info = json.loads(
                    f[f"data/{demo_key}/info/obs_info"][i].decode("utf-8")
                )
                obs_info_list.append(obs_info)

            # Health over time (per-link) – identical to shelve_item
            all_obj_healths = np.array(f[f"data/{demo_key}/obs/health"])
            health_list_link_names = f[f"data/{demo_key}"].attrs["health_list_link_names"]
            health = {}
            for obj_name in target_objects_health_with_links:
                # Check if this object has health data in the HDF5 file
                matching_indices = np.where(health_list_link_names == obj_name)[0]
                if len(matching_indices) > 0:
                    health[obj_name] = all_obj_healths[:, matching_indices[0]]
                    health[obj_name] = health[obj_name][1:]
                else:
                    # Skip objects that don't have health data
                    print(f"Warning: {obj_name} not found in health_list_link_names, skipping")
                    health[obj_name] = None

            # Aggregate health per object (min across links)
            for obj_name in target_objects_health:
                arrays = [v for k, v in health.items() if k.startswith(f"{obj_name}@") and v is not None]
                if arrays:
                    health[obj_name] = np.minimum.reduce(arrays)
                else:
                    health[obj_name] = None

            # Track final env health for debugging / metrics
            current_env_health = 0.0
            for obj_name in target_objects_health:
                if health[obj_name] is None:
                    continue
                final_obj_healths[obj_name].append(health[obj_name][-1])
                current_env_health += health[obj_name][-1]
            if len(target_objects_health) > 0:
                final_env_healths.append(
                    current_env_health / max(1, len(target_objects_health))
                )

            # -------------------------
            # TEMPERATURE FROM damage_info
            # -------------------------
            temperature = {name: [] for name in target_objects_temperature}
            dmg_info_ds = f[f"data/{demo_key}/info/damage_info"]
            for i in range(len(dmg_info_ds)):
                dmg = json.loads(dmg_info_ds[i].decode("utf-8"))
                for full_name in target_objects_temperature:
                    obj_name, link_name = full_name.split("@")
                    temp_val = None
                    try:
                        temp_val = dmg[obj_name][link_name]["thermal"]["temperature"]
                    except Exception:
                        temp_val = None
                    # Use NaN for missing temperature so plotting can still proceed
                    if temp_val is None:
                        temp_val = float("nan")
                    temperature[full_name].append(temp_val)

            # Convert per-link temperature to per-object (min over links)
            obj_temperature = {}
            for obj_name in target_objects_health:
                arrays = [
                    np.array(vals)
                    for link, vals in temperature.items()
                    if link.startswith(f"{obj_name}@")
                ]
                if arrays:
                    obj_temperature[obj_name] = np.nanmin(np.vstack(arrays), axis=0)
                else:
                    # If no thermal data, default to NaNs
                    T = len(next(iter(temperature.values())))
                    obj_temperature[obj_name] = np.full(T, np.nan)

            # Process each camera
            for camera_name in camera_names:
                # Camera frames
                output_video_path = os.path.join(
                    output_video_dir, f"demo_{demo_idx}_{camera_name}_camera_video"
                )
                imgs_ds = f[f"data/{demo_key}/obs/{camera_type}::{camera_name}::rgb"]
                imgs = np.array(imgs_ds[1:])
                if len(imgs) == 0:
                    print(f"No images found for {demo_key} camera {camera_name}")
                    continue

                # Save plain RGB video
                save_rgb_camera_video(output_video_path=output_video_path, imgs=imgs)

                # Save temperature video (RGB + temperature history)
                temp_video_path = os.path.join(
                    output_video_dir, f"demo_{demo_idx}_{camera_name}_temperature_video.mp4"
                )
                save_rgb_temperature_video(
                    output_video_path=temp_video_path,
                    imgs=imgs,
                    target_objects=list(obj_temperature.keys()),
                    temperature=obj_temperature,
                )

        if final_env_healths:
            print("\nAverage environment health across episodes:")
            print(np.mean(final_env_healths))

        og.shutdown()


if __name__ == "__main__":
    __main__()

