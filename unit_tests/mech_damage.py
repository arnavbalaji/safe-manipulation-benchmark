"""
Minimal mech damage teleop scene:
- Loads a coffee table (aoojzy) and a glass plate (pkkgzc)
- Spawns a TIAGo robot facing the table
- Places the plate OnTop of the table
- Runs keyboard teleoperation for 500 steps in a DamageableEnvironment
"""

# Set matplotlib backend to non-interactive before importing pyplot (if ever used downstream)
import matplotlib
matplotlib.use('Agg')

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.object_states import OnTop
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
from omnigibson.utils.sim_utils import place_base_pose
import torch as th
import os
import cv2
import numpy as np
import subprocess
import json
 

from safety_benchmark.damageable_env import DamageableEnvironment
# from safety_benchmark.params.test_params import PARAMS


# Performance knobs similar to other teleop scripts
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

PARAMS = {
    "tiago_robot": {
        "damage_evaluators": ["mechanical", "electrical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 1e10,
            "scale": 1e-10,
            "instant_coefficient": 1.0,
            "creep_coefficient": 1.0,
            "object_type": "brittle",
            "link_thresholds": {
                "arm": {
                    "damage_threshold": 200.0,
                    "scale": 0.0001,
                },
                # "base": {
                #     "damage_threshold": 3.0,
                #     "scale": 0.001,
                # },
                # "wheel": {
                #     "damage_threshold": 10.0,
                #     "scale": 0.01,
                # },
                "gripper": {
                    "damage_threshold": 200.0,
                    "scale": 0.0001,
                }
            }
        },
        "electrical": {
            "damage_threshold": 0.0,  # Minimum particles to cause damage
            "scale": 0.001,  # Damage amount when threshold is exceeded
            "water_system_name": "sludge",
        }
    },
    "plate": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 200.0,
            "scale": 0.5,  # Increased scale for more aggressive damage
            "instant_coefficient": 1.0,
            "creep_coefficient": 0.0,
            "object_type": "brittle",
        }
    },
    "baseball": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 1000,
            "scale": 0.01,
            "instant_coefficient": 1.0,
            "creep_coefficient": 1.0,
            "object_type": "ductile",
        }
    },
    "glass": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 200.0,
            "scale": 0.5,
            "instant_coefficient": 1.0,
            "creep_coefficient": 0.0,
            "object_type": "brittle",
        }
    },
    "book": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 2000,
            "scale": 0.01,
            "instant_coefficient": 1.0,
            "creep_coefficient": 0.0,
            "object_type": "brittle",
        }
    },
    "drawer": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 1000.0,
            "scale": 1e-18,
            "instant_coefficient": 1.0,
            "creep_coefficient": 0.0,
            "link_thresholds": {
                "link_3": {
                    "damage_threshold": 1000.0,
                    "scale": 0.01,
                }
            },
            "object_type": "brittle",
        }
    },
    "default": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.0,
            "scale": 1.0,
            "instant_coefficient": 1.0,
            "creep_coefficient": 1.0,
        }
    },
}


OBJECT_CONFIGS = {
    "coffee_table": {
        "type": "DatasetObject",
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [0.2, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.0],
        # Wood-like table damage behavior
        "damage_params": PARAMS.get("coffee_table", PARAMS["default"]),
    },
}

# Target object templates for easy integration of new targets
TARGET_OBJECT_CONFIGS = {
    # Default plate used in current unit test
    "plate": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "plate",
        "model": "ntedfx",  # glass plate from BEHAVIOR
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 1.0, 1.0],
        # Reuse bowl fragility params for glass-like behavior
        "damage_params": PARAMS["plate"],
    },
    # Example alternatives for quick swapping
    "glass": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "water_glass",
        "model": "uwtdng",
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [2.0, 2.0, 2.0],
        "damage_params": PARAMS["glass"],
    },
    "baseball": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "baseball",
        "model": "zanmar",
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 1.0, 1.0],
        "damage_params": PARAMS["baseball"],
    },
    "book": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "paperback_book",
        "model": "xxknda",
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.5, 1.5, 1.5],
        "damage_params": PARAMS["book"],
    },
    "door": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "door",
        # "model": "ofgpit",
        "model": "ofgodc",
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, -0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 0.5],
        "fixed_base": False,
        "damage_params": PARAMS.get("door", PARAMS["default"]),
    },
    "drawer": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "bottom_cabinet",
        "model": "pkdnbu",
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.5],
        "fixed_base": True,
        "damage_params": PARAMS["drawer"],
    },
}


def main():
    og.log.info(
        f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80
        if main.__doc__
        else f"Demo {__file__}"
    )

    # Easy integration knobs
    # - Choose target object template key from TARGET_OBJECT_CONFIGS (e.g., "plate", "bowl", "baseball")
    target_object_key = "plate"
    # - Toggle whether to load from a previously saved scene file or build fresh from scene_cfg
    use_saved_scene_file = True
    # - Toggle whether to track robot health/strain (only meaningful when target_object_key is not None)
    track_robot_health = True
    # - Explicit head joint init [head_1_joint (yaw), head_2_joint (pitch)] in radians. Set to None to skip.
    # head_joint_init = [0.0, -0.45]

    # Derived names and paths based on target object
    base_name = target_object_key if target_object_key is not None else "robot_only"
    unit_tests_dir = "safe-manipulation-benchmark/unit_tests"
    saved_path = os.path.join(unit_tests_dir, f"json_files/unit_test_{base_name}_mech.json")
    scene_cfg = {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "include_robots": False,
        "load_task_relevant_only": True,
    }
    robot0_cfg = {
        "type": "Tiago",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        # Position near and facing the table at origin
        "position": [0.2, 1.0, 0.0],
        "orientation": [0, 0, -1, 1],  # Facing the table
        "grasping_mode": "assisted",
        "damage_params": PARAMS["tiago_robot"],
    }

    _can_load_saved = (target_object_key is not None) and use_saved_scene_file and os.path.exists(saved_path)
    # Special case: if target is plate, load the scene exactly like rl_test.py
    if target_object_key == "plate":
        save_path = "lift_test.json"
        if not os.path.exists(save_path):
            print(f"Error: Saved state file '{save_path}' not found!")
            print("Please run test_save.py first to create the state file.")
            return
        print(f"Loading simulation state from: {save_path}")

        # Overwrite damage params in lift_test.json to current PARAMS before loading (same mapping as other scenes)
        try:
            with open(save_path, "r") as f:
                scene_dict = json.load(f)
            init_info = scene_dict["objects_info"]["init_info"]

            def set_params(entry_args, params_dict):
                entry_args["params"] = params_dict

            for key, entry in init_info.items():
                args = entry.get("args", {})
                name = args.get("name", "")
                category = args.get("category", "")
                class_name = entry.get("class_name", "")

                if name.startswith("robot_") or class_name == "DamageableTiago":
                    set_params(args, PARAMS["tiago_robot"])  # robot
                elif name == "coffee_table":
                    set_params(args, PARAMS.get("coffee_table", PARAMS["default"]))
                elif name == "target_object" or name == "glass_plate" or category == "plate":
                    # Use the currently selected target's params if available, else fallback by category or default
                    if target_object_key in PARAMS:
                        set_params(args, PARAMS[target_object_key])
                    elif category in PARAMS:
                        set_params(args, PARAMS[category])
                    else:
                        set_params(args, PARAMS["default"])
                else:
                    # Generic: if category maps directly to PARAMS, apply it
                    if category in PARAMS:
                        set_params(args, PARAMS[category])

            with open(save_path, "w") as f:
                json.dump(scene_dict, f)
        except Exception as e:
            print(f"Warning: Could not rewrite params in {save_path}: {e}")

        cfg = {"scene": {"type": "Scene", "scene_file": save_path}}
    elif target_object_key is None:
        # Build scene with only robot and coffee table
        cfg = dict(scene=scene_cfg, robots=[robot0_cfg], objects=[OBJECT_CONFIGS["coffee_table"]])
    elif _can_load_saved:
        # Overwrite damage params in the saved scene JSON to current values before loading
        with open(saved_path, "r") as f:
            scene_dict = json.load(f)
        init_info = scene_dict["objects_info"]["init_info"]

        def set_params(entry_args, params_dict):
            entry_args["params"] = params_dict

        # Map names / categories to current PARAMS
        for key, entry in init_info.items():
            args = entry.get("args", {})
            name = args.get("name", "")
            category = args.get("category", "")
            class_name = entry.get("class_name", "")

            if name.startswith("robot_") or class_name == "DamageableTiago":
                set_params(args, PARAMS["tiago_robot"])  # robot
            elif name == "coffee_table":
                set_params(args, PARAMS.get("coffee_table", PARAMS["default"]))
            elif name == "target_object":
                # Use the currently selected target's params if available, else fallback by category or default
                if target_object_key in PARAMS:
                    set_params(args, PARAMS[target_object_key])
                elif category in PARAMS:
                    set_params(args, PARAMS[category])
                else:
                    set_params(args, PARAMS["default"])
            else:
                # Generic: if category maps directly to PARAMS, apply it
                if category in PARAMS:
                    set_params(args, PARAMS[category])

        with open(saved_path, "w") as f:
            json.dump(scene_dict, f)

        # Load only from scene file (no robots or objects provided explicitly)
        cfg = {"scene": {"type": "Scene", "scene_file": saved_path}}
    else:
        # Build fresh scene from configs
        cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
        target_cfg = TARGET_OBJECT_CONFIGS[target_object_key]
        if target_object_key == "door" or target_object_key == "drawer":
            cfg["objects"] = [target_cfg]
        else:
            cfg["objects"] = [OBJECT_CONFIGS["coffee_table"], target_cfg]

    # Match rl_test.py: ensure a fresh simulator before creating the environment
    if og.sim is None:
        minimal_cfg = {
            "env": {"action_timestep": 1.0 / 60.0, "physics_timestep": 1.0 / 120.0},
            "scene": {"type": "Scene"},
            "robots": [],
        }
        _tmp_env = DamageableEnvironment(configs=minimal_cfg)
        _tmp_env.reset()
        og.clear()
    else:
        og.sim.stop()
        og.clear()

    # Create the environment
    env = DamageableEnvironment(configs=cfg)
    # Reset immediately after creation, like rl_test.py
    env.reset()

    # Configure robot controllers similar to other teleop scripts
    robot = env.robots[0]
    controller_choices = {
        "base": "HolonomicBaseJointController",
        "arm_left": "InverseKinematicsController",
        "arm_right": "InverseKinematicsController",
        "gripper_left": "MultiFingerGripperController",
        "gripper_right": "MultiFingerGripperController",
        "camera": "JointController",
        "trunk": "JointController",
    }
    controller_config = {comp: {"name": name} for comp, name in controller_choices.items()}
    controller_config["gripper_left"]["inverted"] = True
    controller_config["gripper_right"]["inverted"] = True
    # (No IK primitives; default controller modes are sufficient)
    # Increase gripper force by raising joint stiffness / damping on the gripper controllers
    controller_config["gripper_left"]["motor_type"] = "position"
    controller_config["gripper_right"]["motor_type"] = "position"
    controller_config["gripper_left"]["isaac_kp"] = 4000.0
    controller_config["gripper_left"]["isaac_kd"] = 2000.0
    controller_config["gripper_right"]["isaac_kp"] = 4000.0
    controller_config["gripper_right"]["isaac_kd"] = 2000.0
    
    robot.reload_controllers(controller_config=controller_config)

    # Increase friction on gripper fingers to reduce slip during grasp
    try:
        for _link_name, _link in robot.links.items():
            _n = _link_name.lower()
            if ("gripper" in _n) or ("finger" in _n):
                try:
                    _link.set_attribute("physxMaterial:staticFriction", 2.0)
                    _link.set_attribute("physxMaterial:dynamicFriction", 2.0)
                except Exception:
                    pass
    except Exception:
        pass

    # Persist initial state after controller reload
    env.scene.update_initial_file()

    # Reset before teleoperation (robot already loaded from scene); optional extra reset for clean start
    robot.reset()

    # Set TIAGo head link poses at start (world frame)
    robot.set_joint_positions(th.tensor([-0.5, -0.75]), indices=robot.camera_control_idx)
    # Camera teleoperation matching teleop_tiago_bowl.py
    class CustomCameraMover(CameraMover):
        @property
        def input_to_command(self):
            return {
                lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
                lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
                # Exclude T to avoid conflict; use G for vertical move
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }

    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()

    # If this is a fresh scene, place the target object appropriately (when present); otherwise set camera pose
    if (target_object_key is not None) and (not _can_load_saved):
        target_obj = env.scene.object_registry("name", "target_object")
        if target_object_key not in ("door", "drawer"):
            table_obj = env.scene.object_registry("name", "coffee_table")
            assert target_obj.states[OnTop].set_value(table_obj, True), "Failed to set OnTop state for target object"
        else:
            # Land the door / drawer at ground height preserving its current yaw
            pos, quat = target_obj.get_position_orientation()
            pos_t = th.tensor(pos)
            pos_t[2] = 0.0
            place_base_pose(target_obj, pos_t, quat=th.tensor(quat))
            target_obj.keep_still()
    elif _can_load_saved:
        # Set viewer camera pose to last printed values
        cam_pos = th.tensor([-2.008840799331665, 0.2598961591720581, 1.1361593008041382])
        cam_quat = th.tensor([0.4495110809803009, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452])
        og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat)

    # Set starting viewer camera pose to user-specified values
    start_cam_pos = th.tensor([-1.3225984573364258, 0.2645236849784851, 0.9981386065483093])
    start_cam_quat = th.tensor([0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452])
    og.sim.viewer_camera.set_position_orientation(position=start_cam_pos, orientation=start_cam_quat)

    # Let physics settle a bit
    for _ in range(20):
        og.sim.step()

    # Keyboard teleop
    action_generator = KeyboardRobotController(robot=robot)
    # (Removed IK helpers and bindings)
    
    # Register TAB key to print base + EEF poses (world/body) and breakpoint
    def breakpoint_and_print_poses():
        print("=== TAB callback triggered ===", flush=True)
        # Base poses
        base_pos_w, base_quat_w = robot.get_position_orientation()
        base_pos_body = th.tensor([0.0, 0.0, 0.0])
        base_quat_body = th.tensor([0.0, 0.0, 0.0, 1.0])
        print("=== Robot Base Pose ===")
        print(f"World  pos: {base_pos_w.tolist()}  quat: {base_quat_w.tolist()}")
        print(f"Body   pos: {base_pos_body.tolist()}  quat: {base_quat_body.tolist()}")

        # Viewer camera pose
        cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
        print("=== Viewer Camera Pose ===")
        print(f"World  pos: {cam_pos.tolist()}  quat: {cam_quat.tolist()}")

        # End-effector poses for each arm (world and body frames)
        for arm in getattr(robot, 'arm_names', ["right"]):
            eef_pos_w = robot.get_eef_position(arm)
            eef_quat_w = robot.get_eef_orientation(arm)
            eef_pos_body = robot.get_relative_eef_position(arm)
            eef_quat_body = robot.get_relative_eef_orientation(arm)
            print(f"=== {arm} EEF Pose ===")
            print(f"World  pos: {eef_pos_w.tolist()}  quat: {eef_quat_w.tolist()}")
            print(f"Body   pos: {eef_pos_body.tolist()}  quat: {eef_quat_body.tolist()}")

        # Also print current head joint positions
        h1 = robot.links["head_1_link"].get_position_orientation()
        h2 = robot.links["head_2_link"].get_position_orientation()
        print("=== Head Link Positions ===")
        print(f"head_1_link: {h1}")
        print(f"head_2_link: {h2}")

        # Save current scene config/state
        save_dir = "safe-manipulation-benchmark/unit_tests"
        os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"json_files/unit_test_{base_name}_mech_temp.json")
        save_path = f"lift_test.json"
        # Save current scene state to JSON (same pattern as test_save_load_simple.py)
        og.sim.save([save_path])
        print(f"✅ Saved scene to: {save_path}")

        breakpoint()

    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Print base & EEF poses (world/body) and breakpoint",
        callback_fn=breakpoint_and_print_poses,
    )
    action_generator.print_keyboard_teleop_info()
    print("Running mech damage teleop. Press ESC to quit.")

    max_steps = 500
    step = 0
    fps = 30
    frames = []
    # Ego camera recording
    frames_ego = []
    ego_key = None
    # Health tracking
    robot_healths = []
    target_healths = []
    # Resolve target reference if applicable
    target_ref = None
    if target_object_key is not None:
        target_ref = env.scene.object_registry("name", "target_object")
        if target_ref is None:
            target_ref = env.scene.object_registry("name", "glass_plate")
        if target_ref is None:
            target_ref = env.scene.object_registry("name", base_name)
    # Enable CCD and raise friction on target to further reduce slip
    try:
        if target_ref is not None:
            for _tlink in target_ref.links.values():
                try:
                    _tlink.ccd_enabled = True
                except Exception:
                    pass
                try:
                    _tlink.set_attribute("physxMaterial:staticFriction", 1.5)
                    _tlink.set_attribute("physxMaterial:dynamicFriction", 1.5)
                except Exception:
                    pass
    except Exception:
        pass
    # Force metrics tracking (impact and sustained per step)
    robot_strains = []
    target_strains = []
    # Damage status per step (for dynamic overlay color)
    target_statuses = []
    while step != max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1

        # Capture viewer camera RGB and store for video
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        # Scale to at least 1280x720 (HD)
        rgb_np = cv2.resize(rgb_np, (1080, 720))
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Capture robot ego camera RGB (dynamic robot name, eyes:Camera:0)
        rob_obs = robot.get_obs()[0]
        if ego_key is None:
            for k in rob_obs.keys():
                if ":eyes:Camera:0" in k and "rgb" in rob_obs[k]:
                    ego_key = k
                    break
        if ego_key is not None and ego_key in rob_obs:
            ego_rgb = rob_obs[ego_key]["rgb"]
            ego_np = ego_rgb.cpu().numpy()[:, :, :3]
            ego_bgr = cv2.cvtColor(ego_np, cv2.COLOR_RGB2BGR)
            ego_bgr = cv2.resize(ego_bgr, (1080, 720))
            # Dynamic colored border based on current damage status
            if target_ref is not None:
                status = str(target_ref.damage_status).lower()
                if status in ("major", "critical"):
                    bgr = (0, 0, 255)
                elif status in ("minor",):
                    bgr = (0, 255, 255)
                else:
                    bgr = (0, 255, 0)
            else:
                bgr = (0, 255, 0)
            border = 30  # 30px at 1080x720 -> ~10px after scaling to 360x240
            ego_bgr = cv2.copyMakeBorder(ego_bgr, border, border, border, border, cv2.BORDER_CONSTANT, value=bgr)
            frames_ego.append(ego_bgr)

        # Record health (conditionally track robot based on track_robot_health)
        should_track_robot = track_robot_health or (target_object_key is None)
        if should_track_robot:
            robot_healths.append(float(getattr(robot, "health", 100.0)))
        if target_ref is not None:
            target_healths.append(float(getattr(target_ref, "health", 100.0)))
            # Record current damage status string
            target_statuses.append(getattr(target_ref, "damage_status"))

        # Record strain (per-step): take max over per-link last_strain_by_link after this env step
        # For robot, only include links with "arm" or "gripper" in their names
        if should_track_robot:
            r_dmg = robot.damage_evaluators[0]
            if hasattr(r_dmg, "last_strain_by_link") and len(getattr(r_dmg, "last_strain_by_link", {})) > 0:
                # Filter to only arm/gripper links
                arm_gripper_strains = {
                    link_name: strain
                    for link_name, strain in r_dmg.last_strain_by_link.items()
                    if "arm" in link_name.lower() or "gripper" in link_name.lower()
                }
                if arm_gripper_strains:
                    robot_strains.append(float(max(arm_gripper_strains.values())))
                else:
                    # Fallback if no arm/gripper links found
                    robot_strains.append(float(max(r_dmg.last_strain_by_link.values())))
            else:
                # Fallbacks for older evaluators
                if hasattr(r_dmg, "get_current_env_step_strain"):
                    robot_strains.append(float(r_dmg.get_current_env_step_strain()))
                elif hasattr(r_dmg, "strain_values") and len(r_dmg.strain_values) > 0:
                    robot_strains.append(float(r_dmg.strain_values[-1]))
                else:
                    robot_strains.append(0.0)

        if target_ref is not None:
            t_dmg = target_ref.damage_evaluators[0]
            if hasattr(t_dmg, "last_strain_by_link") and len(getattr(t_dmg, "last_strain_by_link", {})) > 0:
                if target_object_key == "drawer":
                    # Only track strain from link_3 when the target is a drawer
                    link3_val = None
                    for k, v in t_dmg.last_strain_by_link.items():
                        if k.lower() == "link_3" or "link_3" in k.lower():
                            link3_val = float(v)
                            break
                    target_strains.append(0.0 if link3_val is None else link3_val)
                else:
                    target_strains.append(float(max(t_dmg.last_strain_by_link.values())))
            else:
                if hasattr(t_dmg, "get_current_env_step_strain"):
                    target_strains.append(float(t_dmg.get_current_env_step_strain()))
                elif hasattr(t_dmg, "strain_values") and len(t_dmg.strain_values) > 0:
                    target_strains.append(float(t_dmg.strain_values[-1]))
                else:
                    target_strains.append(0.0)

    # Clean shutdown
    camera_mover.clear()

    # Write camera video to unit_tests/videos as mp4
    videos_dir = os.path.join("safe-manipulation-benchmark", "unit_tests", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    avi_path = os.path.join(videos_dir, f"{base_name}_camera_obs.avi")
    mp4_path = os.path.join(videos_dir, f"{base_name}_camera_obs.mp4")
    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()
        # Convert AVI to MP4 via ffmpeg
        subprocess.run(["ffmpeg", "-y", "-i", avi_path, "-c:v", "mpeg4", mp4_path], check=True)
        # Remove AVI
        os.remove(avi_path)

    # Write ego camera video
    avi_ego = os.path.join(videos_dir, f"{base_name}_ego_obs.avi")
    mp4_ego = os.path.join(videos_dir, f"{base_name}_ego_obs.mp4")
    if len(frames_ego) > 0:
        he, we = frames_ego[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw_e = cv2.VideoWriter(avi_ego, fourcc, fps, (we, he))
        for f in frames_ego:
            vw_e.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw_e.release()
        subprocess.run(["ffmpeg", "-y", "-i", avi_ego, "-c:v", "mpeg4", mp4_ego], check=True)
        os.remove(avi_ego)

    # Build animated health plot using Matplotlib
    health_mp4 = os.path.join(videos_dir, f'{base_name}_healths.mp4')
    should_track_robot = track_robot_health or (target_object_key is None)
    
    if should_track_robot and len(robot_healths) > 0 and (target_object_key is not None) and len(target_healths) > 0:
        # Both robot and target
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = max(len(robot_healths), len(target_healths))
        # Clamp health plot to [0, 100]
        y_min = 0.0
        y_max = 100.0

        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
        line_p, = ax.plot([], [], lw=6, color='tab:orange', label=target_object_key.capitalize())
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Health', fontsize=20)
        ax.set_title('Health Over Time', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_health():
            line_r.set_data([], [])
            line_p.set_data([], [])
            return line_r, line_p

        def animate_health(i):
            x = [k / fps for k in range(1, i + 2)]
            y_r = robot_healths[: i + 1]
            y_p = target_healths[: i + 1]
            line_r.set_data(x, y_r)
            line_p.set_data(x, y_p)
            return line_r, line_p

        ani = animation.FuncAnimation(
            fig, animate_health, init_func=init_health, frames=T, interval=1000 / fps, blit=True
        )
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)
    elif should_track_robot and len(robot_healths) > 0 and (target_object_key is None):
        # Robot only
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = len(robot_healths)
        y_min = 0.0
        y_max = 100.0
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Health', fontsize=20)
        ax.set_title('Health Over Time', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_health_r():
            line_r.set_data([], [])
            return line_r,

        def animate_health_r(i):
            x = [k / fps for k in range(1, i + 2)]
            y_r = robot_healths[: i + 1]
            line_r.set_data(x, y_r)
            return line_r,

        ani = animation.FuncAnimation(fig, animate_health_r, init_func=init_health_r, frames=T, interval=1000 / fps, blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)
    elif not should_track_robot and (target_object_key is not None) and len(target_healths) > 0:
        # Target only (when track_robot_health is False)
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = len(target_healths)
        y_min = 0.0
        y_max = 100.0
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_p, = ax.plot([], [], lw=6, color='tab:orange', label=target_object_key.capitalize())
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Health', fontsize=20)
        ax.set_title('Health Over Time', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_health_p():
            line_p.set_data([], [])
            return line_p,

        def animate_health_p(i):
            x = [k / fps for k in range(1, i + 2)]
            y_p = target_healths[: i + 1]
            line_p.set_data(x, y_p)
            return line_p,

        ani = animation.FuncAnimation(fig, animate_health_p, init_func=init_health_p, frames=T, interval=1000 / fps, blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)

    # Create side-by-side combined video (camera left, health right)
    # Overlay ego video onto top-right corner of main camera video, then hstack with health plot
    cam_with_ego_mp4 = os.path.join(videos_dir, f'{base_name}_camera_with_ego.mp4')
    if os.path.exists(mp4_path) and os.path.exists(mp4_ego):
        # Ego frames already contain dynamic border and are possibly larger due to copyMakeBorder.
        # Scale ego down and overlay without additional padding.
        subprocess.run([
            'ffmpeg', '-y',
            '-i', mp4_path,
            '-i', mp4_ego,
            '-filter_complex',
            '[0:v]setsar=1[base];[1:v]scale=360:240,setsar=1[ego];[base][ego]overlay=W-w-20:20[out]',
            '-map','[out]',
            '-c:v','mpeg4','-q:v','5', cam_with_ego_mp4
        ], check=True)

    # Rename final combined video to {object}_with_health.mp4
    combined_mp4 = os.path.join(videos_dir, f'{base_name}_with_health.mp4')
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(health_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', cam_with_ego_mp4,
            '-i', health_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_mp4
        ], check=True)

    # Build single strain plot video (robot and target) with damage thresholds
    strain_mp4 = os.path.join(videos_dir, f'{base_name}_strain.mp4')
    should_track_robot = track_robot_health or (target_object_key is None)
    
    if should_track_robot and len(robot_strains) > 0:
        # Robot tracking enabled
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T_strain = max(len(robot_strains), len(target_strains) if target_object_key is not None else 0)

        fig, ax = plt.subplots(figsize=(9.6, 5.4))

        line_r, = ax.plot([], [], lw=4, color='tab:blue', label='Robot')
        line_t = None
        if target_object_key is not None and len(target_strains) > 0:
            line_t, = ax.plot([], [], lw=4, color='tab:orange', label=target_object_key.capitalize())

        ax.set_xlim(0, max(1, T_strain) / fps)

        # Threshold lines (dotted)
        # Use arm or gripper link threshold (they are the same)
        robot_mech_params = PARAMS["tiago_robot"]["mechanical"]
        link_thresholds = robot_mech_params.get("link_thresholds", {})
        robot_thresh = None
        if "arm" in link_thresholds:
            robot_thresh = link_thresholds["arm"]["damage_threshold"]
        elif "gripper" in link_thresholds:
            robot_thresh = link_thresholds["gripper"]["damage_threshold"]
        else:
            # Fallback to main threshold if no arm/gripper thresholds
            robot_thresh = robot_mech_params["damage_threshold"]
        
        target_thresh = None
        if target_object_key is not None:
            target_params = PARAMS.get(target_object_key, PARAMS["default"])
            target_thresh = target_params["mechanical"]["damage_threshold"]

        # y-limits must include thresholds so dotted lines are visible
        extra = []
        if robot_thresh is not None:
            extra.append(robot_thresh)
        if target_thresh is not None:
            extra.append(target_thresh)
        all_strains = robot_strains + (target_strains if target_object_key is not None else [])
        y_min = 0.0
        y_max = max(all_strains + extra + [0.0])
        if y_min == y_max:
            y_min, y_max = (0.0, 1.0)
        ax.set_ylim(y_min, y_max * 1.1)

        if robot_thresh is not None:
            ax.axhline(y=robot_thresh, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Robot Threshold ({robot_thresh})')
        if target_thresh is not None:
            ax.axhline(y=target_thresh, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'{target_object_key.capitalize()} Threshold ({target_thresh})')

        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Strain (proxy)', fontsize=16)
        ax.set_title('Strain Over Time', fontsize=20)
        ax.legend(loc='best')
        plt.tight_layout()

        def init_strain():
            line_r.set_data([], [])
            if line_t is not None:
                line_t.set_data([], [])
            return (line_r, line_t) if line_t is not None else (line_r,)

        def animate_strain(i):
            x = [k / fps for k in range(1, i + 2)]
            y_r = robot_strains[: i + 1]
            line_r.set_data(x, y_r)
            if line_t is not None:
                y_t = target_strains[: i + 1]
                line_t.set_data(x, y_t)
                return line_r, line_t
            return line_r,

        ani = animation.FuncAnimation(fig, animate_strain, init_func=init_strain, frames=T_strain, interval=1000 / fps, blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(strain_mp4, writer=writer)
        plt.close(fig)
    elif not should_track_robot and target_object_key is not None and len(target_strains) > 0:
        # Target only (when track_robot_health is False)
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T_strain = len(target_strains)
        fig, ax = plt.subplots(figsize=(9.6, 5.4))

        line_t, = ax.plot([], [], lw=4, color='tab:orange', label=target_object_key.capitalize())

        ax.set_xlim(0, max(1, T_strain) / fps)

        y_min = 0.0
        y_max = max(target_strains + [0.0])
        if y_min == y_max:
            y_min, y_max = (0.0, 1.0)
        ax.set_ylim(y_min, y_max * 1.1)

        # Threshold line (dotted) for target only
        target_params = PARAMS.get(target_object_key, PARAMS["default"])
        target_thresh = target_params["mechanical"]["damage_threshold"]
        ax.axhline(y=target_thresh, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'{target_object_key.capitalize()} Threshold ({target_thresh})')

        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Strain (proxy)', fontsize=16)
        ax.set_title('Strain Over Time', fontsize=20)
        ax.legend(loc='best')
        plt.tight_layout()

        def init_strain():
            line_t.set_data([], [])
            return line_t,

        def animate_strain(i):
            x = [k / fps for k in range(1, i + 2)]
            y_t = target_strains[: i + 1]
            line_t.set_data(x, y_t)
            return line_t,

        ani = animation.FuncAnimation(fig, animate_strain, init_func=init_strain, frames=T_strain, interval=1000 / fps, blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(strain_mp4, writer=writer)
        plt.close(fig)

    # Second combined video: cam_with_ego (left) + strain plot (right)
    # Rename final combined video to {object}_with_strain.mp4
    combined_forces_mp4 = os.path.join(videos_dir, f'{base_name}_with_strain.mp4')
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(strain_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', cam_with_ego_mp4,
            '-i', strain_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_forces_mp4
        ], check=True)
    
    # Clean up intermediate videos, keeping only the final combined videos
    print("Cleaning up intermediate videos...")
    videos_to_cleanup = [
        mp4_path,  # {base_name}_camera_obs.mp4
        mp4_ego,   # {base_name}_ego_obs.mp4
        cam_with_ego_mp4,  # {base_name}_camera_with_ego.mp4
        health_mp4,  # {base_name}_healths.mp4
        strain_mp4,  # {base_name}_strain.mp4
    ]
    
    # No additional intermediate strain videos
    
    for video_path in videos_to_cleanup:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Deleted: {os.path.basename(video_path)}")
            except OSError as e:
                print(f"Warning: Could not delete {video_path}: {e}")
    
    print(f"Final videos kept:")
    if os.path.exists(combined_mp4):
        print(f"  - {os.path.basename(combined_mp4)}")
    if os.path.exists(combined_forces_mp4):
        print(f"  - {os.path.basename(combined_forces_mp4)}")
    
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main()


