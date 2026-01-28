from ast import Pass
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import json
import h5py
import pickle
import torch as th
import numpy as np
import math

import omnigibson as og
from omnigibson import object_states
# from omnigibson.object_states import IsGrasped
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm

# Commented out - not needed for keyboard teleoperation (causes mediapipe import error)
from telemoma.configs.base_config import teleop_config
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
import omnigibson.lazy as lazy

from omnigibson.controllers.controller_base import IsGraspingState

from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_TRANSITION_RULES = False


def set_laptop_pose(env, target_deg: float = 130.0):
    laptop = env.scene.object_registry("name", "laptop")
    if laptop is None:
        return
    target_rad = math.radians(float(target_deg))
    if hasattr(laptop, "joints"):
        for joint in laptop.joints.values():
            lo = joint.lower_limit
            hi = joint.upper_limit
            target = max(lo, min(hi, target_rad))
            joint.set_pos(target)
            joint.friction = 50000000.0
            if hasattr(joint, "keep_still"):
                joint.keep_still()
    if hasattr(laptop, "keep_still"):
        laptop.keep_still()


LAPTOP_INIT_POS = [6.3, 0.4, 1.1]
LAPTOP_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
COFFEE_CUP_1_INIT_POS = [6.3, 0.6, 1.3]
WATER_GLASS_INIT_POS = [6.3, 0.5, 1.3]

# Task objects are located in BEHAVIOR-1k/datasets/objects/*
TASK_OBJECTS = {
    "laptop": {
        "type": "DatasetObject",
        "name": "laptop",
        "category": "laptop",
        "model": "nvulcs",
        "position": LAPTOP_INIT_POS,
        "orientation": LAPTOP_INIT_ORI,
        "scale": [1.0, 1.0, 1.0],
    },
    "coffee_cup_1": {
        "type": "DatasetObject",
        "name": "coffee_cup_1",
        "category": "coffee_cup",
        "model": "ckkwmj",
        "position": COFFEE_CUP_1_INIT_POS,
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.2, 1.2, 1.2],
    },
    "water_glass": {
        "type": "DatasetObject",
        "name": "water_glass",
        "category": "water_glass",
        "model": "ewgotr",
        "position": WATER_GLASS_INIT_POS,
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.9, 0.9, 0.9],
    },
}

def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_state', action='store_true', help='Load a state')
    args = parser.parse_args()
    if args.load_state:
        load_state_path = f"resources/teleop_data/shelf.hdf5"
        load_f = h5py.File(load_state_path, "r")

    # TODO: Set this
    collect_hdf5_path = f"resources/teleop_data/test_unsafe.hdf5"
    if not os.path.exists(collect_hdf5_path):
        os.makedirs(os.path.dirname(collect_hdf5_path), exist_ok=True)

    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Overwrite any configs here
    cfg["scene"]["scene_model"] = "house_single_floor"
    cfg["scene"]["not_load_object_categories"] = ["ottoman"]
    cfg["scene"]["load_room_instances"] = ["kitchen_0"]
    
    ############### Franka robot ###############
    # TODO(junhong): if we have a better way (a franka-specific config file), we should use that
    # Completely replace robot config to avoid Tiago-specific settings carrying over
    cfg["robots"][0] = {
        "type": "FrankaPanda",
        "name": "franka0",
        "position": [6.8, 0.2, 1.0],  # Match Tiago base position
        "orientation": [0.0, 0.0, 1.0, 0.0],
        "grasping_mode": "assisted",
        "obs_modalities": ["rgb", "depth"],
        "action_normalize": False,
        "self_collisions": True,
        # Franka has single arm (arm_0, gripper_0) instead of left/right
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                # "name": "JointController",
                "command_input_limits": None,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
                "motor_type": "position",
                "isaac_kp": 50000.0,
                "isaac_kd": 5000.0,
            },
        },
    }
    ############### Franka robot ###############

    ############### Tiago robot (commented out) ###############
    # cfg["robots"][0]["name"] = "tiago0"
    # cfg["robots"][0]["position"] = [0.0, 0.0, 0.0]
    # cfg["robots"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]
    # cfg["robots"][0]["default_arm_pose"] = "horizontal"
    # cfg["robots"][0]["grasping_mode"] = "assisted"
    # cfg["robots"][0]["obs_modalities"] = ["rgb", "depth"]
    # cfg["robots"][0]["action_normalize"] = False
    # cfg["robots"][0]["controller_config"] = {
    #     "arm_left": {
    #         "name": "InverseKinematicsController",
    #         "command_input_limits": None,
    #     },
    #     "gripper_left": {
    #         "name": "MultiFingerGripperController",
    #         "command_input_limits": (0.0, 1.0),
    #         "mode": "smooth",
    #     },
    #     "arm_right": {
    #         "name": "InverseKinematicsController",
    #         "command_input_limits": None,
    #     },
    #     "gripper_right": {
    #         "name": "MultiFingerGripperController",
    #         "command_input_limits": (0.0, 1.0),
    #         "mode": "smooth",
    #     },
    # }
    # cfg["robots"][0]["exclude_sensor_names"] = ["left_eef_link"]
    ############### Tiago robot (commented out) ###############

    # Generate external sensors config automatically
    # Get robot name and type from config to construct correct prim path
    robot_name = cfg["robots"][0].get("name", "franka0")
    robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()

    # Set external cameras for videos
    EXTERNAL_CAMERA_CONFIGS = {
        # Side camera (fixed to base_link frame)
        "external_sensor_0": {
            "position": [0.4859, -1.8219,  1.1402],
            "orientation": [ 0.5857, -0.0093, -0.0129,  0.8103],
            "horizontal_aperture": 10.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
        },
        # Left Shoulder (fixed to base_link frame)
        "external_sensor_1": {
            # wrt base frame
            "position": [0.2522, 0.0470, 1.0696],
            "orientation": [ 0.1991, -0.1991, -0.6785,  0.6785],
            "horizontal_aperture": 30.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1",
        },
        # Back camera (fixed to base_link frame)
        "external_sensor_2": {
            "position": [-0.7765, -0.8203,  0.9939],
            "orientation": [ 0.4566, -0.3285, -0.4831,  0.6710],
            "horizontal_aperture": 30.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor2",
        },
        # Front camera (fixed to base_link frame)
        "external_sensor_3": {
            "position": [1.7508, -0.0198,  1.1778],
            "orientation": [0.3821, 0.4173, 0.6080, 0.5570],
            "horizontal_aperture": 20.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor3",
        }
    }
    external_sensors_config = []
    for name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
        i = name.split("_")[-1]
        position = camera_cfg["position"]
        orientation = camera_cfg["orientation"]
        external_sensors_config.append({
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
        })
    cfg["env"]["external_sensors"] = external_sensors_config

    # Add objects here
    cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

    env = DamageableEnvironment(configs=cfg)        
    env = DamageableDataCollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
        # obj_attr_keys=["scale", "visible"],
    )
    # env = og.Environment(configs=cfg)        
    # env = DataCollectionWrapper(
    #     env=env,
    #     output_path=collect_hdf5_path,
    #     only_successes=False,
    #     # obj_attr_keys=["scale", "visible"],
    #     enable_dump_filters=False,
    # )

    robot = env.robots[0]
    # set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([ 7.0659, -0.7141,  1.9185]),
        orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
    )
    for _ in range(10): og.sim.step()

    # Franka default joint positions (7 arm joints + 2 gripper joints)
    robot.set_joint_positions(th.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]))

    # Increase gripper force for better grasping
    try:
        for link_name, link in robot.links.items():
            n = link_name.lower()
            if ("gripper" in n) or ("finger" in n):
                try:
                    link.set_attribute("physxMaterial:staticFriction", 5.0)
                    link.set_attribute("physxMaterial:dynamicFriction", 5.0)
                except Exception:
                    pass
    except Exception:
        pass

    set_laptop_pose(env, target_deg=130.0)
    og.sim.update_handles()

    coffee_cup_1 = env.scene.object_registry("name", "coffee_cup_1")
    water_glass = env.scene.object_registry("name", "water_glass")
    for cup in [coffee_cup_1, water_glass]:
        if cup is not None and hasattr(cup, "joints"):
            for joint in cup.joints.values():
                joint.friction = 50000000.0

    # # load state
    # # with open("shelf_init_state_dict.pkl", "rb") as f: state_dict = pickle.load(f)
    # with open("resources/saved_states/pour_glass_init_state.pkl", "rb") as f: state_flat_array = pickle.load(f)
    # og.sim.load_state(state_flat_array, serialized=True)
    # for _ in range(10): og.sim.step()
    # breakpoint()

    
    # Tiago joint positions (commented out - different DOF count)
    # robot.set_joint_positions(th.tensor([0.2, 0.517,  3.4369e-04,  3.0920e-07, -1.2731e-07,
    #      4.6876e-02,  2.1064e-01,  8.6563e-01,  8.3026e-01,  9.4543e-04,
    #     -5.4681e-01, -2.5515e-01, -9.8940e-01,  2.1207e+00,  1.6975e+00,
    #      1.6488e+00,  1.6281e+00,  8.5711e-01,  2.5175e-01, -9.2479e-01,
    #     -1.4137e+00, -1.0974e+00, -7.0588e-01,  4.5000e-02,  4.5000e-02,
    #      4.5000e-02,  4.5000e-02]))

    robot_joint_positions = th.tensor([ 1.1922, -1.2874, -1.5870, -2.7042,  0.1077,  3.7508, -0.5944])
    robot.set_joint_positions(robot_joint_positions, indices=robot.arm_control_idx["0"])
    for _ in range(10): og.sim.step()
    
    for _ in range(10):
        og.sim.step()

    # controller_config = {"arm_0": {"name": "JointController", "command_input_limits": None}, "gripper_0": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0), "mode": "smooth"},}
    # controller_config = {"arm_0": {"name": "InverseKinematicsController", "command_input_limits": None}, "gripper_0": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0), "mode": "smooth"},}
    # robot.reload_controllers(controller_config=controller_config)

    # Telemoma: Teleoperate robot
    arm_teleop_method = "spacemouse"
    base_teleop_method = "spacemouse"
    # Franka uses arm_0 instead of arm_left/arm_right
    teleop_config.arm_0_controller = arm_teleop_method
    # Tiago config (commented out):
    # teleop_config.arm_left_controller = arm_teleop_method
    # teleop_config.arm_right_controller = arm_teleop_method
    teleop_config.base_controller = base_teleop_method
    teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
    teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.01}
    teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=False)
    teleop_sys.start()

    # Keyboard Teleop
    action_generator = KeyboardRobotController(robot=robot)
    
    # Function to save simulation state to pkl file
    def save_state_to_pkl():
        """Save current simulation state to pkl file"""
        os.makedirs("safe-manipulation-benchmark/resources/saved_states", exist_ok=True)
        save_path = "resources/saved_states/pour_glass_init_state.pkl"
        og.sim.update_handles()  # Update handles before dumping state
        state = og.sim.dump_state(serialized=True)
        f = open(save_path, "wb")
        pickle.dump(state, f)
        print(f"Simulation state saved to: {save_path}")
    
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.S,
        description="Save simulation state to pkl file",
        callback_fn=save_state_to_pkl,
    )
    action_generator.print_keyboard_teleop_info()

    # ======================== Data collection ========================
    n_episodes = 1
    for i in range(n_episodes):
        print(f"Episode {i} starts")
        # if load a state
        if args.load_state:
            # TODO: Remove hardcoding here
            # state = pickle.load(open("resources/saved_states/grasped_sponge.pkl", "rb"))
            # og.sim.load_state(state, serialized=False)

            # state = th.tensor(load_f["data/demo_0/state"][700])
            # og.sim.load_state(state, serialized=True)
            breakpoint()
            for _ in range(50): og.sim.step()

        breakpoint()
        prev_is_grasping = False
        while True:
            # Use KeyboardRobotController for default keybindings (as printed in terminal)
            # Handle different return types from get_teleop_action()
            ret = action_generator.get_teleop_action()
            if isinstance(ret, tuple) and len(ret) == 2:
                action, keypress_str = ret
            else:
                action = ret
                keypress_str = None
            # if robot.is_grasping().value == IsGraspingState.TRUE:
            #     action[robot.gripper_action_idx[robot.default_arm]] = -1.0
            if keypress_str == "TAB":
                # state = og.sim.dump_state(serialized=True)
                # with open("shelf_init_state_2.pkl", "wb") as f: pickle.dump(state, f)

                breakpoint()
                # break
            # Check if grasping state changed and update handles if so
            try:
                from omnigibson.utils.constants import IsGraspingState
                is_grasping = robot.is_grasping().value == IsGraspingState.TRUE
                if is_grasping != prev_is_grasping:
                    if og.sim.is_playing():
                        try:
                            og.sim.update_handles()
                        except Exception:
                            pass
                    prev_is_grasping = is_grasping
            except Exception:
                pass
            # Update handles before step to prevent articulation view errors during state dumping
            if og.sim.is_playing():
                try:
                    og.sim.update_handles()
                except Exception:
                    pass
            # Retry step with handle update if articulation view error occurs
            for attempt in range(3):
                try:
                    env.step(action)
                    break
                except AttributeError as e:
                    if "'NoneType' object has no attribute" in str(e) and attempt < 2:
                        if og.sim.is_playing():
                            try:
                                og.sim.update_handles()
                            except Exception:
                                pass
                    else:
                        raise
    print("Data saved")
    env.save_data()
    og.shutdown()

if __name__ == "__main__":
    __main__()

