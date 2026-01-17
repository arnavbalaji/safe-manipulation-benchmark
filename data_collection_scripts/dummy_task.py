from ast import Pass
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import imageio
import json
import h5py
import pickle
import torch as th
th.set_printoptions(precision=3, sci_mode=False)
import numpy as np
from collections import defaultdict
import omnigibson as og
from omnigibson import object_states
# from omnigibson.object_states import IsGrasped
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm

from telemoma.configs.base_config import teleop_config
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.controllers.controller_base import IsGraspingState
from scipy.spatial.transform import Rotation as R
import cv2
import subprocess

from safety_benchmark.utils.misc_utils import (
    create_panda_eef_cylinders,
    get_external_sensor_paths,
    save_rgb_camera_video,
    save_rgb_force_video,
    save_rgb_health_video,
    save_rgb_force_contact_video,
    setup_viewport_layout,
)
from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper, DamageableDataPlaybackWrapper

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_TRANSITION_RULES = False


FLOUR_INIT_POS = [6.00, 0.35, 1.35]
FLOUR_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
FLOUR_SCALE = [1.0, 1.0, 0.9]

BOTTLE_OF_WINE_INIT_POS = [6.00, 0.2, 1.35]
BOTTLE_OF_WINE_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
BOTTLE_OF_WINE_SCALE = [1.0, 1.0, 1.0]

WINEGLASS_INIT_POS = [6.00, 0.12, 1.35]
WINEGLASS_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
WINEGLASS_SCALE = [1.0, 1.0, 1.0]

BOTTLE_OF_BEER_INIT_POS = [6.00, 0.08, 1.35]
BOTTLE_OF_BEER_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
BOTTLE_OF_BEER_SCALE = [1.0, 1.0, 1.0]

SHELF_INIT_POS = [6.00, 0.2, 1.35]
SHELF_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
SHELF_SCALE = [0.3, 0.7, 0.5]

OBJECT_SCALES = {
    "book": FLOUR_SCALE,
    "bottle_of_wine": BOTTLE_OF_WINE_SCALE,
    "wineglass": WINEGLASS_SCALE,
    "bottle_of_beer": BOTTLE_OF_BEER_SCALE,
}

# Task objects are located in BEHAVIOR-1k/datasets/objects/*
TASK_OBJECTS = {
    "box_of_crackers": {
        "type": "DatasetObject",
        "name": "box_of_crackers",
        "category": "box_of_crackers",
        "model": "cmdigf",
        "position": [6.0, 0.2, 2.0],
        "orientation": [0.0, 0.0, 0.70710678, 0.70710678],
    }, 
    "bag_of_flour": {
        "type": "DatasetObject",
        "name": "book",
        "category": "bag_of_flour",
        "model": "rlejxx",
        "position": FLOUR_INIT_POS,
        "orientation": FLOUR_INIT_ORI,
        "scale": FLOUR_SCALE,
    },
    "bottle_of_wine": {
        "type": "DatasetObject",
        "name": "bottle_of_wine",
        "category": "bottle_of_wine",
        "model": "hnkiog",
        "position": BOTTLE_OF_WINE_INIT_POS,
        "orientation": BOTTLE_OF_WINE_INIT_ORI,
        "scale": [1.0, 1.0, 1.0],
    },
    "wineglass": {
        "type": "DatasetObject",
        "name": "wineglass",
        "category": "wineglass",
        "model": "adiwil",
        "position": WINEGLASS_INIT_POS,
        "orientation": WINEGLASS_INIT_ORI,
        "scale": [1.0, 1.0, 1.0],
    },
    # "bottle_of_whiskey": {
    #     "type": "DatasetObject",
    #     "name": "bottle_of_whiskey",
    #     "category": "bottle_of_whiskey",
    #     # "model": "wfflbd",
    #     "model": "jfjclv",
    #     "position": BOTTLE_OF_WHISKEY_INIT_POS,
    #     "orientation": BOTTLE_OF_WHISKEY_INIT_ORI,
    #     "scale": [0.6, 0.6, 0.6],
    # },
    "bottle_of_beer": {
        "type": "DatasetObject",
        "name": "bottle_of_beer",
        "category": "bottle_of_beer",
        "model": "dqfsgv",
        "position": BOTTLE_OF_BEER_INIT_POS,
        "orientation": BOTTLE_OF_BEER_INIT_ORI,
        "scale": BOTTLE_OF_BEER_SCALE,
    },
    "stand": {
        "type": "DatasetObject",
        "name": "stand",
        "category": "stand",
        "model": "vyrick",
        "position": SHELF_INIT_POS,
        "orientation": SHELF_INIT_ORI,
        "scale": SHELF_SCALE,
        "fixed_base": True,
    },
}

def check_object_upright(obj):
    q = obj.get_position_orientation()[1]
    r = R.from_quat(q)

    # Rotate the up vector
    up_rotated = r.apply([0, 0, 1])
    z_alignment = up_rotated[2]  # should be close to 1 if not toppled

    threshold = 0.995  # cos(small angle) ~1
    upright = z_alignment > threshold
    
    return upright

def reset_env(env):
    env.reset()

    flour = env.scene.object_registry("name", "book")
    wineglass = env.scene.object_registry("name", "wineglass")
    winebottle = env.scene.object_registry("name", "bottle_of_wine")
    beerbottle = env.scene.object_registry("name", "bottle_of_beer")
    stand = env.scene.object_registry("name", "stand")

    # Since the saved state has different laptop and coffee cup positions, setting it here
    beerbottle.set_position_orientation(position=th.tensor(BOTTLE_OF_BEER_INIT_POS))

    objects = [flour, wineglass, winebottle, beerbottle]
    trial_number = 0
    while True:
        print("Reset trial number: ", trial_number)

        # load state
        with open("resources/saved_states/dummy_task_init_state.pkl", "rb") as f: state_flat_array = pickle.load(f)
        og.sim.load_state(state_flat_array, serialized=True)

        for obj in objects:
            pos, orn = obj.get_position_orientation()
            pos_magnitude = [-0.05, 0.05] 
            rot_magnitude = np.pi / 12 # 15 degrees
            pos_diff_xy = np.random.uniform(pos_magnitude[0], pos_magnitude[1], size=2)
            pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
            new_pos = pos + pos_diff
            orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
            new_orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
            obj.set_position_orientation(new_pos, new_orn)

        # randomize scale
        temp_state = og.sim.dump_state(serialized=False)
        og.sim.stop()
        for obj in objects:
            x_scale_magnitude = np.random.uniform(0.9, 1.1)
            y_scale_magnitude = np.random.uniform(0.9, 1.1)
            z_scale_magnitude = np.random.uniform(0.9, 1.1)
            # obtain obj original scales
            original_scale = OBJECT_SCALES[obj.name]
            new_scale = [original_scale[0] * x_scale_magnitude, original_scale[1] * y_scale_magnitude, original_scale[2] * z_scale_magnitude]
            obj.scale = th.tensor(new_scale)

            # scale stand a bit randomly as well
            y_scale_magnitude = np.random.uniform(0.9, 1.0)
            new_scale = [SHELF_SCALE[0], SHELF_SCALE[1] * y_scale_magnitude, SHELF_SCALE[2]]
            stand.scale = th.tensor(new_scale)

        # scale the bar
        bar = env.scene.object_registry("name", "bar_udatjt_0")
        bar.scale = th.tensor([0.85, 0.95, 1.0])
        og.sim.play()
        og.sim.load_state(temp_state)

        for _ in range(50): og.sim.step()

        # Make sure all objects are upright
        all_upright = True
        for obj in objects:
            upright = check_object_upright(obj)
            print("object, upright: ", obj.name, upright)
            if not upright:
                print(f"Object {obj.name} is not upright, randomizing again")
                all_upright = False
                break
        if all_upright:
            print("All objects are upright, breaking")
            break
        trial_number += 1

    for _ in range(50): og.sim.step()

def setup_external_sensors():
    robot_name = "franka0"
    robot_type = "FrankaPanda"
    image_height = 256
    image_width = 256
    # Set external cameras for videos
    EXTERNAL_CAMERA_CONFIGS = {
        # Side camera (fixed to base_link frame)
        "external_sensor_0": {
            "position": [7.3920, -0.6436, 1.7519],
            "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
            "horizontal_aperture": 15.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
        },
        # Left Shoulder (fixed to base_link frame)
        "external_sensor_1": {
            # wrt base frame
            "position": [7.1264, 1.1205, 2.0117],
            "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
            "horizontal_aperture": 15.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1",
        },
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
                "image_height": image_height,
                "image_width": image_width,
                "horizontal_aperture": camera_cfg["horizontal_aperture"],
            },
            "position": th.tensor(position, dtype=th.float32),
            "orientation": th.tensor(orientation, dtype=th.float32),
            "pose_frame": "world",
        })
    return external_sensors_config

def __main__():
    np.random.seed(0)
    th.manual_seed(0)



    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Overwrite any configs here
    cfg["scene"]["scene_model"] = "house_single_floor"
    cfg["scene"]["not_load_object_categories"] = ["ottoman"]
    cfg["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
    
    ############### Franka robot ###############
    # TODO(junhong): if we have a better way (a franka-specific config file), we should use that
    # Completely replace robot config to avoid Tiago-specific settings carrying over
    cfg["robots"][0] = {
        "type": "FrankaPanda",
        "name": "franka0",
        "position": [6.8, 0.2, 1.0],  # Match Tiago base position
        "orientation": [0.0, 0.0, 1.0, 0.0],
        "grasping_mode": "assisted",
        # "obs_modalities": ["rgb", "depth"],
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
            },
        },
    }

    # Generate external sensors config automatically
    # Get robot name and type from config to construct correct prim path
    robot_name = cfg["robots"][0].get("name", "franka0")
    robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()

    # Add objects here
    cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

    # cfg["env"]["external_sensors"] = setup_external_sensors()

    env = DamageableEnvironment(configs=cfg)        

    robot = env.robots[0]
    # set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([ 7.0659, -0.7141,  1.9185]),
        orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
    )
    for _ in range(10): og.sim.step()

    # To set the viewport layout
    setup_viewport_layout()

    # Setting up eef visualization for easier teleop
    eef_vis = create_panda_eef_cylinders(robot, env.scene)
    robot.links["eef_link"].prim.GetAttribute("visibility").Set("inherited")
    # breakpoint()
    for geom_list in eef_vis.values():
        for geom in geom_list:
            geom.prim.GetAttribute("visibility").Set("inherited")  #.set("invisible") for hiding
    for _ in range(10): og.sim.render()

    # Telemoma: Teleoperate robot
    arm_teleop_method = "spacemouse"
    base_teleop_method = "spacemouse"
    # # Franka uses arm_0 instead of arm_left/arm_right
    teleop_config.arm_0_controller = arm_teleop_method
    # # Tiago config (commented out):
    teleop_config.arm_left_controller = arm_teleop_method
    teleop_config.arm_right_controller = arm_teleop_method
    teleop_config.base_controller = base_teleop_method
    teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
    teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.01}
    teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=False)
    teleop_sys.start()

    # Keyboard Teleop
    action_generator = KeyboardRobotController(robot=robot)
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )
    action_generator.print_keyboard_teleop_info()

    # # To debug if reset_env is working correctly
    # for _ in range(20):
    #     reset_env(env)
    #     breakpoint()

    n_episodes = 100
    last_telemoma_grip_action = 1.0 
    completed_episodes = 0
    while completed_episodes < n_episodes:
        print(f"Episode {completed_episodes} starts (target: {n_episodes})")
                    
        reset_env(env)
        action_generator.current_keypress = None

        # If the robot is grasping, set the persistent gripper action to -1.0
        if robot.is_grasping().value == IsGraspingState.TRUE:
            action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
        action = th.zeros(robot.action_dim)
        action[-1] = -1.0
        episode_starts = False

        stand = env.scene.object_registry("name", "stand")
        box_of_crackers = env.scene.object_registry("name", "box_of_crackers")
        
        print("Ready for teleoperation. Press TAB to end episode, BACKSPACE/DELETE to discard and reset.")
        breakpoint()
        while True:
            telemoma_action = teleop_sys.get_action(teleop_sys.get_obs())
            telemoma_grip_action = telemoma_action[-1]
            if telemoma_grip_action != last_telemoma_grip_action:
                action[-1] = -action[-1]
            last_telemoma_grip_action = telemoma_grip_action
            action[:-1] = telemoma_action[:-1]

            _, keypress_str = action_generator.get_teleop_action()
            
            # TAB: end episode and save
            if keypress_str and keypress_str.upper() == "TAB":
                print("TAB pressed - ending episode")
                breakpoint()
                break
            
            env.step(action.clone())

    og.shutdown()

if __name__ == "__main__":
    __main__()
