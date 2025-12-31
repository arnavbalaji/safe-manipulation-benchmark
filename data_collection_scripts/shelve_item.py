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

from omnigibson.controllers.controller_base import IsGraspingState

from safety_benchmark.utils.misc_utils import save_camera_video, save_health_video, save_combined_video, save_forces_video, save_force_contact_video
from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper, DamageableDataPlaybackWrapper

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_TRANSITION_RULES = False


FLOUR_INIT_POS = [6.00, 0.35, 1.3]
FLOUR_INIT_ORI = [0.0, 0.0, 0.0, 1.0]

BOTTLE_OF_WINE_INIT_POS = [6.00, 0.2, 1.3]
BOTTLE_OF_WINE_INIT_ORI = [0.0, 0.0, 0.0, 1.0]

WINEGLASS_INIT_POS = [6.00, 0.12, 1.3]
WINEGLASS_INIT_ORI = [0.0, 0.0, 0.0, 1.0]

BOTTLE_OF_WHISKEY_INIT_POS = [6.00, 0.0, 1.3]
BOTTLE_OF_WHISKEY_INIT_ORI = [0.0, 0.0, 0.0, 1.0]

BOTTLE_OF_BEER_INIT_POS = [6.00, 0.0, 1.3]
BOTTLE_OF_BEER_INIT_ORI = [0.0, 0.0, 0.0, 1.0]

SHELF_INIT_POS = [6.00, 0.2, 1.3]
SHELF_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
SHELF_SCALE = [0.4, 0.8, 0.5]

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
        "scale": [1.0, 1.0, 1.0],
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
        "scale": [1.0, 1.0, 1.0],
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

def reset_env(env):
    env.reset()
    # load state
    with open("resources/saved_states/shelf_init_state.pkl", "rb") as f: state_flat_array = pickle.load(f)
    og.sim.load_state(state_flat_array, serialized=True)

    # TODO: Add object pose and scale randomization

    for _ in range(10): og.sim.step()

def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_hdf5_path', type=str, help='Target hdf5 path', default="resources/teleop_data/shelve_item.hdf5")
    parser.add_argument('--playback_hdf5_path', type=str, help='Output hdf5 path', default="resources/playback_data/shelve_item_playback.hdf5")
    parser.add_argument('--n_episodes', type=int, help='Number of episodes', default=1)
    parser.add_argument('--teleop', action='store_true', help='Teleoperate the robot')
    parser.add_argument('--playback', action='store_true', help='Playback the data')
    parser.add_argument('--visualize', action='store_true', help='Visualize the data')
    args = parser.parse_args()

    if args.teleop:
        # TODO: Set this
        if not os.path.exists(args.collect_hdf5_path):
            os.makedirs(os.path.dirname(args.collect_hdf5_path), exist_ok=True)

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
                },
            },
        }

        # Generate external sensors config automatically
        # Get robot name and type from config to construct correct prim path
        robot_name = cfg["robots"][0].get("name", "franka0")
        robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()

        # Add objects here
        cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

        env = DamageableEnvironment(configs=cfg)        
        env = DamageableDataCollectionWrapper(
            env=env,
            output_path=args.collect_hdf5_path,
            only_successes=False,
            enable_dump_filters=False,
        )

        robot = env.robots[0]
        # set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([ 7.0659, -0.7141,  1.9185]),
            orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
        )
        for _ in range(10): og.sim.step()

        # # Telemoma: Teleoperate robot
        # arm_teleop_method = "spacemouse"
        # base_teleop_method = "spacemouse"
        # # Franka uses arm_0 instead of arm_left/arm_right
        # teleop_config.arm_0_controller = arm_teleop_method
        # # Tiago config (commented out):
        # # teleop_config.arm_left_controller = arm_teleop_method
        # # teleop_config.arm_right_controller = arm_teleop_method
        # teleop_config.base_controller = base_teleop_method
        # teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
        # teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.01}
        # teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=False)
        # teleop_sys.start()

        # Keyboard Teleop
        action_generator = KeyboardRobotController(robot=robot)
        action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=lambda: env.reset(),
        )
        action_generator.print_keyboard_teleop_info()

        # ======================== Data collection ========================
        n_episodes = args.n_episodes
        for i in range(n_episodes):
            print(f"Episode {i} starts")
            reset_env(env)
            breakpoint()
            # If the robot is grasping, set the persistent gripper action to -1.0
            if robot.is_grasping().value == IsGraspingState.TRUE:
                action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
            while True:
                # Not using telemoma for now
                # action = teleop_sys.get_action(teleop_sys.get_obs())
                action, keypress_str = action_generator.get_teleop_action()
                if keypress_str == "TAB":
                    breakpoint()
                    break
                env.step(action)
        env.save_data()
        print("Data saved")
        og.shutdown()

    if args.playback:
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

        # In case we want to modify the robot sensors that were used during data collection
        robot_sensor_config = {
            "VisionSensor": {
                "modalities": ["rgb", "seg_instance"],
                "sensor_kwargs": {
                    "image_height": image_height,
                    "image_width": image_width,
                },
            },
        }
        
        env = DamageableDataPlaybackWrapper.create_from_hdf5(
            input_path=args.collect_hdf5_path,
            output_path=args.playback_hdf5_path,
            robot_obs_modalities=["proprio", "rgb", "seg_instance"],
            robot_sensor_config=robot_sensor_config,
            external_sensors_config=external_sensors_config,
            n_render_iterations=1,
            only_successes=False,
        )

        # set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([ 7.0659, -0.7141,  1.9185]),
            orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
        )
        for _ in range(10): og.sim.step()

        # Playback the dataset
        env.playback_dataset(record_data=True)    
        env.save_data()
        og.shutdown()


    if args.visualize:
        f = h5py.File(args.playback_hdf5_path, "r")
        f_name = "shelve_item"
        scene_file = json.loads(f["data"].attrs["scene_file"])
        # robot_name = [obj_name for obj_name in scene_file["objects_info"]["init_info"].keys() if "robot" in obj_name.lower()][0]
        robot_name = "franka0"       
        camera_type = "external"
        camera_name = "external_sensor0"
        # breakpoint()

        # Parse info to obtain relevant information for visualization
        obs_info_list = []
        for i in range(len(f["data/demo_0/info/obs_info"])):            
            # Obtain observation information 
            obs_info = json.loads(f["data/demo_0/info/obs_info"][i].decode("utf-8"))
            obs_info_list.append(obs_info)
        # breakpoint()

        health = []

        output_video_dir = "resources/videos"
        os.makedirs(output_video_dir, exist_ok=True)
        
        # Save video for rgb camera
        target_objects_health = []
        output_video_path = f"{output_video_dir}/{f_name}_camera_video"
        save_camera_video(hdf5_file=f, 
                    output_video_path=output_video_path,
                    robot_name=robot_name, 
                    camera_type=camera_type, 
                    camera_name=camera_name,
                    target_objects=target_objects_health,
                    obs_info_list=obs_info_list,
                    health=health)


if __name__ == "__main__":
    __main__()
