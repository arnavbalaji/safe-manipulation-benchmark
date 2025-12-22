from ast import Pass
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import yaml
import json
import h5py
import torch as th
import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm

from telemoma.configs.base_config import teleop_config
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
import omnigibson.lazy as lazy

from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.damageable_env import DamageableDataPlaybackWrapper, DamageableDataCollectionWrapper

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_TRANSITION_RULES = False
# gm.ENABLE_HQ_RENDERING = True
# gm.DEFAULT_RENDERING_FREQ = 60

PARAMS = {
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

TASK_OBJECTS = {
    "cup1": {
        "type": "DatasetObject",
        "name": "coffee_cup_1",
        "category": "coffee_cup",
        "model": "rypdvd",
        "position": [-0.6, -1.7, 1.2],
        "orientation": [0.0, 0.0, 0.0, 1.0]
        # "scale": [1.0, 1.0, 1.0],
    },
    "cup2": {
        "type": "DatasetObject",
        "name": "coffee_cup_2",
        "category": "coffee_cup",
        "model": "rypdvd",
        "position": [-0.6, -2.6, 0.7],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        # "scale": [1.0, 1.0, 1.0],
    },
    "plate": {
        "type": "DatasetObject",
        "name": "plate",
        "category": "plate",
        "model": "ntedfx",
        "position": [-0.6, -1.7, 0.415],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 1.0, 1.0],
    },
}


def __main__():
    np.random.seed(0)
    th.manual_seed(0)
    # th.cuda.manual_seed(0)
    # th.backends.cudnn.benchmark = False
    # th.backends.cudnn.deterministic = True
    
    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Overwrite any configs here
    # cfg["scene"]["type"] = "Scene"
    # cfg["scene"]["scene_id"] = "empty"
    cfg["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]


    # Always spawn robot at the origin with no rotation (this is to be compatible with curobo)
    cfg["robots"][0]["name"] = "tiago0"
    cfg["robots"][0]["position"] = [0.0, 0.0, 0.0]
    cfg["robots"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]
    cfg["robots"][0]["default_arm_pose"] = "horizontal"
    cfg["robots"][0]["grasping_mode"] = "assisted"
    cfg["robots"][0]["obs_modalities"] = ["rgb", "depth"]
    cfg["robots"][0]["action_normalize"] = False
    cfg["robots"][0]["controller_config"] = {
        "arm_left": {
            "name": "InverseKinematicsController",
            "command_input_limits": None,
        },
        "gripper_left": {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": "smooth",
        },
        "arm_right": {
            "name": "InverseKinematicsController",
            "command_input_limits": None,
        },
        "gripper_right": {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": "smooth",
        },
    }
    cfg["robots"][0]["exclude_sensor_names"] = ["left_eef_link", "right_eef_link"]

    EXTERNAL_CAMERA_CONFIGS = {
        "external_sensor_0": {
            "position": [1.1553, -2.2072,  1.0119],
            "orientation": [ 0.4284, 0.4160, 0.5588, 0.5755],
            "horizontal_aperture": 20.0,
            "relative_prim_path": f"/external_sensor0",
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
    target_object_name = "drop_mug_on_plate"

    # Load the environment
    # env = og.Environment(configs=cfg)
    teleop = True
    if teleop:
        env = DamageableEnvironment(configs=cfg)
        collect_hdf5_path = f"resources/teleop_data/{target_object_name}.hdf5"
        env = DamageableDataCollectionWrapper(
            env=env,
            output_path=collect_hdf5_path,
            only_successes=False,
            # obj_attr_keys=["scale", "visible"],
            enable_dump_filters=False,
        )
    else:
        collect_hdf5_path = f"resources/teleop_data/{target_object_name}.hdf5"
        output_hdf5_path = f"resources/playback_data/{target_object_name}_playback.hdf5"
        # output_hdf5_path = "temp.hdf5"
        robot_sensor_config = {
            "VisionSensor": {
                "modalities": ["rgb"],
                "sensor_kwargs": {
                    "image_height": 720,
                    "image_width": 720,
                },
            },
        }
        # Create a playback env and playback the data, collecting obs along the way
        # og.sim.stop()
        env = DamageableDataPlaybackWrapper.create_from_hdf5(
            input_path=collect_hdf5_path,
            output_path=output_hdf5_path,
            # robot_obs_modalities=["proprio", "rgb", "depth", "seg_instance"],
            robot_sensor_config=robot_sensor_config,
            # external_sensors_config=external_sensors_config,
            n_render_iterations=1,
            only_successes=False,
            exclude_sensor_names=["left_eef_link", "right_eef_link"]
            # env=env,
        )
   
    robot = env.robots[0]
    # for _ in range(10): og.sim.step()

    # set viewer camera
    og.sim.viewer_camera.set_position_orientation(position=th.tensor([ 1.1553, -2.2072,  1.0119]), orientation=th.tensor([0.4284, 0.4160, 0.5588, 0.5755]))
    
    # To obtain keyboard events
    if teleop:
        action_generator = KeyboardRobotController(robot=robot)

        # Register custom binding to reset the environment
        action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=lambda: env.reset(),
        )

        # Print out relevant keyboard info if using keyboard teleop
        action_generator.print_keyboard_teleop_info()

        # ======================== Data collection ========================
        n_episodes = 1
        for i in range(n_episodes):
            print(f"Episode {i} starts")
            # breakpoint()
            # env_interface.reset()
            breakpoint()
            while True:
            # for i in range(100):
                # print("Step", i)
                # action = teleop_sys.get_action(teleop_sys.get_obs())
                action, keypress_str = action_generator.get_teleop_action()
                if keypress_str == "TAB":
                    breakpoint()
                    # current_traj_history
                    break
                retval = env.step(action)
                # breakpoint()  
                # print("--", env.current_traj_history[-1]["state_size"])
        print("Data saved")
        # breakpoint()
        # teleop_sys.stop()
    # Replay episode
    else:
        # Modify any environment parameters here
        # env.set_object_params()
        breakpoint()

        # set params
        # set_params()

        env.external_sensors["external_sensor0"].horizontal_aperture = 20.0
        env.external_sensors["external_sensor0"].image_height = 1280
        env.external_sensors["external_sensor0"].image_width = 1280
        # Need to reload the observation space to apply the modified sensor parameters
        env.load_observation_space()
        breakpoint()
        env.playback_dataset(record_data=True, replay_for_annotation=False)
        
    breakpoint()
    # obtain obs
    # env.current_traj_history.keys()
    env.save_data()
    # breakpoint()
    og.shutdown()

if __name__ == "__main__":
    __main__()