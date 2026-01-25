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
from omnigibson.macros import gm

from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import omnigibson.lazy as lazy

from omnigibson.controllers.controller_base import IsGraspingState

from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_TRANSITION_RULES = False

# Task objects configuration
TASK_OBJECTS = {
    "fireplace": {
        "type": "DatasetObject",
        "name": "fireplace",
        "category": "wood_fireplace",
        "model": "gpnsij",
        "position": [-1.5, -2.0, 0.5],  # Against back wall, translated -2.0 in y
        "orientation": [0, 0, 0, 1],  # No rotation, straight
        "scale": [1.0, 0.75, 0.75],
        "fixed_base": True,
        "abilities": {
            "heatSource": {
                "temperature": 100.0,
                "heating_rate": 0.1,
                "distance_threshold": 0.5,
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
        "position": [-1.65, -2.0, 0.15],  # Inside fireplace, translated -2.0 in y, moved further back
        "orientation": [0, 0, 0, 1],  # Straight, no rotation
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
        "position": [-1.65, -2.15, 0.17],  # Inside fireplace, translated -2.0 in y, moved further back
        "orientation": [0, 0, 0, 1],  # Straight, no rotation
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
        "position": [-1.0, -2.25, 0.1],  # To the left of robot (robot at y=-2.0)
        "orientation": [0, 0, 0, 1],  # Straight, no rotation
        "scale": [0.7, 0.5, 0.5],
        "abilities": {
            "flammable": {},
        },
        "initial_state": {
            "onFire": False,  # This one is outside, not on fire initially
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

def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_state', action='store_true', help='Load a state')
    args = parser.parse_args()
    if args.load_state:
        load_state_path = f"resources/teleop_data/firewood.hdf5"
        load_f = h5py.File(load_state_path, "r")

    # TODO: Set this
    collect_hdf5_path = f"resources/teleop_data/firewood_test.hdf5"
    if not os.path.exists(collect_hdf5_path):
        os.makedirs(os.path.dirname(collect_hdf5_path), exist_ok=True)

    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Overwrite scene config for Rs_int
    cfg["scene"] = {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "include_robots": False,
        "load_task_relevant_only": True,
    }
    
    ############### Franka robot ###############
    # Completely replace robot config to avoid Tiago-specific settings carrying over
    cfg["robots"][0] = {
        "type": "FrankaPanda",
        "name": "franka0",
            "position": [-0.85, -2.0, 0.0],  # Moved closer to fireplace
        "orientation": [0.0, 0.0, 1.0, 0.0],  # 180 deg rotation around z to face straight towards fireplace
        "grasping_mode": "assisted",
        "obs_modalities": ["rgb", "depth"],
        "action_normalize": False,
        "self_collisions": True,
        # Franka has single arm (arm_0, gripper_0) instead of left/right
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
    ############### Franka robot ###############

    # Generate external sensors config automatically
    # Get robot name and type from config to construct correct prim path
    robot_name = cfg["robots"][0].get("name", "franka0")
    robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()

    # Set external cameras for videos
    # Robot is at [-0.2, 0.0, 0.0] facing negative x (towards fireplace at [-1.5, 0.0, 0.5])
    # Viewer camera is at [1.616, -0.01, 1.036] - positioned behind robot
    EXTERNAL_CAMERA_CONFIGS = {
        # Side camera - left side view (positive y) of robot and fireplace
        "external_sensor_0": {
            "position": [-0.85, -0.8, 1.0],  # Translated -2.0 in y, elevated
            "orientation": [0.5, 0.0, 0.0, 0.866],  # Looking at the interaction
            "horizontal_aperture": 25.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
        },
        # Overhead camera - top-down view, centered
        "external_sensor_1": {
            "position": [-0.85, -2.0, 1.8],  # Translated -2.0 in y, looking straight down
            "orientation": [0.707, 0.0, 0.0, 0.707],  # Looking straight down
            "horizontal_aperture": 30.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1",
        },
        # Back camera - similar to viewer camera position, behind robot
        "external_sensor_2": {
            "position": [1.616, -2.01, 1.036],  # Translated -2.0 in y, behind robot
            "orientation": [0.452, 0.447, 0.543, 0.549],  # Similar orientation to viewer camera
            "horizontal_aperture": 30.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor2",
        },
        # Front camera - from fireplace side, looking back at robot
        "external_sensor_3": {
            "position": [-1.8, -2.0, 1.0],  # Translated -2.0 in y, looking back at robot
            "orientation": [0.0, 0.0, 0.0, 1.0],  # Looking straight back at robot
            "horizontal_aperture": 25.0,
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
        enable_dump_filters=False,
    )

    robot = env.robots[0]
    # set viewer camera - user-provided position
    # Previous camera positions (saved for reference):
    # Position 1: [1.616339921951294, -0.010105276480317116, 1.035615086555481]
    #            Orientation: [0.4517306983470917, 0.44665420055389404, 0.5430005192756653, 0.5491719841957092]
    # Position 2: [-0.4424091577529907, -3.245443105697632, 0.7127112150192261]
    #            Orientation: [0.6173340678215027, 0.136087104678154, 0.16680358350276947, 0.7566739916801453]
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-0.37351322174072266, -0.9105080366134644, 0.9984497427940369]),
        orientation=th.tensor([0.1866627037525177, 0.5293360948562622, 0.7805155515670776, 0.2752378284931183]),
    )
    for _ in range(10): og.sim.step()

    # Franka default joint positions (7 arm joints + 2 gripper joints)
    robot.set_joint_positions(th.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]))

    # Set initial states for objects (onFire for logs inside fireplace)
    log_center = env.scene.object_registry("name", "log_center")
    log_left = env.scene.object_registry("name", "log_left")
    target_object = env.scene.object_registry("name", "target_object")
    
    # Set onFire state for logs inside fireplace
    if log_center is not None and object_states.OnFire in log_center.states:
        log_center.states[object_states.OnFire].set_value(True)
    if log_left is not None and object_states.OnFire in log_left.states:
        log_left.states[object_states.OnFire].set_value(True)
    # target_object should not be on fire initially (it's outside)
    
    # Step simulation to allow states to take effect
    for _ in range(20):
        og.sim.step()

    # Camera teleoperation with WASD keys
    class CustomCameraMover(CameraMover):
        @property
        def input_to_command(self):
            return {
                lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
                lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }

    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()

    # Keyboard Teleop
    action_generator = KeyboardRobotController(robot=robot)
    
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )
    action_generator.print_keyboard_teleop_info()

    fireplace = env.scene.object_registry("name", "fireplace")
    print(f"Fireplace position: {fireplace.get_position_orientation()[0] if fireplace else 'Not found'}")
    
    # Let simulation settle, then set fireplace fixed_base to True
    for _ in range(50):
        og.sim.step()
    
    if fireplace is not None:
        fireplace.fixed_base = True
        fireplace.keep_still()
        print("Fireplace fixed_base set to True after settling")
    
    # ======================== Data collection ========================
    n_episodes = 1
    for i in range(n_episodes):
        print(f"Episode {i} starts")
        # if load a state
        if args.load_state:
            # state = th.tensor(load_f["data/demo_0/state"][700])
            # og.sim.load_state(state, serialized=True)
            breakpoint()
            for _ in range(50): og.sim.step()

        while True:
            action, keypress_str = action_generator.get_teleop_action()
            
            # Skip robot action if camera movement keys are pressed (let camera mover handle them)
            # Camera mover handles these keys via its keyboard event subscription
            camera_keys = {"W", "A", "S", "D", "G"}
            if keypress_str and keypress_str in camera_keys:
                # Just step the environment without robot action to allow camera movement to be visible
                env.step(th.zeros(robot.action_dim))
                continue
            
            if keypress_str == "TAB":
                # Print viewer camera position and orientation
                cam_pos, cam_orn = og.sim.viewer_camera.get_position_orientation()
                print("=" * 50)
                print("Viewer Camera Position:")
                print(f"  Position: {cam_pos.tolist()}")
                print(f"  Orientation: {cam_orn.tolist()}")
                print("=" * 50)
                breakpoint()
                # break
            env.step(action)
    print("Data saved")
    env.save_data()
    camera_mover.clear()
    og.shutdown()

if __name__ == "__main__":
    __main__()

