from ast import Pass
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import yaml
import json
import h5py
import torch as th
import numpy as np
import math
import pickle
import omnigibson as og
from omnigibson import object_states
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm
from omnigibson.controllers.controller_base import IsGraspingState

from telemoma.configs.base_config import teleop_config
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
import omnigibson.lazy as lazy

from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.damageable_env import DamageableDataPlaybackWrapper, DamageableDataCollectionWrapper

gm.USE_GPU_DYNAMICS=True
gm.ENABLE_TRANSITION_RULES = False

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

# rotate laptop 180 degrees around z-axis
LAPTOP_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
LAPTOP_INIT_ORI_180 = [0.0, 0.0, 1.0, 0.0]
TASK_OBJECTS = {
    "laptop": {
        "type": "DatasetObject",
        "name": "laptop",
        "category": "laptop",
        "model": "nvulcs",
        "position": [1.4, 0.4, 0.8],
        "orientation": LAPTOP_INIT_ORI_180,
        "scale": [1.0, 1.0, 1.0],
    },
    "cup1": {
        "type": "DatasetObject",
        "name": "coffee_cup_1",
        "category": "coffee_cup",
        "model": "rypdvd",
        "position": [1.4, 0.2, 0.8],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.2, 1.2, 1.0],
    },
    "teapot": {
        "type": "DatasetObject",
        "name": "teapot",
        "category": "teapot",
        "model": "jlalfc",
        "position": [1.4, 0.6, 0.8],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
}

def add_water(env, robot):
    laptop = env.scene.object_registry("name", "laptop")
    water_system = env.scene.get_system("water", force_init=True)
    # Generate water particles in batches with env.step() for proper simulation
    laptop_pos, _ = laptop.get_position_orientation()
    z_offset = -0.03
    for _ in range(100):
        drop_pos = (laptop_pos + th.tensor([0.0, 0.0, z_offset], dtype=th.float32)).tolist()
        water_system.generate_particles(positions=[drop_pos])
        # robot.set_joint_positions(robot_joint_positions)
        # robot.set_joint_velocities(th.zeros(robot.n_dof))
        # robot.keep_still()
        og.sim.step()
    print(f"Total particles: {water_system.n_particles}")

def set_laptop_pose(env, target_deg: float = 130.0):
    """Open the laptop to a specified angle."""
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
    cfg["scene"]["not_load_object_categories"] = ["straight_chair"]
    # cfg["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]


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

    # Load the environment
    env = og.Environment(configs=cfg)
   
    robot = env.robots[0]
    # for _ in range(10): og.sim.step()

    # set viewer camera
    # (tensor([ 1.8787, -0.4682,  1.6802]), tensor([0.4737, 0.1539, 0.2679, 0.8247]))
    # (Pdb) og.sim.viewer_camera.horizontal_aperture
    # 20.9950008392334
    og.sim.viewer_camera.set_position_orientation(position=th.tensor([ 1.8787, -0.4682,  1.6802]), orientation=th.tensor([0.4737, 0.1539, 0.2679, 0.8247]))
    
    # Telemoma: Teleoperate robot
    arm_teleop_method = "spacemouse"
    base_teleop_method = "spacemouse" 
    teleop_config.arm_left_controller = arm_teleop_method
    teleop_config.arm_right_controller = arm_teleop_method
    teleop_config.base_controller = base_teleop_method
    teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
    teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.02}
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

    # setting all objects
    set_laptop_pose(env, target_deg=100.0)
    for _ in range(10): og.sim.step()

    breakpoint()
    # load state
    with open("resources/saved_states/pull_figure_init_state.pkl", "rb") as f:
        state_flat_array = pickle.load(f)
    og.sim.load_state(state_flat_array, serialized=True)
    for _ in range(10): og.sim.step()

    breakpoint()
    
    add_water(env, robot)
    for _ in range(10): og.sim.step()

    # ======================== Data collection ========================
    n_episodes = 10
    last_telemoma_grip_action = 1.0 
    for i in range(n_episodes):
        print(f"Episode {i} starts")
        action_generator.current_keypress = None
        # If the robot is grasping, set the persistent gripper action to -1.0
        if robot.is_grasping().value == IsGraspingState.TRUE:
            action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
        action = th.zeros(robot.action_dim)
        action[-1] = -1.0
        breakpoint()
        while True:
            telemoma_action = teleop_sys.get_action(teleop_sys.get_obs())
            telemoma_grip_action = telemoma_action[-1]
            if telemoma_grip_action != last_telemoma_grip_action:
                action[-1] = -action[-1]
            last_telemoma_grip_action = telemoma_grip_action
            action[:-1] = telemoma_action[:-1]

            _, keypress_str = action_generator.get_teleop_action()
            if keypress_str == "TAB":
                breakpoint()
                break
            env.step(action)

    breakpoint()
    og.shutdown()
 

if __name__ == "__main__":
    __main__()