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

from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper

gm.USE_GPU_DYNAMICS=True
gm.ENABLE_TRANSITION_RULES = False


TASK_OBJECTS = {
    "pedestal_table": {
        "type": "DatasetObject",
        "name": "pedestal_table",
        "category": "pedestal_table",
        "model": "djflkd",
        "position": [-0.5, 0.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 1.0],
    },
    "vase": {
        "type": "DatasetObject",
        "name": "vase",
        "category": "vase",
        "model": "uuypot",
        "position": [-0.5, -1.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 0.5],
    },
    "swivel_chair": {
        "type": "DatasetObject",
        "name": "swivel_chair",
        "category": "swivel_chair",
        "model": "pkpcew",
        "position": [-0.5, 1.0, 0.50],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
}


def __main__():
    np.random.seed(0)
    th.manual_seed(0)
    
    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Overwrite any configs here
    cfg["scene"]["load_object_categories"] = ["floors", "walls", "breakfast_table"]
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

    # Generate external sensors config automatically
    # Get robot name and type from config to construct correct prim path
    robot_name = cfg["robots"][0].get("name", "tiago0")  # Default to "robot0" if not specified
    robot_type = cfg["robots"][0].get("type", "Tiago").lower()  # Get robot type, default to "tiago"
    
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

    # TODO: Set this 
    collect_hdf5_path = f"resources/teleop_data/nav_to_table.hdf5"
    if not os.path.exists(collect_hdf5_path):
        os.makedirs(os.path.dirname(collect_hdf5_path), exist_ok=True)


    env = DamageableEnvironment(configs=cfg)        
    env = DamageableDataCollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
        # obj_attr_keys=["scale", "visible"],
        enable_dump_filters=False,
    )
    robot = env.robots[0]
    
    # set viewer camera
    og.sim.viewer_camera.set_position_orientation(position=th.tensor([-0.2607, -3.0889,  1.2703]), orientation=th.tensor([ 0.5051, -0.0412, -0.0701,  0.8592])) 
    for _ in range(10): og.sim.step()

    # robot.set_joint_positions(th.tensor([ 8.5932e-02, -1.2322e+00,  3.4371e-04,  3.5505e-07, -2.1036e-07,
    #         -8.5916e-01,  2.3161e-01,  9.0194e-01, -6.3965e-01, -2.7559e-04,
    #         -4.5575e-01,  3.2337e-01, -4.4725e-01,  2.2092e+00,  1.7759e+00,
    #         1.6471e+00,  1.4742e+00,  7.9141e-01,  1.0946e+00, -8.2607e-01,
    #         -6.2780e-01, -1.0932e+00,  1.5404e+00,  4.5000e-02,  4.5000e-02,
    #         4.5000e-02,  4.5000e-02]))
    robot.set_joint_positions(th.tensor([0.0, -1.0]), indices=robot.camera_control_idx)
    robot.set_position_orientation(position=th.tensor([-1.5, 0.0, 0.0]))
    robot.set_joint_positions(robot.default_arm_poses["vertical"], indices=robot.arm_control_idx["right"])

    pedestal_table = env.scene.object_registry("name", "pedestal_table")
    # pedestal_table.root_link.mass = 100.0
    vase = env.scene.object_registry("name", "vase")
    vase_ontop_pedestal = False
    counter = 0
    while not vase_ontop_pedestal:
        vase.states[object_states.OnTop].set_value(pedestal_table, True)
        vase.keep_still()
        for _ in range(70): og.sim.step()
        counter += 1
        print(f"trial: {counter}")
        vase_ontop_pedestal = vase.states[object_states.OnTop].get_value(pedestal_table)

    # breakpoint()
    
    # Telemoma: Teleoperate robot
    arm_teleop_method = "keyboard"
    base_teleop_method = "keyboard" 
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

    # ======================== Data collection ========================
    n_episodes = 1
    for i in range(n_episodes):
        print(f"Episode {i} starts")
        breakpoint()
        while True:
            # Not using telemoma for now
            # action = teleop_sys.get_action(teleop_sys.get_obs())
            action, keypress_str = action_generator.get_teleop_action()
            if keypress_str == "TAB":
                breakpoint()
                break
            env.step(action)
    print("Data saved")
    env.save_data()
    og.shutdown()

if __name__ == "__main__":
    __main__()