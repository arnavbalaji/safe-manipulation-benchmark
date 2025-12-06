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

gm.USE_GPU_DYNAMICS=True
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
    # "cup": {
    #     "type": "DatasetObject",
    #     "name": "cup",
    #     "category": "water_glass",
    #     "model": "ewgotr",
    #     "position": [0.9, 0.0, 0.10],
    #     "orientation": [0.0, 0.0, 0.0, 1.0],
    #     "scale": [1.0, 1.0, 1.0],
    #     "damage_params": PARAMS.get("cup", PARAMS["default"]),
    # },
    # "banana": {
    #     "type": "DatasetObject",
    #     "name": "banana",
    #     "category": "banana",
    #     "model": "vvyyyv",
    #     "position": [0.9, 0.0, 0.10],
    #     "orientation": [0.0, 0.0, 0.0, 1.0],
    #     "scale": [1.0, 1.0, 1.0],
    # },
    # "cup": {
    #     "type": "DatasetObject",
    #     "name": "coffee_cup",
    #     "category": "coffee_cup",
    #     "model": "rypdvd",
    #     "position": [0.9, 0.0, 0.10],
    #     "orientation": [0.0, 0.0, 0.0, 1.0],
    #     # "scale": [1.0, 1.0, 1.0],
    # },
    # "breakfast_table": {
    #     "type": "DatasetObject",
    #     "name": "breakfast_table",
    #     "category": "breakfast_table",
    #     "model": "skczfi",
    #     "position": [1.0, 0.0, 0.10],
    #     "orientation": [0.0, 0.0, 0.0, 1.0],
    #     "scale": [1.0, 1.0, 1.0],
    # },
    # "coffee_table": {
    #     "type": "DatasetObject",
    #     "name": "coffee_table",
    #     "category": "coffee_table",
    #     "model": "aoojzy",
    #     "position": [0.0, -2.0, 0.0],
    #     "orientation": [0, 0, 0.7071068, 0.7071068],
    #     "scale": [1.0, 1.0, 1.0],
    # },
    # "plate": {
    #     "type": "DatasetObject",
    #     "name": "plate",
    #     "category": "plate",
    #     "model": "ntedfx",
    #     "position": [0.8, 0.0, 0.1],
    #     "orientation": [0, 0, 0, 1],
    #     "scale": [0.6, 0.6, 0.6],
    # },
    # "lamp": {
    #     "type": "DatasetObject",
    #     "name": "floor_lamp",
    #     "category": "floor_lamp",
    #     "model": "vdxlda",
    #     "position": [1.0, 0.0, 0.10],
    #     "orientation": [0.0, 0.0, 0.0, 1.0],
    #     "scale": [1.0, 1.0, 1.5],
    # },
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
    # th.cuda.manual_seed(0)
    # th.backends.cudnn.benchmark = False
    # th.backends.cudnn.deterministic = True
    
    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Overwrite any configs here
    # cfg["scene"]["type"] = "Scene"
    # cfg["scene"]["scene_id"] = "empty"
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
    
    # External camera parameters
    # EXTERNAL_CAMERA_CONFIGS = {
    #     "external_sensor_0": {
    #         "position": [-0.4, 0, 2.0],
    #         "orientation": [ 0.2706, -0.2706, -0.6533,  0.6533],
    #     },
    #     "external_sensor_1": {
    #         "position": [-0.2, 0.6, 2.0],
    #         "orientation": [-0.1930, 0.4163, 0.8062, -0.3734],
    #     },
    #     "external_sensor_2": {
    #         "position": [-0.2, -0.6, 2.0],
    #         "orientation": [0.4164, -0.1929, -0.3737, 0.8060],
    #     }
    # }
    EXTERNAL_CAMERA_CONFIGS = {
        # # Side camera (fixed to base_link frame)
        "external_sensor_0": {
            # "position": [-0.4, 0, 2.0],
            # "orientation": [ 0.2706, -0.2706, -0.6533,  0.6533],
            "position": [0.4859, -1.8219,  1.1402],
            "orientation": [ 0.5857, -0.0093, -0.0129,  0.8103],
            "horizontal_aperture": 10.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
        },
        # Left Shoulder (fixed to eyes frame)
        # 1:
        # (Pdb) env.external_sensors["external_sensor1"].get_position_orientation(frame="world")
        # (tensor([ 0.6069, -1.0641,  1.4121]), tensor([-0.0257,  0.4546,  0.8889, -0.0499]))
        # (Pdb) env.external_sensors["external_sensor1"].get_position_orientation(frame="parent")
        # (tensor([0.2130, 0.5043, 1.3359]), tensor([-0.2127,  0.4026,  0.7874, -0.4156]))
        # 2:
        # (Pdb) env.external_sensors["external_sensor1"].get_position_orientation(frame="world")
        # (tensor([ 0.7368, -1.1771,  1.5148]), tensor([0.0655, 0.3936, 0.9044, 0.1510]))
        # (Pdb) env.external_sensors["external_sensor1"].get_position_orientation(frame="parent")
        # (tensor([0.3833, 0.5289, 1.4386]), tensor([-0.1044,  0.3851,  0.8851, -0.2394]))
        "external_sensor_1": {
            # originally
            "position": [-0.2, 0.6, 2.0],
            "orientation": [-0.1930, 0.4163, 0.8062, -0.3734],
            # wrt eyes frame 
            "position": [-0.4819,  0.3097,  0.2396],
            "orientation": [ 0.1083, -0.3191, -0.3823,  0.8604],
            # wrt base frame
            "position": [0.2522, 0.0470, 1.0696],
            "orientation": [ 0.1991, -0.1991, -0.6785,  0.6785],
            "horizontal_aperture": 30.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1",
        },
        # Back camera (fixed to base_link frame)
        "external_sensor_2": {
            # "position": [-0.2, -0.6, 2.0],
            # "orientation": [0.4164, -0.1929, -0.3737, 0.8060],
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
        # Construct prim path: /controllable__{robot_type}__{robot_name}/base_link/external_sensor{i}
        # This attaches the camera to the robot's base_link, so it moves with the robot
        external_sensors_config.append({
            "sensor_type": "VisionSensor",
            "name": f"external_sensor{i}",
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor{i}",
            # "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/head_2_link/external_sensor{i}",
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {
                "image_height": 720,
                "image_width": 720,
                "horizontal_aperture": 40.0,
            },
            "position": th.tensor(position, dtype=th.float32),
            "orientation": th.tensor(orientation, dtype=th.float32),
            "pose_frame": "parent",  # Position/orientation are relative to parent (base_link)
        })
    
    cfg["env"]["external_sensors"] = external_sensors_config
    # Add objects here
    cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]
    target_object_name = "swivel_chair"

    # Load the environment
    # env = og.Environment(configs=cfg)
    teleop = False
    if teleop:
        env = DamageableEnvironment(configs=cfg)
        collect_hdf5_path = f"outputs/{target_object_name}.hdf5"
        env = DataCollectionWrapper(
            env=env,
            output_path=collect_hdf5_path,
            only_successes=False,
            # obj_attr_keys=["scale", "visible"],
            enable_dump_filters=False,
        )
    else:
        collect_hdf5_path = f"outputs/{target_object_name}.hdf5"
        output_hdf5_path = f"outputs/{target_object_name}_playback.hdf5"
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
        env = DataPlaybackWrapper.create_from_hdf5(
            input_path=collect_hdf5_path,
            output_path=output_hdf5_path,
            # robot_obs_modalities=["proprio", "rgb", "depth", "seg_instance"],
            robot_sensor_config=robot_sensor_config,
            external_sensors_config=external_sensors_config,
            n_render_iterations=1,
            only_successes=False,
            exclude_sensor_names=["left_eef_link", "right_eef_link"]
            # env=env,
        )
   
    robot = env.robots[0]
    for _ in range(10): og.sim.step()

    # set viewer camera
    og.sim.viewer_camera.set_position_orientation(position=th.tensor([-0.2607, -3.0889,  1.2703]), orientation=th.tensor([ 0.5051, -0.0412, -0.0701,  0.8592]))
    
    for _ in range(10): og.sim.step()

    if teleop:
        # Slightly further away from the glass
        # TODO: Set head positions
        # robot.set_joint_positions(th.tensor([ 8.5932e-02, -1.2322e+00,  3.4371e-04,  3.5505e-07, -2.1036e-07,
        #         -8.5916e-01,  2.3161e-01,  9.0194e-01, -6.3965e-01, -2.7559e-04,
        #         -4.5575e-01,  3.2337e-01, -4.4725e-01,  2.2092e+00,  1.7759e+00,
        #         1.6471e+00,  1.4742e+00,  7.9141e-01,  1.0946e+00, -8.2607e-01,
        #         -6.2780e-01, -1.0932e+00,  1.5404e+00,  4.5000e-02,  4.5000e-02,
        #         4.5000e-02,  4.5000e-02]))
        robot.set_joint_positions(th.tensor([0.0, -1.0]), indices=robot.camera_control_idx)
        robot.set_position_orientation(position=th.tensor([-1.5, 0.0, 0.0]))
        robot.set_joint_positions(robot.default_arm_poses["vertical"], indices=robot.arm_control_idx["right"])

        target_object = env.scene.object_registry("name", target_object_name)
        # target_object.root_link.mass = 2.0
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
        # target_object.states[object_states.OnTop].set_value(env.scene.object_registry("name", "coffee_table"), True)

        # breakpoint()
        
        # Teleoperate robot
        arm_teleop_method = "keyboard"
        base_teleop_method = "keyboard" 
        # Generate teleop config
        teleop_config.arm_left_controller = arm_teleop_method
        teleop_config.arm_right_controller = arm_teleop_method
        teleop_config.base_controller = base_teleop_method
        teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
        teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.02}

        # Initialize teleoperation system
        teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=False)
        teleop_sys.start()

        # To obtain keyboard events
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
        # breakpoint()
        env.set_object_params()
        env.external_sensors["external_sensor0"].horizontal_aperture = 30.0
        env.playback_dataset(record_data=True, replay_for_annotation=False)
        
    # breakpoint()
    # obtain obs
    # env.current_traj_history.keys()
    env.save_data()
    # breakpoint()
    og.shutdown()

if __name__ == "__main__":
    __main__()