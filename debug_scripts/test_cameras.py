import os
import yaml
import omnigibson as og
import numpy as np
import omnigibson.utils.transform_utils as T
import torch as th
# Load the pre-selected configuration and set the online_sampling flag
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

# Overwrite any configs here
cfg["scene"]["load_object_categories"] = ["floors", "walls", "breakfast_table"]
cfg["robots"][0] = {
    "type": "FrankaPanda",
    "name": "franka0",
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

env = og.Environment(configs=cfg)
env.reset()
robot = env.robots[0]
breakpoint()

# Get the camera names
camera_names = env.simulator.sensors["camera_0"].get_camera_names()
print(camera_names)

# Get the camera extrinsics
cam_pos, cam_ori = robot.sensors["franka0:eef_link:Camera:0"].get_position_orientation()
cam_ori = T.quat2mat(cam_ori)
T_camera_wrt_world = np.eye(4)
T_camera_wrt_world[:3, :3] = cam_ori
T_camera_wrt_world[:3, 3] = cam_pos
print(T_camera_wrt_world)

robot_pos, robot_ori = robot.get_position_orientation()
T_robot_wrt_world = np.eye(4)
T_robot_wrt_world[:3, :3] = T.quat2mat(robot_ori)
T_robot_wrt_world[:3, 3] = robot_pos
print(T_robot_wrt_world)

T_camera_wrt_robot = np.linalg.inv(T_robot_wrt_world) @ T_camera_wrt_world
camera_pos, camera_ori = T.mat2pose(th.from_numpy(T_camera_wrt_robot))
print(camera_pos, camera_ori)
# tensor([0.2945, 0.0018, 0.4901])

joint_positions = robot.get_joint_positions()
# tensor([-7.4434e-10, -1.3000e+00,  3.2457e-09, -2.8700e+00,  1.6150e-10,
#          2.0000e+00,  7.5000e-01,  4.0000e-02,  4.0000e-02])