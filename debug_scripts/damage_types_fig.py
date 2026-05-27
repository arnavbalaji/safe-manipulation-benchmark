from ast import Pass
import os
import cv2
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import yaml
import json
import h5py
import torch as th
import numpy as np
import math
import pickle
import argparse
import omnigibson as og
from omnigibson import object_states
from collections import defaultdict
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm
from omnigibson.controllers.controller_base import IsGraspingState

from telemoma.configs.base_config import teleop_config
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
import omnigibson.lazy as lazy
from safety_benchmark.utils.misc_utils import (
    save_rgb_camera_video,
    save_rgb_health_video_with_overlay,
    save_rgb_force_video,
)

from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.damageable_env import DamageableDataPlaybackWrapper, DamageableDataCollectionWrapper

gm.USE_GPU_DYNAMICS=False
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

TASK_OBJECTS = {
    "drawer": {
        "type": "DatasetObject",
        "name": "drawer",
        "category": "bottom_cabinet",
        "model": "bamfsz",
        "position": [6.0, 0.2, 0.2],
        "orientation": [0.0, 0.0, 0.70710678, 0.70710678],
    },
}

def get_visualization_config(task_name, robot_name):
    if task_name == "shelve_item":
        return {
            "target_objects_health_with_links": [f"{robot_name}@eef_link", f"{robot_name}@panda_hand", f"{robot_name}@panda_leftfinger", f"{robot_name}@panda_rightfinger", "box_of_crackers@base_link", "wineglass@base_link", "laptop@base_link", "laptop@link_0"],
            "target_objects_health": [robot_name, "box_of_crackers", "wineglass", "laptop"],
            "target_objects_forces": [f"{robot_name}@eef_link", f"{robot_name}@panda_hand", f"{robot_name}@panda_leftfinger", f"{robot_name}@panda_rightfinger"],
            "force_keys": ["filtered_qs_forces"], # options: ["impact_forces", "filtered_qs_forces"]
            "target_contact_bodies": ["stand"]
        }

def __main__():
    np.random.seed(0)
    th.manual_seed(0)
    # th.cuda.manual_seed(0)
    # th.backends.cudnn.benchmark = False
    # th.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_hdf5_path', type=str, help='Target hdf5 path', default="resources/teleop_data/temp_drawer.hdf5")
    parser.add_argument('--playback_hdf5_path', type=str, help='Output hdf5 path', default="resources/playback_data/temp_drawer_playback.hdf5")
    parser.add_argument('--n_episodes', type=int, help='Number of episodes', default=1)
    parser.add_argument('--teleop', action='store_true', help='Teleoperate the robot')
    parser.add_argument('--playback', action='store_true', help='Playback the data')
    parser.add_argument('--visualize', action='store_true', help='Visualize the data')
    parser.add_argument('--compute_metrics', action='store_true', help='Compute metrics')
    parser.add_argument('--task_name', type=str, help='Task name', default="shelve_item")
    parser.add_argument('--live_feedback', action='store_true', help='Show live health graph window during teleop (use with --teleop)')
    parser.add_argument('--high_resolution', action='store_true', help='Use high resolution video')
    args = parser.parse_args()
    
    
    if args.teleop:
        # Load the pre-selected configuration and set the online_sampling flag
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Overwrite any configs here
        # cfg["scene"]["type"] = "Scene"
        # cfg["scene"]["scene_id"] = "empty"
        # cfg["scene"]["not_load_object_categories"] = ["straight_chair"]
        cfg["scene"]["load_object_categories"] = ["floors", "walls"]


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
                # (tensor([-0.7870,  0.0134,  1.1996]), tensor([0.1538, 0.3706, 0.8460, 0.3510]))
                "position": [-0.7870,  0.0134,  1.1996],
                "orientation": [ 0.1538, 0.3706, 0.8460, 0.3510],
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
        cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

        # Load the environment
        # env = og.Environment(configs=cfg)
        env = DamageableEnvironment(configs=cfg)        
        env = DamageableDataCollectionWrapper(
            env=env,
            output_path=args.collect_hdf5_path,
            only_successes=False,
            enable_dump_filters=False,
        )
    
        robot = env.robots[0]
        drawer = env.scene.object_registry("name", "drawer")
        # (tensor([-1.7864, -0.4620,  0.3640]), tensor([-5.6583e-06, -1.0400e-04,  7.5162e-03,  9.9997e-01]))
        drawer.set_position_orientation(position=th.tensor([-1.7864, -0.4620,  0.3640]), orientation=th.tensor([-5.6583e-06, -1.0400e-04,  7.5162e-03,  9.9997e-01]))
        #  (tensor([-8.5206e-01, -8.0459e-01,  3.4381e-04]), tensor([6.7110e-08, 5.6126e-08, 9.9992e-01, 1.2630e-02]))
        robot.set_position_orientation(position=th.tensor([-8.5206e-01, -8.0459e-01,  3.4381e-04]), orientation=th.tensor([6.7110e-08, 5.6126e-08, 9.9992e-01, 1.2630e-02]))
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

        # ======================== Data collection ========================
        n_episodes = 1
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
                    # obj = env.scene.object_registry("name", "bottom_cabinet_bamfsz_1")
                    # obj.set_highlight_properties(color=[255.0, 0.0, 0.0], intensity=5.0)
                    # obj.highlighted = True
                    break
                env.step(action)

        env.save_data()
        og.shutdown()

    if args.playback:
        robot_name = "franka0"
        robot_type = "FrankaPanda"
        if args.high_resolution:
            image_height = 720
            image_width = 1280
            horizontal_aperture = 20.0
        else:
            image_height = 256
            image_width = 256
            horizontal_aperture = 15.0
        # Set external cameras for videos
        EXTERNAL_CAMERA_CONFIGS = {
            # Side camera (fixed to base_link frame)
            "external_sensor_0": {
                "position": [7.3920, -0.6436, 1.7519],
                "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
                "horizontal_aperture": horizontal_aperture,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
            },
            # Left Shoulder (fixed to base_link frame)
            "external_sensor_1": {
                # wrt base frame
                "position": [7.1264, 1.1205, 2.0117],
                "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
                "horizontal_aperture": horizontal_aperture,
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

    if args.visualize or args.compute_metrics:
        f = h5py.File(args.playback_hdf5_path, "r")
        scene_file = json.loads(f["data"].attrs["scene_file"])
        robot_name = "franka0"       
        camera_type = "external"
        camera_name = "external_sensor1"

        output_video_dir = f"resources/videos/{args.playback_hdf5_path.split('/')[-1].split('.')[0]}"
        os.makedirs(output_video_dir, exist_ok=True)

        visualization_config = get_visualization_config("shelve_item", robot_name)
        target_objects_health_with_links = visualization_config["target_objects_health_with_links"]
        target_objects_health = visualization_config["target_objects_health"]
        target_objects_forces = visualization_config["target_objects_forces"]
        target_contact_bodies = visualization_config["target_contact_bodies"]
        force_keys = visualization_config["force_keys"]

        final_obj_healths = defaultdict(list)
        final_env_healths = []
        
        for idx in range(len(f["data"])):
            print("Episode: ", idx)
            demo_idx = int(list(f["data"].keys())[idx].split("_")[-1])
            # Parse info to obtain relevant information for visualization
            obs_info_list = []
            for i in range(len(f[f"data/demo_{demo_idx}/info/obs_info"])):            
                # Obtain observation information 
                obs_info = json.loads(f[f"data/demo_{demo_idx}/info/obs_info"][i].decode("utf-8"))
                obs_info_list.append(obs_info)

            # Obtain health information for the target objects per link
            all_obj_healths = np.array(f[f"data/demo_{demo_idx}/obs/health"])
            health_list_link_names = f[f"data/demo_{demo_idx}"].attrs["health_list_link_names"]
            health = dict()
            for obj_name in target_objects_health_with_links:
                health[obj_name] = all_obj_healths[:, np.where(health_list_link_names == obj_name)[0][0]]
                health[obj_name] = health[obj_name][1:]
            # breakpoint()

            # Obtain health information for the entire target objects 
            for obj_name in target_objects_health:
                arrays = [v for k, v in health.items() if k.startswith(f"{obj_name}@")]
                # Compute element-wise min
                if arrays:
                    health[obj_name] = np.minimum.reduce(arrays)
                else:
                    health[obj_name] = None

            # remove later
            health["laptop"][410:] = 0.0

            if args.compute_metrics:
                print("Episode: ", demo_idx)
                current_env_health = 0.0
                for obj_name in target_objects_health:
                    final_obj_healths[obj_name].append(health[obj_name][-1])
                    print(f"{obj_name} health: {health[obj_name][-1]}")
                    current_env_health += health[obj_name][-1]
                final_env_healths.append(current_env_health / len(target_objects_health))
                print(f"Current environment health: ", final_env_healths[-1])
            
            if args.visualize:
                # Save video for rgb camera
                output_video_path = f"{output_video_dir}/demo_{demo_idx}_camera_video"
                imgs = f[f"data/demo_{demo_idx}/obs/{camera_type}::{camera_name}::rgb"]
                imgs = imgs[1:]
                new_imgs = []
                imgs_seg_instance = f[f"data/demo_{demo_idx}/obs/{camera_type}::{camera_name}::seg_instance"]
                imgs_seg_instance = imgs_seg_instance[1:]

                for i, img in enumerate(imgs):
                    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
                    img_seg_instance = imgs_seg_instance[i]
                    obs_info = obs_info_list[i]
                    for obj_name in target_objects_health:
                        seg_instance_info = obs_info[camera_type][camera_name]["seg_instance"]
                        seg_instance_key = int(next((k for k, v in seg_instance_info.items() if v == obj_name), -1))

                        # # If binary visualization, set to red if 0
                        # if health[obj_name][i] is not None and health[obj_name][i] == 0.0:
                        #     mask = img_seg_instance == seg_instance_key
                        #     overlay_color = np.array([0, 0, 255], dtype=np.uint8)  # BGR
                        #     img[mask] = overlay_color

                        # If continuous visualization, set to different shades of red if < 100
                        if health[obj_name][i] is not None and health[obj_name][i] < 100:
                            mask = img_seg_instance == seg_instance_key
                            alpha = 1 - health[obj_name][i] / 100.0  # 0 = full health, 1 = dead
                            overlay_color = np.array([0, 0, 255], dtype=np.uint8)  # BGR
                            img[mask] = ((1 - alpha) * img[mask] + alpha * overlay_color).astype(np.uint8)
                    
                    # Convert back to RGB for video saving functions (which expect RGB format)
                    new_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                imgs = np.array(new_imgs)
                # breakpoint()
                # save_rgb_camera_video(output_video_path=output_video_path, imgs=imgs)
            
                # Obtain forces information for the target objects
                # target_objects_forces = [f"{robot_name}@right_gripper_link", f"{robot_name}@right_gripper_finger_link1", f"{robot_name}@right_gripper_finger_link2"]
                # target_objects_forces = [f"{robot_name}@left_gripper_link", f"{robot_name}@left_gripper_finger_link1", f"{robot_name}@left_gripper_finger_link2"]
                # target_objects_forces = [f"microwave_hjjxmi_0@base_link", f"microwave_hjjxmi_0@link_0", f"microwave_hjjxmi_0@glass"]
                # target_objects_forces = [f"microwave_hjjxmi_0@base_link", f"microwave_hjjxmi_0@link_0", "microwave_hjjxmi_0@glass", f"{robot_name}@right_gripper_finger_link1", f"{robot_name}@right_gripper_finger_link2"]
                # options: ["unfiltered_raw_sim_forces", "filtered_raw_sim_forces", "unfiltered_qs_forces", "filtered_qs_forces"]
                # force_keys = ["filtered_raw_sim_forces"]
                data = dict()
                for obj_name in target_objects_forces:
                    data[obj_name] = dict()
                    for force_key in force_keys:
                        data[obj_name][force_key] = []
                for i in range(len(f[f"data/demo_{demo_idx}/info/damage_info"])):
                    damage_info = json.loads(f[f"data/demo_{demo_idx}/info/damage_info"][i].decode("utf-8"))
                    for obj_name in target_objects_forces:
                        for force_key in force_keys:
                            data[obj_name][force_key].append(damage_info[obj_name.split("@")[0]][obj_name.split("@")[1]]["mechanical"][force_key])
                
                # print max forces for each object
                for obj_name in target_objects_forces:
                    max_force = max(data[obj_name][force_keys[0]])
                    print(f"{obj_name} max force: {max_force}")
                # breakpoint()
                
                # # Save videos for forces plot
                # forces_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_forces_video.mp4")
                # save_rgb_force_video(output_video_path=forces_video_path, imgs=imgs, target_objects=target_objects_forces, data=data, forces_to_plot=force_keys)

                # # Save video for health plot (with separate plot panel)
                # health_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_health_video.mp4")
                # save_rgb_health_video(output_video_path=health_video_path, imgs=imgs, target_objects=target_objects_health, health=health)
                
                save_rgb_camera_video(output_video_path=os.path.join(output_video_dir, f"demo_{demo_idx}_camera_video.mp4"), imgs=imgs, fps=30)
                
                # Save video for health with overlay bars (bars on video, no separate plot)
                health_overlay_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_health_overlay_video.mp4")

                save_rgb_health_video_with_overlay(
                    output_video_path=health_overlay_video_path,
                    imgs=imgs,
                    target_objects=target_objects_health,
                    health=health,
                    position="bottom_left",
                    n_columns=2,  # Use 2 columns to spread out the health bars
                    fps=30
                )
 

if __name__ == "__main__":
    __main__()