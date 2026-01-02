from ast import Pass
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import cv2
import yaml
import json
import h5py
import torch as th
import numpy as np
import argparse
from collections import defaultdict

import omnigibson as og
from omnigibson import object_states
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm
import omnigibson.lazy as lazy

from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataPlaybackWrapper
from safety_benchmark.utils.misc_utils import save_rgb_camera_video, save_rgb_force_contact_video, save_rgb_force_video, save_rgb_health_video

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_TRANSITION_RULES = False

def get_visualization_config(task_name, robot_name):
    if task_name == "clean_a_trumpet": 
        return {
            "target_objects_health_with_links": [f"{robot_name}@left_gripper_link", f"{robot_name}@left_gripper_finger_link1", f"{robot_name}@left_gripper_finger_link2"],  # "scrub@base_link", "trumpet@base_link"
            "target_objects_health": [robot_name],  # "scrub", "trumpet"
            "target_objects_forces": [f"{robot_name}@left_gripper_link", f"{robot_name}@left_gripper_finger_link1", f"{robot_name}@left_gripper_finger_link2"],
            "force_keys": ["filtered_qs_forces"],
            "target_contact_bodies": ["table", "scrub"]
        }
    elif task_name == "make_microwave_popcorn": 
        return {
            "target_objects_health_with_links": [f"{robot_name}@right_gripper_link", f"{robot_name}@right_gripper_finger_link1", f"{robot_name}@right_gripper_finger_link2"],  # "scrub@base_link", "trumpet@base_link"
            "target_objects_health": [robot_name],  # "scrub", "trumpet"
            "target_objects_forces": [f"{robot_name}@right_gripper_link", f"{robot_name}@right_gripper_finger_link1", f"{robot_name}@right_gripper_finger_link2"],
            "force_keys": ["filtered_qs_forces"],
            "target_contact_bodies": ["microwave"]
        }    
    elif task_name == "attach_a_camera_to_a_tripod":
        return {
            "target_objects_health_with_links": [f"{robot_name}@right_gripper_link", f"{robot_name}@right_gripper_finger_link1", f"{robot_name}@right_gripper_finger_link2", "camera_tripod_86@base_link", "digital_camera_87@base_link"],  # "scrub@base_link", "trumpet@base_link"
            "target_objects_health": [robot_name, "camera_tripod_86", "digital_camera_87"],  # "scrub", "trumpet"
            "target_objects_forces": [f"{robot_name}@right_gripper_link", f"{robot_name}@right_gripper_finger_link1", f"{robot_name}@right_gripper_finger_link2", "camera_tripod_86@base_link", "digital_camera_87@base_link"],
            "force_keys": ["filtered_qs_forces", "impact_forces"],
            "target_contact_bodies": ["camera_tripod_86", "digital_camera_87"]
        }

def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, help='Task name') # e.g. rollout_0000_00402500
    parser.add_argument('--rollout_name', type=str, help='Rollout name')
    parser.add_argument('--output_fname', type=str, help='If want to overwrite the output hdf5 file')
    parser.add_argument('--visualize', action='store_true', help='Visualize the data')
    parser.add_argument('--playback', action='store_true', help='Playback the data')
    parser.add_argument('--compute_metrics', action='store_true', help='Compute metrics')
    args = parser.parse_args()

    from pathlib import Path

    # root_path = "/home/arpit"
    # print("Searching for Isaac-GR00T in ", root_path)
    # root = Path(root_path)
    # matches = list(root.rglob("Isaac-GR00T"))
    # for m in matches:
    #     if m.is_dir():
    #         print(m)
    # isaac_gr00t_path = m

    # If want to use a specific path, uncomment the following line
    isaac_gr00t_path = "/home/arpit/projects/Isaac-GR00T"

    # TODO: Set this 
    collect_hdf5_path = f"{isaac_gr00t_path}/gr00t/eval/sim/BEHAVIOR/rollouts/{args.task_name}/{args.rollout_name}.hdf5"
    if args.output_fname:
        output_hdf5_path = f"{isaac_gr00t_path}/gr00t/eval/sim/BEHAVIOR/rollouts/{args.task_name}/{args.output_fname}_playback.hdf5"
    else:
        output_hdf5_path = f"{isaac_gr00t_path}/gr00t/eval/sim/BEHAVIOR/rollouts/{args.task_name}/{args.rollout_name}_playback.hdf5"
    hdf5_file = h5py.File(collect_hdf5_path, "r")

    if args.playback:
    
        cfg = json.loads(hdf5_file["data"].attrs["config"])
        robot_type = cfg["robots"][0]["type"]
        robot_name = cfg["robots"][0]["name"]
        
        # External camera parameters
        EXTERNAL_CAMERA_CONFIGS = {
            # "external_sensor0": {
            #     "position": [-0.4, 0, 2.0],
            #     "orientation": [ 0.2706, -0.2706, -0.6533,  0.6533],
            # },
            # Left Shoulder (fixed to base_link frame)
            "external_sensor1": {
                "position": [-0.2, 0.6, 2.0],
                "orientation": [-0.1930, 0.4163, 0.8062, -0.3734],
            },
            # # Right Shoulder (fixed to base_link frame)
            # "external_sensor2": {
            #     "position": [-0.2, -0.6, 2.0],
            #     "orientation": [0.4164, -0.1929, -0.3737, 0.8060],
            # }
        }
        external_sensors_config = []
        for camera_name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
            external_sensors_config.append({
                "sensor_type": "VisionSensor",
                "name": camera_name,
                "relative_prim_path": f"/controllable__damageable{robot_type.lower()}__{robot_name}/base_link/{camera_name}",
                "modalities": ["rgb", "seg_instance"],
                "sensor_kwargs": {
                    "image_height": 720,
                    "image_width": 720,
                    # "horizontal_aperture": camera_cfg["horizontal_aperture"],
                },
                "position": th.tensor(camera_cfg["position"], dtype=th.float32),
                "orientation": th.tensor(camera_cfg["orientation"], dtype=th.float32),
                "pose_frame": "parent",
            })

        # if output_hdf5_path already exists, ask user if they want to overwrite it
        if os.path.exists(output_hdf5_path):
            overwrite = input(f"Output file {output_hdf5_path} already exists. Do you want to overwrite it? (y/n): ")
            if overwrite != "y":
                print("Exiting...")
                return
        
        env = DamageableDataPlaybackWrapper.create_from_hdf5(
            input_path=collect_hdf5_path,
            output_path=output_hdf5_path,
            # robot_obs_modalities=["proprio", "rgb", "depth", "seg_instance"],
            # robot_sensor_config=robot_sensor_config,
            # NOTE: comment this out to save space by not saving rgb images for all the episodes. Choose the ones you want to visualize later.
            external_sensors_config=external_sensors_config,
            n_render_iterations=1,
            only_successes=False,
            exclude_sensor_names=["left_eef_link", "right_eef_link"]
        )
        robot = env.robots[0]
        # breakpoint()
        # # set viewer camera
        # og.sim.viewer_camera.set_position_orientation(position=th.tensor([-0.2607, -3.0889,  1.2703]), orientation=th.tensor([ 0.5051, -0.0412, -0.0701,  0.8592]))
        for _ in range(10): og.sim.step()

        # Playback the dataset
        env.playback_dataset(record_data=True)
        # breakpoint()
            
        env.save_data()        
    
    # Visualize the episode
    if args.visualize or args.compute_metrics:
        f = h5py.File(output_hdf5_path, "r")
        scene_file = json.loads(f["data"].attrs["scene_file"])
        # robot_name = [obj_name for obj_name in scene_file["objects_info"]["init_info"].keys() if "robot" in obj_name.lower()][0]
        robot_name = "robot_r1"       
        camera_type = "external"
        camera_name = "external_sensor1"
        # breakpoint()

        output_video_dir = f"{isaac_gr00t_path}/gr00t/eval/sim/BEHAVIOR/rollouts/{args.task_name}/videos"
        os.makedirs(output_video_dir, exist_ok=True)
        visualization_config = get_visualization_config(args.task_name, robot_name)
        target_objects_health_with_links = visualization_config["target_objects_health_with_links"]
        target_objects_health = visualization_config["target_objects_health"]
        target_objects_forces = visualization_config["target_objects_forces"]
        target_contact_bodies = visualization_config["target_contact_bodies"]
        force_keys = visualization_config["force_keys"]

        final_obj_healths = defaultdict(list)
        final_env_healths = []
        
        for demo_idx in range(len(f["data"])):
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

            # Obtain health information for the entire target objects 
            for obj_name in target_objects_health:
                arrays = [v for k, v in health.items() if k.startswith(f"{obj_name}@")]
                # Compute element-wise min
                if arrays:
                    health[obj_name] = np.minimum.reduce(arrays)
                else:
                    health[obj_name] = None

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
                output_video_path = f"{output_video_dir}/{args.rollout_name}_demo_{demo_idx}_camera_video"
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
                    
                    new_imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                imgs = np.array(new_imgs)
                save_rgb_camera_video(output_video_path=output_video_path, imgs=imgs)
            
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
                
                # Save videos for forces plot
                forces_video_path = os.path.join(output_video_dir, f"{args.rollout_name}_demo_{demo_idx}_forces_video.mp4")
                save_rgb_force_video(output_video_path=forces_video_path, imgs=imgs, target_objects=target_objects_forces, data=data, forces_to_plot=force_keys)

                # Save video for health plot
                health_video_path = os.path.join(output_video_dir, f"{args.rollout_name}_demo_{demo_idx}_health_video.mp4")
                save_rgb_health_video(output_video_path=health_video_path, imgs=imgs, target_objects=target_objects_health, health=health)

                # Obtain contact information for the target objects
                if args.task_name in ["clean_a_trumpet", "make_microwave_popcorn"]:
                    contact_info = dict()
                    for obj_name in target_objects_forces:
                        contact_info[obj_name] = []
                    for i in range(len(f[f"data/demo_{demo_idx}/info/damage_info"])):
                        damage_info = json.loads(f[f"data/demo_{demo_idx}/info/damage_info"][i].decode("utf-8"))
                        for obj_full_name in target_objects_forces:
                            obj_name = obj_full_name.split("@")[0]
                            obj_link_name = obj_full_name.split("@")[1]
                            contact_list = damage_info[obj_name][obj_link_name]["mechanical"]["contacts"]
                            contact_found = False
                            for contact in contact_list:
                                for target_contact_body in target_contact_bodies:
                                    if target_contact_body in contact[2] or target_contact_body in contact[3]:
                                        contact_found = True
                                        break
                            contact_info[obj_full_name].append(contact_found)

                    # Save video for force and contact plot
                    force_contact_video_path = os.path.join(output_video_dir, f"{args.rollout_name}_demo_{demo_idx}_force_contact_video.mp4")
                    save_rgb_force_contact_video(output_video_path=force_contact_video_path, data=data, imgs=imgs, contact_info=contact_info, target_objects=target_objects_forces, forces_to_plot=force_keys)

    if args.compute_metrics:
        for obj_name in target_objects_health:
            print(f"Average health for {obj_name}: {np.mean(final_obj_healths[obj_name])}")
        print(f"Average environment health: {np.mean(final_env_healths)}")
        breakpoint()
    
    # Shutdown simulation after all processing is complete
    og.shutdown()


if __name__ == "__main__":    
    __main__()
