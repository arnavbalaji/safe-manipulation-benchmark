from ast import Pass
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import yaml
import json
import h5py
import torch as th
import numpy as np
import argparse

import omnigibson as og
from omnigibson import object_states
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm

import omnigibson.lazy as lazy

from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataPlaybackWrapper
from safety_benchmark.utils.misc_utils import save_camera_video, save_health_video, save_combined_video, save_forces_video

gm.USE_GPU_DYNAMICS=True
gm.ENABLE_TRANSITION_RULES = False

def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='Visualize the data')
    parser.add_argument('--playback', action='store_true', help='Playback the data')
    args = parser.parse_args()

    # TODO: Set this 
    f_name = "wipe_table_2"
    collect_hdf5_path = f"resources/teleop_data/{f_name}.hdf5"
    output_hdf5_path = f"resources/playback_data/{f_name}_playback.hdf5"

    if args.playback:
    
        # TODO: remove hardcoding here. Obtian from collected hdf5 file
        robot_name = "tiago0"
        robot_type = "tiago"
        
        # In case we want to modify the external cameras that were used during data collection
        EXTERNAL_CAMERA_CONFIGS = {
            # Side camera (fixed to base_link frame)
            "external_sensor_0": {
                "position": [0.4859, -1.8219,  1.1402],
                "orientation": [ 0.5857, -0.0093, -0.0129,  0.8103],
                "horizontal_aperture": 10.0,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
            },
            # # Left Shoulder (fixed to base_link frame)
            # "external_sensor_1": {
            #     # wrt base frame
            #     "position": [0.2522, 0.0470, 1.0696],
            #     "orientation": [ 0.1991, -0.1991, -0.6785,  0.6785],
            #     "horizontal_aperture": 30.0,
            #     "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1",
            # },
            # # Back camera (fixed to base_link frame)
            # "external_sensor_2": {
            #     "position": [-0.7765, -0.8203,  0.9939],
            #     "orientation": [ 0.4566, -0.3285, -0.4831,  0.6710],
            #     "horizontal_aperture": 30.0,
            #     "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor2",
            # },
            # # Front camera (fixed to base_link frame)
            # "external_sensor_3": {
            #     "position": [1.7508, -0.0198,  1.1778],
            #     "orientation": [0.3821, 0.4173, 0.6080, 0.5570],
            #     "horizontal_aperture": 20.0,
            #     "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor3",
            # }
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

        # In case we want to modify the robot sensors that were used during data collection
        robot_sensor_config = {
            "VisionSensor": {
                "modalities": ["rgb"],
                "sensor_kwargs": {
                    "image_height": 720,
                    "image_width": 720,
                },
            },
        }
        
        env = DamageableDataPlaybackWrapper.create_from_hdf5(
            input_path=collect_hdf5_path,
            output_path=output_hdf5_path,
            # robot_obs_modalities=["proprio", "rgb", "depth", "seg_instance"],
            robot_sensor_config=robot_sensor_config,
            external_sensors_config=external_sensors_config,
            n_render_iterations=1,
            only_successes=False,
            exclude_sensor_names=["left_eef_link", "right_eef_link"]
        )
        robot = env.robots[0]

        # set viewer camera
        og.sim.viewer_camera.set_position_orientation(position=th.tensor([-0.2607, -3.0889,  1.2703]), orientation=th.tensor([ 0.5051, -0.0412, -0.0701,  0.8592]))
        
        for _ in range(10): og.sim.step()

        # # Best to explicitly set the object params here in case object params were modified since data collection
        # env.set_object_params()

        # Playback the dataset
        env.playback_dataset(record_data=True)
            
        env.save_data()
        
    # Visualize the episode
    if args.visualize:
        f = h5py.File(output_hdf5_path, "r")
        scene_file = json.loads(f["data"].attrs["scene_file"])
        # robot_name = [obj_name for obj_name in scene_file["objects_info"]["init_info"].keys() if "robot" in obj_name.lower()][0]
        robot_name = "tiago0"       
        camera_type = "external"
        camera_name = "external_sensor0"
        # breakpoint()

        # Parse info to obtain relevant information for visualization
        obs_info_list = []
        num_steps = len(f["data/demo_0/info/damage_info"])
        for i in range(len(f["data/demo_0/info/damage_info"])):            
            # Obtain observation information 
            obs_info = json.loads(f["data/demo_0/info/obs_info"][i].decode("utf-8"))
            obs_info_list.append(obs_info)

        # Obtain health information for the target objects
        target_objects = ["tiago0@gripper_right_link", "tiago0@gripper_right_left_finger_link", "tiago0@gripper_right_right_finger_link"]
        all_obj_healths = np.array(f["data/demo_0/obs/health"])
        health_list_link_names = f["data/demo_0"].attrs["health_list_link_names"]
        health = dict()
        for obj_name in target_objects:
            health[obj_name] = all_obj_healths[:, np.where(health_list_link_names == obj_name)[0][0]]

        # Obtain health for the whole robot
        arrays = [v for k, v in health.items() if k.startswith("tiago0@")]

        # Compute element-wise min
        if arrays:
            health["tiago0"] = np.minimum.reduce(arrays)
        else:
            health["tiago0"] = None

        output_video_dir = "resources/videos"
        os.makedirs(output_video_dir, exist_ok=True)
        
        # Save video for rgb camera
        target_objects_health = ["tiago0"]
        output_video_path = f"{output_video_dir}/{f_name}_camera_video"
        save_camera_video(hdf5_file=f, 
                    output_video_path=output_video_path,
                    robot_name=robot_name, 
                    camera_type=camera_type, 
                    camera_name=camera_name,
                    target_objects=target_objects_health,
                    obs_info_list=obs_info_list,
                    health=health)

        
        # Obtain forces information for the target objects
        # target_objects_forces = ["tiago0@gripper_right_link", "tiago0@gripper_right_left_finger_link", "tiago0@gripper_right_right_finger_link", "tiago0@arm_right_6_link", "tiago0@arm_right_5_link", "tiago0@arm_right_4_link", "tiago0@arm_right_3_link", "tiago0@arm_right_2_link", "tiago0@arm_right_1_link"]
        target_objects_forces = ["tiago0@gripper_right_link", "tiago0@gripper_right_left_finger_link", "tiago0@gripper_right_right_finger_link"]
        data = dict()
        # options: ["dynamic_forces", "static_forces", "raw_forces_from_sim"]
        force_keys = ["raw_forces_from_sim"]
        for obj_name in target_objects_forces:
            data[obj_name] = dict()
            for force_key in force_keys:
                data[obj_name][force_key] = []
        for i in range(len(f["data/demo_0/info/damage_info"])):
            damage_info = json.loads(f["data/demo_0/info/damage_info"][i].decode("utf-8"))
            for obj_name in target_objects_forces:
                for force_key in force_keys:
                    data[obj_name][force_key].append(damage_info[obj_name.split("@")[0]][obj_name.split("@")[1]]["mechanical"][force_key])
        
        # Save videos for forces plot
        forces_video_path = os.path.join(output_video_dir, f"{f_name}_forces_video.mp4")
        save_forces_video(output_video_path=forces_video_path, target_objects=target_objects_forces, data=data, forces_to_plot=force_keys)
        combined_video_path = os.path.join(output_video_dir, f"{f_name}_combined_video_forces.mp4")
        save_combined_video(video_1=output_video_path+".mp4", video_2=forces_video_path, output_video_path=combined_video_path, delete_intermediate_video_2=True)

        # Save video for health plot
        health_video_path = os.path.join(output_video_dir, f"{f_name}_health_video.mp4")
        save_health_video(output_video_path=health_video_path, target_objects=target_objects_health, health=health)
        combined_video_path = os.path.join(output_video_dir, f"{f_name}_combined_video_health.mp4")
        save_combined_video(video_1=output_video_path+".mp4", video_2=health_video_path, output_video_path=combined_video_path, delete_intermediate_video_2=True)

    # Shutdown simulation after all processing is complete
    og.shutdown()


if __name__ == "__main__":    
    __main__()
