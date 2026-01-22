import torch as th
import numpy as np
import argparse
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import yaml
import json
import h5py
from omnigibson.macros import gm

import omnigibson as og

from safety_benchmark.damageable_env import DamageableDataPlaybackWrapper
from safety_benchmark.utils.misc_utils import save_rgb_camera_video, save_rgb_force_video, save_rgb_health_video, save_rgb_force_contact_video, setup_viewport_layout

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_TRANSITION_RULES = False

collect_hdf5_path = "resources/teleop_data/shelve_item/trial_5.hdf5"
playback_hdf5_path = "resources/playback_data/temp.hdf5"

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
    input_path=collect_hdf5_path,
    output_path=playback_hdf5_path,
    robot_obs_modalities=["proprio", "rgb", "seg_instance"],
    robot_sensor_config=robot_sensor_config,
    external_sensors_config=external_sensors_config,
    n_render_iterations=1,
    only_successes=False,
)

data = [
    {
        "collect_hdf5_path": "resources/teleop_data/shelve_item/trial_5.hdf5",
        "marker_color": (1, 0, 0, 1)
    },
    {
        "collect_hdf5_path": "resources/teleop_data/shelve_item/trial_6.hdf5",
        "marker_color": (0, 1, 0, 1)
    },
        {
        "collect_hdf5_path": "resources/teleop_data/shelve_item/trial_1.hdf5",
        "marker_color": (1, 0, 0, 1)
    },
    {
        "collect_hdf5_path": "resources/teleop_data/shelve_item/trial_2.hdf5",
        "marker_color": (0, 1, 0, 1)
    }

]

input_hdf5 = h5py.File(data[0]["collect_hdf5_path"], "r")
scene_file = json.loads(input_hdf5["data"].attrs["scene_file"])
env.scene.restore(scene_file, update_initial_file=True)

for _ in range(10): og.sim.render()
flour = env.scene.object_registry("name", "book")
wineglass = env.scene.object_registry("name", "wineglass")
winebottle = env.scene.object_registry("name", "bottle_of_wine")
beerbottle = env.scene.object_registry("name", "bottle_of_beer")
box_of_crackers = env.scene.object_registry("name", "box_of_crackers")
wineglass.visible = False
winebottle.visible = False
beerbottle.visible = False
box_of_crackers.visible = False
flour.visible = False
for _ in range(10): og.sim.render()


marker_idx = 0
for data_idx in range(len(data)):
    input_hdf5 = h5py.File(data[data_idx]["collect_hdf5_path"], "r")

    # set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([ 7.0659, -0.7141,  1.9185]),
        orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
    )
    for _ in range(10): og.sim.step()

    num_demos = input_hdf5["data"].attrs["n_episodes"]
    print(f"Number of demos in dataset: {num_demos}")
    box_of_crackers = env.scene.object_registry("name", "box_of_crackers")
    all_box_pos = []
    for episode_id in range(num_demos):
        data_grp = input_hdf5["data"]
        traj_grp = data_grp[f"demo_{episode_id}"]
        last_state = th.tensor(traj_grp["state"][-1])
        last_state_size = traj_grp["state_size"][-1]
        # load the last state into the simulation
        og.sim.load_state(last_state[:int(last_state_size)], serialized=True)
        pos = box_of_crackers.get_position_orientation()[0]
        all_box_pos.append(pos)

    # Show markers
    from omnigibson.objects.primitive_object import PrimitiveObject
    marker_list = []
    for i in range(num_demos):
        marker = PrimitiveObject(
            relative_prim_path=f"/marker_{marker_idx}",
            primitive_type="Cube",
            name=f"marker_{marker_idx}",
            size=th.tensor([0.01, 0.01, 0.01]),
            visual_only=True,
            rgba=th.tensor(data[data_idx]["marker_color"])
        )
        marker_list.append(marker)
        marker_idx += 1

    og.sim.batch_add_objects(marker_list, [env.scene] * len(marker_list))

    for i in range(num_demos):
        pos = all_box_pos[i]
        marker_list[i].set_position_orientation(position=pos)

    for _ in range(5): og.sim.step()

breakpoint()