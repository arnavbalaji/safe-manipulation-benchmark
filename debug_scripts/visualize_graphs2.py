import os
import json
import h5py
import cv2
import subprocess

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# open the hdf5 file
f = h5py.File("outputs/swivel_chair_playback.hdf5", "r")
# raw_data = f["data/demo_0/info/damage_info"][0].decode("utf-8")
# damage_info = json.loads(raw_data)

frames = []

# 1. Get images
# breakpoint()
scene_file = json.loads(f["data"].attrs["scene_file"])
# robot_name = [obj_name for obj_name in scene_file["objects_info"]["init_info"].keys() if "robot" in obj_name.lower()][0]
robot_name = "tiago0"
camera_type = "external"
camera_name = "external_sensor0"
# imgs = f[f"data/demo_0/obs/{robot_name}::{robot_name}:eyes:Camera:0::rgb"]
imgs = f[f"data/demo_0/obs/{camera_type}::{camera_name}::rgb"]
imgs = imgs[1:]
imgs_seg_instance = f[f"data/demo_0/obs/{camera_type}::{camera_name}::seg_instance"]
imgs_seg_instance = imgs_seg_instance[1:]

# 2. Get force values
chair_obj_name = "swivel_chair"
# lamp_obj_name = "floor_lamp"
vase_obj_name = "vase"
target_objects = [chair_obj_name, vase_obj_name]
chair_dynamic_forces = []
lamp_dynamic_forces = []
vase_dynamic_forces = []
obs_info_list = []
num_steps = len(f["data/demo_0/info/damage_info"])
for i in range(len(f["data/demo_0/info/damage_info"])):
    damage_info = json.loads(f["data/demo_0/info/damage_info"][i].decode("utf-8"))
    chair_dynamic_forces.append(damage_info[chair_obj_name]["base_link"]["mechanical"]["dynamic_forces"])
    vase_dynamic_forces.append(damage_info[vase_obj_name]["base_link"]["mechanical"]["dynamic_forces"])
    # lamp_dynamic_forces.append(damage_info[lamp_obj_name]["base_link"]["mechanical"]["dynamic_forces"])

    obs_info = json.loads(f["data/demo_0/info/obs_info"][i].decode("utf-8"))
    obs_info_list.append(obs_info)


# remove later
all_obj_healths = np.array(f["data/demo_0/obs/health"])
health_list_link_names = f["data/demo_0"].attrs["health_list_link_names"]

health = dict()
chair_idx = np.where(health_list_link_names == f"{chair_obj_name}@base_link")[0][0]
vase_idx = np.where(health_list_link_names == f"{vase_obj_name}@base_link")[0][0]
# lamp_idx = np.where(health_list_link_names == f"{lamp_obj_name}@base_link")[0][0]
health[chair_obj_name] = all_obj_healths[:, chair_idx]
health[vase_obj_name] = all_obj_healths[:, vase_idx]
# health[lamp_obj_name] = all_obj_healths[:, lamp_idx]
# health[chair_obj_name] = [100.0] * int(np.floor(num_steps / 2)) + [0.0] * int(np.ceil(num_steps / 2))
# health[vase_obj_name] = [100.0] * int(np.floor(num_steps / 2)) + [0.0] * int(np.ceil(num_steps / 2))
# health[lamp_obj_name] = [100.0] * int(np.floor(num_steps / 2)) + [0.0] * int(np.ceil(num_steps / 2))
breakpoint()

# (Pdb) obs_info["external"]["external_sensor2"]["seg_instance"]
# {'0': 'background', '2': 'tiago0', '3': 'groundPlane', '4': 'pedestal_table', '5': 'floor_lamp', '6': 'vase', '7': 'swivel_chair'}

# Write  camera video
fps = 5
videos_dir = "outputs/"
f_name = "nav_swivel_chair"
avi_ego = os.path.join(videos_dir, f"{f_name}_video.avi")
mp4_ego = os.path.join(videos_dir, f"{f_name}_video.mp4")
if len(imgs) > 0:
    he, we = imgs[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw_e = cv2.VideoWriter(avi_ego, fourcc, fps, (we, he))
    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        # breakpoint()
        # vw_e.write(np.ascontiguousarray(img, dtype=np.uint8))
        img_seg_instance = imgs_seg_instance[i]
        obs_info = obs_info_list[i]
        for obj_name in target_objects:
            seg_instance_info = obs_info[camera_type][camera_name]["seg_instance"]
            # print("seg_instance_info: ", i, seg_instance_info)
            seg_instance_key = int(next((k for k, v in seg_instance_info.items() if v == obj_name), -1))
            if health[obj_name][i] == 0.0:
                # breakpoint()
                img[img_seg_instance == seg_instance_key] = (0, 0, 255)
        vw_e.write(np.ascontiguousarray(img, dtype=np.uint8))

    vw_e.release()
    subprocess.run(["ffmpeg", "-y", "-i", avi_ego, "-c:v", "mpeg4", mp4_ego], check=True)
    os.remove(avi_ego)
# breakpoint()

def plot_forces():
    # 3. Plot them side by side
    T = len(chair_dynamic_forces)
    # Clamp health plot to [0, 100]
    y_min = 0.0
    y_max = 100.0

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    # line_r, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
    # line_p, = ax.plot([], [], lw=6, color='tab:orange', label="Cup")
    chair_dynamic_forces_line, = ax.plot([], [], lw=2, color='tab:blue', label='Chair Acc Forces')
    # lamp_dynamic_forces_line, = ax.plot([], [], lw=2, color='tab:red', label='Lamp Acc Forces')
    vase_dynamic_forces_line, = ax.plot([], [], lw=2, color='tab:purple', label='Vase Acc Forces')
    ax.set_xlim(0, max(1, T) / fps)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Force', fontsize=20)
    ax.set_title('Forces Over Time', fontsize=26)
    ax.legend(loc='best', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    ax.grid(True, linewidth=1.0, alpha=0.3)
    plt.tight_layout()

    def init_forces():
        # line_r.set_data([], [])
        # line_p.set_data([], [])
        chair_dynamic_forces_line.set_data([], [])
        # lamp_dynamic_forces_line.set_data([], [])
        vase_dynamic_forces_line.set_data([], [])
        return chair_dynamic_forces_line, vase_dynamic_forces_line

    def animate_forces(i):
        x = [k / fps for k in range(1, i + 2)]
        y_chair_dynamic_forces = chair_dynamic_forces[: i + 1]
        # y_lamp_dynamic_forces = lamp_dynamic_forces[: i + 1]
        y_vase_dynamic_forces = vase_dynamic_forces[: i + 1]
        chair_dynamic_forces_line.set_data(x, y_chair_dynamic_forces)
        # lamp_dynamic_forces_line.set_data(x, y_lamp_dynamic_forces)
        vase_dynamic_forces_line.set_data(x, y_vase_dynamic_forces)

        return chair_dynamic_forces_line, vase_dynamic_forces_line

    ani = animation.FuncAnimation(
        fig, animate_forces, init_func=init_forces, frames=T, interval=1000 / fps, blit=True
    )
    # breakpoint()
    forces_plot_video = os.path.join("outputs/forces_plot.mp4")
    writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani.save(forces_plot_video, writer=writer)
    plt.close(fig)

    # Rename final combined video to {object}_with_health.mp4
    combined_mp4 = os.path.join(videos_dir, f'{f_name}_with_forces.mp4')
    if os.path.exists(mp4_ego) and os.path.exists(forces_plot_video):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', mp4_ego,
            '-i', forces_plot_video,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_mp4
        ], check=True)

def plot_health():
    # 3. Plot them side by side
    T = len(health[chair_obj_name])
    # Clamp health plot to [0, 100]
    y_min = 0.0
    y_max = 100.0

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    # line_r, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
    # line_p, = ax.plot([], [], lw=6, color='tab:orange', label="Cup")
    chair_health_line, = ax.plot([], [], lw=2, color='tab:blue', label='Chair Health')
    # lamp_dynamic_forces_line, = ax.plot([], [], lw=2, color='tab:red', label='Lamp Acc Forces')
    vase_health_line, = ax.plot([], [], lw=2, color='tab:purple', label='Vase Health')
    ax.set_xlim(0, max(1, T) / fps)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Health', fontsize=20)
    ax.set_title('Health Over Time', fontsize=26)
    ax.legend(loc='best', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    ax.grid(True, linewidth=1.0, alpha=0.3)
    plt.tight_layout()

    def init_health():
        # line_r.set_data([], [])
        # line_p.set_data([], [])
        chair_health_line.set_data([], [])
        # lamp_dynamic_forces_line.set_data([], [])
        vase_health_line.set_data([], [])
        return chair_health_line, vase_health_line

    def animate_health(i):
        x = [k / fps for k in range(1, i + 2)]
        y_chair_health = health[chair_obj_name][: i + 1]
        # y_lamp_dynamic_forces = lamp_dynamic_forces[: i + 1]
        y_vase_health = health[vase_obj_name][: i + 1]
        chair_health_line.set_data(x, y_chair_health)
        # lamp_dynamic_forces_line.set_data(x, y_lamp_dynamic_forces)
        vase_health_line.set_data(x, y_vase_health)

        return chair_health_line, vase_health_line

    ani = animation.FuncAnimation(
        fig, animate_health, init_func=init_health, frames=T, interval=1000 / fps, blit=True
    )
    # breakpoint()
    forces_plot_video = os.path.join("outputs/forces_plot.mp4")
    writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani.save(forces_plot_video, writer=writer)
    plt.close(fig)

    # Rename final combined video to {object}_with_health.mp4
    combined_mp4 = os.path.join(videos_dir, f'{f_name}_with_forces.mp4')
    if os.path.exists(mp4_ego) and os.path.exists(forces_plot_video):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', mp4_ego,
            '-i', forces_plot_video,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_mp4
        ], check=True)

# plot_forces()
plot_health()