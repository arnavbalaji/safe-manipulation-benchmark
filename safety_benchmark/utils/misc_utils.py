import os
import cv2
import torch
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def json_default(o):
    # numpy scalar
    if isinstance(o, (np.float32, np.float64, np.int32, np.int64)):
        return o.item()
    # numpy array
    if isinstance(o, np.ndarray):
        return o.tolist()
    # torch tensor
    if isinstance(o, torch.Tensor):
        return o.tolist()
    # tuple → list
    if isinstance(o, tuple):
        return list(o)
    # fallback: try item()
    if hasattr(o, "item"):
        try:
            return o.item()
        except:
            pass
    raise TypeError(f"Object of type {type(o)} not JSON serializable")

def save_camera_video(hdf5_file, output_video_path, robot_name, camera_type, camera_name, target_objects, obs_info_list, health):
    imgs = hdf5_file[f"data/demo_0/obs/{camera_type}::{camera_name}::rgb"]
    imgs = imgs[1:]
    imgs_seg_instance = hdf5_file[f"data/demo_0/obs/{camera_type}::{camera_name}::seg_instance"]
    imgs_seg_instance = imgs_seg_instance[1:]

    # Write  camera video
    fps = 30
    avi_video = output_video_path + ".avi"
    mp4_video = output_video_path + ".mp4"
    break_loop = False
    if len(imgs) > 0:
        he, we = imgs[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw_e = cv2.VideoWriter(avi_video, fourcc, fps, (we, he))
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
                    if obj_name == "tiago0":
                        break_loop = True
                        break
            vw_e.write(np.ascontiguousarray(img, dtype=np.uint8))
            if break_loop:
                for _ in range(5):
                    vw_e.write(np.ascontiguousarray(img, dtype=np.uint8))                
                break

        vw_e.release()
        subprocess.run(["ffmpeg", "-y", "-i", avi_video, "-c:v", "mpeg4", mp4_video], check=True)
        os.remove(avi_video)

def save_forces_video(output_video_path, target_objects, data, forces_to_plot=["dynamic_forces", "static_forces", "raw_forces_from_sim"]):
    # 3. Plot them side by side
    T = len(data[target_objects[0]][forces_to_plot[0]])
    # Clamp health plot to [0, 100]
    y_min = -1.0
    y_max = 500.0
    fps = 30

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    dynamic_forces_lines = dict()
    lines = dict()
    for obj_name in target_objects:
        for force_key in forces_to_plot:
            lines[f"{obj_name}_{force_key}"], = ax.plot([], [], lw=2, label=obj_name + ' ' + force_key + ' Forces')
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
        for key, value in data.items():
            for force_key in forces_to_plot:
                lines[f"{key}_{force_key}"].set_data([], [])
        return lines.values()

    def animate_forces(i):
        x = [k / fps for k in range(1, i + 2)]
        for key, value in data.items():
            for force_key in forces_to_plot:
                lines[f"{key}_{force_key}"].set_data(x, value[force_key][: i + 1])

        return lines.values()

    ani = animation.FuncAnimation(
        fig, animate_forces, init_func=init_forces, frames=T, interval=1000 / fps, blit=True
    )
    # breakpoint()
    writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani.save(output_video_path, writer=writer)
    plt.close(fig)


def save_health_video(output_video_path, target_objects, health):
    T = len(health[target_objects[0]])
    y_min = -5.0
    y_max = 105.0
    fps = 30

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    health_lines = dict()
    for obj_name in target_objects:
        health_line, = ax.plot([], [], lw=2, label=obj_name + ' Health')
        health_lines[obj_name] = health_line
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
        for obj_name in target_objects:
            health_lines[obj_name].set_data([], [])
        return health_lines.values()

    def animate_health(i):
        x = [k / fps for k in range(1, i + 2)]
        for obj_name in target_objects:
            y_obj_health = health[obj_name][: i + 1]
            health_lines[obj_name].set_data(x, y_obj_health)

        return health_lines.values()

    ani = animation.FuncAnimation(
        fig, animate_health, init_func=init_health, frames=T, interval=1000 / fps, blit=True
    )
    # breakpoint()
    writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani.save(output_video_path, writer=writer)
    plt.close(fig)

def save_combined_video(video_1, video_2, output_video_path, delete_intermediate_video_1=False, delete_intermediate_video_2=False):
    # Rename final combined video to {object}_with_health.mp4
    if os.path.exists(video_1) and os.path.exists(video_2):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', video_1,
            '-i', video_2,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            output_video_path
        ], check=True)

    if delete_intermediate_video_1:
        os.remove(video_1)
    if delete_intermediate_video_2:
        os.remove(video_2)