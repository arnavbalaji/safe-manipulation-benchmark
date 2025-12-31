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

def save_camera_video(hdf5_file, output_video_path, robot_name, camera_type, camera_name, target_objects, obs_info_list, health, demo_idx=0):
    imgs = hdf5_file[f"data/demo_{demo_idx}/obs/{camera_type}::{camera_name}::rgb"]
    imgs = imgs[1:]
    imgs_seg_instance = hdf5_file[f"data/demo_{demo_idx}/obs/{camera_type}::{camera_name}::seg_instance"]
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
                    # if obj_name == "tiago0":
                    #     break_loop = True
                    #     break
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

def save_force_contact_video(
    output_video_path, data, imgs, contact_info, target_objects, forces_to_plot=["dynamic_forces", "static_forces", "raw_forces_from_sim"]):
    T = len(data[target_objects[0]][forces_to_plot[0]])
    fps = 30

    # ---------------------------
    # Figure: 1 row, 2 columns
    # Left  = video
    # Right = your 3 vertical subplots
    # ---------------------------
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # ---------------------------
    # LEFT: RGB VIDEO PANEL
    # ---------------------------
    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.axis("off")
    video_im = ax_video.imshow(imgs[0][:, :, :3])  # will be updated each frame

    # ---------------------------
    # RIGHT: THREE STACKED SUBPLOTS
    # ---------------------------
    right = gs[0, 1].subgridspec(2, 1, hspace=0.45)
    ax1 = fig.add_subplot(right[0, 0])
    ax2 = fig.add_subplot(right[1, 0], sharex=ax1)

    # Fix layout manually so they don't overlap
    fig.subplots_adjust(hspace=0.45, left=0.05, right=0.97)

    # ---------------------------
    # AX1 — Force History
    # ---------------------------
    force_lines = dict()
    ax1.set_title("Force History")
    ax1.set_ylabel("End-effector Force (N)")
    ax1.set_ylim(0, 500.0)
    ax1.set_xlim(0, T)
    ax1.grid(True)

    # ---------------------------
    # AX2 — Contact History
    # ---------------------------
    contact_lines = dict()
    ax2.set_title("Contact History")
    ax2.set_ylabel("Contact (1) or No Contact (0)")
    ax2.set_xlabel("Time step")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True)
    ax2.set_xlim(0, T)

    for obj_name in target_objects:
        for force_key in forces_to_plot:
            force_lines[f"{obj_name}_{force_key}"], = ax1.plot([], [], lw=2, label=obj_name + ' ' + force_key + ' Forces')
            contact_lines[f"{obj_name}"], = ax2.plot([], [], lw=2, label=obj_name + ' Contact')

    ax1.legend(loc="upper right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    # ---------------------------
    # INIT FUNCTION
    # ---------------------------
    def init():
        video_im.set_data(imgs[0])

        for line in force_lines.values():
            line.set_data([], [])

        for line in contact_lines.values():
            line.set_data([], [])
        
        return (
            [video_im]
            + list(force_lines.values())
            + list(contact_lines.values())
        )

    # ---------------------------
    # ANIMATION STEP
    # ---------------------------
    def animate(i):
        # --- RGB video update ---
        video_im.set_data(imgs[i][:, :, :3])

        # --- Force ---
        for obj_name in target_objects:
            for force_key in forces_to_plot:
                force_lines[f"{obj_name}_{force_key}"].set_data(range(i + 1), data[obj_name][force_key][:i + 1])

        # --- Contact ---
        for obj_name in target_objects:
            contact_lines[f"{obj_name}"].set_data(range(i + 1), contact_info[obj_name][:i + 1])

        return (
            [video_im]
            + list(force_lines.values())
            + list(contact_lines.values())
        )

    # ---------------------------
    # SAVE ANIMATION
    # ---------------------------
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=T,
        interval=1000 / fps,
        blit=True
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="mpeg4",
        extra_args=["-vcodec", "mpeg4", "-qscale", "5"]
    )

    ani.save(output_video_path, writer=writer)
    plt.close(fig)
