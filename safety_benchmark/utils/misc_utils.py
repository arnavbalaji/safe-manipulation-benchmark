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


def save_camera_video(output_video_path, imgs, fps=30):
    avi_video = output_video_path + ".avi"
    mp4_video = output_video_path + ".mp4"
    if len(imgs) > 0:
        he, we = imgs[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw_e = cv2.VideoWriter(avi_video, fourcc, fps, (we, he))
        for i, img in enumerate(imgs):
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
            vw_e.write(np.ascontiguousarray(img, dtype=np.uint8))
        vw_e.release()
        subprocess.run([
            "ffmpeg", "-y",
            "-i", avi_video,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            mp4_video
        ], check=True)
        os.remove(avi_video)


def save_rgb_force_video(
    output_video_path,
    imgs,
    target_objects,
    data,
    forces_to_plot=("dynamic_forces", "static_forces", "raw_forces_from_sim"),
    fps=30,
):
    T = len(data[target_objects[0]][forces_to_plot[0]])

    # ---------------------------
    # FIGURE: 1 row, 2 columns
    # ---------------------------
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # ---------------------------
    # LEFT: RGB VIDEO
    # ---------------------------
    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.axis("off")
    video_im = ax_video.imshow(imgs[0][:, :, :3])

    # ---------------------------
    # RIGHT: FORCE PLOT
    # ---------------------------
    ax_force = fig.add_subplot(gs[0, 1])
    ax_force.set_title("Force History")
    ax_force.set_xlabel("Time (s)")
    ax_force.set_ylabel("End-effector Force (N)")
    ax_force.set_xlim(0, T / fps)
    ax_force.set_ylim(0, 500.0)
    ax_force.grid(True)

    force_lines = {}

    for obj_name in target_objects:
        for force_key in forces_to_plot:
            force_lines[f"{obj_name}_{force_key}"], = ax_force.plot(
                [],
                [],
                lw=2,
                label=f"{obj_name} {force_key}",
            )

    ax_force.legend(loc="upper right", fontsize=9)

    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.25)

    # Precompute time axis
    time = [i / fps for i in range(T)]

    # ---------------------------
    # INIT
    # ---------------------------
    def init():
        video_im.set_data(imgs[0][:, :, :3])
        for line in force_lines.values():
            line.set_data([], [])
        return [video_im] + list(force_lines.values())

    # ---------------------------
    # ANIMATE
    # ---------------------------
    def animate(i):
        # RGB frame
        video_im.set_data(imgs[i][:, :, :3])

        # Force plot
        for obj_name in target_objects:
            for force_key in forces_to_plot:
                force_lines[f"{obj_name}_{force_key}"].set_data(
                    time[: i + 1],
                    data[obj_name][force_key][: i + 1],
                )

        return [video_im] + list(force_lines.values())

    # ---------------------------
    # SAVE
    # ---------------------------
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=T,
        interval=1000 / fps,
        blit=True,
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
    )

    ani.save(output_video_path, writer=writer)
    plt.close(fig)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def save_rgb_health_video(
    output_video_path,
    imgs,
    target_objects,
    health,
    fps=30,
):
    T = len(health[target_objects[0]])

    # ---------------------------
    # FIGURE: 1 row, 2 columns
    # ---------------------------
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # ---------------------------
    # LEFT: RGB VIDEO
    # ---------------------------
    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.axis("off")
    video_im = ax_video.imshow(imgs[0][:, :, :3])

    # ---------------------------
    # RIGHT: HEALTH PLOT
    # ---------------------------
    ax_health = fig.add_subplot(gs[0, 1])
    ax_health.set_title("Health Over Time")
    ax_health.set_xlabel("Time (s)")
    ax_health.set_ylabel("Health")
    ax_health.set_xlim(0, T / fps)
    ax_health.set_ylim(-5.0, 105.0)
    ax_health.grid(True)

    health_lines = {}
    for obj_name in target_objects:
        health_lines[obj_name], = ax_health.plot(
            [],
            [],
            lw=2,
            label=f"{obj_name} Health",
        )

    ax_health.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.25)

    # Precompute time axis
    time = [i / fps for i in range(T)]

    # ---------------------------
    # INIT
    # ---------------------------
    def init():
        video_im.set_data(imgs[0][:, :, :3])
        for line in health_lines.values():
            line.set_data([], [])
        return [video_im] + list(health_lines.values())

    # ---------------------------
    # ANIMATE
    # ---------------------------
    def animate(i):
        # RGB frame
        video_im.set_data(imgs[i][:, :, :3])

        # Health plot
        for obj_name in target_objects:
            health_lines[obj_name].set_data(
                time[: i + 1],
                health[obj_name][: i + 1],
            )

        return [video_im] + list(health_lines.values())

    # ---------------------------
    # SAVE
    # ---------------------------
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=T,
        interval=1000 / fps,
        blit=True,
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
    )

    ani.save(output_video_path, writer=writer)
    plt.close(fig)


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
        codec='libx264',
        extra_args=[
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart'
        ]
    )

    ani.save(output_video_path, writer=writer)
    plt.close(fig)
