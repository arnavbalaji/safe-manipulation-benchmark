import os
import json
import h5py
import cv2
import subprocess

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# open the hdf5 file
f = h5py.File("outputs/coffee_cup_playback.hdf5", "r")
# raw_data = f["data/demo_0/info/damage_info"][0].decode("utf-8")
# damage_info = json.loads(raw_data)

frames = []

# 1. Get images
# breakpoint()
scene_file = json.loads(f["data"].attrs["scene_file"])
# robot_name = [obj_name for obj_name in scene_file["objects_info"]["init_info"].keys() if "robot" in obj_name.lower()][0]
robot_name = "tiago0"
# imgs = f[f"data/demo_0/obs/{robot_name}::{robot_name}:eyes:Camera:0::rgb"]
imgs = f[f"data/demo_0/obs/external::external_sensor2::rgb"]
imgs = imgs[1:]
breakpoint()

# 2. Get force values
cup_acc_forces = []
cup_raw_impulse_forces = []
cup_adjusted_impulse_forces = []
contact_changes = [0]
target_obj_name = "coffee_cup"
for i in range(len(f["data/demo_0/info/damage_info"])):
    damage_info = json.loads(f["data/demo_0/info/damage_info"][i].decode("utf-8"))
    cup_acc_forces.append(damage_info[target_obj_name]["base_link"]["mechanical"]["acc_forces"])
    cup_raw_impulse_forces.append(damage_info[target_obj_name]["base_link"]["mechanical"]["raw_impulse_forces"])
    cup_adjusted_impulse_forces.append(damage_info[target_obj_name]["base_link"]["mechanical"]["adjusted_impulse_forces"])
    curr_contacts = damage_info[target_obj_name]["base_link"]["mechanical"]["contacts"]
    unique_contact_bodies = {c[3] for c in curr_contacts}
    # print("unique_contact_bodies: ", unique_contact_bodies)
    curr_num_unique_contacts = len(unique_contact_bodies)
    curr_num_contacts = len(curr_contacts)
    if i > 0:
        delta_contacts = abs(curr_num_unique_contacts - prev_num_unique_contacts)
        # print("i, delta_contacts: ", i, delta_contacts)
        contact_changes.append(delta_contacts)
    prev_num_unique_contacts = curr_num_unique_contacts

    # if i > 200 and i < 205:
    #     print("curr_num_contacts: ", curr_num_contacts, curr_num_unique_contacts)
    #     for contact in damage_info["cup"]["base_link"]["mechanical"]["contacts"]:
    #         print("contact: ", contact[3])


breakpoint()

# Write  camera video
fps = 5
videos_dir = "outputs/"
avi_ego = os.path.join(videos_dir, f"{target_obj_name}_video.avi")
mp4_ego = os.path.join(videos_dir, f"{target_obj_name}_video.mp4")
if len(imgs) > 0:
    he, we = imgs[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw_e = cv2.VideoWriter(avi_ego, fourcc, fps, (we, he))
    for img in imgs:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        vw_e.write(np.ascontiguousarray(img, dtype=np.uint8))
    vw_e.release()
    subprocess.run(["ffmpeg", "-y", "-i", avi_ego, "-c:v", "mpeg4", mp4_ego], check=True)
    os.remove(avi_ego)
# breakpoint()

# 3. Plot them side by side
T = len(cup_acc_forces)
# Clamp health plot to [0, 100]
y_min = 0.0
y_max = 100.0

fig, ax = plt.subplots(figsize=(9.6, 5.4))
# line_r, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
# line_p, = ax.plot([], [], lw=6, color='tab:orange', label="Cup")
line_acc_forces, = ax.plot([], [], lw=2, color='tab:green', label='Acc Forces')
line_raw_impulse_forces, = ax.plot([], [], lw=2, color='tab:red', label='Raw Impulse Forces')
line_adjusted_impulse_forces, = ax.plot([], [], lw=2, color='tab:purple', label='Adjusted Impulse Forces')
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
    line_acc_forces.set_data([], [])
    line_raw_impulse_forces.set_data([], [])
    line_adjusted_impulse_forces.set_data([], [])
    contact_points.set_offsets(np.empty((0, 2)))
    return line_acc_forces, line_raw_impulse_forces, line_adjusted_impulse_forces

def animate_health(i):
    x = [k / fps for k in range(1, i + 2)]
    y_acc_forces = cup_acc_forces[: i + 1]
    y_raw_impulse_forces = cup_raw_impulse_forces[: i + 1]
    y_adjusted_impulse_forces = cup_adjusted_impulse_forces[: i + 1]
    line_acc_forces.set_data(x, y_acc_forces)
    line_raw_impulse_forces.set_data(x, y_raw_impulse_forces)
    line_adjusted_impulse_forces.set_data(x, y_adjusted_impulse_forces)

    # --- Contact markers ---
    xs = [x[k] for k in range(i + 1) if contact_changes[k] > 0]
    ys = [-100.0] * len(xs)           # small negative value works well visually
    coords = np.column_stack([xs, ys]) if len(xs) > 0 else np.empty((0, 2))
    contact_points.set_offsets(coords)

    return line_acc_forces, line_raw_impulse_forces, line_adjusted_impulse_forces, contact_points

contact_points = ax.scatter([], [], s=30, marker='o')
ani = animation.FuncAnimation(
    fig, animate_health, init_func=init_health, frames=T, interval=1000 / fps, blit=True
)
# breakpoint()
forces_plot_video = os.path.join("outputs/forces_plot.mp4")
writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
ani.save(forces_plot_video, writer=writer)
plt.close(fig)

# Rename final combined video to {object}_with_health.mp4
combined_mp4 = os.path.join(videos_dir, f'{target_obj_name}_with_forces.mp4')
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

# # Plot a line graph 
# import plotly.graph_objects as go

# # Replace these with your actual lists
# acc = cup.damage_evaluators[0].acc_forces["base_link"]
# raw_imp = cup.damage_evaluators[0].raw_impulse_forces["base_link"]
# adj_imp = cup.damage_evaluators[0].adjusted_impulse_forces["base_link"]

# x = list(range(len(acc)))   # common x-axis

# fig = go.Figure()

# fig.add_trace(go.Scatter(x=x, y=acc, mode='lines', name='acc_forces'))
# fig.add_trace(go.Scatter(x=x, y=raw_imp, mode='lines', name='raw_impulse_forces'))
# fig.add_trace(go.Scatter(x=x, y=adj_imp, mode='lines', name='adjusted_impulse_forces'))

# fig.update_layout(
#     title="Forces over Time",
#     xaxis_title="Timestep",
#     yaxis_title="Force value",
#     template="plotly_white"
# )

# fig.write_image("outputs/forces_plot.svg")     # SVG (vector)