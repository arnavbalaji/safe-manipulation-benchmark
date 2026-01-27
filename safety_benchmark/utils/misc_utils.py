import os
import cv2
import torch
import subprocess
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for live window
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from typing import Dict, Iterable, Mapping, Optional, Sequence

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


def save_rgb_camera_video(output_video_path, imgs, fps=30):
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
            "-loglevel", "error",
            "-hide_banner",
            "-nostats",
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
            "-loglevel", "error",
            "-hide_banner",
            "-nostats",
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
            "-loglevel", "error",
            "-hide_banner",
            "-nostats",
        ],
    )

    ani.save(output_video_path, writer=writer)
    plt.close(fig)


def save_rgb_temperature_video(
    output_video_path,
    imgs,
    target_objects,
    temperature,
    fps=30,
):
    """
    Save video with RGB frames and temperature history plot.
    
    Args:
        output_video_path: Path to save the video
        imgs: List/array of RGB images
        target_objects: List of object names to plot
        temperature: Dict mapping object_name -> list of temperature values
        fps: Frames per second
    """
    T = len(temperature[target_objects[0]])

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
    # RIGHT: TEMPERATURE PLOT
    # ---------------------------
    ax_temp = fig.add_subplot(gs[0, 1])
    ax_temp.set_title("Temperature Over Time")
    ax_temp.set_xlabel("Time (s)")
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_xlim(0, T / fps)
    
    # Find temperature range for y-axis scaling (handle NaN values)
    temp_values = []
    for obj_name in target_objects:
        if obj_name in temperature:
            temp_array = np.array(temperature[obj_name])
            temp_values.extend(temp_array[~np.isnan(temp_array)])
    
    if temp_values:
        min_temp = min(temp_values)
        max_temp = max(temp_values)
        temp_range = max_temp - min_temp
        ax_temp.set_ylim(min_temp - 0.1 * temp_range, max_temp + 0.1 * temp_range)
    else:
        ax_temp.set_ylim(0, 100)  # Default range
    
    ax_temp.grid(True)

    temp_lines = {}
    for obj_name in target_objects:
        if obj_name in temperature:
            temp_lines[obj_name], = ax_temp.plot(
                [],
                [],
                lw=2,
                label=f"{obj_name} Temperature",
            )

    ax_temp.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.25)

    # Precompute time axis
    time = [i / fps for i in range(T)]

    # ---------------------------
    # INIT
    # ---------------------------
    def init():
        video_im.set_data(imgs[0][:, :, :3])
        for line in temp_lines.values():
            line.set_data([], [])
        return [video_im] + list(temp_lines.values())

    # ---------------------------
    # ANIMATE
    # ---------------------------
    def animate(i):
        # RGB frame
        video_im.set_data(imgs[i][:, :, :3])

        # Temperature plot
        for obj_name in target_objects:
            if obj_name in temp_lines and obj_name in temperature:
                temp_array = np.array(temperature[obj_name])
                # Handle NaN values by only plotting valid data up to current frame
                valid_mask = ~np.isnan(temp_array[:i+1])
                if np.any(valid_mask):
                    valid_time = np.array(time[:i+1])[valid_mask]
                    valid_temp = temp_array[:i+1][valid_mask]
                    temp_lines[obj_name].set_data(valid_time, valid_temp)

        return [video_im] + list(temp_lines.values())

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
            "-loglevel", "error",
            "-hide_banner",
            "-nostats",
        ],
    )

    ani.save(output_video_path, writer=writer)
    plt.close(fig)


def save_rgb_water_contacts_video(
    output_video_path,
    imgs,
    target_objects,
    water_contacts,
    fps=30,
):
    """
    Save video with RGB frames and water particle contacts plot.
    
    Args:
        output_video_path: Path to save the video
        imgs: List/array of RGB images
        target_objects: List of object names to plot
        water_contacts: Dict mapping object_name -> list of particle counts
        fps: Frames per second
    """
    T = len(water_contacts[target_objects[0]])

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
    # RIGHT: WATER CONTACTS PLOT
    # ---------------------------
    ax_water = fig.add_subplot(gs[0, 1])
    ax_water.set_title("Water Particle Contacts Over Time")
    ax_water.set_xlabel("Time (s)")
    ax_water.set_ylabel("Particle Contacts")
    ax_water.set_xlim(0, T / fps)
    
    # Find max contact count for y-axis scaling
    max_contacts = max(max(water_contacts[obj]) for obj in target_objects if len(water_contacts[obj]) > 0)
    ax_water.set_ylim(0, max(max_contacts * 1.1, 10))
    ax_water.grid(True)

    water_lines = {}
    for obj_name in target_objects:
        water_lines[obj_name], = ax_water.plot(
            [],
            [],
            lw=2,
            label=f"{obj_name} Water Contacts",
        )

    ax_water.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.25)

    # Precompute time axis
    time = [i / fps for i in range(T)]

    # ---------------------------
    # INIT
    # ---------------------------
    def init():
        video_im.set_data(imgs[0][:, :, :3])
        for line in water_lines.values():
            line.set_data([], [])
        return [video_im] + list(water_lines.values())

    # ---------------------------
    # ANIMATE
    # ---------------------------
    def animate(i):
        # RGB frame
        video_im.set_data(imgs[i][:, :, :3])

        # Water contacts plot
        for obj_name in target_objects:
            water_lines[obj_name].set_data(
                time[: i + 1],
                water_contacts[obj_name][: i + 1],
            )

        return [video_im] + list(water_lines.values())

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
            "-loglevel", "error",
            "-hide_banner",
            "-nostats",
        ],
    )

    ani.save(output_video_path, writer=writer)
    plt.close(fig)


def save_rgb_force_contact_video(
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


def _load_og_ui_modules():
    """
    Lazy-import OmniGibson UI dependencies to avoid import-time overhead
    for callers that do not need viewport utilities.
    """
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.utils.ui_utils import dock_window

    return og, lazy, dock_window


def get_external_sensor_paths(external_sensors: Optional[Mapping], sensor_names: Sequence[str]) -> Dict[str, str]:
    """
    Extract prim paths for a set of external sensors.

    Args:
        external_sensors: Mapping or iterable of sensor objects. Sensors must expose
            a `name` attribute and `prim_path`.
        sensor_names: Names to look up (e.g., ["external_sensor0", "external_sensor1"]).

    Returns:
        Dict mapping sensor name to prim path for those found.
    """
    if external_sensors is None:
        return {}

    paths: Dict[str, str] = {}

    def _find_sensor(target: str):
        if isinstance(external_sensors, Mapping):
            return external_sensors.get(target)
        if isinstance(external_sensors, Iterable):
            for sensor in external_sensors:
                if getattr(sensor, "name", None) == target:
                    return sensor
        return None

    for name in sensor_names:
        sensor = _find_sensor(name)
        prim_path = getattr(sensor, "prim_path", None) if sensor is not None else None
        if prim_path:
            paths[name] = prim_path

    return paths


def create_docked_viewport(
    parent_window: str,
    dock_position,
    ratio: float,
    camera_path: Optional[str],
    resolution: Optional[Sequence[int]] = None,
):
    """
    Create a viewport, dock it, and bind it to a camera.

    Args:
        parent_window: Workspace window to dock into (e.g., "DockSpace").
        dock_position: Position from lazy.omni.ui.DockPosition.
        ratio: Relative size ratio used by dock_window.
        camera_path: Prim path of the camera to bind. If None, binding is skipped.
        resolution: Optional (H, W) texture resolution for the viewport.

    Returns:
        The created viewport window object.
    """
    og, lazy, dock_window = _load_og_ui_modules()

    viewport = lazy.omni.kit.viewport.utility.create_viewport_window()
    og.sim.render()

    dock_window(
        space=lazy.omni.ui.Workspace.get_window(parent_window),
        name=viewport.name,
        location=dock_position,
        ratio=ratio,
    )
    og.sim.render()

    if camera_path:
        viewport.viewport_api.set_active_camera(camera_path)
    if resolution is not None:
        viewport.viewport_api.set_texture_resolution(tuple(resolution))
    og.sim.render()

    return viewport


def setup_viewport_layout(
):

    og, lazy, _ = _load_og_ui_modules()
    import omni.ui as ui
    from omni.kit.viewport.window import ViewportWindow
    viewports = [w for w in ui.Workspace.get_windows() if isinstance(w, ViewportWindow)]
    for w in ui.Workspace.get_windows():
        if isinstance(w, ViewportWindow):
            w.visible = True
        else:
            w.visible = False

    vp1 = ui.Workspace.get_window("Viewport 1")
    vp1.visible = False
    vp = ui.Workspace.get_window("Viewport")
    vp.height = 890
    vp.width = 1430
    for _ in range(10): og.sim.render()
    # breakpoint()

def setup_robot_visualizers(robot, scene):
    """
    Set up visualization elements for teleop
    
    Args:
        robot: The robot object
        scene: The scene object
        
    Returns:
        dict: Dictionary of visualization elements
    """
    vis_elements = {
        "eef_cylinder_geoms": {},
        "vis_mats": {},
        "vertical_visualizers": {},
        "reachability_visualizers": {}
    }
    
    # Create materials for visualization cylinders
    for arm in robot.arm_names:
        vis_elements["vis_mats"][arm] = []
        for axis, color in zip(("x", "y", "z"), VIS_GEOM_COLORS[False]):
            mat_prim_path = f"{robot.prim_path}/Looks/vis_cylinder_{arm}_{axis}_mat"
            mat = OmniPBRMaterialPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, mat_prim_path),
                name=f"{robot.name}:vis_cylinder_{arm}_{axis}_mat",
            )
            mat.load(scene)
            mat.diffuse_color_constant = color
            vis_elements["vis_mats"][arm].append(mat)

    # Create material for visual sphere
    mat_prim_path = f"{robot.prim_path}/Looks/vis_sphere_mat"
    sphere_mat = OmniPBRMaterialPrim(
        relative_prim_path=absolute_prim_path_to_scene_relative(scene, mat_prim_path),
        name=f"{robot.name}:vis_sphere_mat",
    )
    sphere_color = np.array([252, 173, 76]) / 255.0
    sphere_mat.load(scene)
    sphere_mat.diffuse_color_constant = th.as_tensor(sphere_color)
    vis_elements["sphere_mat"] = sphere_mat

    # Create material for vertical cylinder
    if USE_VERTICAL_VISUALIZERS:
        mat_prim_path = f"{robot.prim_path}/Looks/vis_vertical_mat"
        vert_mat = OmniPBRMaterialPrim(
            relative_prim_path=absolute_prim_path_to_scene_relative(scene, mat_prim_path),
            name=f"{robot.name}:vis_vertical_mat",
        )
        vert_color = np.array([252, 226, 76]) / 255.0
        vert_mat.load(scene)
        vert_mat.diffuse_color_constant = th.as_tensor(vert_color)
        vis_elements["vert_mat"] = vert_mat

    # Extract visualization cylinder settings
    vis_geom_width = VIS_CYLINDER_CONFIG["width"]
    vis_geom_lengths = VIS_CYLINDER_CONFIG["lengths"]
    vis_geom_proportion_offsets = VIS_CYLINDER_CONFIG["proportion_offsets"]
    vis_geom_quat_offsets = VIS_CYLINDER_CONFIG["quat_offsets"]

    # Create visualization cylinders for each arm
    for arm in robot.arm_names:
        hand_link = robot.eef_links[arm]
        vis_elements["eef_cylinder_geoms"][arm] = []
        for axis, length, mat, prop_offset, quat_offset in zip(
            ("x", "y", "z"),
            vis_geom_lengths,
            vis_elements["vis_mats"][arm],
            vis_geom_proportion_offsets,
            vis_geom_quat_offsets,
        ):
            vis_prim_path = f"{hand_link.prim_path}/vis_cylinder_{axis}"
            vis_prim = create_primitive_mesh(
                vis_prim_path,
                "Cylinder",
                extents=1.0
            )
            vis_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, vis_prim_path),
                name=f"{robot.name}:arm_{arm}:vis_cylinder_{axis}"
            )
            vis_geom.load(scene)

            # Attach a material to this prim
            vis_geom.material = mat

            vis_geom.scale = th.tensor([vis_geom_width, vis_geom_width, length])
            vis_geom.set_position_orientation(
                position=th.tensor([0, 0, length * prop_offset]), 
                orientation=quat_offset, 
                frame="parent"
            )
            vis_elements["eef_cylinder_geoms"][arm].append(vis_geom)

        # Add vis sphere around EEF for reachability
        if USE_VISUAL_SPHERES:
            vis_prim_path = f"{hand_link.prim_path}/vis_sphere"
            vis_prim = create_primitive_mesh(
                vis_prim_path,
                "Sphere",
                extents=1.0
            )
            vis_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, vis_prim_path),
                name=f"{robot.name}:arm_{arm}:vis_sphere"
            )
            vis_geom.load(scene)

            # Attach a material to this prim
            sphere_mat.bind(vis_geom.prim_path)

            vis_geom.scale = th.ones(3) * 0.15
            vis_geom.set_position_orientation(
                position=th.zeros(3), 
                orientation=th.tensor([0, 0, 0, 1.0]), 
                frame="parent"
            )

        # Add vertical cylinder at EEF
        if USE_VERTICAL_VISUALIZERS:
            vis_prim_path = f"{hand_link.prim_path}/vis_vertical"
            vis_prim = create_primitive_mesh(
                vis_prim_path,
                "Cylinder",
                extents=1.0
            )
            vis_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, vis_prim_path),
                name=f"{robot.name}:arm_{arm}:vis_vertical"
            )
            
            vis_geom.load(scene)

            # Attach a material to this prim
            vis_elements["vert_mat"].bind(vis_geom.prim_path)

            vis_geom.scale = th.tensor([vis_geom_width, vis_geom_width, 2.0])
            vis_geom.set_position_orientation(
                position=th.zeros(3), 
                orientation=th.tensor([0, 0, 0, 1.0]), 
                frame="parent"
            )
            vis_elements["vertical_visualizers"][arm] = vis_geom

    # Create reachability visualizers
    if USE_REACHABILITY_VISUALIZERS:
        # Create a square formation in front of the robot as reachability signal
        torso_link = robot.links["torso_link4"]
        beam_width = REACHABILITY_VISUALIZER_CONFIG["beam_width"]
        square_distance = REACHABILITY_VISUALIZER_CONFIG["square_distance"]
        square_width = REACHABILITY_VISUALIZER_CONFIG["square_width"]
        square_height = REACHABILITY_VISUALIZER_CONFIG["square_height"]
        beam_color = REACHABILITY_VISUALIZER_CONFIG["beam_color"]

        # Create material for beams
        beam_mat_prim_path = f"{robot.prim_path}/Looks/square_beam_mat"
        beam_mat = OmniPBRMaterialPrim(
            relative_prim_path=absolute_prim_path_to_scene_relative(scene, beam_mat_prim_path),
            name=f"{robot.name}:square_beam_mat",
        )
        beam_mat.load(scene)
        beam_mat.diffuse_color_constant = th.as_tensor(beam_color)
        vis_elements["beam_mat"] = beam_mat

        edges = [
            # name, position, scale, orientation
            ["top", [square_distance, 0, 0.3], [beam_width, beam_width, square_width], [0.0, th.pi/2, th.pi/2]],
            ["bottom", [square_distance, 0, 0.0], [beam_width, beam_width, square_width], [0.0, th.pi/2, th.pi/2]],
            ["left", [square_distance, 0.2, 0.15], [beam_width, beam_width, square_height], [0.0, 0.0, 0.0]],
            ["right", [square_distance, -0.2, 0.15], [beam_width, beam_width, square_height], [0.0, 0.0, 0.0]]
        ]

        for name, position, scale, orientation in edges:
            edge_prim_path = f"{torso_link.prim_path}/square_edge_{name}"
            edge_prim = create_primitive_mesh(
                edge_prim_path,
                "Cylinder",
                extents=1.0
            )
            edge_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, edge_prim_path),
                name=f"{robot.name}:square_edge_{name}"
            )
            edge_geom.load(scene)
            beam_mat.bind(edge_geom.prim_path)
            edge_geom.scale = th.tensor(scale)
            edge_geom.set_position_orientation(
                position=th.tensor(position),
                orientation=T.euler2quat(th.tensor(orientation)),
                frame="parent"
            )
            vis_elements["reachability_visualizers"][name] = edge_geom
    
    return vis_elements


def create_panda_eef_cylinders(
    robot,
    scene,
    width=0.01,
    lengths=(0.4, 0.4, 0.8),
    proportion_offsets=(0.0, 0.0, 0.5),
    colors=(
        (1.0, 0.0, 0.0),  # X-axis
        (0.0, 1.0, 0.0),  # Y-axis
        (0.0, 0.0, 1.0),  # Z-axis
    ),
    quat_offsets=None,
):
    """
    Create end-effector cylinder visualizers (X / Y / Z) for a Panda robot.

    Args:
        robot: Robot instance that exposes arm_names and eef_links.
        scene: OmniGibson scene the robot belongs to.
        width (float): Cylinder radius.
        lengths (tuple[float, float, float]): Cylinder lengths for X, Y, Z.
        proportion_offsets (tuple[float, float, float]): Offset factors along cylinder length for positioning.
        colors (tuple[3-tuples]): RGB tuples in [0, 1] for each axis.
        quat_offsets (optional tuple): Per-axis quaternions; defaults align cylinders with +X/+Y/+Z.

    Returns:
        dict: arm_name -> list[VisualGeomPrim] for the three axis cylinders.
    """
    try:
        import torch
        from omnigibson.prims import VisualGeomPrim
        from omnigibson.prims.material_prim import OmniPBRMaterialPrim
        from omnigibson.utils import transform_utils as T
        from omnigibson.utils.usd_utils import (
            create_primitive_mesh,
            absolute_prim_path_to_scene_relative,
        )
    except ImportError as exc:
        raise ImportError(
            "OmniGibson is required to create Panda EEF cylinder visuals."
        ) from exc

    if quat_offsets is None:
        quat_offsets = (
            T.euler2quat(torch.tensor([0.0, torch.pi / 2, 0.0])),
            T.euler2quat(torch.tensor([-torch.pi / 2, 0.0, 0.0])),
            T.euler2quat(torch.tensor([0.0, 0.0, 0.0])),
        )

    color_tensors = tuple(torch.as_tensor(c, dtype=torch.float32) for c in colors)
    vis_geoms = {}

    arm_names = getattr(robot, "arm_names", ["arm"])
    for arm in arm_names:
        if arm not in robot.eef_links:
            continue
        hand_link = robot.eef_links[arm]
        arm_geoms = []
        for axis, length, color, prop_offset, quat_offset in zip(
            ("x", "y", "z"),
            lengths,
            color_tensors,
            proportion_offsets,
            quat_offsets,
        ):
            mat_prim_path = f"{robot.prim_path}/Looks/panda_eef_vis_{arm}_{axis}_mat"
            mat = OmniPBRMaterialPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(
                    scene, mat_prim_path
                ),
                name=f"{robot.name}:panda_eef_vis_{arm}_{axis}_mat",
            )
            mat.load(scene)
            mat.diffuse_color_constant = color

            vis_prim_path = f"{hand_link.prim_path}/panda_eef_vis_{axis}"
            create_primitive_mesh(
                vis_prim_path,
                "Cylinder",
                extents=1.0,
            )
            vis_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(
                    scene, vis_prim_path
                ),
                name=f"{robot.name}:arm_{arm}:panda_eef_vis_{axis}",
            )
            vis_geom.load(scene)
            vis_geom.material = mat
            vis_geom.scale = torch.tensor([width, width, length], dtype=torch.float32)
            vis_geom.set_position_orientation(
                position=torch.tensor(
                    [0.0, 0.0, length * prop_offset], dtype=torch.float32
                ),
                orientation=quat_offset,
                frame="parent",
            )
            arm_geoms.append(vis_geom)
            # breakpoint()

        vis_geoms[arm] = arm_geoms

    return vis_geoms


def setup_live_health_bars(target_objects_health):
    """
    Set up a live matplotlib window with health bars (HUD-style) for real-time health monitoring.
    
    Args:
        target_objects_health: List of object names to track
        
    Returns:
        tuple: (figure, axes, health_bars_dict)
        health_bars_dict: Dict mapping object names to dict containing:
            - 'background_bar': Rectangle patch for bar background
            - 'shadow_bar': Rectangle patch for shadow effect
            - 'foreground_bar': Rectangle patch for health bar
            - 'label_text': Text object for object name
            - 'value_text': Text object for health value
    """
    plt.ion()  # Enable interactive mode
    
    # Calculate window size based on number of objects
    n_objects = len(target_objects_health)
    bar_height = 40.0  # Increased height for better visibility
    bar_spacing = 18.0  # Increased spacing between bars
    header_height = 60.0  # More space for title
    padding = 25.0  # More padding around edges
    window_width = 600.0  # Wider window for better spacing
    window_height = header_height + n_objects * (bar_height + bar_spacing) + padding * 2
    
    fig, ax = plt.subplots(figsize=(window_width/100, window_height/100))
    fig.canvas.manager.set_window_title("Health Monitor")
    
    # Set dark background with subtle gradient effect
    fig.patch.set_facecolor('#1A1A1A')
    ax.set_facecolor('#1A1A1A')
    
    # Remove axes for HUD look
    ax.axis('off')
    ax.set_xlim(0, window_width)
    ax.set_ylim(0, window_height)
    
    # Add title with subtle underline effect
    title_text = ax.text(
        window_width / 2, window_height - 35,
        'Health Monitor',
        fontsize=18, color='#FFFFFF', weight='bold',
        verticalalignment='center', horizontalalignment='center',
        family='sans-serif'
    )
    # Subtle underline
    title_underline = Rectangle(
        (window_width / 2 - 80, window_height - 50), 160, 2,
        facecolor='#4CAF50', edgecolor='none', linewidth=0,
        zorder=1, alpha=0.6
    )
    ax.add_patch(title_underline)
    
    # Health bar configuration - improved spacing to prevent overlap
    label_width = 160.0  # Wider fixed width for labels
    bar_width = 280.0  # Wider health bar
    gap_after_label = 20.0  # Clear gap between label and bar
    gap_after_bar = 20.0  # Clear gap between bar and value
    bar_x_start = label_width + gap_after_label  # X position where bars start
    label_x = padding  # X position for object name labels
    value_x = bar_x_start + bar_width + gap_after_bar  # X position for health value
    
    health_bars_dict = {}
    
    # Create health bars for each object
    for idx, obj_name in enumerate(target_objects_health):
        # Y position for this bar (from top, accounting for header)
        y_pos = window_height - header_height - (idx + 1) * (bar_height + bar_spacing) - padding / 2
        
        # Background bar container (darker, with border and rounded look via inset)
        bg_bar = Rectangle(
            (bar_x_start, y_pos), bar_width, bar_height,
            facecolor='#0D0D0D', edgecolor='#2A2A2A', linewidth=2.5,
            zorder=1
        )
        ax.add_patch(bg_bar)
        
        # Outer glow effect (subtle highlight on top edge)
        glow_bar = Rectangle(
            (bar_x_start, y_pos + bar_height - 3), bar_width, 3,
            facecolor='#333333', edgecolor='none', linewidth=0, alpha=0.4,
            zorder=2
        )
        ax.add_patch(glow_bar)
        
        # Inner shadow effect (subtle darker border inside)
        shadow_bar = Rectangle(
            (bar_x_start + 2, y_pos + 2), bar_width - 4, bar_height - 4,
            facecolor='none', edgecolor='#000000', linewidth=1, alpha=0.5,
            zorder=2
        )
        ax.add_patch(shadow_bar)
        
        # Foreground bar (will be updated with health color and width)
        fg_bar = Rectangle(
            (bar_x_start + 3, y_pos + 3), 0, bar_height - 6,  # Start with 0 width, inset for border
            facecolor='#4CAF50', edgecolor='none', linewidth=0,
            zorder=3, alpha=0.95
        )
        ax.add_patch(fg_bar)
        
        # Object name label (truncate if too long, with better positioning)
        display_name = obj_name if len(obj_name) <= 20 else obj_name[:17] + '...'
        label_text = ax.text(
            label_x, y_pos + bar_height / 2,
            display_name,
            fontsize=12, color='#E8E8E8', weight='normal',
            verticalalignment='center', horizontalalignment='left',
            family='monospace'  # Monospace for consistent width
        )
        
        # Health value label with better formatting and positioning
        value_text = ax.text(
            value_x, y_pos + bar_height / 2,
            '100.0',
            fontsize=12, color='#FFFFFF', weight='bold',
            verticalalignment='center', horizontalalignment='left',
            family='sans-serif'
        )
        
        health_bars_dict[obj_name] = {
            'background_bar': bg_bar,
            'glow_bar': glow_bar,
            'shadow_bar': shadow_bar,
            'foreground_bar': fg_bar,
            'label_text': label_text,
            'value_text': value_text,
            'bar_x_start': bar_x_start + 3,  # Account for inset
            'bar_width': bar_width - 6,  # Account for inset borders
            'y_pos': y_pos + 3,  # Account for inset
            'bar_height': bar_height - 6  # Account for inset
        }
        
        # Initialize bar to 100% health (green, full width)
        fg_bar.set_width(bar_width - 6)
        fg_bar.set_facecolor('#4CAF50')  # Match the update function color
    
    plt.tight_layout()
    plt.show(block=False)
    
    return fig, ax, health_bars_dict


def update_live_health_bars(fig, ax, health_bars_dict, current_health_values, target_objects_health):
    """
    Update the live health bars with current health values.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        health_bars_dict: Dict mapping object names to bar components (from setup_live_health_bars)
        current_health_values: Dict mapping object names to current health value (float, 0-100)
        target_objects_health: List of object names to track
        
    Returns:
        bool: True if window is still active, False if closed
    """
    # Check if window is still open
    if not plt.fignum_exists(fig.number):
        return False
    
    # Update each health bar
    for obj_name in target_objects_health:
        if obj_name not in health_bars_dict:
            continue
        
        # Get current health value (default to 100 if not available)
        health = current_health_values.get(obj_name, 100.0)
        health = max(0.0, min(100.0, health))  # Clamp to [0, 100]
        
        bar_info = health_bars_dict[obj_name]
        fg_bar = bar_info['foreground_bar']
        value_text = bar_info['value_text']
        bar_width = bar_info['bar_width']
        
        # Calculate bar width as percentage
        health_width = (health / 100.0) * bar_width
        
        # Determine color based on health (more vibrant, game-like colors)
        if health == 0:
            # No health - transparent/empty
            fg_bar.set_facecolor('none')
            fg_bar.set_width(0)
            value_color = '#666666'  # Darker gray for 0 health
        elif health >= 80:
            # Green: 80-100 (bright vibrant green with slight gradient effect)
            fg_bar.set_facecolor('#4CAF50')  # Material Design green
            fg_bar.set_width(health_width)
            value_color = '#E8F5E9'  # Light green tint for text
        elif health >= 40:
            # Yellow/Orange: 40-79 (warning color)
            # Interpolate between yellow and orange for more visual feedback
            if health >= 60:
                # More yellow
                fg_bar.set_facecolor('#FFC107')  # Amber
                value_color = '#FFF9C4'  # Light yellow tint
            else:
                # More orange
                fg_bar.set_facecolor('#FF9800')  # Orange
                value_color = '#FFE0B2'  # Light orange tint
            fg_bar.set_width(health_width)
        else:
            # Red: 1-39 (danger color)
            # Darker red for lower health
            if health >= 20:
                fg_bar.set_facecolor('#F44336')  # Red
                value_color = '#FFCDD2'  # Light red tint
            else:
                fg_bar.set_facecolor('#D32F2F')  # Darker red
                value_color = '#EF9A9A'  # Lighter red for visibility
            fg_bar.set_width(health_width)
        
        # Update health value text with color and better formatting
        value_text.set_text(f'{health:.1f}')
        value_text.set_color(value_color)
        
        # Update label color based on health for better visual feedback
        label_text = bar_info['label_text']
        if health == 0:
            label_text.set_color('#888888')  # Gray for dead objects
        else:
            label_text.set_color('#E8E8E8')  # Normal light gray
    
    # Redraw efficiently
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    
    return True
