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