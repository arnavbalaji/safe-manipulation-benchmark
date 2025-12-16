import os
import sys
import cv2
import numpy as np
import subprocess
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
import torch as th

from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.damage_evaluators.mechanical_damage_evaluator import MechanicalDamageEvaluator

# Force headless plotting
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib

matplotlib.use("Agg")


def main(steps: int = 5000) -> None:
    # Enable object states; GPU dynamics not required for replay
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True
    gm.ENABLE_HQ_RENDERING = False

    # JSON snapshot saved from fill_mug.py via TAB
    scene_path = "safe-manipulation-benchmark/safety_benchmark/task_envs/complicated/fill_mug_saved.json"
    print(f"Loading scene from: {scene_path}")

    # Fresh simulator (mirror simple_task_load.py behavior)
    if og.sim is None:
        minimal_cfg = {
            "env": {"action_timestep": 1.0 / 60.0, "physics_timestep": 1.0 / 120.0},
            "scene": {"type": "Scene"},
            "robots": [],
        }
        tmp_env = DamageableEnvironment(configs=minimal_cfg)
        tmp_env.reset()
        og.clear()
    else:
        og.sim.stop()
        og.clear()

    cfg = {"scene": {"type": "Scene", "scene_file": scene_path}}
    env = DamageableEnvironment(configs=cfg)
    env.reset()

    # Set viewer camera pose to match simple_task.py
    start_cam_pos = th.tensor([-1.3225984573364258, 0.2645236849784851, 0.9981386065483093])
    start_cam_quat = th.tensor([0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452])
    og.sim.viewer_camera.set_position_orientation(position=start_cam_pos, orientation=start_cam_quat)

    # Grab robot
    assert len(env.robots) > 0, "No robots found after loading scene."
    robot = env.robots[0]

    # Configure robot controllers with higher gripper force (mirror simple_task.py)
    controller_config = {
        "arm_left": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        "arm_right": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        "gripper_left": {
            "motor_type": "position",
            "isaac_kp": 4000.0,
            "isaac_kd": 2000.0,
            "inverted": True,
        },
        "gripper_right": {
            "motor_type": "position",
            "isaac_kp": 4000.0,
            "isaac_kd": 2000.0,
            "inverted": True,
        },
    }
    robot.reload_controllers(controller_config=controller_config)

    # Increase friction on gripper fingers to reduce slip during grasp
    gripper_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
        prim_path=f"{robot.prim_path}/Looks/gripper_mat",
        name="gripper_material",
        static_friction=2.0,
        dynamic_friction=2.0,
    )
    for link_name, link in robot.links.items():
        link_n = link_name.lower()
        if ("gripper" in link_n) or ("finger" in link_n):
            for mesh in link.collision_meshes.values():
                mesh.apply_physics_material(gripper_mat)

    # Persist initial state after controller reload
    env.scene.update_initial_file()

    # Target mug for mechanical damage tracking (from fill_mug.yaml lines 106-112)
    target = env.scene.object_registry("name", "mug_ppzttc")
    target_eval = None
    if target is not None:
        for ev in getattr(target, "damage_evaluators", []):
            if isinstance(ev, MechanicalDamageEvaluator) or getattr(ev, "name", "") == "mechanical":
                target_eval = ev
                break
        if target_eval is None:
            print("⚠️ Target mug 'mug_ppzttc' has no MechanicalDamageEvaluator; strain tracking will be zero.")
    else:
        print("⚠️ Target mug 'mug_ppzttc' not found; health and strain tracking disabled.")

    # Teleoperation
    controller = KeyboardRobotController(robot=robot)
    controller.print_keyboard_teleop_info()

    fps = 30
    teleop_steps = steps

    frames = []
    frames_rgb = []  # Store RGB frames for PNG export
    frames_ego = []
    ego_key = None
    target_healths = []
    target_strains = []

    print(f"Starting {teleop_steps} steps of teleoperation with video + mechanical strain tracking...")
    for step in range(teleop_steps):
        action = controller.get_teleop_action()
        env.step(action=action)

        # Capture viewer camera frame
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        rgb_np = cv2.resize(rgb_np, (1080, 720))  # HD resolution
        frames_rgb.append(rgb_np.copy())  # Store RGB for PNG export
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Capture robot ego camera RGB (dynamic robot name, eyes:Camera:0)
        rob_obs = robot.get_obs()[0]
        if ego_key is None:
            for k in rob_obs.keys():
                if ":eyes:Camera:0" in k and "rgb" in rob_obs[k]:
                    ego_key = k
                    break
        if ego_key is not None and ego_key in rob_obs:
            ego_rgb = rob_obs[ego_key]["rgb"]
            ego_np = ego_rgb.cpu().numpy()[:, :, :3]
            ego_bgr = cv2.cvtColor(ego_np, cv2.COLOR_RGB2BGR)
            ego_bgr = cv2.resize(ego_bgr, (1080, 720))
            # Dynamic colored border based on current damage status (target mug)
            if target is not None:
                status = str(getattr(target, "damage_status", "")).lower()
                if status in ("major", "critical"):
                    bgr = (0, 0, 255)  # Red border
                elif status in ("minor",):
                    bgr = (0, 255, 255)  # Yellow border
                else:
                    bgr = (0, 255, 0)  # Green border
            else:
                bgr = (0, 255, 0)
            border = 30
            ego_bgr = cv2.copyMakeBorder(
                ego_bgr, border, border, border, border, cv2.BORDER_CONSTANT, value=bgr
            )
            frames_ego.append(ego_bgr)

        # Track target mug health
        if target is not None:
            target_healths.append(float(getattr(target, "health", 100.0)))

        # Track mechanical strain proxy for the target mug
        if target_eval is not None:
            if hasattr(target_eval, "last_strain_by_link") and getattr(
                target_eval, "last_strain_by_link", {}
            ):
                target_strains.append(float(max(target_eval.last_strain_by_link.values())))
            elif hasattr(target_eval, "get_current_env_step_strain"):
                target_strains.append(float(target_eval.get_current_env_step_strain()))
            elif hasattr(target_eval, "strain_values") and getattr(target_eval, "strain_values", []):
                target_strains.append(float(target_eval.strain_values[-1]))
            else:
                target_strains.append(0.0)
        else:
            target_strains.append(0.0)

        # Progress indicator
        if step % 50 == 0:
            latest_strain = target_strains[-1] if target_strains else 0.0
            latest_health = target_healths[-1] if target_healths else 100.0
            print(
                f"Step {step}/{teleop_steps} - Target mug strain: {latest_strain:.3f}, Health: {latest_health:.1f}%"
            )

    print("Teleoperation complete. Generating videos...")

    # Save high-quality PNG images from first, middle, and last frames
    if len(frames_rgb) > 0:
        out_dir = os.path.join(os.path.dirname(__file__), "videos")
        os.makedirs(out_dir, exist_ok=True)
        
        # Helper to convert RGB frame to uint8 (handle both 0-1 and 0-255 ranges)
        def prepare_frame_for_png(frame):
            frame_max = np.max(frame)
            if frame_max <= 1.0:
                # Values are in 0-1 range, convert to 0-255
                frame_uint8 = (frame * 255).astype(np.uint8)
            else:
                # Values are already in 0-255 range
                frame_uint8 = frame.astype(np.uint8)
            return frame_uint8
        
        # Save first frame
        first_frame = prepare_frame_for_png(frames_rgb[0])
        first_path = os.path.join(out_dir, "fill_mug_frame_first.png")
        cv2.imwrite(first_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved first frame: {first_path}")
        
        # Save middle frame
        mid_idx = len(frames_rgb) // 2
        mid_frame = prepare_frame_for_png(frames_rgb[mid_idx])
        mid_path = os.path.join(out_dir, "fill_mug_frame_middle.png")
        cv2.imwrite(mid_path, cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved middle frame: {mid_path}")
        
        # Save last frame
        last_frame = prepare_frame_for_png(frames_rgb[-1])
        last_path = os.path.join(out_dir, "fill_mug_frame_last.png")
        cv2.imwrite(last_path, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved last frame: {last_path}")

    # Generate videos (camera + ego + strain / health plots)
    if len(frames) > 0:
        generate_videos(frames, frames_ego, target_strains, target_healths, fps)

    og.clear()
    og.shutdown()


def generate_videos(frames, frames_ego, target_strains, target_healths, fps):
    """Generate high quality videos with mechanical strain and health plots."""
    out_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(out_dir, exist_ok=True)

    # Write main camera video
    sim_mp4 = os.path.join(out_dir, "fill_mug_camera_obs.mp4")
    if len(frames) > 0:
        h0, w0 = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(sim_mp4, fourcc, fps, (w0, h0))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()
        print(f"✅ Saved camera video: {sim_mp4}")

    # Write ego camera video
    ego_mp4 = os.path.join(out_dir, "fill_mug_ego_obs.mp4")
    if len(frames_ego) > 0:
        he, we = frames_ego[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw_e = cv2.VideoWriter(ego_mp4, fourcc, fps, (we, he))
        for f in frames_ego:
            vw_e.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw_e.release()
        print(f"✅ Saved ego camera video: {ego_mp4}")

    # Generate strain and health plot videos
    strain_mp4 = generate_strain_plot(target_strains, fps, out_dir)
    health_mp4 = generate_health_plot(target_healths, fps, out_dir)

    # Create camera with ego overlay
    cam_with_ego_mp4 = os.path.join(out_dir, "fill_mug_camera_with_ego.mp4")
    if os.path.exists(sim_mp4) and os.path.exists(ego_mp4):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                sim_mp4,
                "-i",
                ego_mp4,
                "-filter_complex",
                "[0:v]setsar=1[base];[1:v]scale=360:240,setsar=1[ego];[base][ego]overlay=W-w-20:20[out]",
                "-map",
                "[out]",
                "-c:v",
                "mpeg4",
                "-q:v",
                "5",
                cam_with_ego_mp4,
            ],
            check=True,
        )
        print(f"✅ Saved camera with ego overlay: {cam_with_ego_mp4}")

    # Combined video: camera + strain plot
    combined_strain_mp4 = os.path.join(out_dir, "fill_mug_with_strain.mp4")
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(strain_mp4):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                cam_with_ego_mp4,
                "-i",
                strain_mp4,
                "-filter_complex",
                "[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]",
                "-map",
                "[v]",
                "-c:v",
                "mpeg4",
                "-q:v",
                "5",
                combined_strain_mp4,
            ],
            check=True,
        )
        print(f"✅ Saved combined video (strain): {combined_strain_mp4}")

    # Combined video: camera + health plot
    combined_health_mp4 = os.path.join(out_dir, "fill_mug_with_health.mp4")
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(health_mp4):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                cam_with_ego_mp4,
                "-i",
                health_mp4,
                "-filter_complex",
                "[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]",
                "-map",
                "[v]",
                "-c:v",
                "mpeg4",
                "-q:v",
                "5",
                combined_health_mp4,
            ],
            check=True,
        )
        print(f"✅ Saved combined video (health): {combined_health_mp4}")

    # Clean up intermediate videos
    for video_path in [sim_mp4, ego_mp4, cam_with_ego_mp4, strain_mp4, health_mp4]:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass


def generate_strain_plot(target_strains, fps, out_dir):
    """Generate animated plot of mechanical strain proxy over time for the target mug."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    strain_mp4 = os.path.join(out_dir, "fill_mug_strain.mp4")

    strain_series = target_strains if len(target_strains) > 0 else [0.0]
    max_len = len(strain_series)

    fig_s, ax_s = plt.subplots(figsize=(9.6, 5.4))

    line_target, = ax_s.plot([], [], lw=6, color="tab:blue", label="Target mug strain")

    ax_s.set_xlim(0, max(1, max_len) / fps)
    y_min_s = 0.0
    y_max_s = max(float(max(strain_series + [0.0])), 1.0)
    if y_min_s == y_max_s:
        y_min_s, y_max_s = (0.0, 1.0)
    ax_s.set_ylim(y_min_s, y_max_s * 1.1)
    ax_s.set_xlabel("Time (s)", fontsize=20)
    ax_s.set_ylabel("Mechanical strain proxy", fontsize=20)
    ax_s.set_title("Target Mug Mechanical Strain Over Time", fontsize=26)

    ax_s.legend(loc="best", fontsize=16)
    ax_s.tick_params(axis="both", which="major", labelsize=16, width=1.5)
    ax_s.grid(True, linewidth=1.0, alpha=0.3)
    plt.tight_layout()

    def init_s():
        line_target.set_data([], [])
        return (line_target,)

    def animate_s(i):
        x = [k / fps for k in range(1, i + 2)]
        y = strain_series[: i + 1]
        line_target.set_data(x, y)
        return (line_target,)

    ani_s = animation.FuncAnimation(
        fig_s, animate_s, init_func=init_s, frames=max_len, interval=1000 / fps, blit=True
    )
    writer_s = animation.FFMpegWriter(
        fps=fps, codec="mpeg4", extra_args=["-vcodec", "mpeg4", "-qscale", "5"]
    )
    ani_s.save(strain_mp4, writer=writer_s)
    plt.close(fig_s)

    print(f"✅ Saved strain plot: {strain_mp4}")
    return strain_mp4


def generate_health_plot(target_healths, fps, out_dir):
    """Generate animated plot of target mug health over time."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    health_mp4 = os.path.join(out_dir, "fill_mug_health.mp4")

    health_series = target_healths if len(target_healths) > 0 else [100.0]
    max_len = len(health_series)

    fig_h, ax_h = plt.subplots(figsize=(9.6, 5.4))

    line_target, = ax_h.plot([], [], lw=6, color="tab:orange", label="Target mug health")

    ax_h.set_xlim(0, max(1, max_len) / fps)
    ax_h.set_ylim(0.0, 100.0)
    ax_h.set_xlabel("Time (s)", fontsize=20)
    ax_h.set_ylabel("Health (%)", fontsize=20)
    ax_h.set_title("Target Mug Health Over Time", fontsize=26)

    ax_h.legend(loc="best", fontsize=16)
    ax_h.tick_params(axis="both", which="major", labelsize=16, width=1.5)
    ax_h.grid(True, linewidth=1.0, alpha=0.3)
    plt.tight_layout()

    def init_h():
        line_target.set_data([], [])
        return (line_target,)

    def animate_h(i):
        x = [k / fps for k in range(1, i + 2)]
        y = health_series[: i + 1]
        line_target.set_data(x, y)
        return (line_target,)

    ani_h = animation.FuncAnimation(
        fig_h, animate_h, init_func=init_h, frames=max_len, interval=1000 / fps, blit=True
    )
    writer_h = animation.FFMpegWriter(
        fps=fps, codec="mpeg4", extra_args=["-vcodec", "mpeg4", "-qscale", "5"]
    )
    ani_h.save(health_mp4, writer=writer_h)
    plt.close(fig_h)

    print(f"✅ Saved health plot: {health_mp4}")
    return health_mp4


if __name__ == "__main__":
    main(steps=400)


