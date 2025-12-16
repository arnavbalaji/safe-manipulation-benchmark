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
import math

from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.damage_evaluators.electrical_damage_evaluator import ElectricalDamageEvaluator
from omnigibson.object_states import ToggledOn, Filled
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

# Force headless plotting
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")


def main(steps: int = 500) -> None:
    # Enable object states and GPU dynamics for particle correctness if present
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True
    gm.ENABLE_HQ_RENDERING = False

    scene_path = "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/move_glass.json"
    print(f"Loading scene from: {scene_path}")

    # Fresh simulator (match rl_test.py behavior)
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

    # Manually turn the faucet (furniture_sink) on at the start
    sink = env.scene.object_registry("name", "furniture_sink")
    if sink is not None and ToggledOn in sink.states:
        sink.states[ToggledOn].set_value(True)

    # Set viewer camera pose to match simple_task.py
    start_cam_pos = th.tensor([-1.3225984573364258, 0.2645236849784851, 0.9981386065483093])
    start_cam_quat = th.tensor([0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452])
    og.sim.viewer_camera.set_position_orientation(position=start_cam_pos, orientation=start_cam_quat)

    # Grab robot
    assert len(env.robots) > 0, "No robots found after loading scene."
    robot = env.robots[0]

    # Configure robot controllers with higher gripper force (match simple_task.py)
    # Note: grasping_mode should already be set in the saved JSON file
    controller_config = {
        # Set arms to IK so teleop supports EEF translation and rotation (arrows/P/;/N/B/O/U/V/C)
        "arm_left": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        "arm_right": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        # Stronger gripper position control
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
    # Create a PhysicsMaterial and apply it to gripper/finger collision meshes
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

    # Helper to hold laptop lid at target angle, mirroring PPO env behavior
    def set_laptop_pose(env, target_deg: float = 130.0):
        laptop = env.scene.object_registry("name", "laptop")
        if laptop is None:
            return
        target_rad = math.radians(float(target_deg))
        if hasattr(laptop, "joints"):
            for joint in laptop.joints.values():
                lo = joint.lower_limit
                hi = joint.upper_limit
                target = max(lo, min(hi, target_rad))
                joint.set_pos(target)
                if hasattr(joint, "keep_still"):
                    joint.keep_still()
        if hasattr(laptop, "keep_still"):
            laptop.keep_still()

    # Lock health changes during settle + initialization primitives, like PPO reset
    env.lock_health_changes()
    print("Stepping simulation for 20 steps to let physics settle...")
    for _ in range(20):
        action = th.zeros(robot.action_dim)
        set_laptop_pose(env)
        env.step(action=action)

    # Helper modeled after rl_test.py: execute a linear Cartesian EEF move with action primitives
    def execute_controller(env, controller, robot, delta_pose: th.Tensor, ignore_failure: bool = True, max_steps: int = 50):
        current_eef_pos = robot.get_eef_position("right")
        current_eef_orn = robot.get_eef_orientation("right")
        target_eef_pos = current_eef_pos + delta_pose
        target_eef_pose = (target_eef_pos, current_eef_orn)
        step = 0
        for action in controller._move_hand_linearly_cartesian(target_eef_pose, ignore_failure=ignore_failure):
            env.step(action=action)
            step += 1
            if step > max_steps:
                return

    # Instantiate action primitives and move the right arm to the requested target in two segments
    target_world_pos = th.tensor([0.4794032573699951, 0.15221568942070007, 0.6986250281333923], dtype=th.float32)
    prims = StarterSemanticActionPrimitives(env=env, robot=robot, skip_curobo_initilization=True)
    prims.arm = "right"
    current_pos = robot.get_eef_position("right")
    delta_total = target_world_pos - current_pos
    # First: move up (z only)
    delta_up = th.tensor([0.0, 0.0, float(delta_total[2])], dtype=th.float32)

    print(f"Executing UP move")
    execute_controller(env, prims, robot, th.tensor([0.0, 0.0, 0.25], dtype=th.float32))  
    execute_controller(env, prims, robot, th.tensor([0.0, -0.1, 0.0], dtype=th.float32))
    # # print(f"Executing FORWARD move")
    execute_controller(env, prims, robot, th.tensor([0.1, 0.0, 0.0], dtype=th.float32), max_steps=200)
    # print("Executing primitives complete")
    env.unlock_health_changes()

    # Generate 200 water particles above the mug and set it to Filled
    mug = env.scene.object_registry("name", "mug")
    if mug is not None:
        water_system = env.scene.get_system("water", force_init=True)
        mug.states[Filled].set_value(water_system, True)
        mug_pos, _ = mug.get_position_orientation()
        z_offset = 0.05 if not isinstance(mug_pos, th.Tensor) else 0.05
        for _ in range(200):
            if isinstance(mug_pos, th.Tensor):
                drop_pos = (mug_pos + th.tensor([0.0, 0.0, z_offset], dtype=th.float32)).tolist()
            else:
                drop_pos = [mug_pos[0], mug_pos[1], mug_pos[2] + z_offset]
            water_system.generate_particles(positions=[drop_pos])
            for _ in range(2):
                og.sim.step()
    
    for _ in range(50):
        og.sim.step()

    # Teleoperation
    controller = KeyboardRobotController(robot=robot)
    # Register TAB key to save scene JSON and breakpoint (same as simple_task.py)
    def save_and_breakpoint():
        print("=== TAB callback triggered ===", flush=True)
        save_path = os.path.join(os.path.dirname(__file__), "simple_task_saved.json")
        print(f"Saving to: {save_path}")
        og.sim.save([save_path])
        # Print right EEF pose, and mug / laptop poses, then breakpoint (no saving)
        # Right EEF pose: pick a link that looks like the right gripper / eef / tool
        right_eef_pos, right_eef_quat = None, None
        for link_name, link in robot.links.items():
            ln = link_name.lower()
            if ("right" in ln) and ("gripper" in ln or "eef" in ln or "tool" in ln):
                right_eef_pos, right_eef_quat = link.get_position_orientation()
                break
        if right_eef_pos is None:
            for link_name, link in robot.links.items():
                if "right" in link_name.lower():
                    right_eef_pos, right_eef_quat = link.get_position_orientation()
                    break
        pos_print = right_eef_pos.tolist() if hasattr(right_eef_pos, "tolist") else right_eef_pos
        quat_print = right_eef_quat.tolist() if hasattr(right_eef_quat, "tolist") else right_eef_quat
        print(f"Right EEF position: {pos_print}")
        print(f"Right EEF orientation (quat xyzw): {quat_print}")

        mug = env.scene.object_registry("name", "mug")
        laptop = env.scene.object_registry("name", "laptop")
        if mug is not None:
            mpos, mo = mug.get_position_orientation()
            print(f"Mug position: {mpos.tolist() if hasattr(mpos, 'tolist') else mpos}")
            print(f"Mug orientation (quat xyzw): {mo.tolist() if hasattr(mo, 'tolist') else mo}")
        else:
            print("Mug not found.")
        if laptop is not None:
            lpos, lo = laptop.get_position_orientation()
            print(f"Laptop position: {lpos.tolist() if hasattr(lpos, 'tolist') else lpos}")
            print(f"Laptop orientation (quat xyzw): {lo.tolist() if hasattr(lo, 'tolist') else lo}")
        else:
            print("Laptop not found.")
        # Print current camera position and orientation
        cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
        print(f"📷 Current camera position: {cam_pos.tolist() if hasattr(cam_pos, 'tolist') else cam_pos}")
        print(f"📷 Current camera orientation (quaternion): {cam_quat.tolist() if hasattr(cam_quat, 'tolist') else cam_quat}")
        breakpoint()
    controller.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Print EEF/mug/laptop poses and breakpoint",
        callback_fn=save_and_breakpoint,
    )
    controller.print_keyboard_teleop_info()

    # Get laptop object and setup electrical damage evaluator for water contact tracking
    laptop = env.scene.object_registry("name", "laptop")
    laptop_eval = None
    if laptop is not None:
        # Prefer an existing evaluator already attached by the environment (mimic electrical_damage.py style)
        existing_evals = getattr(laptop, "damage_evaluators", [])
        for ev in existing_evals:
            if isinstance(ev, ElectricalDamageEvaluator):
                laptop_eval = ev
                break
        # Never create a new evaluator — fail fast if missing
        assert laptop_eval is not None, "Laptop is missing ElectricalDamageEvaluator; this should never happen."
        print(f"✅ Tracking water contacts for laptop: {laptop.name}")
    else:
        print("⚠️ No laptop found in scene - water contact tracking disabled")

    # Data collection for video generation
    frames = []
    frames_rgb = []  # Store RGB frames for PNG export
    frames_ego = []
    ego_key = None
    laptop_contact_counts = []
    laptop_healths = []
    # Track discrete laptop damage status each step for status-border video
    laptop_statuses = []
    fps = 30
    teleop_steps = 400

    print(f"Starting {teleop_steps} steps of teleoperation with video recording...")
    for step in range(teleop_steps):
        action = controller.get_teleop_action()
        # Hold laptop pose each step (mirrors PPO env behavior)
        set_laptop_pose(env)
        env.step(action=action)

        # Capture viewer camera frame
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        rgb_np = cv2.resize(rgb_np, (1080, 720))  # HD resolution
        frames_rgb.append(rgb_np.copy())  # Store RGB for PNG export
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Cache laptop damage status this step
        if laptop is not None:
            status_this_step = str(getattr(laptop, "damage_status", "none")).lower()
        else:
            status_this_step = "none"
        laptop_statuses.append(status_this_step)

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
            # Dynamic colored border based on current damage status (laptop is target)
            if status_this_step in ("major", "critical"):
                bgr = (0, 0, 255)  # Red border
            elif status_this_step in ("minor",):
                bgr = (0, 255, 255)  # Yellow border
            else:
                bgr = (0, 255, 0)  # Green border
            border = 30  # 30px at 1080x720 -> ~10px after scaling to 360x240
            ego_bgr = cv2.copyMakeBorder(ego_bgr, border, border, border, border, cv2.BORDER_CONSTANT, value=bgr)
            frames_ego.append(ego_bgr)

        # Track water particle contacts for laptop
        laptop_contact_count = 0
        if laptop_eval is not None:
            summary = laptop_eval.get_contact_summary()
            details = summary.get("link_details", {})
            laptop_contact_count = max((v.get("particle_count", 0) for v in details.values()), default=0)
        laptop_contact_counts.append(int(laptop_contact_count))

        # Track laptop health
        laptop_health = 100.0
        if laptop is not None:
            laptop_health = float(getattr(laptop, "health", 100.0))
        laptop_healths.append(laptop_health)

        # Progress indicator
        if step % 50 == 0:
            print(f"Step {step}/{teleop_steps} - Laptop water contacts: {laptop_contact_count}, Health: {laptop_health:.1f}%")

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
        first_path = os.path.join(out_dir, "electrical_frame_first.png")
        cv2.imwrite(first_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved first frame: {first_path}")
        
        # Save middle frame
        mid_idx = len(frames_rgb) // 2
        mid_frame = prepare_frame_for_png(frames_rgb[mid_idx])
        mid_path = os.path.join(out_dir, "electrical_frame_middle.png")
        cv2.imwrite(mid_path, cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved middle frame: {mid_path}")
        
        # Save last frame
        last_frame = prepare_frame_for_png(frames_rgb[-1])
        last_path = os.path.join(out_dir, "electrical_frame_last.png")
        cv2.imwrite(last_path, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved last frame: {last_path}")

    # Generate videos
    if len(frames) > 0 and laptop is not None:
        generate_videos(frames, frames_ego, laptop_contact_counts, laptop_healths, laptop_statuses, fps)

    og.clear()
    og.shutdown()


def generate_videos(frames, frames_ego, laptop_contact_counts, laptop_healths, laptop_statuses, fps):
    """Generate high quality video with water contacts, health plots, and a status-border sim video."""
    out_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(out_dir, exist_ok=True)

    # Write main camera video
    sim_mp4 = os.path.join(out_dir, "laptop_camera_obs.mp4")
    if len(frames) > 0:
        h0, w0 = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(sim_mp4, fourcc, fps, (w0, h0))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()
        print(f"✅ Saved camera video: {sim_mp4}")

        # Also create a status-border sim video where the border color encodes laptop damage status.
        border_mp4 = os.path.join(out_dir, "laptop_camera_with_status_border.mp4")
        border_width = 30
        vw_border = cv2.VideoWriter(
            border_mp4,
            fourcc,
            fps,
            (w0 + 2 * border_width, h0 + 2 * border_width),
        )

        # Fallback if statuses are missing for some reason
        if len(laptop_statuses) == 0:
            laptop_statuses = ["none"] * len(frames)

        for f, status in zip(frames, laptop_statuses):
            s = (status or "none").lower()
            if s in ("major", "critical"):
                bgr = (0, 0, 255)  # Red
            elif s == "minor":
                bgr = (0, 255, 255)  # Yellow
            else:
                bgr = (0, 255, 0)  # Green

            bordered = cv2.copyMakeBorder(
                f,
                border_width,
                border_width,
                border_width,
                border_width,
                cv2.BORDER_CONSTANT,
                value=bgr,
            )
            vw_border.write(np.ascontiguousarray(bordered, dtype=np.uint8))

        vw_border.release()
        print(f"✅ Saved camera video with status border: {border_mp4}")

    # Write ego camera video
    ego_mp4 = os.path.join(out_dir, "laptop_ego_obs.mp4")
    if len(frames_ego) > 0:
        he, we = frames_ego[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw_e = cv2.VideoWriter(ego_mp4, fourcc, fps, (we, he))
        for f in frames_ego:
            vw_e.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw_e.release()
        print(f"✅ Saved ego camera video: {ego_mp4}")

    # Generate water contacts plot video
    water_mp4 = generate_water_contacts_plot(laptop_contact_counts, fps, out_dir)
    
    # Generate health plot video
    health_mp4 = generate_health_plot(laptop_healths, fps, out_dir)
    
    # Create camera with ego overlay
    cam_with_ego_mp4 = os.path.join(out_dir, "laptop_camera_with_ego.mp4")
    if os.path.exists(sim_mp4) and os.path.exists(ego_mp4):
        # Ego frames already contain dynamic border and are possibly larger due to copyMakeBorder.
        # Scale ego down and overlay without additional padding.
        subprocess.run([
            'ffmpeg', '-y',
            '-i', sim_mp4,
            '-i', ego_mp4,
            '-filter_complex',
            '[0:v]setsar=1[base];[1:v]scale=360:240,setsar=1[ego];[base][ego]overlay=W-w-20:20[out]',
            '-map','[out]',
            '-c:v','mpeg4','-q:v','5', cam_with_ego_mp4
        ], check=True)
        print(f"✅ Saved camera with ego overlay: {cam_with_ego_mp4}")

    # Create final combined video (camera + ego + water contacts plot)
    combined_water_mp4 = os.path.join(out_dir, "laptop_with_water_contacts.mp4")
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(water_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', cam_with_ego_mp4,
            '-i', water_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_water_mp4
        ], check=True)
        print(f"✅ Saved combined video (water): {combined_water_mp4}")

    # Create final combined video (camera + ego + health plot)
    combined_health_mp4 = os.path.join(out_dir, "laptop_with_health.mp4")
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(health_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', cam_with_ego_mp4,
            '-i', health_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_health_mp4
        ], check=True)
        print(f"✅ Saved combined video (health): {combined_health_mp4}")

    # Clean up intermediate videos
    for video_path in [sim_mp4, ego_mp4, cam_with_ego_mp4, water_mp4, health_mp4]:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass


def generate_water_contacts_plot(laptop_contact_counts, fps, out_dir):
    """Generate animated plot of water particle contacts for laptop."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    water_mp4 = os.path.join(out_dir, 'laptop_water_contacts.mp4')
    
    # Prepare data series
    contact_series = laptop_contact_counts if len(laptop_contact_counts) > 0 else [0]
    max_len = len(contact_series)
    
    # Create figure with professional styling
    fig_w, ax_w = plt.subplots(figsize=(9.6, 5.4))
    
    line_laptop, = ax_w.plot([], [], lw=6, color='tab:cyan', label='Laptop water contacts')
    
    # Configure axes
    ax_w.set_xlim(0, max(1, max_len) / fps)
    y_min_w = 0.0
    y_max_w = max(float(max(contact_series + [0])), 1.0)
    if y_min_w == y_max_w:
        y_min_w, y_max_w = (0.0, 1.0)
    ax_w.set_ylim(y_min_w, y_max_w * 1.1)
    ax_w.set_xlabel('Time (s)', fontsize=20)
    ax_w.set_ylabel('Water particle contacts', fontsize=20)
    ax_w.set_title('Laptop Water Particle Contacts Over Time', fontsize=26)
    
    # Add damage threshold line (using typical laptop threshold from electrical_damage.py)
    damage_threshold = 10.0
    ax_w.axhline(y=damage_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                 label=f'Damage Threshold ({damage_threshold} particles)')
    
    ax_w.legend(loc='best', fontsize=16)
    ax_w.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    ax_w.grid(True, linewidth=1.0, alpha=0.3)
    plt.tight_layout()

    # Animation functions
    def init_w():
        line_laptop.set_data([], [])
        return line_laptop,

    def animate_w(i):
        x = [k / fps for k in range(1, i + 2)]
        y_laptop = contact_series[: i + 1]
        line_laptop.set_data(x, y_laptop)
        return line_laptop,

    # Create and save animation
    ani_w = animation.FuncAnimation(fig_w, animate_w, init_func=init_w, frames=max_len, 
                                    interval=1000 / fps, blit=True)
    writer_w = animation.FFMpegWriter(fps=fps, codec='mpeg4', 
                                       extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani_w.save(water_mp4, writer=writer_w)
    plt.close(fig_w)
    
    print(f"✅ Saved water contacts plot: {water_mp4}")
    return water_mp4


def generate_health_plot(laptop_healths, fps, out_dir):
    """Generate animated plot of laptop health over time."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    health_mp4 = os.path.join(out_dir, 'laptop_healths.mp4')
    
    # Prepare data series
    health_series = laptop_healths if len(laptop_healths) > 0 else [100.0]
    max_len = len(health_series)
    
    # Create figure with professional styling
    fig_h, ax_h = plt.subplots(figsize=(9.6, 5.4))
    
    line_laptop, = ax_h.plot([], [], lw=6, color='tab:orange', label='Laptop health')
    
    # Configure axes
    ax_h.set_xlim(0, max(1, max_len) / fps)
    ax_h.set_ylim(0.0, 100.0)
    ax_h.set_xlabel('Time (s)', fontsize=20)
    ax_h.set_ylabel('Health (%)', fontsize=20)
    ax_h.set_title('Laptop Health Over Time', fontsize=26)
    
    ax_h.legend(loc='best', fontsize=16)
    ax_h.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    ax_h.grid(True, linewidth=1.0, alpha=0.3)
    plt.tight_layout()

    # Animation functions
    def init_h():
        line_laptop.set_data([], [])
        return line_laptop,

    def animate_h(i):
        x = [k / fps for k in range(1, i + 2)]
        y_laptop = health_series[: i + 1]
        line_laptop.set_data(x, y_laptop)
        return line_laptop,

    # Create and save animation
    ani_h = animation.FuncAnimation(fig_h, animate_h, init_func=init_h, frames=max_len, 
                                    interval=1000 / fps, blit=True)
    writer_h = animation.FFMpegWriter(fps=fps, codec='mpeg4', 
                                       extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani_h.save(health_mp4, writer=writer_h)
    plt.close(fig_h)
    
    print(f"✅ Saved health plot: {health_mp4}")
    return health_mp4


if __name__ == "__main__":
    main(steps=10000)


