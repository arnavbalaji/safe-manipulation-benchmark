import os
import sys
import json
import matplotlib
import torch as th
# Use non-interactive backend to avoid Qt/xcb plugin requirement in headless runs
matplotlib.use('Agg')

from safety_benchmark.damageable_env import DamageableEnvironment

def damage_reward_fn(env, obs):
    terminated = False
    plate_health_states = obs["object_health_states"]["glass_plate"]
    total_damage = 0.0
    for damage_type, damage_info in plate_health_states["damage_info"].items():
        for link_name, damage in damage_info.items():
            total_damage += damage
    if total_damage == 0.0 and plate_health_states["health"] == 100.0:
        return 0.0001, terminated
    if plate_health_states["health"] == 0.0:
        terminated = True
    return -total_damage, terminated


def distance_reward_fn(env, obs):
    eps = 0.1
    terminated = False
    plate_obj = env.scene.object_registry("name", "glass_plate")
    plate_pos, plate_orn = plate_obj.get_position_orientation()
    target_pos = th.tensor([-0.0966, 0.0843, 0.4593])
    distance = th.norm(plate_pos - target_pos).item()
    if distance < eps:
        terminated = True
        return 100.0, terminated
    return -distance, terminated

def reward_fn(env, obs):
    return distance_reward_fn(env, obs)

def _ensure_omnigibson_on_path():
    # Allow running without installing the package by appending the local OmniGibson repo
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    og_src = os.path.join(repo_root, "OmniGibson")
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


def execute_controller(env, controller, robot, delta_pose, ignore_failure=True, max_steps=500):
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


def main():
    _ensure_omnigibson_on_path()

    import omnigibson as og
    from omnigibson.utils.ui_utils import KeyboardRobotController

    # Initialize the simulation first
    if og.sim is None:
        # Create a minimal environment to initialize the simulator
        minimal_cfg = {
            "env": {"action_timestep": 1.0 / 60.0, "physics_timestep": 1.0 / 120.0},
            "scene": {"type": "Scene"},
            "robots": [],
        }
        temp_env = DamageableEnvironment(configs=minimal_cfg)
        temp_env.reset()
        # Now we can safely clear
        og.clear()
    else:
        # Stop and clear existing simulation
        og.sim.stop()
        og.clear()

    # Path to the saved state file
    save_path = "lift_test.json"
    
    if not os.path.exists(save_path):
        print(f"Error: Saved state file '{save_path}' not found!")
        print("Please run test_save.py first to create the state file.")
        return

    print(f"Loading simulation state from: {save_path}")

    # Overwrite damage params in the saved scene JSON to current values before loading (match mech_damage.py)
    try:
        with open(save_path, "r") as f:
            scene_dict = json.load(f)
        init_info = scene_dict["objects_info"]["init_info"]

        # Define PARAMS consistent with mech_damage.py for robot and plate
        PARAMS = {
            "tiago_robot": {
                "damage_evaluators": ["mechanical", "electrical"],
                "health_thresholds": [90.0, 60.0, 30.0],
                "mechanical": {
                    "damage_threshold": 1e10,
                    "scale": 1e-10,
                    "instant_coefficient": 1.0,
                    "creep_coefficient": 1.0,
                    "object_type": "brittle",
                    "link_thresholds": {
                        "arm": {
                            "damage_threshold": 200.0,
                            "scale": 0.05,
                        },
                        "gripper": {
                            "damage_threshold": 200.0,
                            "scale": 0.05,
                        },
                    },
                },
                "electrical": {"damage_threshold": 0.0, "scale": 0.001, "water_system_name": "sludge"},
            },
            "plate": {
                "damage_evaluators": ["mechanical"],
                "health_thresholds": [90.0, 60.0, 30.0],
                "mechanical": {
                    "damage_threshold": 200.0,
                    "scale": 0.5,
                    "instant_coefficient": 1.0,
                    "creep_coefficient": 0.0,
                    "object_type": "brittle",
                },
            },
            "default": {
                "damage_evaluators": ["mechanical"],
                "health_thresholds": [90.0, 60.0, 30.0],
                "mechanical": {
                    "damage_threshold": 0.0,
                    "scale": 1.0,
                    "instant_coefficient": 1.0,
                    "creep_coefficient": 1.0,
                },
            },
        }

        def set_params(entry_args, params_dict):
            entry_args["params"] = params_dict

        for key, entry in init_info.items():
            args = entry.get("args", {})
            name = args.get("name", "")
            category = args.get("category", "")
            class_name = entry.get("class_name", "")

            if name.startswith("robot_") or class_name == "DamageableTiago":
                set_params(args, PARAMS["tiago_robot"])  # robot
            elif name == "coffee_table":
                set_params(args, PARAMS.get("coffee_table", PARAMS["default"]))
            elif name == "target_object" or name == "glass_plate" or category == "plate":
                # Plate target
                set_params(args, PARAMS["plate"])
            else:
                # Generic: if category maps directly to PARAMS, apply it
                if category in PARAMS:
                    set_params(args, PARAMS[category])

        with open(save_path, "w") as f:
            json.dump(scene_dict, f)
    except Exception as e:
        print(f"Warning: Could not rewrite params in {save_path}: {e}")

    # Create an Environment from the saved scene so we can step actions
    try:
        cfg = {"scene": {"type": "Scene", "scene_file": save_path}}
        env = DamageableEnvironment(configs=cfg, reward_fn=reward_fn)
        env.reset()
    except Exception as e:
        print(f"Failed to create environment from scene file: {e}")
        return

    # Grab robot
    if len(env.robots) == 0:
        print("Error: No robots found after loading scene!")
        return
    robot = env.robots[0]
    print(f"Found robot: {robot.name}")

    # Toggle whether to track robot health/strain (only meaningful when target object exists)
    track_robot_health = True

    # Match viewer camera pose from mech_damage.py after load
    try:
        og.sim.viewer_camera.set_position_orientation(
            position=[-1.3225984573364258, 0.2645236849784851, 0.9981386065483093],
            orientation=[0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452],
        )
    except Exception:
        pass

    # Match gripper force settings from mech_damage.py (higher stiffness/damping)
    # Also set arm controllers to pose_absolute_ori mode for action primitives
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
            "name": "MultiFingerGripperController",
            "motor_type": "position",
            "isaac_kp": 20000.0,
            "isaac_kd": 1000.0,
            "inverted": True,
        },
        "gripper_right": {
            "name": "MultiFingerGripperController",
            "motor_type": "position",
            "isaac_kp": 20000.0,
            "isaac_kd": 1000.0,
            "inverted": True,
        },
    }
    robot.reload_controllers(controller_config=controller_config)
    # Persist controller state so subsequent saves keep these settings
    env.scene.update_initial_file()

    # Set up keyboard teleoperation for the restored robot
    teleop = KeyboardRobotController(robot=robot)
    teleop.print_keyboard_teleop_info()

    print("Press ESC to quit")

    # Find the plate object for health tracking
    plate_obj = None
    for obj_name in ["glass_plate", "target_object", "plate"]:
        plate_obj = env.scene.object_registry("name", obj_name)
        if plate_obj is not None:
            break
    
    if plate_obj is None:
        print("Warning: No plate object found for health tracking")
    else:
        print(f"Tracking health for: {plate_obj.name}")

    # Video recording setup
    import cv2
    import numpy as np
    import subprocess
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import torch as th
    
    frames = []
    # Mech-style extras
    frames_ego = []
    ego_key = None
    robot_healths = []
    target_healths = []
    robot_strains = []
    target_strains = []
    # RL reward tracking (existing)
    total_rewards = []
    total_damage_rewards = []
    total_distance_rewards = []
    fps = 30
    total_reward = 0.0
    total_damage_reward = 0.0
    total_distance_reward = 0.0

    # Let physics settle for 20 steps
    print("Letting physics settle for 20 steps...")
    for _ in range(20):
        action = teleop.get_teleop_action()
        env.step(action=action)

    # Move right end effector up by 0.3m using action primitives
    print("Moving right end effector up by 0.3m...")
    from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
    
    # Create action primitives instance
    action_primitives = StarterSemanticActionPrimitives(env=env, robot=robot, skip_curobo_initilization=True)
    
    # Set the action primitives to use the right arm
    action_primitives.arm = "right"
    
    # Move the hand to the target pose using direct IK
    print("Executing upward movement with right arm...")
    execute_controller(env, action_primitives, robot, th.tensor([0.0, 0.0, 0.45]))
    print("Upward movement completed. Starting teleoperation and video recording...")

    execute_controller(env, action_primitives, robot, th.tensor([0.0, -0.15, 0.0]))
    print("Forward movement completed. Starting teleoperation and video recording...")

    # Resolve target object (for status / health)
    target_ref = None
    for name in ["target_object", "glass_plate", "plate"]:
        try:
            target_ref = env.scene.object_registry("name", name)
        except Exception:
            target_ref = None
        if target_ref is not None:
            break

    # Run simulation loop
    steps = 500
    for step in range(steps):
        action = teleop.get_teleop_action()
        obs, reward, terminated, truncated, info = env.step(action=action)

        plate_pos, plate_orn = plate_obj.get_position_orientation()

        # Track cumulative (total) episode reward and individual components
        total_reward += float(reward)
        total_rewards.append(total_reward)
        
        # Track individual reward components
        damage_reward, _ = damage_reward_fn(env, obs)
        distance_reward, _ = distance_reward_fn(env, obs)
        total_damage_reward += damage_reward
        total_distance_reward += distance_reward
        total_damage_rewards.append(total_damage_reward)
        total_distance_rewards.append(total_distance_reward)

        # Optional plate tracking if available
        if plate_obj is not None:
            obj_pos, obj_orn = plate_obj.get_position_orientation()

        # Capture viewer camera frame
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        rgb_np = cv2.resize(rgb_np, (1080, 720))
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Ego camera with dynamic border based on target damage status
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
            status = str(getattr(target_ref, "damage_status", "none")).lower() if target_ref is not None else "none"
            if status in ("major", "critical"):
                bgr = (0, 0, 255)
            elif status in ("minor",):
                bgr = (0, 255, 255)
            else:
                bgr = (0, 255, 0)
            border = 30
            ego_bgr = cv2.copyMakeBorder(ego_bgr, border, border, border, border, cv2.BORDER_CONSTANT, value=bgr)
            frames_ego.append(ego_bgr)

        # Health tracking (conditionally track robot based on track_robot_health)
        should_track_robot = track_robot_health or (target_ref is None)
        if should_track_robot:
            robot_healths.append(float(getattr(robot, "health", 100.0)))
        if target_ref is not None:
            target_healths.append(float(getattr(target_ref, "health", 100.0)))

        # Strain tracking: take max over per-link last_strain_by_link after the step
        # For robot, only include links with "arm" or "gripper" in their names
        if should_track_robot and hasattr(robot, "damage_evaluators") and len(getattr(robot, "damage_evaluators", [])) > 0:
            r_dmg = robot.damage_evaluators[0]
            if hasattr(r_dmg, "last_strain_by_link") and len(getattr(r_dmg, "last_strain_by_link", {})) > 0:
                # Filter to only arm/gripper links
                arm_gripper_strains = {
                    link_name: strain
                    for link_name, strain in r_dmg.last_strain_by_link.items()
                    if "arm" in link_name.lower() or "gripper" in link_name.lower()
                }
                if arm_gripper_strains:
                    robot_strains.append(float(max(arm_gripper_strains.values())))
                else:
                    # Fallback if no arm/gripper links found
                    robot_strains.append(float(max(r_dmg.last_strain_by_link.values())))
            elif hasattr(r_dmg, "get_current_env_step_strain"):
                robot_strains.append(float(r_dmg.get_current_env_step_strain()))
            elif hasattr(r_dmg, "strain_values") and len(r_dmg.strain_values) > 0:
                robot_strains.append(float(r_dmg.strain_values[-1]))
            else:
                robot_strains.append(0.0)
        elif should_track_robot:
            robot_strains.append(0.0)

        if target_ref is not None and hasattr(target_ref, "damage_evaluators") and len(getattr(target_ref, "damage_evaluators", [])) > 0:
            t_dmg = target_ref.damage_evaluators[0]
            if hasattr(t_dmg, "last_strain_by_link") and len(getattr(t_dmg, "last_strain_by_link", {})) > 0:
                target_strains.append(float(max(t_dmg.last_strain_by_link.values())))
            elif hasattr(t_dmg, "get_current_env_step_strain"):
                target_strains.append(float(t_dmg.get_current_env_step_strain()))
            elif hasattr(t_dmg, "strain_values") and len(t_dmg.strain_values) > 0:
                target_strains.append(float(t_dmg.strain_values[-1]))
            else:
                target_strains.append(0.0)
        elif target_ref is not None:
            target_strains.append(0.0)

    # Generate videos
    videos_dir = "safe-manipulation-benchmark/rl/videos"
    os.makedirs(videos_dir, exist_ok=True)
    
    # Write camera video
    avi_path = os.path.join(videos_dir, "rl_test_camera.avi")
    mp4_path = os.path.join(videos_dir, "rl_test_camera.mp4")
    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()
        # Convert AVI to MP4
        subprocess.run(["ffmpeg", "-y", "-i", avi_path, "-c:v", "mpeg4", mp4_path], check=True)
        os.remove(avi_path)

    # Mech-style videos: ego overlay, health plot, strain plot, and combined outputs
    base_name = "plate"
    # Ego camera video
    avi_ego = os.path.join(videos_dir, f"{base_name}_ego_obs.avi")
    mp4_ego = os.path.join(videos_dir, f"{base_name}_ego_obs.mp4")
    if len(frames_ego) > 0:
        he, we = frames_ego[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw_e = cv2.VideoWriter(avi_ego, fourcc, fps, (we, he))
        for f in frames_ego:
            vw_e.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw_e.release()
        subprocess.run(["ffmpeg", "-y", "-i", avi_ego, "-c:v", "mpeg4", mp4_ego], check=True)
        os.remove(avi_ego)

    # Camera with ego overlay (top-right), like mech_damage
    cam_with_ego_mp4 = os.path.join(videos_dir, f"{base_name}_camera_with_ego.mp4")
    if os.path.exists(mp4_path) and os.path.exists(mp4_ego):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', mp4_path,
            '-i', mp4_ego,
            '-filter_complex',
            '[0:v]setsar=1[base];[1:v]scale=360:240,setsar=1[ego];[base][ego]overlay=W-w-20:20[out]',
            '-map','[out]',
            '-c:v','mpeg4','-q:v','5', cam_with_ego_mp4
        ], check=True)

    # Health plot (robot + plate)
    health_mp4 = os.path.join(videos_dir, f'{base_name}_healths.mp4')
    should_track_robot = track_robot_health or (target_ref is None)
    
    if should_track_robot and len(robot_healths) > 0 and len(target_healths) > 0:
        # Both robot and target
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = max(len(robot_healths), len(target_healths))
        y_min, y_max = 0.0, 100.0
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
        line_p, = ax.plot([], [], lw=6, color='tab:orange', label='Plate')
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
            line_r.set_data([], [])
            line_p.set_data([], [])
            return line_r, line_p

        def animate_health(i):
            x = [k / fps for k in range(1, i + 2)]
            y_r = robot_healths[: i + 1]
            y_p = target_healths[: i + 1]
            line_r.set_data(x, y_r)
            line_p.set_data(x, y_p)
            return line_r, line_p

        ani = animation.FuncAnimation(fig, animate_health, init_func=init_health, frames=T, interval=1000 / fps, blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)
    elif should_track_robot and len(robot_healths) > 0 and (target_ref is None):
        # Robot only
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = len(robot_healths)
        y_min, y_max = 0.0, 100.0
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Health', fontsize=20)
        ax.set_title('Health Over Time', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_health_r():
            line_r.set_data([], [])
            return line_r,

        def animate_health_r(i):
            x = [k / fps for k in range(1, i + 2)]
            y_r = robot_healths[: i + 1]
            line_r.set_data(x, y_r)
            return line_r,

        ani = animation.FuncAnimation(fig, animate_health_r, init_func=init_health_r, frames=T, interval=1000 / fps, blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)
    elif not should_track_robot and target_ref is not None and len(target_healths) > 0:
        # Target only (when track_robot_health is False)
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = len(target_healths)
        y_min, y_max = 0.0, 100.0
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_p, = ax.plot([], [], lw=6, color='tab:orange', label='Plate')
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Health', fontsize=20)
        ax.set_title('Health Over Time', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_health_p():
            line_p.set_data([], [])
            return line_p,

        def animate_health_p(i):
            x = [k / fps for k in range(1, i + 2)]
            y_p = target_healths[: i + 1]
            line_p.set_data(x, y_p)
            return line_p,

        ani = animation.FuncAnimation(fig, animate_health_p, init_func=init_health_p, frames=T, interval=1000 / fps, blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)

    # Side-by-side cam_with_ego + health
    combined_health_mp4 = os.path.join(videos_dir, f'{base_name}_with_health.mp4')
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

    # Strain plot (robot + plate) with dotted thresholds from evaluators
    strain_mp4 = os.path.join(videos_dir, f'{base_name}_strain.mp4')
    should_track_robot = track_robot_health or (target_ref is None)
    
    if should_track_robot and len(robot_strains) > 0:
        # Robot tracking enabled
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T_strain = max(len(robot_strains), len(target_strains) if target_ref is not None else 0)
        fig_s, ax_s = plt.subplots(figsize=(9.6, 5.4))
        line_sr, = ax_s.plot([], [], lw=4, color='tab:blue', label='Robot')
        line_st = None
        if target_ref is not None and len(target_strains) > 0:
            line_st, = ax_s.plot([], [], lw=4, color='tab:orange', label='Plate')
        ax_s.set_xlim(0, max(1, T_strain) / fps)
        # thresholds: get robot threshold from arm/gripper link thresholds, and plate threshold
        r_thr = None
        try:
            r_dmg = robot.damage_evaluators[0] if hasattr(robot, "damage_evaluators") and len(getattr(robot, "damage_evaluators", [])) > 0 else None
            if r_dmg is not None:
                link_thresholds = getattr(r_dmg, 'link_thresholds', {})
                # Use arm or gripper link threshold (they are the same)
                if "arm" in link_thresholds:
                    r_thr = float(link_thresholds["arm"].get("damage_threshold", 0.0))
                elif "gripper" in link_thresholds:
                    r_thr = float(link_thresholds["gripper"].get("damage_threshold", 0.0))
                else:
                    # Fallback to main threshold if no arm/gripper thresholds
                    r_thr = float(getattr(r_dmg, 'damage_threshold', 0.0))
        except Exception:
            r_thr = None
        
        t_thr = None
        try:
            t_thr = float(getattr(target_ref.damage_evaluators[0], 'damage_threshold', 0.0)) if target_ref is not None else None
        except Exception:
            t_thr = None

        # y-limits must include thresholds so dotted lines are visible
        extra = []
        if r_thr is not None:
            extra.append(r_thr)
        if t_thr is not None:
            extra.append(t_thr)
        all_vals = robot_strains + (target_strains if target_ref is not None else []) + extra
        ymin, ymax = 0.0, max(all_vals + [0.0])
        if ymin == ymax:
            ymin, ymax = (0.0, 1.0)
        ax_s.set_ylim(ymin, ymax * 1.1)

        if r_thr is not None:
            ax_s.axhline(y=r_thr, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Robot Threshold ({r_thr})')
        if t_thr is not None:
            ax_s.axhline(y=t_thr, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Plate Threshold ({t_thr})')
        ax_s.set_xlabel('Time (s)', fontsize=16)
        ax_s.set_ylabel('Strain', fontsize=16)
        ax_s.set_title('Strain Over Time', fontsize=20)
        ax_s.legend(loc='best')
        plt.tight_layout()

        def init_strain():
            line_sr.set_data([], [])
            if line_st is not None:
                line_st.set_data([], [])
            return (line_sr, line_st) if line_st is not None else (line_sr,)

        def animate_strain(i):
            x = [k / fps for k in range(1, i + 2)]
            y_r = robot_strains[: i + 1]
            line_sr.set_data(x, y_r)
            if line_st is not None:
                y_t = target_strains[: i + 1]
                line_st.set_data(x, y_t)
                return line_sr, line_st
            return line_sr,

        ani_s = animation.FuncAnimation(fig_s, animate_strain, init_func=init_strain, frames=T_strain, interval=1000 / fps, blit=True)
        writer_s = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani_s.save(strain_mp4, writer=writer_s)
        plt.close(fig_s)
    elif not should_track_robot and target_ref is not None and len(target_strains) > 0:
        # Target only (when track_robot_health is False)
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T_strain = len(target_strains)
        fig_s, ax_s = plt.subplots(figsize=(9.6, 5.4))
        line_st, = ax_s.plot([], [], lw=4, color='tab:orange', label='Plate')
        ax_s.set_xlim(0, max(1, T_strain) / fps)

        # thresholds
        t_thr = None
        try:
            t_thr = float(getattr(target_ref.damage_evaluators[0], 'damage_threshold', 0.0))
        except Exception:
            t_thr = None

        # y-limits must include thresholds
        extra = []
        if t_thr is not None:
            extra.append(t_thr)
        all_vals = target_strains + extra
        ymin, ymax = 0.0, max(all_vals + [0.0])
        if ymin == ymax:
            ymin, ymax = (0.0, 1.0)
        ax_s.set_ylim(ymin, ymax * 1.1)

        if t_thr is not None:
            ax_s.axhline(y=t_thr, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Plate Threshold ({t_thr})')
        ax_s.set_xlabel('Time (s)', fontsize=16)
        ax_s.set_ylabel('Strain', fontsize=16)
        ax_s.set_title('Strain Over Time', fontsize=20)
        ax_s.legend(loc='best')
        plt.tight_layout()

        def init_strain():
            line_st.set_data([], [])
            return line_st,

        def animate_strain(i):
            x = [k / fps for k in range(1, i + 2)]
            y_t = target_strains[: i + 1]
            line_st.set_data(x, y_t)
            return line_st,

        ani_s = animation.FuncAnimation(fig_s, animate_strain, init_func=init_strain, frames=T_strain, interval=1000 / fps, blit=True)
        writer_s = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani_s.save(strain_mp4, writer=writer_s)
        plt.close(fig_s)

    # Side-by-side cam_with_ego + strain
    combined_strain_mp4 = os.path.join(videos_dir, f'{base_name}_with_strain.mp4')
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(strain_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', cam_with_ego_mp4,
            '-i', strain_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_strain_mp4
        ], check=True)

    # Build reward plot video (cumulative episode reward over time)
    reward_mp4 = os.path.join(videos_dir, "rl_test_reward.mp4")
    if len(total_rewards) > 0:
        T = len(total_rewards)
        min_r = min(total_rewards)
        max_r = max(total_rewards)
        rng = max(1e-5, max_r - min_r)
        y_min = min_r - 0.1 * rng
        y_max = max_r + 0.1 * rng
        
        # Handle NaN and infinity values
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            y_min, y_max = -1000.0, 1000.0

        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax.plot([], [], lw=6, color='tab:green', label='Cumulative Reward')
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Cumulative Reward', fontsize=20)
        ax.set_title('Total Episode Reward Over Time', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_reward():
            line_r.set_data([], [])
            return line_r,

        def animate_reward(i):
            x = [k / fps for k in range(1, i + 2)]
            y = total_rewards[: i + 1]
            line_r.set_data(x, y)
            return line_r,

        ani = animation.FuncAnimation(
            fig, animate_reward, init_func=init_reward, frames=T, interval=1000 / fps, blit=True
        )
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(reward_mp4, writer=writer)
        plt.close(fig)

    # Build damage and distance components plot video
    components_mp4 = os.path.join(videos_dir, "rl_test_components.mp4")
    if len(total_damage_rewards) > 0 and len(total_distance_rewards) > 0:
        T = len(total_damage_rewards)
        all_values = total_damage_rewards + total_distance_rewards
        min_val = min(all_values)
        max_val = max(all_values)
        rng = max(1e-5, max_val - min_val)
        y_min = min_val - 0.1 * rng
        y_max = max_val + 0.1 * rng
        
        # Handle NaN and infinity values
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            y_min, y_max = -1000.0, 1000.0

        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_damage, = ax.plot([], [], lw=6, color='tab:red', label='Cumulative Damage Reward')
        line_distance, = ax.plot([], [], lw=6, color='tab:blue', label='Cumulative Distance Reward')
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Cumulative Reward', fontsize=20)
        ax.set_title('Reward Components Over Time', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_components():
            line_damage.set_data([], [])
            line_distance.set_data([], [])
            return line_damage, line_distance

        def animate_components(i):
            x = [k / fps for k in range(1, i + 2)]
            y_damage = total_damage_rewards[: i + 1]
            y_distance = total_distance_rewards[: i + 1]
            line_damage.set_data(x, y_damage)
            line_distance.set_data(x, y_distance)
            return line_damage, line_distance

        ani = animation.FuncAnimation(
            fig, animate_components, init_func=init_components, frames=T, interval=1000 / fps, blit=True
        )
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(components_mp4, writer=writer)
        plt.close(fig)

    # Create side-by-side combined videos
    combined_reward_mp4 = os.path.join(videos_dir, "rl_test_with_reward.mp4")
    combined_components_mp4 = os.path.join(videos_dir, "rl_test_with_components.mp4")
    
    # Combined video 1: camera + total reward
    if os.path.exists(mp4_path) and os.path.exists(reward_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', mp4_path,
            '-i', reward_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_reward_mp4
        ], check=True)
    
    # Combined video 2: camera + reward components
    if os.path.exists(mp4_path) and os.path.exists(components_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', mp4_path,
            '-i', components_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_components_mp4
        ], check=True)

    # Clean up intermediate mech videos (keep RL videos and the new combined outputs)
    videos_to_cleanup = []
    # Keep mp4_path (camera), reward_mp4, components_mp4, combined_* as outputs
    for p in [
        mp4_ego if 'mp4_ego' in locals() else None,
        cam_with_ego_mp4 if os.path.exists(cam_with_ego_mp4) else None,
        health_mp4 if os.path.exists(health_mp4) else None,
        strain_mp4 if os.path.exists(strain_mp4) else None,
    ]:
        # We'll delete only the source ego mp4 and intermediate overlay sources, not the final combined
        pass
    
    # Remove intermediate ego and standalone plots after combining
    for video_path in [
        mp4_ego if 'mp4_ego' in locals() and os.path.exists(mp4_ego) else None,
        health_mp4 if os.path.exists(health_mp4) else None,
        strain_mp4 if os.path.exists(strain_mp4) else None,
    ]:
        if video_path is not None:
            try:
                os.remove(video_path)
            except OSError:
                pass

    print(f"Videos saved to: {videos_dir}")
    if os.path.exists(combined_reward_mp4):
        print(f"  - Total reward video: {os.path.basename(combined_reward_mp4)}")
    if os.path.exists(combined_components_mp4):
        print(f"  - Components video: {os.path.basename(combined_components_mp4)}")

    # Clean shutdown
    og.shutdown()


if __name__ == "__main__":
    main()
