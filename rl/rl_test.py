import os
import sys
import matplotlib
import torch as th
# Use non-interactive backend to avoid Qt/xcb plugin requirement in headless runs
matplotlib.use('Agg')

from safety_benchmark.damageable_env import DamageableEnvironment

def damage_reward_fn(env, obs):
    plate_health_states = obs["object_health_states"]["glass_plate"]
    total_damage = 0.0
    for damage_type, damage_info in plate_health_states["damage_info"].items():
        for link_name, damage in damage_info.items():
            total_damage += damage
    return -total_damage
    # if plate_health_states["health"] == 0.0:
    #     # return float('-inf')
    #     return -1000.0
    # else:
    #     damage = 0.0
    #     for link_name, damage_info in plate_health_states["damage_info"].items():
    #         for link_name, damage in damage_info.items():
    #             damage += damage
    #     return -damage

def distance_reward_fn(env,obs):
    plate_obj = env.scene.object_registry("name", "glass_plate")
    plate_pos, plate_orn = plate_obj.get_position_orientation()
    target_pos = th.tensor([-0.0966,  0.0843,  0.4593])
    distance = th.norm(plate_pos - target_pos).item()
    return -distance

def reward_fn(env, obs):
    return damage_reward_fn(env, obs) + distance_reward_fn(env, obs)

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
    total_rewards = []
    total_damage_rewards = []
    total_distance_rewards = []
    fps = 15
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

    # Run simulation loop
    steps = 300
    for step in range(steps):
        action = teleop.get_teleop_action()
        obs, reward, terminated, truncated, info = env.step(action=action)

        plate_pos, plate_orn = plate_obj.get_position_orientation()

        # Track cumulative (total) episode reward and individual components
        total_reward += float(reward)
        total_rewards.append(total_reward)
        
        # Track individual reward components
        damage_reward = damage_reward_fn(env, obs)
        distance_reward = distance_reward_fn(env, obs)
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

        # No health tracking; replaced by cumulative reward tracking

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

    # Clean up intermediate videos, keep only the final combined videos
    videos_to_cleanup = [mp4_path, reward_mp4, components_mp4]
    
    for video_path in videos_to_cleanup:
        if os.path.exists(video_path):
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
