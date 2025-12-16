import os
import sys
import cv2
import numpy as np
import subprocess
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import torch as th
from omnigibson import object_states
from omnigibson.object_states import OnFire

from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.damage_evaluators.thermal_damage_evaluator import ThermalDamageEvaluator

# Force headless plotting
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main(steps: int = 500) -> None:
    # Enable GPU dynamics + object states (required for thermal simulation)
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True
    gm.ENABLE_HQ_RENDERING = False

    scene_path = os.path.join(os.path.dirname(__file__), "temp.json")
    print(f"Loading scene from: {scene_path}")

    # Fresh simulator
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

    # Camera teleoperation
    class CustomCameraMover(CameraMover):
        @property
        def input_to_command(self):
            return {
                # Strafe left/right
                lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
                # Forward / backward
                lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
                lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
                # Vertical move (use G to avoid conflicts)
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }

    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()
    # Set starting viewer camera pose
    start_cam_pos = th.tensor([-0.5867, 1.0141, 0.8647])
    start_cam_quat = th.tensor([-0.2521, 0.5779, 0.7115, -0.3104])
    og.sim.viewer_camera.set_position_orientation(position=start_cam_pos, orientation=start_cam_quat)

    # Grab robot
    assert len(env.robots) > 0, "No robots found after loading scene."
    robot = env.robots[0]

    # Check if robot has Temperature state enabled (required for thermal damage)
    # If not, add it along with its dependencies
    if object_states.Temperature in robot.states:
        initial_temp = robot.states[object_states.Temperature].get_value()
        print(f"✅ Robot has Temperature state enabled. Initial temperature: {initial_temp:.2f}°C")
    else:
        print("⚠️  Robot does NOT have Temperature state enabled. Adding it now...")
        
        # Check if robot is a StatefulObject (it should be)
        from omnigibson.objects.stateful_object import StatefulObject
        if not isinstance(robot, StatefulObject):
            print("❌ ERROR: Robot is not a StatefulObject, cannot add Temperature state!")
        else:
            # Temperature requires AABB as a dependency - check and add if needed
            if object_states.AABB not in robot.states:
                print("   Adding AABB state (required dependency)...")
                compatible, reason = object_states.AABB.is_compatible(obj=robot)
                if compatible:
                    aabb_state = object_states.AABB(obj=robot)
                    robot.add_state(aabb_state)
                    aabb_state.initialize()
                    print("   ✅ AABB state added and initialized")
                else:
                    print(f"   ❌ Cannot add AABB state: {reason}")
            
            # Now add Temperature state
            compatible, reason = object_states.Temperature.is_compatible(obj=robot)
            if compatible:
                temp_state = object_states.Temperature(obj=robot)
                robot.add_state(temp_state)
                temp_state.initialize()
                initial_temp = temp_state.get_value()
                print(f"   ✅ Temperature state added and initialized. Initial temperature: {initial_temp:.2f}°C")
            else:
                print(f"   ❌ Cannot add Temperature state: {reason}")
                print(f"   Available robot states: {[s.__name__ for s in robot.states.keys()]}")

    # Force the initial robot temperature to 20°C for this script
    if object_states.Temperature in robot.states:
        try:
            robot.states[object_states.Temperature].set_value(20.0)
            initial_temp = robot.states[object_states.Temperature].get_value()
            print(f"🌡️  Set robot initial temperature to {initial_temp:.2f}°C (target: 20°C)")
        except Exception as e:
            print(f"⚠️  Failed to set robot initial temperature to 20°C: {e}")

    # Record thermal thresholds from the robot's thermal damage evaluator (if present)
    heating_threshold = None
    cooling_threshold = None
    if hasattr(robot, "damage_evaluators"):
        for ev in robot.damage_evaluators:
            if isinstance(ev, ThermalDamageEvaluator):
                heating_threshold = float(getattr(ev, "heating_threshold", None))
                cooling_threshold = float(getattr(ev, "cooling_threshold", None))
                break

    # Filter collisions between robot and fireplace's fillable meta link only
    # This allows the robot arm to reach inside the fireplace interior to place logs
    # while keeping collisions with the base and other structural parts
    fireplace = env.scene.object_registry("name", "fireplace")
    if fireplace is not None:
        # Find the fillable meta link (interior of fireplace)
        target_link_name = "meta__base_link_fillable_0_0_link"
        if target_link_name in fireplace.links:
            target_link = fireplace.links[target_link_name]
            for robot_link_name, robot_link in robot.links.items():
                robot_link.add_filtered_collision_pair(target_link)
                print(f"✅ Filtered collisions between robot link '{robot_link_name}' and fireplace fillable link '{target_link_name}'")
        else:
            print(f"⚠️  Fireplace fillable link '{target_link_name}' not found. Available links: {list(fireplace.links.keys())}")

    # Configure robot controllers with higher gripper force (match simple_task.py)
    controller_config = {
        # Set arms to IK so teleop supports EEF translation and rotation
        "arm_left": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        "arm_right": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        # Stronger gripper position control (higher gains for easier grasping)
        "gripper_left": {
            "motor_type": "position",
            "isaac_kp": 8000.0,
            "isaac_kd": 4000.0,
            "inverted": True,
        },
        "gripper_right": {
            "motor_type": "position",
            "isaac_kp": 8000.0,
            "isaac_kd": 4000.0,
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

    # Let physics settle
    print("Stepping simulation for 20 steps to let physics settle...")
    for _ in range(20):
        action = th.zeros(robot.action_dim)
        env.step(action=action)

    # Teleoperation
    controller = KeyboardRobotController(robot=robot)
    
    # Register TAB key to save sim state and breakpoint
    def save_and_breakpoint():
        print("=== TAB callback triggered ===", flush=True)
        save_path = os.path.join(os.path.dirname(__file__), "temp.json")
        og.sim.save([save_path])
        print(f"✅ Saved simulation state to: {save_path}")
        # Print current camera position and orientation
        cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
        print(f"📷 Current camera position: {cam_pos}")
        print(f"📷 Current camera orientation (quaternion): {cam_quat}")

        target_log_pos, target_log_quat = target_log.get_position_orientation()
        print(f"🪵 Target log position: {target_log_pos}")
        print(f"🪵 Target log orientation (quaternion): {target_log_quat}")
        breakpoint()
    
    controller.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Save simulation state to JSON and breakpoint",
        callback_fn=save_and_breakpoint,
    )
    
    controller.print_keyboard_teleop_info()

    # Get target log object for OnFire state tracking
    target_log = env.scene.object_registry("name", "target_object")
    if target_log is not None:
        print(f"✅ Found target log object: {target_log.name}")
        if OnFire in target_log.states:
            initial_fire_state = target_log.states[OnFire].get_value()
            print(f"   Initial OnFire state: {initial_fire_state}")
        else:
            print("   ⚠️  Target log does not have OnFire state")
    else:
        print("⚠️  Could not find target_object in scene")

    # Data collection for video generation
    frames = []
    frames_rgb = []  # Store RGB frames for PNG export
    frames_ego = []
    ego_key = None
    robot_healths = []
    robot_temperatures = []
    # Track discrete damage status (none / negligible / minor / major / critical) per step
    robot_statuses = []
    log_fire_states = []
    fps = 30
    teleop_steps = steps

    print(f"Starting {teleop_steps} steps of teleoperation with video recording...")
    for step in range(teleop_steps):
        action = controller.get_teleop_action()
        env.step(action=action)

        # Capture viewer camera frame
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        rgb_np = cv2.resize(rgb_np, (1080, 720))  # HD resolution
        frames_rgb.append(rgb_np.copy())  # Store RGB for PNG export
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Track robot damage status each step for status-border coloring
        status_this_step = str(getattr(robot, "damage_status", "none")).lower()
        robot_statuses.append(status_this_step)

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
            # Dynamic colored border based on current damage status (robot is target)
            if status_this_step in ("major", "critical"):
                bgr = (0, 0, 255)  # Red border
            elif status_this_step in ("minor",):
                bgr = (0, 255, 255)  # Yellow border
            else:
                bgr = (0, 255, 0)  # Green border
            border = 30  # 30px at 1080x720 -> ~10px after scaling to 360x240
            ego_bgr = cv2.copyMakeBorder(ego_bgr, border, border, border, border, cv2.BORDER_CONSTANT, value=bgr)
            frames_ego.append(ego_bgr)

        # Track robot health
        robot_health = float(getattr(robot, "health", 100.0))
        robot_healths.append(robot_health)

        # Track robot temperature (check if robot has Temperature state)
        robot_temp = 20.0  # Default room temperature
        if object_states.Temperature in robot.states:
            try:
                temp_value = robot.states[object_states.Temperature].get_value()
                # Handle both tensor and scalar returns
                if hasattr(temp_value, 'item'):
                    robot_temp = float(temp_value.item())
                elif isinstance(temp_value, (int, float)):
                    robot_temp = float(temp_value)
                else:
                    # If it's a tensor with multiple elements, take the first
                    robot_temp = float(temp_value[0].item() if hasattr(temp_value, '__getitem__') else temp_value)
            except Exception as e:
                print(f"⚠️  Error reading temperature at step {step}: {e}")
                robot_temp = 20.0  # Fallback to default
        robot_temperatures.append(robot_temp)

        # Track target log OnFire state
        fire_state = 0  # Default to not on fire
        if target_log is not None and OnFire in target_log.states:
            try:
                fire_value = target_log.states[OnFire].get_value()
                fire_state = 1 if fire_value else 0
            except Exception as e:
                print(f"⚠️  Error reading OnFire state at step {step}: {e}")
                fire_state = 0
        log_fire_states.append(fire_state)

        # Progress indicator with temperature change tracking
        if step % 50 == 0:
            temp_change = robot_temperatures[-1] - robot_temperatures[0] if len(robot_temperatures) > 1 else 0.0
            fire_str = "🔥" if fire_state == 1 else "❄️"
            print(f"Step {step}/{teleop_steps} - Robot temp={robot_temp:.2f}°C (change: {temp_change:+.2f}°C), Health={robot_health:.1f}%, Log Fire: {fire_str}")
            if object_states.Temperature in robot.states:
                # Also print raw value for debugging
                raw_temp = robot.states[object_states.Temperature].get_value()
                print(f"  Raw temperature value: {raw_temp} (type: {type(raw_temp)})")

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
        first_path = os.path.join(out_dir, "thermal_frame_first.png")
        cv2.imwrite(first_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved first frame: {first_path}")
        
        # Save middle frame
        mid_idx = len(frames_rgb) // 2
        mid_frame = prepare_frame_for_png(frames_rgb[mid_idx])
        mid_path = os.path.join(out_dir, "thermal_frame_middle.png")
        cv2.imwrite(mid_path, cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved middle frame: {mid_path}")
        
        # Save last frame
        last_frame = prepare_frame_for_png(frames_rgb[-1])
        last_path = os.path.join(out_dir, "thermal_frame_last.png")
        cv2.imwrite(last_path, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved last frame: {last_path}")

    # Generate videos
    if len(frames) > 0:
        generate_videos(
            frames,
            frames_ego,
            robot_healths,
            robot_temperatures,
            robot_statuses,
            log_fire_states,
            fps,
            heating_threshold,
            cooling_threshold,
        )

    # Clean up camera mover and simulator
    camera_mover.clear()
    # Only shutdown - og.clear() can cause registry errors during cleanup
    og.shutdown()


def generate_videos(
    frames,
    frames_ego,
    robot_healths,
    robot_temperatures,
    robot_statuses,
    log_fire_states,
    fps,
    heating_threshold,
    cooling_threshold,
):
    """Generate high quality video with health, temperature, fire state plots, and a status-border sim video."""
    out_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(out_dir, exist_ok=True)

    # Write main camera video
    sim_mp4 = os.path.join(out_dir, "thermal_camera_obs.mp4")
    if len(frames) > 0:
        h0, w0 = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(sim_mp4, fourcc, fps, (w0, h0))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()
        print(f"✅ Saved camera video: {sim_mp4}")

        # Also create a status-border sim video where the border color encodes temperature state,
        # and the current robot temperature is printed at the top-center of each frame.
        border_mp4 = os.path.join(out_dir, "thermal_camera_with_temp_border.mp4")
        border_width = 30
        vw_border = cv2.VideoWriter(
            border_mp4,
            fourcc,
            fps,
            (w0 + 2 * border_width, h0 + 2 * border_width),
        )

        for f, status, temp in zip(frames, robot_statuses, robot_temperatures):
            # Map health / damage status to border color (match rl_test.py)
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
            # Overlay current temperature text at top-center
            text = f"Robot Temp: {float(temp):.2f} C"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            # Place text inside the sim image, near the top-left (not in the colored border)
            x = border_width + 20
            y = border_width + text_h + 20
            cv2.putText(
                bordered,
                text,
                (x, y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
            vw_border.write(np.ascontiguousarray(bordered, dtype=np.uint8))
        vw_border.release()
        print(f"✅ Saved camera video with temperature status border: {border_mp4}")

    # Write ego camera video
    ego_mp4 = os.path.join(out_dir, "thermal_ego_obs.mp4")
    if len(frames_ego) > 0:
        he, we = frames_ego[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw_e = cv2.VideoWriter(ego_mp4, fourcc, fps, (we, he))
        for f in frames_ego:
            vw_e.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw_e.release()
        print(f"✅ Saved ego camera video: {ego_mp4}")

    # Generate health plot video
    health_mp4 = generate_health_plot(robot_healths, fps, out_dir)
    
    # Generate temperature plot video
    temp_mp4 = generate_temperature_plot(robot_temperatures, fps, out_dir)
    
    # Generate fire state plot video
    fire_mp4 = generate_fire_state_plot(log_fire_states, fps, out_dir)
    
    # Create camera with ego overlay
    cam_with_ego_mp4 = os.path.join(out_dir, "thermal_camera_with_ego.mp4")
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

    # Create final combined video (camera + ego + health plot)
    combined_health_mp4 = os.path.join(out_dir, "thermal_with_health.mp4")
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

    # Create final combined video (camera + ego + temperature plot)
    combined_temp_mp4 = os.path.join(out_dir, "thermal_with_temperature.mp4")
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(temp_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', cam_with_ego_mp4,
            '-i', temp_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_temp_mp4
        ], check=True)
        print(f"✅ Saved combined video (temperature): {combined_temp_mp4}")

    # Create final combined video (camera + ego + fire state plot)
    combined_fire_mp4 = os.path.join(out_dir, "thermal_with_fire_state.mp4")
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(fire_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', cam_with_ego_mp4,
            '-i', fire_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_fire_mp4
        ], check=True)
        print(f"✅ Saved combined video (fire state): {combined_fire_mp4}")

    # Clean up intermediate videos
    for video_path in [sim_mp4, ego_mp4, cam_with_ego_mp4, health_mp4, temp_mp4, fire_mp4]:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass


def generate_health_plot(robot_healths, fps, out_dir):
    """Generate animated plot of robot health over time."""
    health_mp4 = os.path.join(out_dir, 'robot_healths.mp4')
    
    # Prepare data series
    health_series = robot_healths if len(robot_healths) > 0 else [100.0]
    max_len = len(health_series)
    
    # Create figure with professional styling
    fig_h, ax_h = plt.subplots(figsize=(9.6, 5.4))
    
    line_robot, = ax_h.plot([], [], lw=6, color='tab:blue', label='Robot health')
    
    # Configure axes
    ax_h.set_xlim(0, max(1, max_len) / fps)
    ax_h.set_ylim(0.0, 100.0)
    ax_h.set_xlabel('Time (s)', fontsize=20)
    ax_h.set_ylabel('Health (%)', fontsize=20)
    ax_h.set_title('Robot Health Over Time', fontsize=26)
    
    ax_h.legend(loc='best', fontsize=16)
    ax_h.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    ax_h.grid(True, linewidth=1.0, alpha=0.3)
    plt.tight_layout()

    # Animation functions
    def init_h():
        line_robot.set_data([], [])
        return line_robot,

    def animate_h(i):
        x = [k / fps for k in range(1, i + 2)]
        y_robot = health_series[: i + 1]
        line_robot.set_data(x, y_robot)
        return line_robot,

    # Create and save animation
    ani_h = animation.FuncAnimation(fig_h, animate_h, init_func=init_h, frames=max_len, 
                                    interval=1000 / fps, blit=True)
    writer_h = animation.FFMpegWriter(fps=fps, codec='mpeg4', 
                                       extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani_h.save(health_mp4, writer=writer_h)
    plt.close(fig_h)
    
    print(f"✅ Saved health plot: {health_mp4}")
    return health_mp4


def generate_temperature_plot(robot_temperatures, fps, out_dir):
    """Generate animated plot of robot temperature over time."""
    temp_mp4 = os.path.join(out_dir, 'robot_temperatures.mp4')
    
    # Prepare data series
    temp_series = robot_temperatures if len(robot_temperatures) > 0 else [20.0]
    max_len = len(temp_series)
    
    # Create figure with professional styling
    fig_t, ax_t = plt.subplots(figsize=(9.6, 5.4))
    
    line_robot, = ax_t.plot([], [], lw=6, color='tab:orange', label='Robot temperature')
    
    # Configure axes
    all_temps = temp_series
    y_min = min(all_temps) if all_temps else 0
    y_max = max(all_temps) * 1.1 if all_temps else 100
    
    ax_t.set_xlim(0, max(1, max_len) / fps)
    ax_t.set_ylim(y_min, y_max)
    ax_t.set_xlabel('Time (s)', fontsize=20)
    ax_t.set_ylabel('Temperature (°C)', fontsize=20)
    ax_t.set_title('Robot Temperature Over Time', fontsize=26)
    
    # Add thermal damage threshold line (from robot thermal damage params: heating_threshold = 60.0)
    damage_threshold = 60.0
    ax_t.axhline(y=damage_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                 label=f'Damage Threshold ({damage_threshold}°C)')
    
    ax_t.legend(loc='best', fontsize=16)
    ax_t.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    ax_t.grid(True, linewidth=1.0, alpha=0.3)
    plt.tight_layout()

    # Animation functions
    def init_t():
        line_robot.set_data([], [])
        return line_robot,

    def animate_t(i):
        x = [k / fps for k in range(1, i + 2)]
        y_robot = temp_series[: i + 1]
        line_robot.set_data(x, y_robot)
        return line_robot,

    # Create and save animation
    ani_t = animation.FuncAnimation(fig_t, animate_t, init_func=init_t, frames=max_len, 
                                    interval=1000 / fps, blit=True)
    writer_t = animation.FFMpegWriter(fps=fps, codec='mpeg4', 
                                       extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani_t.save(temp_mp4, writer=writer_t)
    plt.close(fig_t)
    
    print(f"✅ Saved temperature plot: {temp_mp4}")
    return temp_mp4


def generate_fire_state_plot(log_fire_states, fps, out_dir):
    """Generate animated plot of target log OnFire state over time (0 or 1)."""
    fire_mp4 = os.path.join(out_dir, 'log_fire_state.mp4')
    
    # Prepare data series
    fire_series = log_fire_states if len(log_fire_states) > 0 else [0]
    max_len = len(fire_series)
    
    # Create figure with professional styling
    fig_f, ax_f = plt.subplots(figsize=(9.6, 5.4))
    
    line_fire, = ax_f.plot([], [], lw=6, color='tab:red', label='Log OnFire state', drawstyle='steps-post')
    
    # Configure axes
    ax_f.set_xlim(0, max(1, max_len) / fps)
    ax_f.set_ylim(-0.1, 1.1)
    ax_f.set_yticks([0, 1])
    ax_f.set_yticklabels(['Off (0)', 'On (1)'])
    ax_f.set_xlabel('Time (s)', fontsize=20)
    ax_f.set_ylabel('Fire State', fontsize=20)
    ax_f.set_title('Target Log OnFire State Over Time', fontsize=26)
    
    ax_f.legend(loc='best', fontsize=16)
    ax_f.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    ax_f.grid(True, linewidth=1.0, alpha=0.3)
    
    # Add color bands for visual clarity
    ax_f.axhspan(0.5, 1.1, alpha=0.15, color='red', label='_nolegend_')
    ax_f.axhspan(-0.1, 0.5, alpha=0.15, color='blue', label='_nolegend_')
    
    plt.tight_layout()

    # Animation functions
    def init_f():
        line_fire.set_data([], [])
        return line_fire,

    def animate_f(i):
        x = [k / fps for k in range(1, i + 2)]
        y_fire = fire_series[: i + 1]
        line_fire.set_data(x, y_fire)
        return line_fire,

    # Create and save animation
    ani_f = animation.FuncAnimation(fig_f, animate_f, init_func=init_f, frames=max_len, 
                                    interval=1000 / fps, blit=True)
    writer_f = animation.FFMpegWriter(fps=fps, codec='mpeg4', 
                                       extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani_f.save(fire_mp4, writer=writer_f)
    plt.close(fig_f)
    
    print(f"✅ Saved fire state plot: {fire_mp4}")
    return fire_mp4


if __name__ == "__main__":
    main()

