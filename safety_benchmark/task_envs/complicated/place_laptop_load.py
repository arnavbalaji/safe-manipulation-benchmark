import os
import subprocess
import math

import cv2
import numpy as np
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.object_states import Filled

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover


def main(steps: int = 500) -> None:
    # Enable GPU dynamics + object states (same as place_laptop.py / simple_task_load.py)
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True
    gm.ENABLE_HQ_RENDERING = False

    scene_path = os.path.join(os.path.dirname(__file__), "temp.json")
    print(f"Loading saved scene from: {scene_path}")

    # Fresh simulator (match thermal/simple_task_load.py pattern)
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

    # Create environment from saved scene file
    cfg = {"scene": {"type": "Scene", "scene_file": scene_path}}
    env = DamageableEnvironment(configs=cfg)
    env.reset()

    # Try to restore a nicer background / lighting similar to fresh Rs_int scenes.
    # Two steps:
    #   1) Ensure a skybox exists (og.sim.add_skybox)
    #   2) Brighten / warm it, like the B1K QA viewer.
    try:
        # Ensure there is a skybox at all (some restored scenes may not add one)
        if getattr(og.sim, "skybox", None) is None:
            og.sim.add_skybox()
        dome_light = getattr(og.sim, "skybox", None)
        if dome_light is not None:
            # Bright, warm-toned dome light for better contrast (matches QA tools)
            dome_light.intensity = 0.5e4
            dome_light.color = (1.07, 0.85, 0.61)
    except Exception:
        # Lighting tweaks are best-effort and shouldn't break loading
        pass

    # Ensure the water glass is marked as filled and seeded with particles
    water_glass = None
    try:
        water_glass = env.scene.object_registry("name", "water_glass")
    except Exception:
        water_glass = None
    water_system = env.scene.get_system("water", force_init=True)
    if water_glass is not None and Filled in water_glass.states and water_system is not None:
        # Mark logical filled state
        water_glass.states[Filled].set_value(water_system, True)
        # Drop exactly 100 particles centered above the glass with a small vertical offset
        glass_pos, _ = water_glass.get_position_orientation()
        if isinstance(glass_pos, th.Tensor):
            glass_pos = glass_pos.tolist()
        z_offset = 0.05
        drop_pos = [glass_pos[0], glass_pos[1], glass_pos[2] + z_offset]
        for _ in range(100):
            water_system.generate_particles(positions=[drop_pos])
            for _ in range(2):
                og.sim.step()

    # Grab robot
    assert len(env.robots) > 0, "No robots found after loading saved scene."
    robot = env.robots[0]

    # Configure robot controllers similarly to place_laptop.py
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

    # Persist initial state after controller reload so subsequent saves keep these settings
    env.scene.update_initial_file()

    # Camera teleoperation (same style as place_laptop.py)
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

    # Use the same viewer camera pose as in place_laptop.py
    start_cam_pos = th.tensor(
        [0.5973, -0.7110, 1.3160]
    )
    start_cam_quat = th.tensor(
        [0.5281, -0.1252, -0.1938, 0.8173]
    )
    og.sim.viewer_camera.set_position_orientation(
        position=start_cam_pos, orientation=start_cam_quat
    )

    controller = KeyboardRobotController(robot=robot)

    # Register TAB key to save sim state and breakpoint (mirrors other teleop scripts)
    def save_and_breakpoint():
        print("=== TAB callback triggered ===", flush=True)
        save_path = os.path.join(os.path.dirname(__file__), "place_laptop_saved.json")
        og.sim.save([save_path])
        print(f"✅ Saved simulation state to: {save_path}")
        # Print current viewer camera pose
        cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
        print(f"📷 Current camera position: {cam_pos}")
        print(f"📷 Current camera orientation (quaternion): {cam_quat}")
        breakpoint()

    controller.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Save simulation state to JSON and breakpoint",
        callback_fn=save_and_breakpoint,
    )

    controller.print_keyboard_teleop_info()

    # Helper to hold laptop lid at target angle, mirroring electrical task behavior
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

    # Let the scene settle for a bit
    for _ in range(50):
        og.sim.step_physics()

    # Simple teleop loop with video capture from the main viewer camera
    frames = []
    frames_rgb = []  # Store RGB frames for PNG export
    fps = 30
    for _ in range(steps):
        action = controller.get_teleop_action()
        # Hold laptop pose each step (mirrors electrical task behavior)
        set_laptop_pose(env)
        env.step(action=action)
        # Capture viewer camera RGB
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        rgb_np = cv2.resize(rgb_np, (1080, 720))
        frames_rgb.append(rgb_np.copy())  # Store RGB for PNG export
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

    # Save high-quality PNG images from first, middle, and last frames
    if len(frames_rgb) > 0:
        videos_dir = os.path.join(os.path.dirname(__file__), "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
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
        first_path = os.path.join(videos_dir, "place_laptop_frame_first.png")
        cv2.imwrite(first_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved first frame: {first_path}")
        
        # Save middle frame
        mid_idx = len(frames_rgb) // 2
        mid_frame = prepare_frame_for_png(frames_rgb[mid_idx])
        mid_path = os.path.join(videos_dir, "place_laptop_frame_middle.png")
        cv2.imwrite(mid_path, cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved middle frame: {mid_path}")
        
        # Save last frame
        last_frame = prepare_frame_for_png(frames_rgb[-1])
        last_path = os.path.join(videos_dir, "place_laptop_frame_last.png")
        cv2.imwrite(last_path, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved last frame: {last_path}")

    # Save high-quality sim-only video (match quality settings from rl_test.py)
    if len(frames) > 0:
        videos_dir = os.path.join(os.path.dirname(__file__), "videos")
        os.makedirs(videos_dir, exist_ok=True)

        avi_path = os.path.join(videos_dir, "place_laptop_sim.avi")
        mp4_path = os.path.join(videos_dir, "place_laptop_sim.mp4")

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()

        # High-quality MPEG-4 encoding with low quantizer (same as rl_test.py)
        subprocess.run(
            ["ffmpeg", "-y", "-i", avi_path, "-c:v", "mpeg4", "-q:v", "2", mp4_path],
            check=True,
        )
        # Optional cleanup of intermediate AVI
        try:
            os.remove(avi_path)
        except OSError:
            pass

    # Clean up camera mover and simulator
    camera_mover.clear()
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main(steps=100)


