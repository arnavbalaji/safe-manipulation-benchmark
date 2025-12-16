import os
import subprocess
import math

import cv2
import numpy as np
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson import object_states
from omnigibson.object_states import OnTop, OnFire

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover


def main(steps: int = 500) -> None:
    # Enable GPU dynamics + object states (same as place_pan_on_stove.py)
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True
    gm.ENABLE_HQ_RENDERING = False

    scene_path = os.path.join(os.path.dirname(__file__), "place_pan_on_stove_saved.json")
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

    # Re-apply object placements so the pan sits properly on an inactive burner
    stove = env.scene.object_registry("name", "stove")
    pan = env.scene.object_registry("name", "pan")
    target_obj = env.scene.object_registry("name", "target_object")

    # Place pan on the stove, on an inactive burner (not the one that's on)
    # Use exact position and orientation from YAML for consistency
    if pan is not None and stove is not None:
        # Exact position from YAML: [0.3, -1.0, 0.8]
        pan_position = [0.3, -1.0, 0.8]
        # Exact orientation from YAML: [0, 0, 0.7071068, 0.7071068]
        pan_orientation = [0, 0, 0.7071068, 0.7071068]
        pan.set_position(pan_position)
        pan.set_orientation(pan_orientation)
        # Set OnTop state to ensure logical relationship
        if OnTop in pan.states:
            pan.states[OnTop].set_value(stove, True)
        pan.keep_still()
        # Step physics to let pan settle on stove
        for _ in range(10):
            og.sim.step()
    
    # Place spatula (target_object) on top of the pan
    # Get pan position after it has settled to ensure proper placement
    if target_obj is not None and pan is not None:
        # Get pan position after physics settling
        pan_pos, pan_orn = pan.get_position_orientation()
        if isinstance(pan_pos, th.Tensor):
            pan_pos = pan_pos.tolist()
        # Place spatula on top of pan with a small offset above the pan surface
        target_z_offset = 0.05  # Small offset above pan surface
        target_position = [pan_pos[0], pan_pos[1], pan_pos[2] + target_z_offset]
        target_obj.set_position(target_position)
        # Set OnTop state to ensure logical relationship
        if OnTop in target_obj.states:
            target_obj.states[OnTop].set_value(pan, True)
        target_obj.keep_still()
        # Step physics to let target settle on pan
        for _ in range(10):
            og.sim.step()

        # Ensure the spatula's OnFire state is enabled (match thermal/simple_task.py behavior)
        if OnFire in target_obj.states:
            target_obj.states[OnFire].set_value(True)

    # Turn on the stove and close the oven door, mirroring place_pan_on_stove.py
    if stove is not None:
        if object_states.ToggledOn in stove.states:
            stove.states[object_states.ToggledOn].set_value(True)
        if object_states.Open in stove.states:
            stove.states[object_states.Open].set_value(False)
        else:
            # Fallback: explicitly set any door / oven joints to zero
            for jname, joint in stove.joints.items():
                if "door" in jname.lower() or "oven" in jname.lower():
                    joint.set_joint_position(0.0)

    # Initialize target temperature to room temp (20°C) if the state exists
    if target_obj is not None and (object_states.Temperature in target_obj.states):
        target_obj.states[object_states.Temperature].set_value(20.0)

    # Let physics settle briefly after repositioning
    for _ in range(50):
        og.sim.step()

    # Grab robot
    assert len(env.robots) > 0, "No robots found after loading saved scene."
    robot = env.robots[0]

    # Configure robot controllers similarly to place_pan_on_stove.py
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

    # Raise the robot's trunk to its maximum limit before starting teleop
    if hasattr(robot, "trunk_control_idx") and hasattr(robot, "joint_upper_limits"):
        trunk_idx = robot.trunk_control_idx
        trunk_upper_limits = robot.joint_upper_limits[trunk_idx]
        # Set trunk joints to their maximum positions
        robot.set_joint_positions(positions=trunk_upper_limits / 1.5, indices=trunk_idx, drive=False)
        # Step simulation a few times to let the trunk settle
        for _ in range(10):
            og.sim.step()
        print("✅ Trunk set to highest position")

    # Camera teleoperation (same style as place_pan_on_stove.py)
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

    # Set viewer camera pose (from user-provided values)
    start_cam_pos = th.tensor([1.2066, -0.7955, 1.2996])
    start_cam_quat = th.tensor([0.4323, 0.4049, 0.5508, 0.5881])
    og.sim.viewer_camera.set_position_orientation(
        position=start_cam_pos, orientation=start_cam_quat
    )

    controller = KeyboardRobotController(robot=robot)

    # Register TAB key to save sim state and breakpoint (mirrors other teleop scripts)
    def save_and_breakpoint():
        print("=== TAB callback triggered ===", flush=True)
        save_path = os.path.join(os.path.dirname(__file__), "place_pan_on_stove_saved.json")
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

    # Let the scene settle a bit more with the robot controllers active
    for _ in range(50):
        og.sim.step_physics()

    # Simple teleop loop with video capture from the main viewer camera
    frames = []
    frames_rgb = []  # Store RGB frames for PNG export
    fps = 30
    for _ in range(steps):
        action = controller.get_teleop_action()
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
        first_path = os.path.join(videos_dir, "place_pan_on_stove_frame_first.png")
        cv2.imwrite(first_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved first frame: {first_path}")
        
        # Save middle frame
        mid_idx = len(frames_rgb) // 2
        mid_frame = prepare_frame_for_png(frames_rgb[mid_idx])
        mid_path = os.path.join(videos_dir, "place_pan_on_stove_frame_middle.png")
        cv2.imwrite(mid_path, cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved middle frame: {mid_path}")
        
        # Save last frame
        last_frame = prepare_frame_for_png(frames_rgb[-1])
        last_path = os.path.join(videos_dir, "place_pan_on_stove_frame_last.png")
        cv2.imwrite(last_path, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved last frame: {last_path}")

    # Save high-quality sim-only video (match quality settings from rl_test.py)
    if len(frames) > 0:
        videos_dir = os.path.join(os.path.dirname(__file__), "videos")
        os.makedirs(videos_dir, exist_ok=True)

        avi_path = os.path.join(videos_dir, "place_pan_on_stove_sim.avi")
        mp4_path = os.path.join(videos_dir, "place_pan_on_stove_sim.mp4")

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
    main(steps=500)

