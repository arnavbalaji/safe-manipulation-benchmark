"""
Load a saved nav_to_table state and teleoperate.

You need an init state file to run this script. To create one:
  1. Run: python teleop_nav_to_table.py
  2. Wait for the scene to load (Tiago, pedestal_table, vase, etc.).
  3. Press S to save the state to safe-manipulation-benchmark/resources/saved_states/nav_to_table_init_state.pkl
  4. Quit, then run: python teleop_nav_to_table_load.py [--state_path <path>]

TAB: save 3840x2160 viewer camera PNG. WASD+G: move camera. R: reset.
"""
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import pickle
import yaml
from datetime import datetime
import torch as th
import numpy as np
import cv2

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

from telemoma.configs.base_config import teleop_config
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import omnigibson.lazy as lazy

from safety_benchmark.damageable_env import DamageableEnvironment, load_sim_state_with_size_fallback
from safety_benchmark.utils.misc_utils import save_rgb_camera_video
import json

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False

TASK_OBJECTS = {
    "pedestal_table": {
        "type": "DatasetObject",
        "name": "pedestal_table",
        "category": "pedestal_table",
        "model": "djflkd",
        "position": [-0.5, 0.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 1.0],
    },
    "vase": {
        "type": "DatasetObject",
        "name": "vase",
        "category": "vase",
        "model": "uuypot",
        "position": [-0.5, -1.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 0.5],
    },
    "swivel_chair": {
        "type": "DatasetObject",
        "name": "swivel_chair",
        "category": "swivel_chair",
        "model": "pkpcew",
        "position": [-0.5, 1.0, 0.50],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
}


def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", type=str, default="safe-manipulation-benchmark/resources/saved_states/nav_to_table_init_state.pkl", help="Path to saved state pkl")
    args = parser.parse_args()

    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["load_object_categories"] = ["floors", "walls", "breakfast_table"]
    cfg["robots"][0]["name"] = "tiago0"
    cfg["robots"][0]["position"] = [0.0, 0.0, 0.0]
    cfg["robots"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]
    cfg["robots"][0]["default_arm_pose"] = "horizontal"
    cfg["robots"][0]["grasping_mode"] = "assisted"
    cfg["robots"][0]["obs_modalities"] = ["rgb", "depth"]
    cfg["robots"][0]["action_normalize"] = False
    cfg["robots"][0]["controller_config"] = {
        "arm_left": {"name": "InverseKinematicsController", "command_input_limits": None},
        "gripper_left": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0), "mode": "smooth"},
        "arm_right": {"name": "InverseKinematicsController", "command_input_limits": None},
        "gripper_right": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0), "mode": "smooth"},
    }
    cfg["robots"][0]["exclude_sensor_names"] = ["left_eef_link", "right_eef_link"]

    robot_name = cfg["robots"][0].get("name", "tiago0")
    robot_type = cfg["robots"][0].get("type", "Tiago").lower()
    EXTERNAL_CAMERA_CONFIGS = {
        "external_sensor_0": {"position": [0.4859, -1.8219, 1.1402], "orientation": [0.5857, -0.0093, -0.0129, 0.8103], "horizontal_aperture": 10.0, "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0"},
        "external_sensor_1": {"position": [0.2522, 0.0470, 1.0696], "orientation": [0.1991, -0.1991, -0.6785, 0.6785], "horizontal_aperture": 30.0, "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1"},
        "external_sensor_2": {"position": [-0.7765, -0.8203, 0.9939], "orientation": [0.4566, -0.3285, -0.4831, 0.6710], "horizontal_aperture": 30.0, "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor2"},
        "external_sensor_3": {"position": [1.7508, -0.0198, 1.1778], "orientation": [0.3821, 0.4173, 0.6080, 0.5570], "horizontal_aperture": 20.0, "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor3"},
    }
    external_sensors_config = []
    for name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
        i = name.split("_")[-1]
        external_sensors_config.append({
            "sensor_type": "VisionSensor",
            "name": f"external_sensor{i}",
            "relative_prim_path": camera_cfg["relative_prim_path"],
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {"image_height": 720, "image_width": 720, "horizontal_aperture": camera_cfg["horizontal_aperture"]},
            "position": th.tensor(camera_cfg["position"], dtype=th.float32),
            "orientation": th.tensor(camera_cfg["orientation"], dtype=th.float32),
            "pose_frame": "world",
        })
    cfg["env"]["external_sensors"] = external_sensors_config
    cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

    env = DamageableEnvironment(configs=cfg)
    env.reset()
    robot = env.robots[0]
    for _ in range(10):
        og.sim.step()

    state_path = args.state_path
    if not os.path.exists(state_path):
        print(f"Error: State file not found at {state_path}")
        print("Run teleop_nav_to_table.py first, then press S to save state to nav_to_table_init_state.pkl.")
        return
    with open(state_path, "rb") as f:
        state_flat_array = pickle.load(f)
    load_sim_state_with_size_fallback(state_flat_array)
    print(f"Loaded state from {state_path}")
    
    # Re-run damageable object init so vase, pedestal_table, swivel_chair get track_damage (config in damageable_objects.yaml)
    env.initialize_damageable_objects()
    env.set_damageable_object_params()
    
    # Disable robot (tiago) damage tracking for nav_to_table task
    robot = env.robots[0]  # Re-get robot reference after state load
    if hasattr(robot, "set_track_damage"):
        robot.set_track_damage(False)
        print(f"Disabled damage tracking for robot '{robot.name}' (tiago health tracking disabled for nav_to_table)")

    robot_pos, robot_orn = robot.get_position_orientation()
    robot_joint_positions = robot.get_joint_positions()
    robot.keep_still()
    for _ in range(20):
        robot.keep_still()
        og.sim.step()
    robot.set_position_orientation(robot_pos, robot_orn)
    robot.set_joint_positions(robot_joint_positions)
    robot.set_joint_velocities(th.zeros(robot.n_dof))
    for ctrl_name in ["arm_left", "arm_right", "gripper_left", "gripper_right"]:
        ctrl = robot.controllers.get(ctrl_name)
        if ctrl is not None:
            ctrl.reset()
    for _ in range(10):
        og.sim.step()

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.5345655679702759, -2.3398592472076416, 1.3116816282272339]),
        orientation=th.tensor([0.605172872543335, 0.14635765552520752, 0.18393288552761078, 0.7606010437011719]),
    )
    
    # Enable seg_instance modality for viewer camera to enable health-based coloring
    try:
        viewer_camera_modalities = og.sim.viewer_camera.modalities
        if "seg_instance" not in viewer_camera_modalities:
            og.sim.viewer_camera.modalities = list(viewer_camera_modalities) + ["seg_instance"]
            print("Enabled seg_instance modality for viewer camera")
    except Exception as e:
        print(f"Warning: Could not enable seg_instance for viewer camera: {e}")

    class CustomCameraMover(CameraMover):
        @property
        def input_to_command(self):
            return {
                lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
                lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }

    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()

    arm_teleop_method = "keyboard"
    base_teleop_method = "keyboard"
    teleop_config.arm_left_controller = arm_teleop_method
    teleop_config.arm_right_controller = arm_teleop_method
    teleop_config.base_controller = base_teleop_method
    teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
    teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.02}
    teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=False)
    teleop_sys.start()

    action_generator = KeyboardRobotController(robot=robot)

    TASK_NAME = "nav_to_table"
    SCREENSHOT_DIR = "safe-manipulation-benchmark/resources/teleop_data/nav_to_table_screenshots"
    VIDEO_DIR = "safe-manipulation-benchmark/resources/teleop_data/nav_to_table_videos"
    RGB_SIZE = (3840, 2160)  # high-quality single frame
    VIDEO_SIZE = (1920, 1080)  # High resolution for video

    # Video recording state - track health for task objects only (robot tracking disabled)
    recording_video = False
    video_frames = []
    video_health_history = {"pedestal_table": [], "vase": [], "swivel_chair": []}
    HEALTH_THRESHOLD = 100.0  # Objects colored red when health < this value

    def start_recording():
        """Start recording video with health-based coloring"""
        nonlocal recording_video, video_frames, video_health_history
        recording_video = True
        video_frames = []
        video_health_history = {"pedestal_table": [], "vase": [], "swivel_chair": []}
        print("=" * 60)
        print("VIDEO RECORDING STARTED")
        print("Recording high-resolution video with health-based coloring")
        print("Press TAB again to stop recording and save video")
        print("=" * 60)

    def stop_recording_and_save():
        """Stop recording and save video with health-based coloring"""
        nonlocal recording_video, video_frames, video_health_history

        if not recording_video or len(video_frames) == 0:
            print("No video recording in progress or no frames captured")
            return

        recording_video = False
        print("=" * 60)
        print("STOPPING VIDEO RECORDING")
        print(f"Captured {len(video_frames)} frames")
        print("Saving videos...")
        print("=" * 60)

        # Save videos
        os.makedirs(VIDEO_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main video with health-based coloring
        video_path = os.path.join(VIDEO_DIR, f"{TASK_NAME}_video_{timestamp}")
        video_array = np.array(video_frames)
        save_rgb_camera_video(video_path, video_array, fps=30)
        print(f"Main video saved: {video_path}.mp4")
        print(f"Full path: {os.path.abspath(video_path)}.mp4")

        print(f"Video resolution: {VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}")
        print(f"Total frames: {len(video_frames)}")
        print("=" * 60)

        # Clear recording state
        video_frames = []
        video_health_history = {"pedestal_table": [], "vase": [], "swivel_chair": []}

    def breakpoint_and_print_poses():
        nonlocal recording_video

        # Stop recording and save video when TAB is pressed
        if recording_video:
            stop_recording_and_save()
            # Also save viewer_camera RGB as PNG
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
            rgb_np = np.asarray(rgb.cpu().numpy()[:, :, :3], dtype=np.uint8)
            frame = cv2.resize(rgb_np, RGB_SIZE)
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = os.path.join(SCREENSHOT_DIR, f"{TASK_NAME}_{timestamp}.png")
            cv2.imwrite(png_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Viewer camera screenshot saved: {png_path} ({RGB_SIZE[0]}x{RGB_SIZE[1]})")
            print("\n" + "=" * 60)
            print("POSE INFORMATION (TAB pressed)")
            print("=" * 60)
            base_pos, base_orn = robot.get_position_orientation()
            print("Robot base Pose:")
            print(f"  Position: {base_pos.tolist()}")
            print(f"  Orientation: {base_orn.tolist()}")
            vase_obj = env.scene.object_registry("name", "vase")
            if vase_obj is not None:
                p, o = vase_obj.get_position_orientation()
                print("Vase Pose:")
                print(f"  Position: {p.tolist()}")
                print(f"  Orientation: {o.tolist()}")
            cam_pos, cam_orn = og.sim.viewer_camera.get_position_orientation()
            print("Viewer Camera Pose:")
            print(f"  Position: {cam_pos.tolist()}")
            print(f"  Orientation: {cam_orn.tolist()}")
            print("=" * 60 + "\n")
            breakpoint()

    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.R, description="Reset", callback_fn=lambda: env.reset())
    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.TAB, description="Stop video recording, save video, print poses, and breakpoint", callback_fn=breakpoint_and_print_poses)
    action_generator.print_keyboard_teleop_info()

    # Ensure robot damage tracking is disabled (tiago health tracking disabled for nav_to_table)
    if hasattr(robot, "set_track_damage"):
        robot.set_track_damage(False)

    # Enable health visualization for robot and objects
    print("=" * 60)
    print("ENABLING HEALTH VISUALIZATION")
    print("=" * 60)
    damageable_objects = env.get_damageable_objects()
    print(f"Found {len(damageable_objects)} damageable objects: {[obj.name for obj in damageable_objects]}")
    if len(damageable_objects) > 0:
        env.enable_health_visualization()
        print("Health bars enabled for damageable objects")
    else:
        print("Warning: No damageable objects found. Health visualization not enabled.")
        print("Robot track_damage:", hasattr(robot, "track_damage") and robot.track_damage)

    # Start recording automatically from the beginning of teleop
    start_recording()

    print("=" * 60)
    print("Nav to table load: Teleoperate, WASD+G camera, TAB stop recording/save video, ESC quit")
    print("Video recording started automatically")
    print("Live health bars enabled for robot and objects")
    print(f"Videos will be saved to: {VIDEO_DIR}")
    print("=" * 60)

    step_count = 0
    while True:
        ret = action_generator.get_teleop_action()
        if isinstance(ret, tuple) and len(ret) == 2:
            action, keypress_str = ret
        else:
            action = ret
            keypress_str = None
        camera_keys = {"W", "A", "S", "D", "G"}
        if keypress_str and keypress_str in camera_keys:
            env.step(th.zeros(robot.action_dim))
            continue
        
        # Step environment
        try:
            step_result = env.step(action)
            env.update_health_visualization()
        except AttributeError as e:
            if "'NoneType' object has no attribute" in str(e):
                step_result = env.step(action)
                env.update_health_visualization()
            else:
                raise
        
        # Record video frame if recording (after step to capture updated state)
        if recording_video:
            try:
                # Get viewer camera observations
                viewer_obs_result = og.sim.viewer_camera.get_obs()
                if isinstance(viewer_obs_result, tuple):
                    viewer_obs = viewer_obs_result[0]
                    viewer_info = viewer_obs_result[1] if len(viewer_obs_result) > 1 else {}
                else:
                    viewer_obs = viewer_obs_result
                    viewer_info = {}

                rgb = viewer_obs["rgb"]
                rgb_np = np.asarray(rgb.cpu().numpy()[:, :, :3], dtype=np.uint8)

                # Resize to video resolution
                frame = cv2.resize(rgb_np, VIDEO_SIZE)

                # Get segmentation instance if available
                seg_instance_np = None
                seg_instance_info = {}
                if "seg_instance" in viewer_obs:
                    seg_instance = viewer_obs["seg_instance"]
                    seg_instance_np = np.asarray(seg_instance.cpu().numpy(), dtype=np.int64)
                    if isinstance(viewer_info, dict) and "seg_instance" in viewer_info:
                        seg_instance_info = viewer_info["seg_instance"]

                # Get health values for tracked objects (robot tracking disabled)
                health_values = {}
                tracked_objects = ["pedestal_table", "vase", "swivel_chair"]

                # Object healths
                for obj_name in ["pedestal_table", "vase", "swivel_chair"]:
                    try:
                        obj = env.scene.object_registry("name", obj_name)
                        if obj is not None and hasattr(obj, "health"):
                            try:
                                health_values[obj_name] = float(obj.health)
                            except Exception:
                                health_values[obj_name] = 100.0
                        else:
                            health_values[obj_name] = 100.0
                    except Exception:
                        health_values[obj_name] = 100.0

                # Store health history
                for obj_name in tracked_objects:
                    video_health_history[obj_name].append(health_values.get(obj_name, 100.0))

                # Apply health-based coloring if segmentation is available
                if seg_instance_np is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Color objects red when health is low
                    for obj_name in tracked_objects:
                        if obj_name not in health_values:
                            continue

                        health = health_values[obj_name]
                        if health < HEALTH_THRESHOLD:
                            seg_key = None
                            if seg_instance_info:
                                for k, v in seg_instance_info.items():
                                    if isinstance(v, str):
                                        # Check if object name appears in the mapping value
                                        if obj_name.lower() in v.lower():
                                            try:
                                                seg_key = int(k)
                                                break
                                            except (ValueError, TypeError):
                                                continue

                            if seg_key is not None:
                                mask = seg_instance_np == seg_key

                                if np.any(mask):
                                    alpha = 1.0 - (health / HEALTH_THRESHOLD)
                                    alpha = max(0.0, min(1.0, alpha))

                                    overlay_color = np.array([0, 0, 255], dtype=np.uint8)  # BGR red
                                    frame_bgr[mask] = ((1 - alpha) * frame_bgr[mask] + alpha * overlay_color).astype(np.uint8)

                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Append frame to video frames
                video_frames.append(frame)

            except Exception as e:
                print(f"Warning: Error capturing video frame: {e}")
                import traceback
                traceback.print_exc()

        step_count += 1

    camera_mover.clear()
    og.shutdown()


if __name__ == "__main__":
    __main__()
