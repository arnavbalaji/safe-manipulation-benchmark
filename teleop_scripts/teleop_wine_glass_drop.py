from ast import Pass
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import json
import h5py
import pickle
from datetime import datetime
import torch as th
import numpy as np
import cv2

import omnigibson as og
from omnigibson import object_states
# from omnigibson.object_states import IsGrasped
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm

from telemoma.configs.base_config import teleop_config
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
import omnigibson.lazy as lazy

from omnigibson.controllers.controller_base import IsGraspingState

from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper
from safety_benchmark.utils.misc_utils import save_rgb_camera_video, save_rgb_health_video_with_overlay

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_TRANSITION_RULES = False


SHELF_INIT_POS = [6.00, 0.2, 1.3]
SHELF_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
SHELF_SCALE = [0.4, 0.6, 0.5]

# Top shelf surface z so objects stay OnTop when shelf scale changes (stand model top ~1.4 * scale_z above base)
TOP_SHELF_Z = SHELF_INIT_POS[2] + SHELF_SCALE[2] * 1.4

# Task objects: only stand, box_of_crackers, and wineglass on top shelf
TASK_OBJECTS = {
    "box_of_crackers": {
        "type": "DatasetObject",
        "name": "box_of_crackers",
        "category": "box_of_crackers",
        "model": "cmdigf",
        "position": [6.0, 0.2, TOP_SHELF_Z],
        "orientation": [0.0, 0.0, 0.70710678, 0.70710678],
    },
    "wineglass": {
        "type": "DatasetObject",
        "name": "wineglass",
        "category": "wineglass",
        "model": "adiwil",
        "position": [6.0, 0.35, TOP_SHELF_Z],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
    "stand": {
        "type": "DatasetObject",
        "name": "stand",
        "category": "stand",
        "model": "vyrick",
        "position": SHELF_INIT_POS,
        "orientation": SHELF_INIT_ORI,
        "scale": SHELF_SCALE,
        "fixed_base": True,
    },
}

def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_state', action='store_true', help='Load a state')
    args = parser.parse_args()
    if args.load_state:
        load_state_path = f"resources/teleop_data/wine_glass_drop.hdf5"
        load_f = h5py.File(load_state_path, "r")

    # TODO: Set this
    collect_hdf5_path = f"resources/teleop_data/wine_glass_drop.hdf5"
    if not os.path.exists(collect_hdf5_path):
        os.makedirs(os.path.dirname(collect_hdf5_path), exist_ok=True)

    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Overwrite any configs here
    cfg["scene"]["scene_model"] = "house_single_floor"
    cfg["scene"]["not_load_object_categories"] = ["ottoman"]
    cfg["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]

    ############### Franka robot ###############
    # TODO(junhong): if we have a better way (a franka-specific config file), we should use that
    # Completely replace robot config to avoid Tiago-specific settings carrying over
    cfg["robots"][0] = {
        "type": "FrankaPanda",
        "name": "franka0",
        "position": [6.7, 0.2, 1.0],  # 0.1 closer to shelf than 6.8
        "orientation": [0.0, 0.0, 1.0, 0.0],
        "grasping_mode": "assisted",
        "obs_modalities": ["rgb", "depth"],
        "action_normalize": False,
        "self_collisions": True,
        # Franka has single arm (arm_0, gripper_0) instead of left/right
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                # "name": "JointController",
                "command_input_limits": None,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            },
        },
    }
    ############### Franka robot ###############

    # Generate external sensors config automatically
    # Get robot name and type from config to construct correct prim path
    robot_name = cfg["robots"][0].get("name", "franka0")
    robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()

    # Set external cameras for videos
    EXTERNAL_CAMERA_CONFIGS = {
        # Side camera (fixed to base_link frame)
        "external_sensor_0": {
            "position": [0.4859, -1.8219,  1.1402],
            "orientation": [ 0.5857, -0.0093, -0.0129,  0.8103],
            "horizontal_aperture": 10.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
        },
        # Left Shoulder (fixed to base_link frame)
        "external_sensor_1": {
            # wrt base frame
            "position": [0.2522, 0.0470, 1.0696],
            "orientation": [ 0.1991, -0.1991, -0.6785,  0.6785],
            "horizontal_aperture": 30.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1",
        },
        # Back camera (fixed to base_link frame)
        "external_sensor_2": {
            "position": [-0.7765, -0.8203,  0.9939],
            "orientation": [ 0.4566, -0.3285, -0.4831,  0.6710],
            "horizontal_aperture": 30.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor2",
        },
        # Front camera (fixed to base_link frame)
        "external_sensor_3": {
            "position": [1.7508, -0.0198,  1.1778],
            "orientation": [0.3821, 0.4173, 0.6080, 0.5570],
            "horizontal_aperture": 20.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor3",
        }
    }
    external_sensors_config = []
    for name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
        i = name.split("_")[-1]
        position = camera_cfg["position"]
        orientation = camera_cfg["orientation"]
        external_sensors_config.append({
            "sensor_type": "VisionSensor",
            "name": f"external_sensor{i}",
            "relative_prim_path": camera_cfg["relative_prim_path"],
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {
                "image_height": 720,
                "image_width": 720,
                "horizontal_aperture": camera_cfg["horizontal_aperture"],
            },
            "position": th.tensor(position, dtype=th.float32),
            "orientation": th.tensor(orientation, dtype=th.float32),
            "pose_frame": "world",
        })
    cfg["env"]["external_sensors"] = external_sensors_config

    # Add objects here
    cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

    env = DamageableEnvironment(configs=cfg)
    env = DamageableDataCollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
        # obj_attr_keys=["scale", "visible"],
        enable_dump_filters=False,
    )

    robot = env.robots[0]
    # set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([7.0659, -0.7141, 2.0185]),  # z +0.1
        orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
    )
    for _ in range(10): og.sim.step()

    # Franka default joint positions (7 arm joints + 2 gripper joints)
    robot.set_joint_positions(th.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]))

    # load state
    if args.load_state:
        # state = th.tensor(load_f["data/demo_0/state"][700])
        # og.sim.load_state(state, serialized=True)
        breakpoint()
        for _ in range(50): og.sim.step()

    for _ in range(10):
        og.sim.step()

    # Enable seg_instance on viewer camera for health-based coloring
    try:
        viewer_camera_modalities = og.sim.viewer_camera.modalities
        if "seg_instance" not in viewer_camera_modalities:
            og.sim.viewer_camera.modalities = list(viewer_camera_modalities) + ["seg_instance"]
            print("Enabled seg_instance modality for viewer camera")
    except Exception as e:
        print(f"Warning: Could not enable seg_instance for viewer camera: {e}")

    TASK_NAME = "wine_glass_drop"
    STATE_SAVE_DIR = "safe-manipulation-benchmark/resources/saved_states"
    STATE_SAVE_PATH = os.path.join(STATE_SAVE_DIR, "wine_glass_drop_init_state.pkl")
    SCREENSHOT_DIR = "safe-manipulation-benchmark/resources/teleop_data/wine_glass_drop_screenshots"
    VIDEO_DIR = "safe-manipulation-benchmark/resources/teleop_data/wine_glass_drop_videos"
    RGB_SIZE = (3840, 2160)
    VIDEO_SIZE = (1920, 1080)
    TRACKED_OBJECTS = ["wineglass", "box_of_crackers", "franka0"]
    recording_video = False
    trajectory_saved = False  # set True on TAB after save_data(); exit loop so we don't step with closed file
    video_frames_raw = []
    video_frames_colored = []
    video_health_history = {obj: [] for obj in TRACKED_OBJECTS}
    HEALTH_THRESHOLD = 100.0

    def start_recording():
        nonlocal recording_video, video_frames_raw, video_frames_colored, video_health_history
        recording_video = True
        video_frames_raw = []
        video_frames_colored = []
        video_health_history = {obj: [] for obj in TRACKED_OBJECTS}
        print("=" * 60)
        print("VIDEO RECORDING STARTED")
        print("Recording high-resolution video (raw + colored + health bars on TAB)")
        print("Press TAB to stop recording and save three videos")
        print("=" * 60)

    def stop_recording_and_save():
        nonlocal recording_video, video_frames_raw, video_frames_colored, video_health_history
        if not recording_video or len(video_frames_raw) == 0:
            print("No video recording in progress or no frames captured")
            return
        recording_video = False
        print("=" * 60)
        print("STOPPING VIDEO RECORDING")
        print(f"Captured {len(video_frames_raw)} frames")
        print("Saving three videos...")
        print("=" * 60)
        os.makedirs(VIDEO_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_array = np.array(video_frames_raw)
        colored_array = np.array(video_frames_colored)
        # 1) High-quality sim video, no object coloring
        path_raw = os.path.join(VIDEO_DIR, f"{TASK_NAME}_raw_{timestamp}")
        save_rgb_camera_video(path_raw, raw_array, fps=30)
        print(f"1) Raw video (no coloring): {path_raw}.mp4")
        # 2) High-quality sim video with health-based coloring
        path_colored = os.path.join(VIDEO_DIR, f"{TASK_NAME}_colored_{timestamp}")
        save_rgb_camera_video(path_colored, colored_array, fps=30)
        print(f"2) Colored video: {path_colored}.mp4")
        # 3) High-quality sim video with coloring and health bars
        path_overlay = os.path.join(VIDEO_DIR, f"{TASK_NAME}_colored_health_bars_{timestamp}")
        save_rgb_health_video_with_overlay(
            output_video_path=path_overlay,
            imgs=colored_array,
            target_objects=TRACKED_OBJECTS,
            health=video_health_history,
            position="bottom_center",
            n_columns=min(3, len(TRACKED_OBJECTS)),
            fps=30,
        )
        print(f"3) Colored + health bars: {path_overlay}.mp4")
        print(f"Full paths under: {os.path.abspath(VIDEO_DIR)}")
        print(f"Video resolution: {VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}, {len(video_frames_raw)} frames")
        print("=" * 60)
        video_frames_raw = []
        video_frames_colored = []
        video_health_history = {obj: [] for obj in TRACKED_OBJECTS}

    def breakpoint_and_print_poses():
        nonlocal recording_video, trajectory_saved
        if recording_video:
            stop_recording_and_save()
        # Save teleop trajectory to HDF5 for replay (see shelve_item.py / DataCollectionWrapper)
        if len(env.current_traj_history) > 0:
            env.save_data()
            trajectory_saved = True  # exit loop after breakpoint; file is closed
            print("=" * 60)
            print(f"Trajectory saved to HDF5: {os.path.abspath(collect_hdf5_path)}")
            print("Replay later with a playback script using this path.")
            print("=" * 60)
        else:
            print("No trajectory steps recorded; skipping HDF5 save.")
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = np.asarray(rgb.cpu().numpy()[:, :, :3], dtype=np.uint8)
        frame = cv2.resize(rgb_np, RGB_SIZE)
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(SCREENSHOT_DIR, f"{TASK_NAME}_{timestamp}.png")
        cv2.imwrite(png_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Screenshot: {png_path}")
        print("TAB pressed – breakpoint")
        breakpoint()

    def save_state_callback():
        os.makedirs(STATE_SAVE_DIR, exist_ok=True)
        state = og.sim.dump_state(serialized=True)
        with open(STATE_SAVE_PATH, "wb") as f:
            pickle.dump(state, f)
        print("=" * 60)
        print(f"State saved to {os.path.abspath(STATE_SAVE_PATH)}")
        print("Run teleop_wine_glass_drop_load.py [--state_path ...] to load this state.")
        print("=" * 60)

    # Keyboard Teleop
    action_generator = KeyboardRobotController(robot=robot)
    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.R, description="Reset", callback_fn=lambda: env.reset())
    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.S, description="Save state to pkl (for load script)", callback_fn=save_state_callback)
    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.TAB, description="Stop recording, save 3 videos + trajectory HDF5, screenshot, breakpoint", callback_fn=breakpoint_and_print_poses)
    action_generator.print_keyboard_teleop_info()

    # Enable health visualization
    damageable_objects = env.get_damageable_objects()
    print(f"Damageable objects: {[obj.name for obj in damageable_objects]}")
    if len(damageable_objects) > 0:
        env.enable_health_visualization()
        print("Health bars enabled")
    start_recording()

    stand = env.scene.object_registry("name", "stand")
    print(f"Stand position: {stand.get_position_orientation()[0]}")
    print("Wine glass drop: teleop; S = save state; TAB = stop recording, save 3 videos + trajectory HDF5")
    print(f"Videos: {VIDEO_DIR}; Trajectory HDF5: {os.path.abspath(collect_hdf5_path)}")
    print("=" * 60)

    step_count = 0
    while True:
        if trajectory_saved:
            print("Trajectory already saved; exiting teleop loop. Restart script to record again.")
            break
        action, keypress_str = action_generator.get_teleop_action()
        # S = save state (callback); W/A/D/G = camera only (no robot step)
        camera_keys = {"W", "A", "D", "G"}
        if keypress_str == "S":
            continue  # S handled by save_state_callback
        if keypress_str and keypress_str in camera_keys:
            env.step(th.zeros(robot.action_dim))
            continue
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
                # Get viewer camera observations (same as nav_to_table_load / pick_place_load / pick_bowl_load)
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
                frame_raw = cv2.resize(rgb_np, VIDEO_SIZE)
                frame_colored = frame_raw.copy()

                # Get segmentation instance if available
                seg_instance_np = None
                seg_instance_info = {}
                if "seg_instance" in viewer_obs:
                    seg_instance = viewer_obs["seg_instance"]
                    seg_instance_np = np.asarray(seg_instance.cpu().numpy(), dtype=np.int64)
                    if isinstance(viewer_info, dict) and "seg_instance" in viewer_info:
                        seg_instance_info = viewer_info["seg_instance"]

                # Get health values for tracked objects
                health_values = {}
                for obj_name in TRACKED_OBJECTS:
                    try:
                        if obj_name == "franka0":
                            obj = robot
                        else:
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
                for obj_name in TRACKED_OBJECTS:
                    video_health_history[obj_name].append(health_values.get(obj_name, 100.0))

                # Apply health-based coloring if segmentation is available
                if seg_instance_np is not None:
                    frame_bgr = cv2.cvtColor(frame_colored, cv2.COLOR_RGB2BGR)

                    # Color objects red when health is low
                    for obj_name in TRACKED_OBJECTS:
                        if obj_name not in health_values:
                            continue

                        health = health_values[obj_name]
                        if health < HEALTH_THRESHOLD:
                            seg_key = None
                            if seg_instance_info:
                                for k, v in seg_instance_info.items():
                                    if isinstance(v, str):
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

                    frame_colored = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Append frame to video frames
                video_frames_raw.append(frame_raw)
                video_frames_colored.append(frame_colored)

            except Exception as e:
                print(f"Warning: Error capturing video frame: {e}")
                import traceback
                traceback.print_exc()
        step_count += 1

if __name__ == "__main__":
    __main__()
