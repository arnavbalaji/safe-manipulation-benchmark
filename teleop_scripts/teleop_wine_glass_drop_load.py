"""
Load a saved wine_glass_drop state and teleoperate (same as teleop_wine_glass_drop but from saved state).

To create a state file:
  1. Run: python safe-manipulation-benchmark/teleop_scripts/teleop_wine_glass_drop.py
  2. Arrange scene as desired, then press S to save state to
     safe-manipulation-benchmark/resources/saved_states/wine_glass_drop_init_state.pkl
  3. Quit, then run: python safe-manipulation-benchmark/teleop_scripts/teleop_wine_glass_drop_load.py [--state_path <path>]

Same behavior as main script: high-quality video recording, TAB saves 4 videos (raw, colored, colored+health bars, impact force graph)
and trajectory HDF5. S still saves state (overwrites or new path).
"""
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import pickle
from datetime import datetime
import torch as th
import numpy as np
import cv2

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy

from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.controllers.controller_base import IsGraspingState

from safety_benchmark.damageable_env import DamageableEnvironment, load_sim_state_with_size_fallback
from safety_benchmark.utils.misc_utils import (
    save_rgb_camera_video,
    save_rgb_health_video_with_overlay,
    save_rgb_force_video,
)

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False

SHELF_INIT_POS = [6.00, 0.2, 1.3]
SHELF_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
SHELF_SCALE = [0.4, 0.6, 0.5]
TOP_SHELF_Z = SHELF_INIT_POS[2] + SHELF_SCALE[2] * 1.4

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
    parser.add_argument(
        "--state_path",
        type=str,
        default="safe-manipulation-benchmark/resources/saved_states/wine_glass_drop_init_state.pkl",
        help="Path to saved state pkl (save with teleop_wine_glass_drop.py, press S)",
    )
    args = parser.parse_args()

    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "house_single_floor"
    cfg["scene"]["not_load_object_categories"] = ["ottoman"]
    cfg["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]

    cfg["robots"][0] = {
        "type": "FrankaPanda",
        "name": "franka0",
        "position": [6.7, 0.2, 1.0],
        "orientation": [0.0, 0.0, 1.0, 0.0],
        "grasping_mode": "assisted",
        "obs_modalities": ["rgb", "depth"],
        "action_normalize": False,
        "self_collisions": True,
        "controller_config": {
            "arm_0": {"name": "InverseKinematicsController", "command_input_limits": None},
            "gripper_0": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0), "mode": "smooth"},
        },
    }

    robot_name = cfg["robots"][0].get("name", "franka0")
    robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()
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

    # Same as nav_to_table_load / pick_place_load / pick_bowl_load: NO DataCollectionWrapper.
    # The wrapper switches viewport/rendering and makes viewer_camera.get_obs() capture the wrong scene.
    env = DamageableEnvironment(configs=cfg)
    env.reset()
    robot = env.robots[0]
    for _ in range(10):
        og.sim.step()

    state_path = args.state_path
    if not os.path.exists(state_path):
        print(f"Error: State file not found at {state_path}")
        print("Run teleop_wine_glass_drop.py first, then press S to save state.")
        return
    with open(state_path, "rb") as f:
        state_flat_array = pickle.load(f)
    load_sim_state_with_size_fallback(state_flat_array)
    print(f"Loaded state from {state_path}")

    env.initialize_damageable_objects()
    env.set_damageable_object_params()
    robot = env.robots[0]

    robot_pos, robot_orn = robot.get_position_orientation()
    robot_joint_positions = robot.get_joint_positions()
    robot.keep_still()
    for _ in range(20):
        robot.keep_still()
        og.sim.step()
    robot.set_position_orientation(robot_pos, robot_orn)
    robot.set_joint_positions(robot_joint_positions)
    robot.set_joint_velocities(th.zeros(robot.n_dof))
    for ctrl_name in ["arm_0", "gripper_0"]:
        ctrl = robot.controllers.get(ctrl_name)
        if ctrl is not None:
            ctrl.reset()
    for _ in range(10):
        og.sim.step()

    # Keep gripper closed after load (apply closed action, then set joints to closed)
    close_gripper_action = th.zeros(robot.action_dim)
    close_gripper_action[robot.gripper_action_idx[robot.default_arm]] = -1.0
    for _ in range(20):
        robot.apply_action(close_gripper_action)
        og.sim.step()
        robot.keep_still()
    try:
        gripper_joint_indices = robot.gripper_joint_indices[robot.default_arm]
        if len(gripper_joint_indices) > 0:
            current_joint_positions = robot.get_joint_positions()
            for gripper_joint_idx in gripper_joint_indices:
                current_joint_positions[gripper_joint_idx] = 0.0
            robot.set_joint_positions(current_joint_positions)
            robot.keep_still()
            for _ in range(5):
                og.sim.step()
            print("Gripper joints set to closed position")
    except (AttributeError, KeyError, IndexError) as e:
        print(f"Could not set gripper joints directly (ok): {e}")

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([7.0659, -0.7141, 2.0185]),
        orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
    )
    for _ in range(10):
        og.sim.step()

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
    # Force targets: object@link for impact force graph video (Franka EEF/hand + task objects)
    TARGET_OBJECTS_FORCES = [
        f"{robot_name}@eef_link",
        f"{robot_name}@panda_hand",
        "wineglass@base_link",
        "box_of_crackers@base_link",
    ]
    FORCE_KEYS = ["impact_forces", "filtered_qs_forces"]
    recording_video = False
    video_frames_raw = []
    video_frames_colored = []
    video_health_history = {obj: [] for obj in TRACKED_OBJECTS}
    video_force_data = {}  # filled each step when recording; used for impact force video
    HEALTH_THRESHOLD = 100.0

    def _extract_force_value(mech_dict, force_key):
        """One scalar per step: use max of list or the scalar."""
        val = mech_dict.get(force_key, 0.0)
        if isinstance(val, (list, np.ndarray)):
            return float(np.max(val)) if len(val) > 0 else 0.0
        return float(val) if isinstance(val, (int, float)) else 0.0

    def start_recording():
        nonlocal recording_video, video_frames_raw, video_frames_colored, video_health_history, video_force_data
        recording_video = True
        video_frames_raw = []
        video_frames_colored = []
        video_health_history = {obj: [] for obj in TRACKED_OBJECTS}
        video_force_data = {obj: {k: [] for k in FORCE_KEYS} for obj in TARGET_OBJECTS_FORCES}
        print("=" * 60)
        print("VIDEO RECORDING STARTED (load script)")
        print("Press TAB to stop recording and save four videos (raw, colored, colored+health, impact force)")
        print("=" * 60)

    def stop_recording_and_save():
        nonlocal recording_video, video_frames_raw, video_frames_colored, video_health_history, video_force_data
        if not recording_video or len(video_frames_raw) == 0:
            return
        recording_video = False
        print("=" * 60)
        print("STOPPING VIDEO RECORDING")
        print(f"Captured {len(video_frames_raw)} frames")
        print("=" * 60)
        os.makedirs(VIDEO_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_array = np.array(video_frames_raw)
        colored_array = np.array(video_frames_colored)
        path_raw = os.path.join(VIDEO_DIR, f"{TASK_NAME}_load_raw_{timestamp}")
        save_rgb_camera_video(path_raw, raw_array, fps=30)
        print(f"1) Raw: {path_raw}.mp4")
        path_colored = os.path.join(VIDEO_DIR, f"{TASK_NAME}_load_colored_{timestamp}")
        save_rgb_camera_video(path_colored, colored_array, fps=30)
        print(f"2) Colored: {path_colored}.mp4")
        path_overlay = os.path.join(VIDEO_DIR, f"{TASK_NAME}_load_colored_health_bars_{timestamp}")
        save_rgb_health_video_with_overlay(output_video_path=path_overlay, imgs=colored_array, target_objects=TRACKED_OBJECTS, health=video_health_history, position="bottom_left", n_columns=1, fps=30)
        print(f"3) Colored + health bars: {path_overlay}.mp4")
        # 4) Impact force graph video (same frames as colored, with force plot)
        n_force_steps = len(video_force_data[TARGET_OBJECTS_FORCES[0]][FORCE_KEYS[0]]) if TARGET_OBJECTS_FORCES else 0
        if n_force_steps > 0 and n_force_steps == len(colored_array):
            path_forces = os.path.join(VIDEO_DIR, f"{TASK_NAME}_load_forces_{timestamp}.mp4")
            save_rgb_force_video(
                output_video_path=path_forces,
                imgs=colored_array,
                target_objects=TARGET_OBJECTS_FORCES,
                data=video_force_data,
                forces_to_plot=FORCE_KEYS,
                fps=30,
                force_ylim=(0, 100.0),
            )
            print(f"4) Impact force graph: {path_forces}")
        else:
            print("4) Impact force video skipped (force data length != frame count)")
        print("=" * 60)
        video_frames_raw = []
        video_frames_colored = []
        video_health_history = {obj: [] for obj in TRACKED_OBJECTS}
        video_force_data = {obj: {k: [] for k in FORCE_KEYS} for obj in TARGET_OBJECTS_FORCES}

    def breakpoint_and_print_poses():
        nonlocal recording_video
        if recording_video:
            stop_recording_and_save()
        viewer_obs_result = og.sim.viewer_camera.get_obs()
        viewer_obs = viewer_obs_result[0] if isinstance(viewer_obs_result, tuple) else viewer_obs_result
        rgb = viewer_obs["rgb"]
        rgb_np = np.asarray(rgb.cpu().numpy()[:, :, :3], dtype=np.uint8)
        frame = cv2.resize(rgb_np, RGB_SIZE)
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(SCREENSHOT_DIR, f"{TASK_NAME}_load_{timestamp}.png")
        cv2.imwrite(png_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Screenshot: {png_path}")
        breakpoint()

    def save_state_callback():
        os.makedirs(STATE_SAVE_DIR, exist_ok=True)
        state = og.sim.dump_state(serialized=True)
        with open(STATE_SAVE_PATH, "wb") as f:
            pickle.dump(state, f)
        print("=" * 60)
        print(f"State saved to {os.path.abspath(STATE_SAVE_PATH)}")
        print("=" * 60)

    action_generator = KeyboardRobotController(robot=robot)
    # Keep gripper closed by default (persistent action -1.0) so it stays closed after load
    if hasattr(action_generator, "binary_grippers") and len(action_generator.binary_grippers) > 0:
        action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
        if hasattr(action_generator, "gripper_direction"):
            action_generator.gripper_direction[action_generator.binary_grippers[0]] = -1.0
        print("Gripper set to stay closed (persistent -1.0); press T to toggle if needed.")
    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.R, description="Reset", callback_fn=lambda: env.reset())
    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.S, description="Save state to pkl", callback_fn=save_state_callback)
    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.TAB, description="Stop recording, save 4 videos (incl. impact force), screenshot, breakpoint", callback_fn=breakpoint_and_print_poses)
    action_generator.print_keyboard_teleop_info()

    damageable_objects = env.get_damageable_objects()
    if len(damageable_objects) > 0:
        env.enable_health_visualization()
    start_recording()

    print("Wine glass drop (load): teleop; S = save state; TAB = stop recording, save 4 videos (raw, colored, health bars, impact force)")
    print(f"Videos: {VIDEO_DIR}")
    print("=" * 60)

    step_count = 0
    while True:
        action, keypress_str = action_generator.get_teleop_action()
        camera_keys = {"W", "A", "D", "G"}
        if keypress_str == "S":
            continue
        if keypress_str and keypress_str in camera_keys:
            env.step(th.zeros(robot.action_dim))
            continue
        step_info = {}
        try:
            result = env.step(action)
            if isinstance(result, (list, tuple)) and len(result) >= 5:
                _obs, _reward, _term, _trunc, step_info = result[0], result[1], result[2], result[3], result[4]
            else:
                step_info = getattr(env, "_last_info", {})
            env.update_health_visualization()
        except AttributeError as e:
            if "'NoneType' object has no attribute" in str(e):
                result = env.step(action)
                if isinstance(result, (list, tuple)) and len(result) >= 5:
                    step_info = result[4]
                else:
                    step_info = {}
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

                # Get health values for tracked objects (same structure as main script / pick_place_load)
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

                # Store force data for impact force graph video (from step_info)
                damage_info = step_info.get("damage_info") or {}
                for obj_link in TARGET_OBJECTS_FORCES:
                    parts = obj_link.split("@", 1)
                    obj_name = parts[0] if len(parts) == 2 else obj_link
                    link_name = parts[1] if len(parts) == 2 else "base_link"
                    obj_dmg = damage_info.get(obj_name, {})
                    link_dmg = obj_dmg.get(link_name, {})
                    mech = link_dmg.get("mechanical", {})
                    for fk in FORCE_KEYS:
                        val = _extract_force_value(mech, fk)
                        video_force_data[obj_link][fk].append(val)

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
