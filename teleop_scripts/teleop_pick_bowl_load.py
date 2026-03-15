"""
Load a saved pick_bowl state and teleoperate (same flow as teleop_pick_place_load.py for bowl task).
Run teleop_pick_bowl.py, press S to save state, then run this script with --state_path to load and continue.
TAB saves high-quality (3840x2160) viewer camera PNG to pick_bowl_screenshots.
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
from omnigibson.macros import gm
from omnigibson import object_states
from omnigibson.object_states import ParticleSource
from omnigibson.systems import FluidSystem

from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import omnigibson.lazy as lazy

from omnigibson.controllers.controller_base import IsGraspingState

from safety_benchmark.damageable_env import DamageableEnvironment, load_sim_state_with_size_fallback
from safety_benchmark.utils.misc_utils import save_rgb_camera_video, save_rgb_water_contacts_video
import json

# Match teleop_pick_bowl: GPU dynamics for faucet/water
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_OBJECT_STATES = True

from omnigibson.robots import manipulation_robot
manipulation_robot.m.MAX_ASSIST_FORCE = 500
print(f"Assisted grasp force increased: MAX_ASSIST_FORCE = {manipulation_robot.m.MAX_ASSIST_FORCE}")


def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--state_path', type=str,
                       default='safe-manipulation-benchmark/resources/saved_states/pick_bowl_init_state.pkl',
                       help='Path to the saved state pkl file')
    args = parser.parse_args()

    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    cfg["scene"]["scene_model"] = "house_single_floor"
    cfg["scene"]["not_load_object_categories"] = ["ottoman"]
    cfg["scene"]["load_room_instances"] = ["kitchen_0"]

    cfg["robots"][0] = {
        "type": "FrankaMounted",
        "name": "franka0",
        "position": [5.7, -1.4, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "grasping_mode": "assisted",
        "obs_modalities": ["rgb", "depth"],
        "action_normalize": False,
        "self_collisions": True,
        "controller_config": {
            "arm_0": {"name": "InverseKinematicsController", "command_input_limits": None},
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            },
        },
    }
    cfg["env"]["external_sensors"] = []

    # Include bowl and place_mat (same location as ppo_train_place_plate.py)
    PLACE_MAT_POS = [5.185426712036133, -1.8776537656784058, 0.9251976013183594]
    PLACE_MAT_SCALE = [0.3, 0.3, 0.3]
    cfg["objects"] = [
        {
            "type": "DatasetObject",
            "name": "bowl",
            "category": "bowl",
            "model": "jblalf",
            "position": [5.4, -1.7, 0.92],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "type": "DatasetObject",
            "name": "place_mat",
            "category": "place_mat",
            "model": "nxzfmz",
            "position": PLACE_MAT_POS,
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "scale": PLACE_MAT_SCALE,
        },
    ]

    env = DamageableEnvironment(configs=cfg)
    env.reset()

    robot = env.robots[0]

    for _ in range(10):
        og.sim.step()

    state_path = args.state_path
    if not os.path.exists(state_path):
        print(f"Error: State file not found at {state_path}")
        print("Provide --state_path to a saved state pkl from pick_bowl (or similar) teleop.")
        return

    try:
        with open(state_path, "rb") as f:
            state_flat_array = pickle.load(f)
        load_sim_state_with_size_fallback(state_flat_array)
        print(f"Successfully loaded state from {state_path}")
        
        # After loading state, ensure robot is still configured for damage tracking
        robot = env.robots[0]  # Re-get robot reference after state load
        if hasattr(robot, "set_track_damage"):
            robot.set_track_damage(True)
        if hasattr(robot, "_initialize_health"):
            robot._initialize_health()
        if hasattr(robot, "_initialize_damage_evaluators"):
            robot._initialize_damage_evaluators()
        # Re-set params after state load
        env.set_damageable_object_params()
        print(f"Re-configured robot '{robot.name}' for damage tracking after state load")
    except Exception as e:
        print(f"Error loading state: {e}")
        raise

    robot_pos, robot_orn = robot.get_position_orientation()
    robot_joint_positions = robot.get_joint_positions()

    robot.keep_still()
    for _ in range(10):
        robot.keep_still()
        og.sim.step()
    robot.keep_still()
    og.sim.step()

    keep_gripper_action = th.zeros(robot.action_dim)
    if robot.is_grasping().value == IsGraspingState.TRUE:
        keep_gripper_action[robot.gripper_action_idx[robot.default_arm]] = -1.0
        print("Gripper kept closed to maintain saved grasping state")
    else:
        keep_gripper_action[robot.gripper_action_idx[robot.default_arm]] = -1.0
        print("Gripper set to closed (default)")

    for _ in range(10):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        robot.apply_action(keep_gripper_action)
        og.sim.step()
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()

    for _ in range(30):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        robot.apply_action(keep_gripper_action)
        og.sim.step()
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()

    # Greatly increased gripper force for reliable bowl grasp
    controller_config = {
        "arm_0": {"name": "InverseKinematicsController", "command_input_limits": None},
        "gripper_0": {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": "smooth",
            "motor_type": "position",
            "isaac_kp": 30000.0,   # greatly increased (was 5000)
            "isaac_kd": 15000.0,   # greatly increased (was 2500)
        },
    }
    robot.reload_controllers(controller_config=controller_config)

    # High friction on gripper/finger links for strong grip
    try:
        for link_name, link in robot.links.items():
            n = link_name.lower()
            if ("gripper" in n) or ("finger" in n):
                try:
                    link.set_attribute("physxMaterial:staticFriction", 6.0)
                    link.set_attribute("physxMaterial:dynamicFriction", 6.0)
                except Exception:
                    pass
    except Exception:
        pass

    bowl = env.scene.object_registry("name", "bowl")
    if bowl is not None:
        try:
            for link in bowl.links.values():
                try:
                    link.set_attribute("physxMaterial:staticFriction", 3.0)
                    link.set_attribute("physxMaterial:dynamicFriction", 3.0)
                except Exception:
                    pass
        except Exception:
            pass

    # Set place_mat fixed_base after settling (like in ppo_train_place_plate.py)
    place_mat_obj = None
    try:
        place_mat_obj = env.scene.object_registry("name", "place_mat")
    except Exception:
        place_mat_obj = None
    if place_mat_obj is not None:
        place_mat_obj.fixed_base = True
        place_mat_obj.keep_still()
        print(f"Set place_mat fixed_base=True")

    robot.set_position_orientation(robot_pos, robot_orn)
    robot.set_joint_positions(robot_joint_positions)
    robot.set_joint_velocities(th.zeros(robot.n_dof))

    arm_controller = robot.controllers.get("arm_0")
    if arm_controller is not None:
        arm_controller.reset()
    gripper_controller = robot.controllers.get("gripper_0")
    if gripper_controller is not None:
        gripper_controller.reset()

    robot.keep_still()
    for _ in range(10):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        og.sim.step()

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
            print("Gripper joints directly set to closed position")
    except (AttributeError, KeyError, IndexError) as e:
        print(f"Could not directly set gripper joints (this is okay): {e}")

    # Viewer camera (same as teleop_pick_bowl)
    camera_pos = th.tensor([6.764060974121094, -1.9225226640701294, 1.3960963487625122])
    camera_orn = th.tensor([0.44636261463165283, 0.4237414598464966, 0.542643666267395, 0.5716130137443542])
    og.sim.viewer_camera.set_position_orientation(
        position=camera_pos,
        orientation=camera_orn,
    )
    
    # Enable seg_instance modality for viewer camera to enable health-based coloring
    try:
        # Add seg_instance to viewer camera modalities if not already present
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

    action_generator = KeyboardRobotController(robot=robot)

    if robot.is_grasping().value == IsGraspingState.TRUE:
        print("Robot is grasping bowl - setting persistent gripper action to closed")
        action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
        action_generator.gripper_direction[action_generator.binary_grippers[0]] = -1.0
    else:
        print("Setting persistent gripper action to closed (default)")
        action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
        action_generator.gripper_direction[action_generator.binary_grippers[0]] = -1.0

    TASK_NAME = "pick_bowl"
    SCREENSHOT_DIR = "safe-manipulation-benchmark/resources/teleop_data/pick_bowl_screenshots"
    VIDEO_DIR = "safe-manipulation-benchmark/resources/teleop_data/pick_bowl_videos"
    RGB_SIZE = (3840, 2160)  # high-quality single frame
    VIDEO_SIZE = (1920, 1080)  # High resolution for video (can be adjusted)
    
    # Video recording state
    recording_video = False
    video_frames = []
    video_health_history = {"bowl": [], "franka0": []}
    video_water_contacts_history = {"franka0": []}  # Track water contacts for robot
    HEALTH_THRESHOLD = 100.0  # Objects colored red when health < this value

    def start_recording():
        """Start recording video with health-based coloring"""
        nonlocal recording_video, video_frames, video_health_history, video_water_contacts_history
        recording_video = True
        video_frames = []
        video_health_history = {"bowl": [], "franka0": []}
        video_water_contacts_history = {"franka0": []}
        print("=" * 60)
        print("VIDEO RECORDING STARTED")
        print("Recording high-resolution video with health-based coloring")
        print("Press TAB again to stop recording and save video")
        print("=" * 60)
    
    def stop_recording_and_save():
        """Stop recording and save video with health-based coloring"""
        nonlocal recording_video, video_frames, video_health_history, video_water_contacts_history
        
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
        
        # Save water contacts video (using same frames)
        if len(video_water_contacts_history["franka0"]) > 0:
            water_contacts_video_path = os.path.join(VIDEO_DIR, f"{TASK_NAME}_water_contacts_{timestamp}.mp4")
            save_rgb_water_contacts_video(
                output_video_path=water_contacts_video_path,
                imgs=video_array,
                target_objects=["franka0"],
                water_contacts=video_water_contacts_history,
                fps=30
            )
            print(f"Water contacts video saved: {water_contacts_video_path}")
            print(f"Full path: {os.path.abspath(water_contacts_video_path)}")
        
        print(f"Video resolution: {VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}")
        print(f"Total frames: {len(video_frames)}")
        print("=" * 60)
        
        # Clear recording state
        video_frames = []
        video_health_history = {"bowl": [], "franka0": []}
        video_water_contacts_history = {"franka0": []}
    
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
            eef_pos = robot.get_eef_position(arm="default")
            eef_orn = robot.get_eef_orientation(arm="default")
            print("Robot End Effector Pose:")
            print(f"  Position (x, y, z): {eef_pos.tolist()}")
            print(f"  Orientation (quaternion w, x, y, z): {eef_orn.tolist()}")
            print()
            bowl_obj = env.scene.object_registry("name", "bowl")
            if bowl_obj is not None:
                bowl_pos, bowl_orn = bowl_obj.get_position_orientation()
                print("Bowl Pose:")
                print(f"  Position (x, y, z): {bowl_pos.tolist()}")
                print(f"  Orientation (quaternion w, x, y, z): {bowl_orn.tolist()}")
            else:
                print("Bowl: Not found in scene")
            print()
            cam_pos, cam_orn = og.sim.viewer_camera.get_position_orientation()
            print("Viewer Camera Pose:")
            print(f"  Position (x, y, z): {cam_pos.tolist()}")
            print(f"  Orientation (quaternion w, x, y, z): {cam_orn.tolist()}")
            print("=" * 60 + "\n")
            breakpoint()

    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Stop video recording, save video, print poses, and breakpoint",
        callback_fn=breakpoint_and_print_poses,
    )
    action_generator.print_keyboard_teleop_info()

    sink = None
    for obj in env.scene.objects:
        if "sink" in obj.name.lower():
            sink = obj
            print(f"Found sink: {obj.name}")
            break
    if sink is None:
        print("Warning: Could not find sink in scene")

    # Slightly faster faucet: spawn every 5 steps (default is every N steps, N large)
    if sink is not None and ParticleSource in sink.states:
        try:
            ps = sink.states[ParticleSource]
            ps._n_steps_per_modification = 4
            print("Faucet rate increased: ParticleSource n_steps_per_modification=5")
        except Exception as e:
            print(f"Could not set ParticleSource n_steps: {e}")

    # Ensure robot is still tracking damage after state load
    if hasattr(robot, "set_track_damage"):
        robot.set_track_damage(True)
    if hasattr(robot, "_initialize_health"):
        robot._initialize_health()
    # Re-initialize damage evaluators after state load
    if hasattr(robot, "_initialize_damage_evaluators"):
        robot._initialize_damage_evaluators()
    # Re-set params after state load
    env.set_damageable_object_params()
    
    # Enable health visualization for robot
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
    print("Teleoperation ready! (pick_bowl load)")
    print("Video recording started automatically")
    print("Live health bars enabled for robot")
    print(f"Videos will be saved to: {VIDEO_DIR}")
    print("Press TAB to stop recording, save video, print poses, and breakpoint")
    print("Press ESC to quit")
    print("=" * 60)

    prev_is_grasping = False
    prev_gripper_action = -1.0  # Track previous gripper action to detect closing
    gripper_close_steps_remaining = 0  # Counter for extra closing steps
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

        # Check if gripper is being closed (transition from open to closed)
        current_gripper_action = action[robot.gripper_action_idx[robot.default_arm]]
        gripper_is_closing = prev_gripper_action >= 0.0 and current_gripper_action < -0.5
        
        if gripper_is_closing:
            # Gripper is being closed - apply extra closing force
            gripper_close_steps_remaining = 25  # Apply extra closing for 25 steps
            print("Gripper closing detected - applying extra closing force and setting joints to closed")
            # Directly set gripper joints to closed position immediately
            try:
                gripper_joint_indices = robot.gripper_joint_indices[robot.default_arm]
                if len(gripper_joint_indices) > 0:
                    current_joint_positions = robot.get_joint_positions()
                    for gripper_joint_idx in gripper_joint_indices:
                        current_joint_positions[gripper_joint_idx] = 0.0
                    robot.set_joint_positions(current_joint_positions)
                    robot.keep_still()
                    # Step a few times to let joints settle
                    for _ in range(3):
                        og.sim.step()
            except (AttributeError, KeyError, IndexError):
                pass
        
        prev_gripper_action = current_gripper_action

        # Apply extra closing force if needed (during closing phase)
        if gripper_close_steps_remaining > 0:
            action[robot.gripper_action_idx[robot.default_arm]] = -1.0  # Force fully closed
            gripper_close_steps_remaining -= 1

        try:
            is_grasping = robot.is_grasping().value == IsGraspingState.TRUE
            if is_grasping != prev_is_grasping:
                if og.sim.is_playing():
                    try:
                        og.sim.update_handles()
                    except Exception:
                        pass
                prev_is_grasping = is_grasping
        except Exception:
            pass

        if og.sim.is_playing():
            try:
                og.sim.update_handles()
            except Exception:
                pass

        # Re-apply place_mat fixed_base every step (fixed_base from initialization may not persist)
        try:
            place_mat_obj = env.scene.object_registry("name", "place_mat")
            if place_mat_obj is not None:
                place_mat_obj.fixed_base = True
                place_mat_obj.keep_still()
        except Exception:
            pass

        try:
            step_result = env.step(action)
            # Update health visualization
            env.update_health_visualization()
        except AttributeError as e:
            if "'NoneType' object has no attribute" in str(e):
                if og.sim.is_playing():
                    try:
                        og.sim.update_handles()
                    except Exception:
                        pass
                step_result = env.step(action)
                # Update health visualization
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
                    # Get seg_instance mapping from viewer_info
                    if isinstance(viewer_info, dict) and "seg_instance" in viewer_info:
                        seg_instance_info = viewer_info["seg_instance"]
                
                # Get health values for tracked objects
                health_values = {}
                bowl_obj = env.scene.object_registry("name", "bowl")
                if bowl_obj is not None and hasattr(bowl_obj, "health"):
                    try:
                        health_values["bowl"] = float(bowl_obj.health)
                    except Exception:
                        health_values["bowl"] = 100.0
                else:
                    health_values["bowl"] = 100.0
                
                if robot is not None and hasattr(robot, "health"):
                    try:
                        health_values["franka0"] = float(robot.health)
                    except Exception:
                        health_values["franka0"] = 100.0
                else:
                    health_values["franka0"] = 100.0
                
                # Store health history
                video_health_history["bowl"].append(health_values["bowl"])
                video_health_history["franka0"].append(health_values["franka0"])
                
                # Track water contacts for robot
                water_contact_count = 0
                if robot is not None and hasattr(robot, "damage_evaluators"):
                    for evaluator in robot.damage_evaluators:
                        if evaluator.name == "electrical":
                            contact_summary = evaluator.get_contact_summary()
                            water_contact_count = contact_summary.get("total_contact", 0)
                            break
                video_water_contacts_history["franka0"].append(water_contact_count)
                
                # Apply health-based coloring if segmentation is available
                if seg_instance_np is not None:
                    # Convert RGB to BGR for OpenCV operations
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Color objects red when health is low
                    for obj_name in ["bowl", "franka0"]:
                        if obj_name not in health_values:
                            continue
                        
                        health = health_values[obj_name]
                        if health < HEALTH_THRESHOLD:
                            # Try to find segmentation key from seg_instance_info
                            seg_key = None
                            if seg_instance_info:
                                for k, v in seg_instance_info.items():
                                    if isinstance(v, str):
                                        # Check if object name appears in the mapping value
                                        # The value might be a full path like "/World/scene/bowl" or just "bowl"
                                        if obj_name.lower() in v.lower():
                                            try:
                                                seg_key = int(k)
                                                break
                                            except (ValueError, TypeError):
                                                continue
                            
                            if seg_key is not None:
                                # Create mask for this object
                                mask = seg_instance_np == seg_key
                                
                                if np.any(mask):  # Only apply if mask has any True values
                                    # Calculate alpha based on health (lower health = more red)
                                    alpha = 1.0 - (health / HEALTH_THRESHOLD)
                                    alpha = max(0.0, min(1.0, alpha))
                                    
                                    # Apply red overlay
                                    overlay_color = np.array([0, 0, 255], dtype=np.uint8)  # BGR red
                                    frame_bgr[mask] = ((1 - alpha) * frame_bgr[mask] + alpha * overlay_color).astype(np.uint8)
                    
                    # Convert back to RGB
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Append frame to video frames
                video_frames.append(frame)
                
            except Exception as e:
                print(f"Warning: Error capturing video frame: {e}")
                import traceback
                traceback.print_exc()

        step_count += 1
        try:
            bowl_obj = env.scene.object_registry("name", "bowl")
            if bowl_obj is not None:
                bowl_pos, _ = bowl_obj.get_position_orientation()
                is_inside_sink = False
                if sink is not None and object_states.Inside in bowl_obj.states:
                    try:
                        is_inside_sink = bowl_obj.states[object_states.Inside].get_value(sink)
                    except Exception:
                        pass
                print(f"\r[Step {step_count}] Bowl pos: [{bowl_pos[0]:.2f}, {bowl_pos[1]:.2f}, {bowl_pos[2]:.2f}] | Inside sink: {is_inside_sink}", end="", flush=True)
        except Exception:
            pass

    camera_mover.clear()
    og.shutdown()


if __name__ == "__main__":
    __main__()
