"""
Teleop script: pick bowl from sink (spawned Inside sink) and place it back or elsewhere.
Uses GPU dynamics so the sink faucet can run (water particles). Bowl model: bowl-jblalf (BEHAVIOR).
"""
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
from omnigibson.object_states import ToggledOn, Inside, ParticleSource
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T

from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import omnigibson.lazy as lazy

from omnigibson.controllers.controller_base import IsGraspingState

from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper
from safety_benchmark.utils.misc_utils import save_rgb_camera_video

# GPU dynamics required for faucet water flow (FluidSystem / particle source)
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_OBJECT_STATES = True  # Required for OnTop, Inside

# Increase assisted grasp force for better grasping
from omnigibson.robots import manipulation_robot
manipulation_robot.m.MAX_ASSIST_FORCE = 500
print(f"Assisted grasp force increased: MAX_ASSIST_FORCE = {manipulation_robot.m.MAX_ASSIST_FORCE}")


def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_state', action='store_true', help='Load a state')
    args = parser.parse_args()
    if args.load_state:
        load_state_path = f"resources/teleop_data/pick_bowl.hdf5"
        load_f = h5py.File(load_state_path, "r")

    collect_hdf5_path = f"resources/teleop_data/pick_bowl.hdf5"
    if not os.path.exists(collect_hdf5_path):
        os.makedirs(os.path.dirname(collect_hdf5_path), exist_ok=True)

    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    cfg["scene"]["scene_model"] = "house_single_floor"
    cfg["scene"]["not_load_object_categories"] = ["ottoman"]
    cfg["scene"]["load_room_instances"] = ["kitchen_0"]

    ############### FrankaMounted robot ###############
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
            "arm_0": {
                "name": "InverseKinematicsController",
                "command_input_limits": None,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            },
        },
    }
    ############### FrankaMounted robot ###############

    robot_name = cfg["robots"][0].get("name", "franka0")
    robot_type = cfg["robots"][0].get("type", "FrankaMounted").lower()
    cfg["env"]["external_sensors"] = []

    # Bowl: BEHAVIOR bowl-jblalf (https://behavior.stanford.edu/knowledgebase/objects/jblalf.html)
    # Initial position approximate; will be set Inside sink after scene load
    cfg["objects"] = [
        {
            "type": "DatasetObject",
            "name": "bowl",
            "category": "bowl",
            "model": "jblalf",
            "position": [5.4, -1.7, 0.92],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
    ]

    env = DamageableEnvironment(configs=cfg)
    env = DamageableDataCollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
        enable_dump_filters=False,
    )

    robot = env.robots[0]

    for _ in range(10):
        og.sim.step()

    # Rotate robot 90 degrees to the right (face sink area)
    current_pos, current_orn = robot.get_position_orientation()
    rotation_90_right = T.euler2quat(th.tensor([0.0, 0.0, -np.pi/2]))
    new_robot_orn = T.quat_multiply(current_orn, rotation_90_right)
    robot.set_position_orientation(position=current_pos, orientation=new_robot_orn)

    for _ in range(10):
        og.sim.step()

    # Viewer camera (chosen for sink/bowl view)
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

    robot.set_joint_positions(th.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]))

    # Stronger gripper force for reliable bowl grasp (like teleop_pick_place_load / teleop_pick_place)
    controller_config = {
        "arm_0": {
            "name": "InverseKinematicsController",
            "command_input_limits": None,
        },
        "gripper_0": {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": "smooth",
            "motor_type": "position",
            "isaac_kp": 5000.0,   # Increased for stronger grasp (pick_place uses 4000)
            "isaac_kd": 2500.0,   # Increased for stronger grasp (pick_place uses 2000)
        },
    }
    robot.reload_controllers(controller_config=controller_config)

    # Gripper friction (increased to 3.0 like plate friction in pick_place_load for better grip on bowl)
    try:
        for link_name, link in robot.links.items():
            n = link_name.lower()
            if ("gripper" in n) or ("finger" in n):
                try:
                    link.set_attribute("physxMaterial:staticFriction", 3.0)
                    link.set_attribute("physxMaterial:dynamicFriction", 3.0)
                except Exception:
                    pass
    except Exception:
        pass

    for _ in range(10):
        og.sim.step()

    # Find countertop and sink
    countertop = None
    sink = None
    for obj in env.scene.objects:
        if "countertop" in obj.name.lower() or "counter" in obj.name.lower():
            if countertop is None:
                countertop = obj
                print(f"Found countertop: {obj.name}")
        if "sink" in obj.name.lower():
            if sink is None:
                sink = obj
                print(f"Found sink: {obj.name} at position {obj.get_position_orientation()[0]}")

    if countertop is None:
        print("Warning: Could not find countertop in scene, trying table/bar")
        for obj in env.scene.objects:
            if "table" in obj.name.lower() or "bar" in obj.name.lower():
                countertop = obj
                print(f"Found table/bar: {obj.name}")
                break

    bowl = env.scene.object_registry("name", "bowl")

    # Spawn bowl Inside the sink using object_states.Inside (like teleop_pick_place_load.py checks for Inside sink)
    if sink and bowl and object_states.Inside in bowl.states:
        try:
            if bowl.states[Inside].set_value(sink, True):
                print("Bowl spawned Inside sink ✓")
                for _ in range(20):
                    og.sim.step()
            else:
                print("Warning: Inside.set_value(sink, True) failed; placing bowl on countertop as fallback")
                if countertop:
                    countertop_aabb_min, countertop_aabb_max = countertop.aabb
                    countertop_top_z = countertop_aabb_max[2].item()
                    bowl_aabb_min, bowl_aabb_max = bowl.aabb
                    bowl_half_height = (bowl_aabb_max[2] - bowl_aabb_min[2]).item() / 2.0
                    desired_z = countertop_top_z + bowl_half_height + 0.001
                    bowl.set_position_orientation(
                        position=th.tensor([5.4, -1.7, desired_z]),
                        orientation=th.tensor([0.0, 0.0, 0.0, 1.0])
                    )
                    for _ in range(20):
                        og.sim.step()
        except Exception as e:
            print(f"Warning: Could not set bowl Inside sink: {e}; placing on countertop as fallback")
            if countertop and bowl:
                countertop_aabb_min, countertop_aabb_max = countertop.aabb
                countertop_top_z = countertop_aabb_max[2].item()
                bowl_aabb_min, bowl_aabb_max = bowl.aabb
                bowl_half_height = (bowl_aabb_max[2] - bowl_aabb_min[2]).item() / 2.0
                desired_z = countertop_top_z + bowl_half_height + 0.001
                bowl.set_position_orientation(
                    position=th.tensor([5.4, -1.7, desired_z]),
                    orientation=th.tensor([0.0, 0.0, 0.0, 1.0])
                )
                for _ in range(20):
                    og.sim.step()
    elif countertop and bowl:
        # No sink or bowl has no Inside state: place on counter
        countertop_aabb_min, countertop_aabb_max = countertop.aabb
        countertop_top_z = countertop_aabb_max[2].item()
        bowl_aabb_min, bowl_aabb_max = bowl.aabb
        bowl_half_height = (bowl_aabb_max[2] - bowl_aabb_min[2]).item() / 2.0
        desired_z = countertop_top_z + bowl_half_height + 0.001
        bowl.set_position_orientation(
            position=th.tensor([5.4, -1.7, desired_z]),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0])
        )
        print(f"Set bowl position on counter to [5.4, -1.7, {desired_z:.3f}]")
        for _ in range(20):
            og.sim.step()
    else:
        if not sink:
            print("Warning: Could not find sink in scene")
        if not countertop:
            print("Warning: Could not find countertop/table in scene")
        if not bowl:
            print("Warning: Could not find bowl object")

    # Bowl friction (increased to 3.0 like plate in teleop_pick_place_load to reduce slip during grasp)
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

    # Turn sink faucet on (required for water flow; sink may have ToggledOn state)
    faucet_obj = None
    if sink is not None and ToggledOn in sink.states:
        try:
            sink.states[ToggledOn].set_value(True)
            print("Sink faucet turned ON (ToggledOn on sink)")
            faucet_obj = sink
        except Exception as e:
            print(f"Warning: Could not set sink ToggledOn: {e}")
    else:
        # Fallback: any object with "faucet" in name that has ToggledOn
        for obj in env.scene.objects:
            if "faucet" in obj.name.lower() and ToggledOn in obj.states:
                try:
                    obj.states[ToggledOn].set_value(True)
                    print(f"Faucet turned ON: {obj.name}")
                    faucet_obj = obj
                    break
                except Exception:
                    pass
    if faucet_obj is None and sink is not None:
        print("Note: Sink/faucet has no ToggledOn state; water will not run (GPU dynamics still enabled).")

    # Slightly faster faucet: spawn every 5 steps (default is every N steps, N large)
    if sink is not None and ParticleSource in sink.states:
        try:
            ps = sink.states[ParticleSource]
            ps._n_steps_per_modification = 5
            print("Faucet rate increased: ParticleSource n_steps_per_modification=5")
        except Exception as e:
            print(f"Could not set ParticleSource n_steps: {e}")

    # Camera teleoperation
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

    # TAB: save high-quality viewer camera PNG (3840x2160), then print poses and breakpoint (same capture as ppo_eval/ppo_train)
    TASK_NAME = "pick_bowl"
    SCREENSHOT_DIR = "safe-manipulation-benchmark/resources/teleop_data/pick_bowl_screenshots"
    VIDEO_DIR = "safe-manipulation-benchmark/resources/teleop_data/pick_bowl_videos"
    RGB_SIZE = (3840, 2160)  # high-quality single frame
    VIDEO_SIZE = (1920, 1080)  # High resolution for video (can be adjusted)
    
    # Video recording state
    recording_video = False
    video_frames = []
    video_health_history = {"bowl": [], "franka0": []}
    HEALTH_THRESHOLD = 100.0  # Objects colored red when health < this value

    def start_recording():
        """Start recording video with health-based coloring"""
        nonlocal recording_video, video_frames, video_health_history
        recording_video = True
        video_frames = []
        video_health_history = {"bowl": [], "franka0": []}
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
        print("Saving video...")
        print("=" * 60)
        
        # Save video
        os.makedirs(VIDEO_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(VIDEO_DIR, f"{TASK_NAME}_video_{timestamp}")
        
        # Convert frames to numpy array
        video_array = np.array(video_frames)
        save_rgb_camera_video(video_path, video_array, fps=30)
        
        print(f"Video saved: {video_path}.mp4")
        print(f"Video resolution: {VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}")
        print(f"Total frames: {len(video_frames)}")
        print("=" * 60)
        
        # Clear recording state
        video_frames = []
        video_health_history = {"bowl": [], "franka0": []}
    
    def breakpoint_and_print_poses():
        nonlocal recording_video
        
        # If recording, stop recording and save video
        if recording_video:
            stop_recording_and_save()
            return
        
        # Otherwise, start recording
        start_recording()
        
        # Also save viewer_camera RGB as PNG (same code as ppo_eval_place_plate.py / ppo_train_place_plate.py, different size)
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

    def save_state_to_pkl():
        """Save current simulation state to pkl file (init state for training/eval)"""
        os.makedirs("safe-manipulation-benchmark/resources/saved_states", exist_ok=True)
        save_path = "safe-manipulation-benchmark/resources/saved_states/pick_bowl_init_state.pkl"
        og.sim.update_handles()
        state = og.sim.dump_state(serialized=True)
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        print(f"Simulation state saved to: {save_path}")

    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Start/Stop video recording (first press starts, second press stops and saves)",
        callback_fn=breakpoint_and_print_poses,
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.S,
        description="Save simulation state to pkl file",
        callback_fn=save_state_to_pkl,
    )
    action_generator.print_keyboard_teleop_info()

    print(f"Robot position: {robot.get_position_orientation()[0]}")
    if sink is None:
        print("No sink found in scene")

    print("=" * 60)
    print("Teleoperation ready!")
    print("Press TAB to start video recording (press TAB again to stop and save)")
    print("Press ESC to quit")
    print("=" * 60)

    # Data collection loop
    n_episodes = 1
    for i in range(n_episodes):
        print(f"Episode {i} starts")
        if args.load_state:
            breakpoint()
            for _ in range(50):
                og.sim.step()

        prev_is_grasping = False
        while True:
            ret = action_generator.get_teleop_action()
            if isinstance(ret, tuple) and len(ret) == 2:
                action, keypress_str = ret
            else:
                action = ret
                keypress_str = None

            # Skip robot action when camera keys pressed (WASD+G) so camera mover can drive
            camera_keys = {"W", "A", "S", "D", "G"}
            if keypress_str and keypress_str in camera_keys:
                env.step(th.zeros(robot.action_dim))
                continue

            try:
                from omnigibson.utils.constants import IsGraspingState
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

            for attempt in range(3):
                try:
                    step_result = env.step(action)
                    break
                except AttributeError as e:
                    if "'NoneType' object has no attribute" in str(e) and attempt < 2:
                        if og.sim.is_playing():
                            try:
                                og.sim.update_handles()
                            except Exception:
                                pass
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
    print("Data saved")
    env.save_data()
    camera_mover.clear()
    og.shutdown()


if __name__ == "__main__":
    __main__()
