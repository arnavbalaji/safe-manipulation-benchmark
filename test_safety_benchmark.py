"""
Example script demonstrating robot control with damage functionality.
Tests the safety benchmark features including damage generators and health tracking.
"""

import os
import sys
import torch as th
import cv2
import numpy as np
import subprocess
import argparse
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options, draw_box
from omnigibson.object_states import OnTop
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS
from omnigibson.utils.control_utils import IKSolver
from omnigibson.utils.transform_utils import quat2mat, mat2quat, quat2axisangle, axisangle2quat
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure OmniGibson settings
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

# Available scenes
SCENES = ["Rs_int", "empty"]

OBJECT_CONFIGS = {
    "baseball": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "baseball",
        "model": "zanmar",
        "position": [0.1, 0.0, 3.0],
        "orientation": [0, 0, 0, 1],
        "scale": [2.0, 2.0, 2.0],
        "damage_params": PARAMS["baseball"]
    },
    "bowl": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "bowl",
        "model": "bexgtn",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],  # Identity quaternion for upright orientation
        "scale": [0.75, 0.75, 0.65],
        "damage_params": PARAMS["bowl"]
    },
    "coffee_table": {
        "type": "DatasetObject",    
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 1.0, 1.0],
        "damage_params": PARAMS["coffee_table"]
    }
}
def choose_scene():
    """Let user choose a scene."""
    return "empty"

def choose_robot():
    """Let user choose a robot."""
    return "Tiago"

def create_environment(scene_name, robot_name, target_object):
    """Create and configure the environment."""
    # Load Tiago configuration from YAML
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "omnigibson/configs/tiago_primitives.yaml"), 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Modify scene configuration
    cfg["scene"]["type"] = "Scene" if scene_name == "empty" else "InteractiveTraversableScene"
    cfg["scene"]["scene_model"] = None if scene_name == "empty" else scene_name
    
    # Add objects
    objects = [OBJECT_CONFIGS["coffee_table"]]
    if target_object in OBJECT_CONFIGS:
        objects.append(OBJECT_CONFIGS[target_object])
    cfg["objects"] = objects
    
    # Update robot configuration for starter semantic action primitives
    robot_cfg = cfg["robots"][0]
    robot_cfg["position"] = [0., 0.75, 0.0]  # Position next to the table
    robot_cfg["orientation"] = [0, 0, -1, 1]  # Facing the table
    robot_cfg["grasping_mode"] = "assisted"
    robot_cfg["params"] = PARAMS["tiago_robot"]
    
    # Update controller configuration for starter semantic action primitives
    robot_cfg["controller_config"] = {
        "base": {
            "name": "JointController",
            "motor_type": "position",
            "use_delta_commands": False,
            "command_output_limits": None
        },
        "camera": {
            "name": "JointController",
            "motor_type": "position",
            "use_delta_commands": True,
            "command_output_limits": None
        },
        "arm_left": {
            "name": "InverseKinematicsController",
            "mode": "pose_absolute_ori",
            "smoothing_filter_size": 5,
            "use_impedances": True,
            "command_output_limits": None
        },
        "arm_right": {
            "name": "InverseKinematicsController",
            "mode": "pose_absolute_ori",
            "smoothing_filter_size": 5,
            "use_impedances": True,
            "command_output_limits": None
        },
        "gripper_left": {
            "name": "MultiFingerGripperController",
            "mode": "independent",
            "command_output_limits": None
        },
        "gripper_right": {
            "name": "MultiFingerGripperController",
            "mode": "independent",
            "command_output_limits": None
        }
    }
    env = DamageableEnvironment(configs=cfg)
    return env

def setup_robot_controllers(robot):
    """Set up robot controllers."""
    # The controllers are already configured in the environment creation
    # Just make sure the robot is properly initialized
    robot.reset()
    robot.keep_still()

def capture_frame(obs=None, env=None):
    """Capture and process frames from both the camera and robot eyes."""
    frames = {}
    
    # Capture from main camera
    rgb_img = og.sim.viewer_camera.get_obs()[0]["rgb"]
    rgb_img = rgb_img.cpu().numpy()[:, :, :3]
    rgb_img = cv2.resize(rgb_img, (512, 512))
    frames['camera'] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    
    # Capture from robot eyes if obs is provided
    if obs is not None:
        robot_key = next(k for k in obs.keys() if k.startswith('robot_'))
        camera_key = f"{robot_key}:eyes:Camera:0"
        rgb_img = obs[robot_key][camera_key]['rgb']
        rgb_img = rgb_img.cpu().numpy()  # Convert tensor to numpy array
        rgb_img = cv2.resize(rgb_img, (512, 512))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        frames['robot'] = rgb_img
    
    # Capture table and object health
    coffee_table = env.scene.object_registry("name", "coffee_table")
    obj = env.scene.object_registry("name", "target_object")

    frames["table_health"] = coffee_table.health
    frames["table_link_health"] = coffee_table.link_healths.copy()
    frames["table_damage_status"] = coffee_table.damage_status
    frames["obj_link_health"] = obj.link_healths.copy()
    frames["obj_health"] = obj.health
    frames["obj_damage_status"] = obj.damage_status
    
    return frames

def save_video(frames_dict, output_prefix):
    """Save frames as images and videos for both camera and robot views."""
    if not frames_dict:
        return
    height, width = frames_dict["camera"][0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    avi_path1 = f'videos_and_images/{output_prefix}_table.avi'
    mp4_path1 = f'videos_and_images/{output_prefix}_table.mp4'
    out1 = cv2.VideoWriter(avi_path1, fourcc, 30, (width, height))

    avi_path2 = f'videos_and_images/{output_prefix}_object.avi'
    mp4_path2 = f'videos_and_images/{output_prefix}_object.mp4'
    out2 = cv2.VideoWriter(avi_path2, fourcc, 30, (width, height))

    def format_value(val):
        """Helper function to format values to 2 decimal places if they are numbers."""
        try:
            return f"{float(val):.2f}"
        except (ValueError, TypeError):
            return str(val)

    for i in range(len(frames_dict["camera"])):
        frame_copy = frames_dict["camera"][i].copy()
            
        # Add table health captions
        table_frame = frame_copy.copy()
        y_pos = 30
        cv2.putText(table_frame, f"Table Health: {format_value(frames_dict['table_health'][i])}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Handle table link health wrapping
        y_pos += 30
        link_health_text = f"Table Link Health: {', '.join([f'{key}: {format_value(value)}' for key, value in frames_dict['table_link_health'][i].items()])}"
        words = link_health_text.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) * 10 < width - 20:  # Approximate character width
                current_line = test_line
            else:
                cv2.putText(table_frame, current_line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 25
                current_line = word
        if current_line:
            cv2.putText(table_frame, current_line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 25
        
        # Add table damage status after link health
        damage_status = frames_dict['table_damage_status'][i]
        cv2.putText(table_frame, f"Table Damage Status: {damage_status}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out1.write(np.ascontiguousarray(table_frame, dtype=np.uint8))

        # Add object health captions
        obj_frame = frame_copy.copy()
        y_pos = 30
        cv2.putText(obj_frame, f"Object Health: {format_value(frames_dict['obj_health'][i])}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Handle object link health wrapping
        y_pos += 30
        link_health_text = f"Object Link Health: {', '.join([f'{key}: {format_value(value)}' for key, value in frames_dict['obj_link_health'][i].items()])}"
        words = link_health_text.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) * 10 < width - 20:  # Approximate character width
                current_line = test_line
            else:
                cv2.putText(obj_frame, current_line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 25
                current_line = word
        if current_line:
            cv2.putText(obj_frame, current_line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 25
        
        # Add object damage status after link health
        damage_status = frames_dict['obj_damage_status'][i]
        cv2.putText(obj_frame, f"Object Damage Status: {damage_status}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out2.write(np.ascontiguousarray(obj_frame, dtype=np.uint8))

    out1.release()
    out2.release() 

    subprocess.run([
        'ffmpeg', '-y', '-i', avi_path1,
        '-c:v', 'mpeg4', mp4_path1
    ], check=True)
    subprocess.run([
        'ffmpeg', '-y', '-i', avi_path2,
        '-c:v', 'mpeg4', mp4_path2
    ], check=True)
    
    # Clean up AVI files
    os.remove(avi_path1)
    os.remove(avi_path2)
    
def execute_controller(target_eef_pos, target_eef_orn, controller, env, frames, step_limit=200):
    """Execute controller actions."""
    obs = None
    step_count = 0
    for action in controller._move_hand_direct_ik((target_eef_pos, target_eef_orn), stop_on_contact=False, ignore_failure=True):
        if step_count >= step_limit:
            break
        obs, _, _, _, _ = env.step(action)
        frame = capture_frame(obs, env)
        for view_type, view_frame in frame.items():
            frames[view_type].append(view_frame)
        step_count += 1
    
    for _ in range(50):
        og.sim.step()
        if obs is not None:
            frame = capture_frame(obs, env)
            for view_type, view_frame in frame.items():
                frames[view_type].append(view_frame)
        step_count += 1
    
    return frames

def execute_grasp_sequence(env, controller, frames, obj):
    """Execute a grasp sequence with step-by-step movements."""
    print("Executing grasp sequence...")
    robot = env.robots[0]
    obj_pos, obj_orn = obj.get_position_orientation()
    current_pos = robot.get_eef_position()
    current_orn = robot.get_eef_orientation()
    
    # Helper function to check if we reached the target pose
    def check_pose_success(target_pos, threshold=0.02):
        current_pos = robot.get_eef_position()
        distance = th.norm(target_pos - current_pos)
        return distance < threshold
    
    # Step 1: Move to z position above object
    print("Step 1: Moving to z position above object")
    target_pos = th.tensor([current_pos[0], current_pos[1], obj_pos[2] + 0.2], dtype=th.float32)
    frames = execute_controller(target_pos, current_orn, controller, env, frames, step_limit=100)
    print("Step 1: " + ("SUCCESS" if check_pose_success(target_pos) else "FAILURE"))
    
    # Step 2: Move to object's y position
    print("Step 2: Moving to object's y position")
    target_pos = th.tensor([current_pos[0], obj_pos[1], obj_pos[2] + 0.2], dtype=th.float32)
    frames = execute_controller(target_pos, current_orn, controller, env, frames, step_limit=100)
    print("Step 2: " + ("SUCCESS" if check_pose_success(target_pos) else "FAILURE"))
    
    # Step 3: Move to object's x position
    print("Step 3: Moving to object's x position")
    target_pos = th.tensor([obj_pos[0], obj_pos[1], obj_pos[2] + 0.2], dtype=th.float32)
    frames = execute_controller(target_pos, current_orn, controller, env, frames, step_limit=100)
    print("Step 3: " + ("SUCCESS" if check_pose_success(target_pos) else "FAILURE"))
    
    # Step 4: Move down to grasp position
    print("Step 4: Moving down to grasp position")
    target_pos = th.tensor([obj_pos[0], obj_pos[1], obj_pos[2]], dtype=th.float32)
    frames = execute_controller(target_pos, current_orn, controller, env, frames, step_limit=100)
    print("Step 4: " + ("SUCCESS" if check_pose_success(target_pos) else "FAILURE"))
    
    # Close gripper to grasp
    print("Closing gripper to grasp")
    action = controller._empty_action()
    action[robot.controller_action_idx["gripper_left"]] = -1.0  # Close gripper
    for _ in range(50):  # Quick gripper movement
        obs, _, _, _, _ = env.step(action)
        frame = capture_frame(obs, env)
        for view_type, view_frame in frame.items():
            frames[view_type].append(view_frame)
    print("Grasp: " + ("SUCCESS" if robot.get_eef_position()[2] > obj_pos[2] else "FAILURE"))
    
    # Step 5: Lift the object
    print("Step 5: Lifting object")
    target_pos = th.tensor([obj_pos[0], obj_pos[1], obj_pos[2] + 0.2], dtype=th.float32)
    frames = execute_controller(target_pos, current_orn, controller, env, frames, step_limit=100)
    print("Step 5: " + ("SUCCESS" if check_pose_success(target_pos) else "FAILURE"))
    
    # Open gripper to release
    print("Opening gripper to release")
    action = controller._empty_action()
    action[robot.controller_action_idx["gripper_left"]] = 1.0  # Open gripper
    for _ in range(50):  # Quick gripper movement
        obs, _, _, _, _ = env.step(action)
        frame = capture_frame(obs, env)
        for view_type, view_frame in frame.items():
            frames[view_type].append(view_frame)
    print("Release: " + ("SUCCESS" if robot.get_eef_position()[2] > obj_pos[2] else "FAILURE"))
    
    return frames

def execute_random_actions(env, controller, frames, bowl, num_actions=10):
    """Execute random arm movements to make contact with the bowl."""
    print("Executing random actions...")
    robot = env.robots[0]
    bowl_pos, bowl_orn = bowl.get_position_orientation()
    
    # Get current position
    current_pos = robot.get_eef_position()
    current_orn = robot.get_eef_orientation()
    
    # Define bounds for random movements
    x_bounds = (bowl_pos[0] - 0.2, bowl_pos[0] + 0.2)  # 20cm range around bowl
    y_bounds = (bowl_pos[1] - 0.2, bowl_pos[1] + 0.2)
    z_bounds = (bowl_pos[2] - 0.1, bowl_pos[2] + 0.3)  # From below bowl to above it
    
    for i in range(num_actions):
        print(f"Random action {i+1}/{num_actions}")
        
        # Generate random target position within bounds
        target_pos = th.tensor([
            th.rand(1).item() * (x_bounds[1] - x_bounds[0]) + x_bounds[0],
            th.rand(1).item() * (y_bounds[1] - y_bounds[0]) + y_bounds[0],
            th.rand(1).item() * (z_bounds[1] - z_bounds[0]) + z_bounds[0]
        ], dtype=th.float32)
        
        # Keep current orientation
        target_orn = current_orn
        
        # Execute movement
        frames = execute_controller(target_pos, target_orn, controller, env, frames, step_limit=100)
        
        # Randomly open/close gripper
        if th.rand(1).item() < 0.3:  # 30% chance to change gripper state
            action = controller._empty_action()
            action[robot.controller_action_idx["gripper_left"]] = 1.0 if th.rand(1).item() < 0.5 else -1.0
            for _ in range(20):  # Quick gripper movement
                obs, _, _, _, _ = env.step(action)
                frame = capture_frame(obs, env)
                for view_type, view_frame in frame.items():
                    frames[view_type].append(view_frame)
    
    return frames

def main(quickstart=False):
    """Main function to run the safety benchmark."""
    # Initialize environment
    scene_name = "Rs_int" if quickstart else "empty"
    robot_name = "Tiago"
    target_object = "bowl"

    # Create and setup environment
    env = create_environment(scene_name, robot_name, target_object)
    robot = env.robots[0]
    setup_robot_controllers(robot)

    # Initialize simulation
    for _ in range(10):
        og.sim.step()

    # Place object on coffee table
    coffee_table = env.scene.object_registry("name", "coffee_table")
    obj = env.scene.object_registry("name", "target_object")
    
    # First set the OnTop state
    assert obj.states[OnTop].set_value(coffee_table, True), "Failed to set OnTop state"
    
    # Get table top position and place bowl there
    table_pos = coffee_table.get_position()

    # Let physics settle
    for _ in range(20):  # Increased settling steps
        og.sim.step()

    # Initialize frames dictionary
    frames = {'camera': [], 'robot': [], "table_health": [], 
    "table_link_health": [], "table_damage_status": [], 
    "obj_link_health": [], "obj_health": [], "obj_damage_status": []}
    
    # Initialize controller AFTER scene is set up
    controller = StarterSemanticActionPrimitives(env)
    
    # Initialize action space to set up addressable objects
    controller.addressable_objects = sorted(env.scene.object_registry.objects, key=lambda obj: obj.name)

    # Save initial camera observation
    initial_camera_obs = og.sim.viewer_camera.get_obs()[0]["rgb"]
    initial_camera_obs = initial_camera_obs.cpu().numpy()[:, :, :3]
    cv2.imwrite('videos_and_images/initial_camera_obs.png', cv2.cvtColor(initial_camera_obs, cv2.COLOR_RGB2BGR))
    
    # Add breakpoint
    breakpoint()
    
    # Execute random actions instead of grasp sequence
    # frames = execute_random_actions(env, controller, frames, obj, num_actions=10)

    # # Save the video
    # save_video(frames, f'{target_object}_random_actions')

    frames = execute_grasp_sequence(env, controller, frames, obj)
    save_video(frames, f'{target_object}_grasp_sequence')

    # Clean up
    og.clear()
    og.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quickstart", action="store_true", help="Use default settings without user input")
    args = parser.parse_args()
    main(quickstart=args.quickstart) 