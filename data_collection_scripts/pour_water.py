from ast import Pass
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import imageio
import json
import h5py
import pickle
import torch as th
import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.object_states import Filled
# from omnigibson.object_states import IsGrasped
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm

# Commented out - not needed for keyboard teleoperation (causes mediapipe import error)
from telemoma.configs.base_config import teleop_config
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
import omnigibson.lazy as lazy
from omnigibson.controllers.controller_base import IsGraspingState
from omnigibson.utils.vision_utils import segmentation_to_rgb
import omnigibson.utils.transform_utils as T
import cv2
import subprocess


from safety_benchmark.utils.misc_utils import setup_viewport_layout

from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper, DamageableDataPlaybackWrapper
from collections import defaultdict
import math
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for live window
import matplotlib.pyplot as plt


gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False

def get_visualization_config(task_name, robot_name):
    if task_name == "pour_water":
        return {
            "target_objects_health_with_links": [f"{robot_name}@eef_link", f"{robot_name}@panda_hand", f"{robot_name}@panda_leftfinger", f"{robot_name}@panda_rightfinger", "laptop@base_link", "laptop@link_0", "coffee_cup_1@base_link"],
            # TODO: Add water_glass
            "target_objects_health": [robot_name, "laptop", "coffee_cup_1"],
            "target_objects_water_contacts": ["laptop@link_0", "laptop@base_link"],
        }


def setup_live_health_graph(target_objects_health, fps=30):
    """
    Set up a live matplotlib window for real-time health monitoring during teleop.
    
    Args:
        target_objects_health: List of object names to track
        fps: Frames per second for time axis calculation
        
    Returns:
        tuple: (figure, axes, health_lines dict, time_data list)
    """
    plt.ion()  # Enable interactive mode
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title("Live Health Monitor")
    ax.set_title("Health Over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Health", fontsize=12)
    ax.set_ylim(-5.0, 105.0)
    ax.set_xlim(0, 1.0)  # Will auto-expand
    ax.grid(True, alpha=0.3)
    
    # Create lines for each object
    health_lines = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_objects_health)))
    for obj_name, color in zip(target_objects_health, colors):
        line, = ax.plot([], [], label=f"{obj_name} Health", lw=2, color=color)
        health_lines[obj_name] = line
    
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show(block=False)
    
    # Time data will be populated as we go
    time_data = []
    
    return fig, ax, health_lines, time_data


def update_live_health_graph(fig, ax, health_lines, time_data, health_data, target_objects_health, fps=30):
    """
    Update the live health graph with new data.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        health_lines: Dict mapping object names to line objects
        time_data: List of time values (will be updated)
        health_data: Dict mapping object names to lists of health values
        target_objects_health: List of object names to track
        fps: Frames per second for time calculation
        
    Returns:
        bool: True if window is still active, False if closed
    """
    # Check if window is still open
    if not plt.fignum_exists(fig.number):
        return False
    
    # Find the maximum length of health data across all objects
    max_length = max([len(health_data[obj]) for obj in target_objects_health if obj in health_data], default=0)
    
    if max_length == 0:
        return True  # No data yet, but window is still open
    
    # Build time data array based on current data length
    # time_data list is maintained but we regenerate time points to match health_data length
    time_points = [i / fps for i in range(max_length)]
    
    # Update each line
    for obj_name in target_objects_health:
        if obj_name in health_data and len(health_data[obj_name]) > 0:
            # Use time points matching the length of this object's health data
            n_points = len(health_data[obj_name])
            obj_time_points = time_points[:n_points]
            health_lines[obj_name].set_data(obj_time_points, health_data[obj_name])
    
    # Auto-scale x-axis based on current time range
    if max_length > 0:
        max_time = (max_length - 1) / fps
        ax.set_xlim(0, max(max_time * 1.1, 1.0))
    
    # Redraw
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return True


def setup_live_water_contacts_graph(target_objects_water_contacts, fps=30):
    """
    Set up a live matplotlib window for real-time water particle contact monitoring during teleop.
    
    Args:
        target_objects_water_contacts: List of object@link names to track (e.g., ["laptop@link_0", "laptop@base_link"])
        fps: Frames per second for time axis calculation
        
    Returns:
        tuple: (figure, axes, water_lines dict, time_data list)
    """
    plt.ion()  # Enable interactive mode
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title("Live Water Contact Monitor")
    ax.set_title("Water Particle Contacts Over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Particle Count", fontsize=12)
    ax.set_ylim(0, 100)  # Initial scale, will auto-expand
    ax.set_xlim(0, 1.0)  # Will auto-expand
    ax.grid(True, alpha=0.3)
    
    # Create lines for each object@link
    water_lines = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_objects_water_contacts)))
    for obj_link_name, color in zip(target_objects_water_contacts, colors):
        line, = ax.plot([], [], label=f"{obj_link_name}", lw=2, color=color)
        water_lines[obj_link_name] = line
    
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show(block=False)
    
    # Time data will be populated as we go
    time_data = []
    
    return fig, ax, water_lines, time_data


def update_live_water_contacts_graph(fig, ax, water_lines, time_data, water_contacts_data, target_objects_water_contacts, fps=30):
    """
    Update the live water contacts graph with new data.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        water_lines: Dict mapping object@link names to line objects
        time_data: List of time values (will be updated)
        water_contacts_data: Dict mapping object@link names to lists of particle counts
        target_objects_water_contacts: List of object@link names to track
        fps: Frames per second for time calculation
        
    Returns:
        bool: True if window is still active, False if closed
    """
    # Check if window is still open
    if not plt.fignum_exists(fig.number):
        return False
    
    # Find the maximum length of data across all tracked links
    max_length = max([len(water_contacts_data[obj_link]) for obj_link in target_objects_water_contacts if obj_link in water_contacts_data], default=0)
    
    if max_length == 0:
        return True  # No data yet, but window is still open
    
    # Build time data array based on current data length
    time_points = [i / fps for i in range(max_length)]
    
    # Update each line
    max_particle_count = 0
    for obj_link_name in target_objects_water_contacts:
        if obj_link_name in water_contacts_data and len(water_contacts_data[obj_link_name]) > 0:
            n_points = len(water_contacts_data[obj_link_name])
            obj_time_points = time_points[:n_points]
            water_lines[obj_link_name].set_data(obj_time_points, water_contacts_data[obj_link_name])
            # Track max for y-axis scaling
            if len(water_contacts_data[obj_link_name]) > 0:
                max_particle_count = max(max_particle_count, max(water_contacts_data[obj_link_name]))
    
    # Auto-scale x-axis based on current time range
    if max_length > 0:
        max_time = (max_length - 1) / fps
        ax.set_xlim(0, max(max_time * 1.1, 1.0))
    
    # Auto-scale y-axis based on particle counts
    if max_particle_count > 0:
        ax.set_ylim(0, max(max_particle_count * 1.2, 10))
    
    # Redraw
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return True


def get_water_contacts_per_link(tracked_objects, target_objects_water_contacts):
    """
    Get water particle contacts for each tracked object@link.
    
    Args:
        tracked_objects: Dict mapping object names to object instances
        target_objects_water_contacts: List of "object@link" strings to track
        
    Returns:
        Dict mapping "object@link" to particle count
    """
    contacts = {}
    for obj_link_name in target_objects_water_contacts:
        parts = obj_link_name.split("@")
        if len(parts) != 2:
            contacts[obj_link_name] = 0
            continue
        obj_name, link_name = parts
        if obj_name not in tracked_objects:
            contacts[obj_link_name] = 0
            continue
        
        obj = tracked_objects[obj_name]
        # Find electrical damage evaluator for this object
        particle_count = 0
        for evaluator in getattr(obj, 'damage_evaluators', []):
            if evaluator.name == "electrical":
                contact_summary = evaluator.get_contact_summary()
                link_details = contact_summary.get("link_details", {})
                if link_name in link_details:
                    particle_count = link_details[link_name].get("particle_count", 0)
                break
        contacts[obj_link_name] = particle_count
    
    return contacts


def save_camera_video(hdf5_file, output_video_path, robot_name, camera_type, camera_name, target_objects, obs_info_list, health, demo_idx=0):
    """Save RGB camera frames from HDF5 to video file."""
    imgs = hdf5_file[f"data/demo_{demo_idx}/obs/{camera_type}::{camera_name}::rgb"]
    imgs = np.array(imgs[1:])  # Skip first frame, convert to numpy
    
    if len(imgs) == 0:
        print(f"No images found for demo_{demo_idx}")
        return
    
    mp4_video = output_video_path + ".mp4"
    imageio.mimsave(mp4_video, imgs)
    print(f"Saved video: {mp4_video} ({len(imgs)} frames)")


def set_laptop_pose(env, target_deg: float = 130.0):
    """Open the laptop to a specified angle."""
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
            joint.friction = 50000000.0
            if hasattr(joint, "keep_still"):
                joint.keep_still()
    if hasattr(laptop, "keep_still"):
        laptop.keep_still()


LAPTOP_INIT_POS = [6.3, 0.2, 1.3]
LAPTOP_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
COFFEE_CUP_1_INIT_POS = [6.3, 0.0, 1.3]
WATER_GLASS_INIT_POS = [6.3, 0.5, 1.3]

# Task objects are located in BEHAVIOR-1k/datasets/objects/*
TASK_OBJECTS = {
    "laptop": {
        "type": "DatasetObject",
        "name": "laptop",
        "category": "laptop",
        "model": "nvulcs",
        "position": LAPTOP_INIT_POS,
        "orientation": LAPTOP_INIT_ORI,
        "scale": [1.0, 1.0, 1.0],
    },
    "coffee_cup_1": {
        "type": "DatasetObject",
        "name": "coffee_cup_1",
        "category": "coffee_cup",
        "model": "ckkwmj",
        "position": COFFEE_CUP_1_INIT_POS,
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.2, 1.2, 1.2],
    },
    "water_glass": {
        "type": "DatasetObject",
        "name": "water_glass",
        "category": "water_glass",
        "model": "ewgotr",
        "position": WATER_GLASS_INIT_POS,
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.9, 0.9, 0.9],
    },
}

def add_water_to_glass(env, robot):
    robot_joint_positions = th.tensor([ 1.8706, -1.3073, -1.6741, -2.5116,  0.1573,  3.7525, -0.5432,  0.0400, 0.0261])
    inp = input("Fill water glass with water? (y/n)")
    if inp == "n":
        reset_env(env)

    # Fill water glass with water
    water_glass = env.scene.object_registry("name", "water_glass")
    if water_glass is not None:
        water_system = env.scene.get_system("water", force_init=True)
        # Set the Filled state to True
        if Filled in water_glass.states:
            water_glass.states[Filled].set_value(water_system, True)
        # Generate water particles in batches with env.step() for proper simulation
        glass_pos, _ = water_glass.get_position_orientation()
        z_offset = 0.05
        for _ in range(100):
            if isinstance(glass_pos, th.Tensor):
                drop_pos = (glass_pos + th.tensor([0.0, 0.0, z_offset], dtype=th.float32)).tolist()
            else:
                drop_pos = [glass_pos[0], glass_pos[1], glass_pos[2] + z_offset]
            water_system.generate_particles(positions=[drop_pos])
            robot.set_joint_positions(robot_joint_positions)
            robot.set_joint_velocities(th.zeros(robot.n_dof))
            robot.keep_still()
            og.sim.step()
            # env.step(keep_gripper_closed_action)
            # Force restore after EVERY step to guarantee no movement
            robot.set_joint_positions(robot_joint_positions)
            robot.set_joint_velocities(th.zeros(robot.n_dof))
            robot.keep_still()
        print(f"Total particles: {water_system.n_particles}")
    
    # Let simulation settle with env.step() for proper physics
    for _ in range(30):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        og.sim.step()
        # env.step(keep_gripper_closed_action)
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()

    # # If we want water particles to be invisible
    # if "water" in env.scene.systems:
    #     water_system = env.scene.get_system("water")
    #     for prototype in water_system.particle_prototypes: prototype.visible = False
    #     for instancer in water_system.particle_instancers.values(): instancer.visible = False

def reset_env(env):
    env.reset()
    # load state
    with open("resources/saved_states/pour_water_init_state.pkl", "rb") as f:
        state_flat_array = pickle.load(f)
    og.sim.load_state(state_flat_array, serialized=True)
    
    # CRITICAL: Get all original positions IMMEDIATELY after loading state, before any sim steps
    # This prevents NaN quaternion issues from simulation instability
    robot = env.robots[0]
    robot_pos, robot_orn = robot.get_position_orientation()
    robot_joint_positions = robot.get_joint_positions()  # Save joint positions to restore later
    
    water_glass = env.scene.object_registry("name", "water_glass")    
    laptop = env.scene.object_registry("name", "laptop")
    coffee_cup = env.scene.object_registry("name", "coffee_cup_1")

    # Since the saved state has different laptop and coffee cup positions, setting it here
    laptop.set_position_orientation(position=th.tensor([6.2, 0.3, 1.1]))
    coffee_cup.set_position_orientation(position=th.tensor([6.2, 0.5, 1.1]))
    
    # Save original positions from the freshly loaded state (before any sim steps)
    laptop_orig_pos, laptop_orig_orn = None, None
    cup_orig_pos, cup_orig_orn = None, None
    if laptop is not None:
        laptop_orig_pos, laptop_orig_orn = laptop.get_position_orientation()
    if coffee_cup is not None:
        cup_orig_pos, cup_orig_orn = coffee_cup.get_position_orientation()
    
    # Now sync robot and run sim steps
    robot.keep_still()
    for _ in range(10):
        robot.keep_still()
        og.sim.step()
    
    # Open the laptop
    set_laptop_pose(env, target_deg=130.0)
    
    min_distance = 0.15  # Minimum distance to prevent collision    
    # Randomize positions of laptop and coffee_cup_1 using absolute position limits
    # Retry loop: randomize both, check distance, retry if too close
    max_trials = 50
    for trial in range(max_trials):
        for obj in [laptop, coffee_cup]:
            pos, orn = obj.get_position_orientation()
            pos_magnitude = [-0.03, 0.03] 
            rot_magnitude = np.pi / 12 # 15 degrees
            pos_diff_xy = np.random.uniform(pos_magnitude[0], pos_magnitude[1], size=2)
            pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
            new_pos = pos + pos_diff
            orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
            new_orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
            obj.set_position_orientation(new_pos, new_orn)
        
        laptop_new_pos, _ = laptop.get_position_orientation()
        cup_new_pos, _ = coffee_cup.get_position_orientation()
        # Check distance between laptop and coffee cup
        final_distance = None
        laptop_xy = laptop_new_pos[:2].cpu().numpy()
        cup_xy = cup_new_pos[:2].cpu().numpy()
        distance = np.linalg.norm(laptop_xy - cup_xy)
        
        if distance >= min_distance:
            # Distance OK, apply positions and break
            laptop.set_position_orientation(laptop_new_pos, laptop_orig_orn)
            coffee_cup.set_position_orientation(cup_new_pos, cup_orig_orn)
            final_distance = distance
            print(f"Randomization trial {trial}: distance={distance:.3f} >= {min_distance} (OK)")
            break
        else:
            if trial < max_trials - 1:
                print(f"Randomization trial {trial}: distance={distance:.3f} < {min_distance} (retrying...)")
    else:
        # Max trials reached, use last positions anyway
        print(f"Warning: Could not satisfy distance constraint after {max_trials} trials, using last positions")
        laptop.set_position_orientation(laptop_new_pos, laptop_orig_orn)
        coffee_cup.set_position_orientation(cup_new_pos, cup_orig_orn)
        # Calculate distance for scale even if constraint wasn't satisfied
        laptop_xy = laptop_new_pos[:2].cpu().numpy()
        cup_xy = cup_new_pos[:2].cpu().numpy()
        final_distance = np.linalg.norm(laptop_xy - cup_xy)
    
    # # Randomize laptop scale based on distance between objects
    # # Further apart = larger scale allowed (up to 1.2), closer = smaller scale (min 0.9)
    # if laptop is not None:
    #     old_scale = laptop.scale.tolist()
    #     # Dump state AFTER position randomization to preserve new positions
    #     temp_state = og.sim.dump_state(serialized=False)
    #     og.sim.stop()
        
    #     # Calculate max scale based on distance
    #     # Estimate max possible distance: sqrt((x_range)^2 + (y_range)^2)
    #     # x_range = 6.35 - 6.25 = 0.1, y_range = 0.25 - (-0.05) = 0.3
    #     max_possible_distance = np.sqrt(0.1**2 + 0.3**2)  # ~0.316
    #     # Map distance from [min_distance, max_possible_distance] to scale max [0.9, 1.2]
    #     if final_distance is not None:
    #         # Clamp distance to reasonable range
    #         distance_clamped = np.clip(final_distance, min_distance, max_possible_distance)
    #         # Linear interpolation: distance -> max_scale
    #         max_scale = 0.9 + (distance_clamped - min_distance) / (max_possible_distance - min_distance) * (1.1 - 0.9)
    #         max_scale = np.clip(max_scale, 0.9, 1.1)  # Ensure within bounds
    #     else:
    #         max_scale = 1.2  # Default if distance not calculated
        
    #     x_scale_mult = np.random.uniform(0.9, max_scale)
    #     y_scale_mult = np.random.uniform(0.9, max_scale)
    #     z_scale_mult = np.random.uniform(0.9, max_scale)
    #     new_scale = [old_scale[0] * x_scale_mult, old_scale[1] * y_scale_mult, old_scale[2] * z_scale_mult]
    #     laptop.scale = th.tensor(new_scale)
    #     print(f"Randomizing laptop scale: distance={final_distance:.3f}, max_scale={max_scale:.3f}, old={old_scale}, multipliers=[{x_scale_mult:.3f}, {y_scale_mult:.3f}, {z_scale_mult:.3f}], new={new_scale}")
    #     og.sim.play()
    #     og.sim.load_state(temp_state)
    #     robot.keep_still()

    # for _ in range(10):
    #     robot.keep_still()
    #     og.sim.step()
    
    # Sync robot controller state after loading - prevents random movement
    robot.keep_still()
    og.sim.step()
    
    # Ensure gripper is closed to match saved grasping state
    # The saved state should have the grasping constraint, but we need to close the gripper controller
    # Create action to keep gripper closed and arm still
    keep_gripper_closed_action = th.zeros(robot.action_dim)
    keep_gripper_closed_action[robot.gripper_action_idx[robot.default_arm]] = -1.0  # Close gripper
    for _ in range(10):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        og.sim.step()
        # env.step(keep_gripper_closed_action)
        # Force restore after step to prevent any drift
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
    print("Gripper closed to match saved grasping state")

    # set joint positons
    robot_joint_positions = th.tensor([ 1.8706, -1.3073, -1.6741, -2.5116,  0.1573,  3.7525, -0.5432,  0.0400, 0.0261])
    robot.set_joint_positions(robot_joint_positions)
    for _ in range(20): og.sim.step()


def save_camera_images(env, output_dir="safe-manipulation-benchmark/resources/debug_images"):
    """
    Directly capture and save RGB + segmentation images from all cameras.
    """
    os.makedirs(output_dir, exist_ok=True)
    robot = env.robots[0]
    
    # Get robot camera observations
    robot_obs, _ = robot.get_obs()
    for sensor_name, sensor_data in robot_obs.items():
        if isinstance(sensor_data, dict) and "rgb" in sensor_data and "seg_instance" in sensor_data:
            rgb = sensor_data["rgb"]
            seg = sensor_data["seg_instance"]
            
            # Convert to numpy
            rgb_np = rgb.cpu().numpy() if isinstance(rgb, th.Tensor) else np.array(rgb)
            if rgb_np.shape[-1] == 4:
                rgb_np = rgb_np[:, :, :3]
            if rgb_np.dtype != np.uint8:
                rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
            
            # Convert seg to RGB visualization
            seg_tensor = seg if isinstance(seg, th.Tensor) else th.tensor(seg)
            seg_rgb = segmentation_to_rgb(seg_tensor, N=256)
            seg_np = seg_rgb.cpu().numpy() if isinstance(seg_rgb, th.Tensor) else np.array(seg_rgb)
            if seg_np.dtype != np.uint8:
                seg_np = (seg_np * 255).astype(np.uint8) if seg_np.max() <= 1.0 else seg_np.astype(np.uint8)
            
            # Resize if needed and combine side by side
            if rgb_np.shape[:2] != seg_np.shape[:2]:
                seg_np = cv2.resize(seg_np, (rgb_np.shape[1], rgb_np.shape[0]))
            combined = np.concatenate([rgb_np, seg_np], axis=1)
            
            # Save
            safe_name = sensor_name.replace(":", "_")
            filepath = os.path.join(output_dir, f"robot_{safe_name}_rgb_seg.png")
            cv2.imwrite(filepath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print(f"Saved: {filepath}")
    
    # Get external sensor observations
    if hasattr(env, "external_sensors") and env.external_sensors:
        for sensor_name, sensor in env.external_sensors.items():
            sensor_obs, _ = sensor.get_obs()
            if "rgb" in sensor_obs and "seg_instance" in sensor_obs:
                rgb = sensor_obs["rgb"]
                seg = sensor_obs["seg_instance"]
                
                # Convert to numpy
                rgb_np = rgb.cpu().numpy() if isinstance(rgb, th.Tensor) else np.array(rgb)
                if rgb_np.shape[-1] == 4:
                    rgb_np = rgb_np[:, :, :3]
                if rgb_np.dtype != np.uint8:
                    rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
                
                # Convert seg to RGB visualization
                seg_tensor = seg if isinstance(seg, th.Tensor) else th.tensor(seg)
                seg_rgb = segmentation_to_rgb(seg_tensor, N=256)
                seg_np = seg_rgb.cpu().numpy() if isinstance(seg_rgb, th.Tensor) else np.array(seg_rgb)
                if seg_np.dtype != np.uint8:
                    seg_np = (seg_np * 255).astype(np.uint8) if seg_np.max() <= 1.0 else seg_np.astype(np.uint8)
                
                # Resize if needed and combine side by side
                if rgb_np.shape[:2] != seg_np.shape[:2]:
                    seg_np = cv2.resize(seg_np, (rgb_np.shape[1], rgb_np.shape[0]))
                combined = np.concatenate([rgb_np, seg_np], axis=1)
                
                # Save
                filepath = os.path.join(output_dir, f"external_{sensor_name}_rgb_seg.png")
                cv2.imwrite(filepath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                print(f"Saved: {filepath}")
    
    print(f"Images saved to {output_dir}")


def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_hdf5_path', type=str, help='Target hdf5 path', default="resources/teleop_data/pour_water.hdf5")
    parser.add_argument('--playback_hdf5_path', type=str, help='Output hdf5 path', default="resources/playback_data/pour_water_playback.hdf5")
    parser.add_argument('--n_episodes', type=int, help='Number of episodes', default=1)
    parser.add_argument('--teleop', action='store_true', help='Teleoperate the robot')
    parser.add_argument('--playback', action='store_true', help='Playback the data')
    parser.add_argument('--visualize', action='store_true', help='Visualize the data')
    parser.add_argument('--compute_metrics', action='store_true', help='Compute metrics for the data')
    parser.add_argument('--live_feedback', action='store_true', help='Show live health graph window during teleop (use with --teleop)')
    args = parser.parse_args()

    if args.teleop:
        # TODO: Set this
        if not os.path.exists(args.collect_hdf5_path):
            os.makedirs(os.path.dirname(args.collect_hdf5_path), exist_ok=True)

        # Load the pre-selected configuration and set the online_sampling flag
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Overwrite any configs here
        cfg["scene"]["scene_model"] = "house_single_floor"
        cfg["scene"]["not_load_object_categories"] = ["ottoman"]
        cfg["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        
        ############### Franka robot ###############
        # Completely replace robot config to avoid Tiago-specific settings carrying over
        cfg["robots"][0] = {
            "type": "FrankaPanda",
            "name": "franka0",
            "position": [6.8, 0.2, 1.0],  # Match Tiago base position
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "seg_instance", "proprio"],
            "action_normalize": False,
            "self_collisions": True,
            # Franka has single arm (arm_0, gripper_0) instead of left/right
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

        # Generate external sensors config automatically
        # Get robot name and type from config to construct correct prim path
        robot_name = cfg["robots"][0].get("name", "franka0")
        robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()

        # Set external cameras for data collection
        EXTERNAL_CAMERA_CONFIGS = {
            # Side camera (fixed to base_link frame)
            "external_sensor_0": {
                "position": [7.3920, -0.6436, 1.7519],
                "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
                "horizontal_aperture": 15.0,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
            },
            # Left Shoulder (fixed to base_link frame)
            "external_sensor_1": {
                "position": [7.1264, 1.1205, 2.0117],
                "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
                "horizontal_aperture": 15.0,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1",
            },
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
                    "image_height": 256,
                    "image_width": 256,
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
            output_path=args.collect_hdf5_path,
            only_successes=False,
            enable_dump_filters=False,
        )

        robot = env.robots[0]
        # set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([ 7.0659, -0.7141,  1.9185]),
            orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
        )
        for _ in range(10): og.sim.step()

        setup_viewport_layout()

        # # trying
        # eef_vis = create_panda_eef_cylinders(robot, env.scene)
        # robot.links["eef_link"].prim.GetAttribute("visibility").Set("inherited")
        # for _ in range(10): og.sim.render()
        # breakpoint()
        # for geom_list in eef_vis.values():
        #     for geom in geom_list:
        #         geom.prim.GetAttribute("visibility").Set("inherited")
        # for _ in range(10): og.sim.render()
        # breakpoint()


        # Telemoma: Teleoperate robot
        arm_teleop_method = "spacemouse"
        base_teleop_method = "spacemouse"
        # # Franka uses arm_0 instead of arm_left/arm_right
        teleop_config.arm_0_controller = arm_teleop_method
        # # Tiago config (commented out):
        teleop_config.arm_left_controller = arm_teleop_method
        teleop_config.arm_right_controller = arm_teleop_method
        teleop_config.base_controller = base_teleop_method
        teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
        teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.02}
        teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=False)
        teleop_sys.start()
        
        # Keyboard Teleop only (telemoma commented out due to mediapipe issues)
        action_generator = KeyboardRobotController(robot=robot)
        action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=lambda: env.reset(),
        )
        action_generator.print_keyboard_teleop_info()

        # ======================== Health visualization setup (if enabled) ========================
        enable_health_graph = args.live_feedback
        target_objects_water_contacts = ["laptop"]
        tracked_objects = {}
        water_contacts_data = defaultdict(list)  # object_name -> list of total particle counts
        
        if enable_health_graph:
            # Enable health visualization using the general environment method
            env.enable_health_visualization()
            
            # Get references to objects we want to track electrical damage for (for water contacts)
            for obj_name in target_objects_water_contacts:
                obj = env.scene.object_registry("name", obj_name)
                if obj is not None:
                    tracked_objects[obj_name] = obj
                    print(f"Found object '{obj_name}' for water contact tracking")
                else:
                    print(f"Warning: Object '{obj_name}' not found in scene")
        # ====================================================================================

        # Initialize gripper to CLOSED state (-1.0) instead of default open (1.0)
        # This ensures gripper stays closed until user presses T to toggle
        for gripper_name in action_generator.binary_grippers:
            action_generator.gripper_direction[gripper_name] = -1.0
            action_generator.persistent_gripper_action[gripper_name] = -1.0
        

        # ======================== Data collection ========================
        n_episodes = args.n_episodes
        completed_episodes = 0
        while completed_episodes < n_episodes:
            print(f"Episode {completed_episodes} starts (target: {n_episodes})")
            
            # Reset health tracking data
            if enable_health_graph:
                water_contacts_data.clear()
            
            inp = "n"
            while inp != "y":
                reset_env(env)
                inp = input("Fill water glass with water? (y/n)")
            add_water_to_glass(env, robot)
            coffee_cup = env.scene.object_registry("name", "coffee_cup_1")
            water_system = env.scene.get_system("water")
            
            # CRITICAL: Sync robot one more time after reset before teleop starts
            # Zero velocities and reset controller to prevent any residual movement
            robot.set_joint_velocities(th.zeros(robot.n_dof))
            arm_controller = robot.controllers.get(f"arm_{robot.default_arm}")
            if arm_controller is not None:
                arm_controller.reset()
            robot.keep_still()
            og.sim.step()
            robot.set_joint_velocities(th.zeros(robot.n_dof))
            robot.keep_still()
            
            # If the robot is grasping, ensure persistent gripper action is closed
            if robot.is_grasping().value == IsGraspingState.TRUE:
                action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
                action_generator.gripper_direction[action_generator.binary_grippers[0]] = -1.0
            
            action = th.zeros(robot.action_dim)
            action[-1] = -1.0            
            discard_episode = False
            episode_starts = False
            last_telemoma_grip_action = 1.0

            # If we are saving obs directly during teleop, let's set the water particles to not visible
            water_system = env.scene.get_system("water")
            for prototype in water_system.particle_prototypes: prototype.visible = False
            for instancer in water_system.particle_instancers.values(): instancer.visible = False

            # Do a zero action
            obs, reward, terminated, truncated, info = env.step(action)
            for _ in range(30): og.sim.render()
            
            # Print initial water particles in contact with the laptop
            print("Initial water particles in contact with the laptop:")
            print("base_link: ", info["damage_info"]["laptop"]["base_link"]["electrical"]["particle_count"])
            print("link_0: ", info["damage_info"]["laptop"]["link_0"]["electrical"]["particle_count"])

            print("Ready for teleoperation. Press TAB to end episode, BACKSPACE/DELETE to discard and reset.")
            breakpoint()

            env.initialize_env_health()
            
            discard_episode = False
            while True:
                # if step_count % 200 == 0:
                #     print(robot.get_joint_positions(normalized=True))
                telemoma_action = teleop_sys.get_action(teleop_sys.get_obs())
                telemoma_grip_action = telemoma_action[-1]
                if telemoma_grip_action != last_telemoma_grip_action:
                    action[-1] = -action[-1]
                last_telemoma_grip_action = telemoma_grip_action
                action[:-1] = telemoma_action[:-1]
                if not episode_starts:
                    episode_starts = (action[:-1].sum() > 0).item()

                _, keypress_str = action_generator.get_teleop_action()
                
                # TAB: end episode and save
                if keypress_str and keypress_str.upper() == "TAB":
                    print("TAB pressed - ending episode")
                    breakpoint()
                    inp = input("Do you want to save the data? (y/n)")
                    if inp == "y":
                        print("Saving as task success being True")
                        env.task._success = True
                        break
                    else:
                        print("Saving but as task success being False")
                        steps_to_remove = len(env.current_traj_history)
                        env.current_traj_history = []
                        env.step_count -= steps_to_remove
                        print(f"Discarded {steps_to_remove} steps from current trajectory")
                        discard_episode = True
                        break
                
                # BACKSPACE/DELETE: discard current trajectory and reset
                if keypress_str and keypress_str.upper() in ("BACKSPACE", "DEL", "DELETE"):
                    print("BACKSPACE/DELETE pressed - discarding current trajectory and resetting...")
                    breakpoint()
                    # Clear the current trajectory data without saving
                    steps_to_remove = len(env.current_traj_history)
                    env.current_traj_history = []
                    env.step_count -= steps_to_remove
                    print(f"Discarded {steps_to_remove} steps from current trajectory")
                    discard_episode = True
                    # # Reset health tracking data
                    # if enable_health_graph:
                    #     water_contacts_data.clear()
                    #     # Health visualization will be reset on next env.reset()
                    break
                
                if episode_starts:
                    print("action: ", action)
                    print("telemoma_action: ", telemoma_action)
                    obs, reward, terminated, truncated, info = env.step(action.clone())
                
                # Health visualization updates automatically in env.step() if enabled
                # Track water particle contacts via electrical damage evaluators (if needed)
                if enable_health_graph:
                    for obj_name, obj in tracked_objects.items():
                        total_contacts = 0
                        # Find electrical damage evaluator for this object
                        for evaluator in getattr(obj, 'damage_evaluators', []):
                            if evaluator.name == "electrical":
                                # Get contact summary from the evaluator
                                contact_summary = evaluator.get_contact_summary()
                                total_contacts = contact_summary.get("total_contact", 0)
                                break
                        water_contacts_data[obj_name].append(total_contacts)

                # Check success condition
                particles_in_coffee_cup = coffee_cup.states[object_states.ContainedParticles].get_value(system=water_system).n_in_volume
                if particles_in_coffee_cup > 10:
                    print("Coffee cup is filled with water. Success!")
                    env.task._success = True
                    breakpoint()
                    inp = input("Do you want to save the data? (y/n)")
                    if inp == "y":
                        print("Saving as task success being True")
                        break
                    else:
                        print("Discarding current trajectory and resetting...")
                        # Clear the current trajectory data without saving
                        steps_to_remove = len(env.current_traj_history)
                        env.current_traj_history = []
                        env.step_count -= steps_to_remove
                        print(f"Discarded {steps_to_remove} steps from current trajectory")
                        discard_episode = True

            
            # Only count completed episodes (not discarded ones)
            if not discard_episode:
                completed_episodes += 1
                print(f"Episode completed ({completed_episodes}/{n_episodes})")
            else:
                print(f"Episode discarded, redoing...")
        
        env.save_data()
        print("Data saved")
        
        # Close live health visualization if it was enabled
        if enable_health_graph:
            env.disable_health_visualization()
        
    if args.playback:
        robot_name = "franka0"
        robot_type = "FrankaPanda"
        image_height = 256
        image_width = 256
        # Set external cameras for videos
        EXTERNAL_CAMERA_CONFIGS = {
            # Side camera (fixed to base_link frame)
            "external_sensor_0": {
                "position": [7.3920, -0.6436, 1.7519],
                "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
                "horizontal_aperture": 15.0,
                "relative_prim_path": f"/controllable__damageable{robot_type.lower()}__{robot_name}/base_link/external_sensor0",
            },
            # Left Shoulder (fixed to base_link frame)
            "external_sensor_1": {
                "position": [7.1264, 1.1205, 2.0117],
                "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
                "horizontal_aperture": 15.0,
                "relative_prim_path": f"/controllable__damageable{robot_type.lower()}__{robot_name}/base_link/external_sensor1",
            },
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
                    "image_height": image_height,
                    "image_width": image_width,
                    "horizontal_aperture": camera_cfg["horizontal_aperture"],
                },
                "position": th.tensor(position, dtype=th.float32),
                "orientation": th.tensor(orientation, dtype=th.float32),
                "pose_frame": "world",
            })

        # In case we want to modify the robot sensors that were used during data collection
        robot_sensor_config = {
            "VisionSensor": {
                "modalities": ["rgb", "seg_instance"],
                "sensor_kwargs": {
                    "image_height": image_height,
                    "image_width": image_width,
                },
            },
        }
        
        env = DamageableDataPlaybackWrapper.create_from_hdf5(
            input_path=args.collect_hdf5_path,
            output_path=args.playback_hdf5_path,
            robot_obs_modalities=["proprio", "rgb", "seg_instance"],
            robot_sensor_config=robot_sensor_config,
            external_sensors_config=external_sensors_config,
            n_render_iterations=1,
            only_successes=False,
        )

        # set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([ 7.0659, -0.7141,  1.9185]),
            orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
        )
        for _ in range(10): og.sim.step()

        # Playback the dataset
        env.playback_dataset(record_data=True, demo_ids=[1])    
        env.save_data()

    if args.visualize:
        f = h5py.File(args.playback_hdf5_path, "r")
        f_collect = h5py.File(args.collect_hdf5_path, "r")
        scene_file = json.loads(f["data"].attrs["scene_file"])
        robot_name = "franka0"       
        camera_type = "external"
        camera_name = "external_sensor0"

        output_video_dir = "resources/videos/pour_water"
        os.makedirs(output_video_dir, exist_ok=True)
        
        visualization_config = get_visualization_config("pour_water", robot_name)
        target_objects_health_with_links = visualization_config["target_objects_health_with_links"]
        target_objects_health = visualization_config["target_objects_health"]
        target_objects_water_contacts = visualization_config["target_objects_water_contacts"]

        final_obj_healths = defaultdict(list)
        final_env_healths = []
        
        for idx in range(len(f["data"])):
            print("Episode: ", idx)
            demo_idx = int(list(f["data"].keys())[idx].split("_")[-1])
            # Parse info to obtain relevant information for visualization
            obs_info_list = []
            for i in range(len(f[f"data/demo_{demo_idx}/info/obs_info"])):            
                # Obtain observation information 
                obs_info = json.loads(f[f"data/demo_{demo_idx}/info/obs_info"][i].decode("utf-8"))
                obs_info_list.append(obs_info)

            # Obtain health information for the target objects per link
            all_obj_healths = np.array(f[f"data/demo_{demo_idx}/obs/health"])
            health_list_link_names = f_collect[f"data/demo_{demo_idx}"].attrs["health_list_link_names"]
            
            health = dict()
            for obj_name in target_objects_health_with_links:
                health[obj_name] = all_obj_healths[:, np.where(health_list_link_names == obj_name)[0][0]]
                health[obj_name] = health[obj_name][1:]
            breakpoint()

            # Obtain health information for the entire target objects 
            for obj_name in target_objects_health:
                arrays = [v for k, v in health.items() if k.startswith(f"{obj_name}@")]
                # Compute element-wise min
                if arrays:
                    health[obj_name] = np.minimum.reduce(arrays)
                else:
                    health[obj_name] = None

            if args.compute_metrics:
                print("Episode: ", demo_idx)
                current_env_health = 0.0
                for obj_name in target_objects_health:
                    final_obj_healths[obj_name].append(health[obj_name][-1])
                    print(f"{obj_name} health: {health[obj_name][-1]}")
                    current_env_health += health[obj_name][-1]
                final_env_healths.append(current_env_health / len(target_objects_health))
                print(f"Current environment health: ", final_env_healths[-1])

            if args.visualize:
                # Save video for rgb camera
                output_video_path = f"{output_video_dir}/demo_{demo_idx}_camera_video"
                imgs = f[f"data/demo_{demo_idx}/obs/{camera_type}::{camera_name}::rgb"]
                imgs = imgs[1:]
                new_imgs = []
                imgs_seg_instance = f[f"data/demo_{demo_idx}/obs/{camera_type}::{camera_name}::seg_instance"]
                imgs_seg_instance = imgs_seg_instance[1:]

                for i, img in enumerate(imgs):
                    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
                    img_seg_instance = imgs_seg_instance[i]
                    obs_info = obs_info_list[i]
                    for obj_name in target_objects_health:
                        seg_instance_info = obs_info[camera_type][camera_name]["seg_instance"]
                        seg_instance_key = int(next((k for k, v in seg_instance_info.items() if v == obj_name), -1))

                        # # If binary visualization, set to red if 0
                        # if health[obj_name][i] is not None and health[obj_name][i] == 0.0:
                        #     mask = img_seg_instance == seg_instance_key
                        #     overlay_color = np.array([0, 0, 255], dtype=np.uint8)  # BGR
                        #     img[mask] = overlay_color

                        # If continuous visualization, set to different shades of red if < 100
                        if health[obj_name][i] is not None and health[obj_name][i] < 100:
                            mask = img_seg_instance == seg_instance_key
                            alpha = 1 - health[obj_name][i] / 100.0  # 0 = full health, 1 = dead
                            overlay_color = np.array([0, 0, 255], dtype=np.uint8)  # BGR
                            img[mask] = ((1 - alpha) * img[mask] + alpha * overlay_color).astype(np.uint8)
                    
                    new_imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                imgs = np.array(new_imgs)
                # save_rgb_camera_video(output_video_path=output_video_path, imgs=imgs)

                data = dict()
                for obj_name in target_objects_water_contacts:
                    data[obj_name] = dict()
                    data[obj_name] = []
                for i in range(len(f[f"data/demo_{demo_idx}/info/damage_info"])):
                    damage_info = json.loads(f[f"data/demo_{demo_idx}/info/damage_info"][i].decode("utf-8"))
                    for obj_name in target_objects_water_contacts:
                        data[obj_name].append(damage_info[obj_name.split("@")[0]][obj_name.split("@")[1]]["electrical"]["particle_count"])
  
                # breakpoint()
                # Save video for water contacts plot
                water_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_water_contacts_video.mp4")
                save_rgb_water_contacts_video(output_video_path=water_video_path, imgs=imgs, target_objects=target_objects_water_contacts, water_contacts=data, fps=30)
            
                # Save video for health plot
                health_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_health_video.mp4")
                save_rgb_health_video(output_video_path=health_video_path, imgs=imgs, target_objects=target_objects_health, health=health)
        
    og.shutdown()


if __name__ == "__main__":
    __main__()
