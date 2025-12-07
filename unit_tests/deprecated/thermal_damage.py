"""
Thermal damage test scene:
- Loads a stove, frying pan, and egg
- Tracks health and temperature over time
- Generates plots and videos similar to mech_damage.py
- No robot - just thermal simulation
"""

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson import object_states
from omnigibson.object_states import OnTop
from omnigibson.utils.ui_utils import CameraMover
import torch as th
import os
import cv2
import numpy as np
import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from safety_benchmark.damageable_env import DamageableEnvironment

# Performance knobs similar to other scripts
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True

# Thermal damage parameters - using correct parameter names from ThermalDamageEvaluator
PARAMS = {
    "egg": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "damage_threshold": 60.0,  # Start taking damage at 60°C
            "scale": 0.01,  # Damage scale factor
        }
    },
    "pan": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "damage_threshold": 150.0,  # Higher threshold for metal pan
            "scale": 0.001,  # Slower damage rate for metal
        }
    },
    "stove": {
        "damage_evaluators": [],
        "health_thresholds": [90.0, 60.0, 30.0],
    },
    "apple": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "damage_threshold": 50.0,  # Lower threshold for fruit
            "scale": 0.02,  # Higher damage rate for fruit
        }
    },
    "bread": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "damage_threshold": 40.0,  # Very low threshold for bread
            "scale": 0.05,  # High damage rate for bread
        }
    },
    "toy": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "damage_threshold": 100.0,  # Moderate threshold for plastic toy
            "scale": 0.005,  # Moderate damage rate for plastic
        }
    },
    "fork": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "damage_threshold": 100.0,
            "scale": 0.0,
        }
    },
    "default": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "damage_threshold": 80.0,  # Default threshold
            "scale": 0.01,  # Default scale
        }
    },
}

# Target object templates for easy integration of new targets
TARGET_OBJECT_CONFIGS = {
    # Default egg used in current unit test
    "egg": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "egg",
        "model": "brkitw",
        "bounding_box": [0.07, 0.06, 0.06],
        "position": [0.0, -1.0, 0.85],  # On top of pan
        "initial_state": {
            "OnTop": "pan",
            "temperature": 20.0,  # Room temperature
        },
        "damage_params": PARAMS["egg"],
    },
    # Apple alternative
    "apple": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "apple",
        "model": "brkitw",  # Using same model as egg for now
        "bounding_box": [0.08, 0.08, 0.08],
        "position": [0.0, -1.0, 0.85],
        "initial_state": {
            "OnTop": "pan",
            "temperature": 20.0,
        },
        "damage_params": PARAMS["apple"],
    },
    # Bread alternative
    "bread": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "bread",
        "model": "brkitw",  # Using same model as egg for now
        "bounding_box": [0.12, 0.08, 0.04],
        "position": [0.0, -1.0, 0.85],
        "initial_state": {
            "OnTop": "pan",
            "temperature": 20.0,
        },
        "damage_params": PARAMS["bread"],
    },
    # Fork alternative (tablefork-pqrtkc)
    "fork": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "tablefork",
        "model": "pqrtkc",
        "bounding_box": [0.20, 0.03, 0.02],
        "position": [0.0, -1.0, 0.85],
        "initial_state": {
            "OnTop": "pan",
            "temperature": 20.0,
        },
        "abilities": {
            "heatable": {}
        },
        "damage_params": PARAMS["fork"],
    },
    # Toy from BEHAVIOR dataset
    "toy": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "toy_figure",
        "model": "txgpxx",  # BEHAVIOR dataset model
        "bounding_box": [0.15, 0.09, 0.08],  # From BEHAVIOR object page
        "position": [0.0, -1.0, 0.85],
        "initial_state": {
            "OnTop": "pan",
            "temperature": 20.0,
        },
        # Ensure Temperature state exists via dependency (Heated depends on Temperature)
        "abilities": {
            "heatable": {}
        },
        "damage_params": PARAMS["toy"],
    },
}

def main():
    og.log.info(
        f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80
        if main.__doc__
        else f"Demo {__file__}"
    )

    # Easy integration knobs - Choose target object template key from TARGET_OBJECT_CONFIGS
    target_object_key = "toy"  # Change this to switch between "egg", "apple", "bread", "toy", "fork", or None for robot-only
    
    # Derived names and paths based on target object
    base_name = target_object_key if target_object_key is not None else "robot_only"

    # Scene configuration - Rs_int background (minimal change)
    scene_cfg = {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "include_robots": False,
        "load_task_relevant_only": True,
    }
    
    # Robot configuration
    robot0_cfg = {
        "type": "Tiago",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        "position": [0.0, 0.0, 0.0],  # Positioned to view the stove
        "orientation": [0, 0, -1, 1],  # Facing the stove
        "grasping_mode": "assisted",
        "damage_params": {
            "damage_evaluators": ["mechanical"],
            "health_thresholds": [90.0, 60.0, 30.0],
            "mechanical": {
                "impact_threshold": 30.0,
                "impact_scale": 0.001,
                "crushing_threshold": 10.0,
                "crushing_scale": 0.001,
            }
        },
    }
    
    # Base objects (stove and pan)
    objects = [
        {
            "type": "DatasetObject",
            "name": "stove",
            "category": "stove",
            "model": "yhjzwg",
            "position": [0.0, -1.0, 0.69],  # Match Rs_int floor height to avoid popping
            "orientation": [0, 0, 0.7071068, 0.7071068],  # Same orientation as coffee table
            "abilities": {
                "heatSource": {
                    "temperature": 200.0,
                    "heating_rate": 0.1,  # Much slower heating
                    "distance_threshold": 0.2,
                    "requires_toggled_on": True
                },
                "toggleable": {},
            },
            "initial_state": {
                "joints": {"door": 0.0},  # 0.0 = closed, 1.0 = open
                "toggleable": True,  # Allow stove to be turned on/off
                "temperature": 200.0,  # Set initial temperature for heating
            },
            "damage_params": PARAMS["stove"],
        },
        {
            "type": "DatasetObject",
            "name": "pan",
            "category": "frying_pan",
            "model": "aewpzn",
            "scale": [0.9, 0.9, 0.9],  # Slightly smaller pan for better fit on burner
            "position": [0.2, -1.1, 0.8],  # Position on the active burning stove ring
            "initial_state": {
                "OnTop": "stove"  # Ensure pan is on top of the stove
            },
            "damage_params": PARAMS["pan"],
        }
    ]

    # Add target object if specified
    if target_object_key is not None:
        if target_object_key not in TARGET_OBJECT_CONFIGS:
            print(f"⚠️ Unknown target object '{target_object_key}', using 'egg'")
            target_object_key = "egg"
        
        target_obj = TARGET_OBJECT_CONFIGS[target_object_key].copy()
        objects.append(target_obj)

    # Compile config (objects still explicitly added; Rs_int only changes background)
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg], objects=objects)

    # Create the environment
    env = DamageableEnvironment(configs=cfg, debug_physics_frequency=True)

    # Setup robot controllers
    robot = env.robots[0]
    controller_choices = {
        "base": "HolonomicBaseJointController",
        "arm_left": "InverseKinematicsController",
        "arm_right": "InverseKinematicsController",
        "gripper_left": "MultiFingerGripperController",
        "gripper_right": "MultiFingerGripperController",
        "camera": "JointController",
        "trunk": "JointController",
    }
    controller_config = {comp: {"name": name} for comp, name in controller_choices.items()}
    controller_config["gripper_left"]["inverted"] = True
    controller_config["gripper_right"]["inverted"] = True
    robot.reload_controllers(controller_config=controller_config)

    # Persist initial state after controller reload
    env.scene.update_initial_file()

    # Reset environment and robot first
    env.reset()
    robot.reset()

    # Get references to objects after reset
    stove = env.scene.object_registry("name", "stove")
    pan = env.scene.object_registry("name", "pan")
    
    # Get target object reference (only if target_object_key is not None)
    target_obj = None
    if target_object_key is not None:
        target_obj = env.scene.object_registry("name", "target_object")

    # Ensure placements conform and match empty-scene burner location
    if pan is not None and stove is not None:
        pan.states[OnTop].set_value(stove, True)
        pan.set_position([0.19, -1.1, 0.8])
        pan.keep_still()
    if target_obj is not None and pan is not None:
        target_obj.states[OnTop].set_value(pan, True)
        target_obj.set_position([0.21, -1.1, 0.85])
        target_obj.keep_still()

    # Turn on the stove
    stove.states[object_states.ToggledOn].set_value(True)

    # Ensure the oven door is closed (some datasets default to open)
    if object_states.Open in stove.states:
        stove.states[object_states.Open].set_value(False)
    else:
        # Fallback: explicitly set any joint containing "door" or "oven" to zero position
        for jname, joint in stove.joints.items():
            if "door" in jname.lower() or "oven" in jname.lower():
                joint.set_joint_position(0.0)

    # Set initial temperature of the target object to room temperature (20°C)
    if target_obj is not None and (object_states.Temperature in target_obj.states):
        target_obj.states[object_states.Temperature].set_value(20.0)

    # Let physics settle
    for _ in range(50):
        og.sim.step()

    # Add physics damping to prevent objects from flying away
    for obj in env.scene.objects:
        if hasattr(obj, 'solver_position_iteration_count'):
            obj.solver_position_iteration_count = 8
        if hasattr(obj, 'solver_velocity_iteration_count'):
            obj.solver_velocity_iteration_count = 1

    # Setup camera controls
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

    # Set TIAGo head link poses at start
    robot.set_joint_positions(th.tensor([0.0, -0.5]), indices=robot.camera_control_idx)

    # Register TAB key to print camera pose and breakpoint
    def breakpoint_and_print_poses():
        print("=== TAB callback triggered ===", flush=True)
        
        # Viewer camera pose
        cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
        print("=== Viewer Camera Pose ===")
        print(f"World  pos: {cam_pos.tolist()}  quat: {cam_quat.tolist()}")

        # Save current scene config/state
        save_dir = "safe-manipulation-benchmark/unit_tests"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "unit_test_thermal_temp.json")
        og.sim.save([save_path])
        print(f"✅ Saved scene to: {save_path}")

        breakpoint()

    # Create a dummy action generator for keyboard controls
    from omnigibson.utils.ui_utils import KeyboardRobotController
    action_generator = KeyboardRobotController(robot=robot)
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Print camera pose and breakpoint",
        callback_fn=breakpoint_and_print_poses,
    )

    # Set starting camera pose - updated from latest TAB callback
    start_cam_pos = th.tensor([1.1924761533737183, -1.0572415590286255, 1.2996164560317993])
    start_cam_quat = th.tensor([0.4386861324310303, 0.3979824185371399, 0.5413640141487122, 0.5967323184013367])
    og.sim.viewer_camera.set_position_orientation(position=start_cam_pos, orientation=start_cam_quat)

    print("Running thermal damage simulation. Press ESC to quit.")
    print("Press TAB to print camera pose and breakpoint.")

    # Simulation parameters
    max_steps = 500
    step = 0
    fps = 10
    frames = []
    
    # Ego camera recording
    frames_ego = []
    ego_key = None
    
    # Health and temperature tracking - only robot and target object
    robot_healths = []
    target_healths = []
    robot_temperatures = []
    target_temperatures = []
    
    # Damage status tracking
    target_statuses = []

    while step != max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1

        # Capture viewer camera RGB
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        rgb_np = cv2.resize(rgb_np, (1080, 720))
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Capture robot ego camera RGB
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
            # Dynamic colored border based on current damage status
            if target_obj is not None:
                status = str(target_obj.damage_status).lower()
                if status in ("major", "critical"):
                    bgr = (0, 0, 255)  # Red border
                elif status in ("minor",):
                    bgr = (0, 255, 255)  # Yellow border
                else:
                    bgr = (0, 255, 0)  # Green border
            else:
                bgr = (0, 255, 0)
            border = 30
            ego_bgr = cv2.copyMakeBorder(ego_bgr, border, border, border, border, cv2.BORDER_CONSTANT, value=bgr)
            frames_ego.append(ego_bgr)

        # Record health and temperature - only robot and target object
        robot_health = float(getattr(robot, "health", 100.0))
        target_health = float(getattr(target_obj, "health", 100.0)) if target_obj is not None else 100.0
        
        # Get temperatures - only track for objects that have Temperature state
        robot_temp = 20.0  # Robot doesn't have temperature, use room temp
        target_temp = float(target_obj.states[object_states.Temperature].get_value()) if target_obj is not None else 20.0
        
        robot_healths.append(robot_health)
        target_healths.append(target_health)
        robot_temperatures.append(robot_temp)
        target_temperatures.append(target_temp)
        
        # Record damage status
        target_statuses.append(getattr(target_obj, "damage_status", "healthy") if target_obj is not None else "healthy")

        # Print progress every 100 steps
        if step % 100 == 0:
            if target_object_key is not None:
                print(f"Step {step}: Robot health={robot_health:.1f}%, {target_object_key.capitalize()} temp={target_temp:.1f}°C, health={target_health:.1f}%")
            else:
                print(f"Step {step}: Robot health={robot_health:.1f}%")

    # Clean shutdown
    camera_mover.clear()

    # Create videos directory
    videos_dir = os.path.join("safe-manipulation-benchmark", "unit_tests", "videos")
    os.makedirs(videos_dir, exist_ok=True)

    # Write main camera video
    avi_path = os.path.join(videos_dir, "thermal_camera_obs.avi")
    mp4_path = os.path.join(videos_dir, "thermal_camera_obs.mp4")
    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()
        # Convert AVI to MP4
        subprocess.run(["ffmpeg", "-y", "-i", avi_path, "-c:v", "mpeg4", mp4_path], check=True)
        os.remove(avi_path)

    # Write ego camera video
    avi_ego = os.path.join(videos_dir, "thermal_ego_obs.avi")
    mp4_ego = os.path.join(videos_dir, "thermal_ego_obs.mp4")
    if len(frames_ego) > 0:
        he, we = frames_ego[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw_e = cv2.VideoWriter(avi_ego, fourcc, fps, (we, he))
        for f in frames_ego:
            vw_e.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw_e.release()
        subprocess.run(["ffmpeg", "-y", "-i", avi_ego, "-c:v", "mpeg4", mp4_ego], check=True)
        os.remove(avi_ego)

    # Create health plot animation - like mech_damage.py
    health_mp4 = os.path.join(videos_dir, f'{base_name}_healths.mp4')
    if len(robot_healths) > 0:
        T = max(len(robot_healths), len(target_healths))
        y_min = 0.0
        y_max = 100.0

        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_robot, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
        if target_object_key is not None:
            line_target, = ax.plot([], [], lw=6, color='tab:orange', label=target_object_key.capitalize())
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Health', fontsize=20)
        ax.set_title('Health Over Time', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_health():
            line_robot.set_data([], [])
            if target_object_key is not None:
                line_target.set_data([], [])
            if target_object_key is not None:
                return line_robot, line_target
            else:
                return line_robot,

        def animate_health(i):
            x = [k / fps for k in range(1, i + 2)]
            y_robot = robot_healths[: i + 1]
            y_target = target_healths[: i + 1] if target_object_key is not None else []
            line_robot.set_data(x, y_robot)
            if target_object_key is not None:
                line_target.set_data(x, y_target)
            if target_object_key is not None:
                return line_robot, line_target
            else:
                return line_robot,

        ani = animation.FuncAnimation(
            fig, animate_health, init_func=init_health, frames=T, interval=1000 / fps, blit=True
        )
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)

    # Create temperature plot animation - like electrical_damage.py water contacts
    temp_mp4 = os.path.join(videos_dir, f'{base_name}_temperatures.mp4')
    if len(robot_temperatures) > 0 or len(target_temperatures) > 0:
        T = max(len(robot_temperatures), len(target_temperatures))
        all_temps = robot_temperatures + target_temperatures
        y_min = min(all_temps) if all_temps else 0
        y_max = max(all_temps) * 1.1 if all_temps else 100

        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_robot, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
        if target_object_key is not None:
            line_target, = ax.plot([], [], lw=6, color='tab:orange', label=target_object_key.capitalize())
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Temperature (°C)', fontsize=20)
        ax.set_title('Temperature Over Time', fontsize=26)
        # Add dotted damage threshold line for target object temperature only
        if target_object_key is not None and target_object_key in PARAMS:
            thr = PARAMS[target_object_key].get("thermal", {}).get("damage_threshold", None)
            if thr is not None:
                ax.axhline(y=thr, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Damage Threshold')
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_temp():
            line_robot.set_data([], [])
            if target_object_key is not None:
                line_target.set_data([], [])
            if target_object_key is not None:
                return line_robot, line_target
            else:
                return line_robot,

        def animate_temp(i):
            x = [k / fps for k in range(1, i + 2)]
            y_robot = robot_temperatures[: i + 1]
            y_target = target_temperatures[: i + 1] if target_object_key is not None else []
            line_robot.set_data(x, y_robot)
            if target_object_key is not None:
                line_target.set_data(x, y_target)
            if target_object_key is not None:
                return line_robot, line_target
            else:
                return line_robot,

        ani = animation.FuncAnimation(
            fig, animate_temp, init_func=init_temp, frames=T, interval=1000 / fps, blit=True
        )
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(temp_mp4, writer=writer)
        plt.close(fig)

    # Create camera with ego overlay
    cam_with_ego_mp4 = os.path.join(videos_dir, f'{base_name}_camera_with_ego.mp4')
    if os.path.exists(mp4_path) and os.path.exists(mp4_ego):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', mp4_path,
            '-i', mp4_ego,
            '-filter_complex',
            '[0:v]setsar=1[base];[1:v]scale=360:240,setsar=1[ego];[base][ego]overlay=W-w-20:20[out]',
            '-map','[out]',
            '-c:v','mpeg4','-q:v','5', cam_with_ego_mp4
        ], check=True)

    # Create health video (camera + health plot) - like mech_damage.py
    combined_health_mp4 = os.path.join(videos_dir, f'{base_name}_with_health.mp4')
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

    # Create temperature video (camera + temperature plot) - like mech_damage.py
    combined_temp_mp4 = os.path.join(videos_dir, f'{base_name}_with_temperature.mp4')
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

    # Clean up intermediate videos
    print("Cleaning up intermediate videos...")
    videos_to_cleanup = [
        mp4_path,  # {base_name}_camera_obs.mp4
        mp4_ego,   # {base_name}_ego_obs.mp4
        cam_with_ego_mp4,  # {base_name}_camera_with_ego.mp4
        health_mp4,  # {base_name}_healths.mp4
        temp_mp4,  # {base_name}_temperatures.mp4
    ]
    
    for video_path in videos_to_cleanup:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Deleted: {os.path.basename(video_path)}")
            except OSError as e:
                print(f"Warning: Could not delete {video_path}: {e}")
    
    print(f"Final videos kept:")
    if os.path.exists(combined_health_mp4):
        print(f"  - {os.path.basename(combined_health_mp4)}")
    if os.path.exists(combined_temp_mp4):
        print(f"  - {os.path.basename(combined_temp_mp4)}")
    
    # Print final statistics
    print("\n=== Final Statistics ===")
    print(f"Robot: Final health={robot_healths[-1]:.1f}%")
    if target_object_key is not None:
        print(f"{target_object_key.capitalize()}: Final temp={target_temperatures[-1]:.1f}°C, Final health={target_healths[-1]:.1f}%")
    
    try:
        og.clear()
    except Exception as e:
        print(f"Warning during og.clear(): {e}")
    og.shutdown()


if __name__ == "__main__":
    main()
