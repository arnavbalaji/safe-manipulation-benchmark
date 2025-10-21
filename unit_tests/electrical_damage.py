import os
import sys
import yaml
import json
import cv2
import numpy as np
import subprocess

import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.object_states import ToggledOn, Inside, ContactParticles
from omnigibson.utils.constants import ParticleModifyCondition
from omnigibson.utils.ui_utils import KeyboardRobotController

from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.damage_evaluators.electrical_damage_evaluator import ElectricalDamageEvaluator

# Force headless plotting to avoid Qt/xcb issues
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")

# Try to import ParticleSource (may not be available in all OmniGibson versions)
try:
    from omnigibson.object_states import ParticleSource  # type: ignore
except Exception:
    ParticleSource = None

# Configure OmniGibson for stable simulation
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_HQ_RENDERING = False


# Electrical damage parameters (similar structure to mech_damage.py)
PARAMS = {
    "tiago_robot": {
        "damage_evaluators": ["electrical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "electrical": {
            "damage_threshold": 0.0,
            "scale": 0.0005,
            "water_system_name": "water",
            "link_thresholds": {},
        },
    },
    "laptop": {
        "damage_evaluators": ["electrical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "electrical": {
            "damage_threshold": 700.0,
            "scale": 5.0,
            "water_system_name": "water",
            "link_thresholds": {},
        },
    },
    "toy_car": {
        "damage_evaluators": ["electrical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "electrical": {
            "damage_threshold": 0.0,
            "scale": 0.0,  # No damage from water
            "water_system_name": "water",
            "link_thresholds": {},
        },
    },
    "default": {
        "damage_evaluators": ["electrical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "electrical": {
            "damage_threshold": 0.0,
            "scale": 0.001,
            "water_system_name": "water",
            "link_thresholds": {},
        },
    },
}

# Target object templates for easy integration of new targets
TARGET_OBJECT_CONFIGS = {
    # Default laptop used in current unit test
    "laptop": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "laptop",
        "model": "nvulcs",
        "position": [0.35, 0.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
        "damage_params": PARAMS["laptop"],
    },
    # Toy car from BEHAVIOR dataset
    "toy_car": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "toy_car",
        "model": "nhtywr",
        "position": [0.25, 0.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 0.5],
        "damage_params": PARAMS["toy_car"],
    },
}

def create_scene_config(target_object_key="laptop"):
    """Create the scene configuration with robot, sink, and target object."""
    # Configure TIAGo robot
    robot_cfg = {
        "type": "Tiago",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        "grasping_mode": "assisted",
        "damage_params": PARAMS["tiago_robot"],
        "position": [0.0, 1.0, 0.0],
        "orientation": [0, 0, -0.7071068, 0.7071068],
    }

    # Configure commercial kitchen sink with water particle system
    sink_obj = {
        "type": "DatasetObject",
        "name": "commercial_kitchen_sink",
        "category": "commercial_kitchen_sink",
        "model": "xecfyh",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 0.6],
        "abilities": {
            "toggleable": {},
            "particleSource": {
                "conditions": {"water": []},  # Always on to avoid CUDA errors
                "initial_speed": 0.2,  # Low speed for stability
            },
            "particleSink": {"conditions": {"water": []}}
        }
    }

    # Get target object configuration
    objects = [sink_obj]
    if target_object_key is not None:
        if target_object_key not in TARGET_OBJECT_CONFIGS:
            print(f"⚠️ Unknown target object '{target_object_key}', using 'laptop'")
            target_object_key = "laptop"
        
        target_obj = TARGET_OBJECT_CONFIGS[target_object_key].copy()
        objects.append(target_obj)
    
    # Return complete scene configuration
    return {
        "scene": {"type": "Scene", "scene_id": "empty"},
        "robots": [robot_cfg],
        "objects": objects,
    }


def setup_robot_controllers(robot):
    """Configure robot controllers for teleoperation."""
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
    # TIAGo grippers are inverted
    controller_config["gripper_left"]["inverted"] = True
    controller_config["gripper_right"]["inverted"] = True
    
    robot.reload_controllers(controller_config=controller_config)


def setup_camera_controls():
    """Setup camera movement controls."""
    from omnigibson.utils.ui_utils import CameraMover
    
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
    return camera_mover


def setup_keyboard_controls(teleop, env, sink, water_system):
    """Setup keyboard controls for robot and sink."""
    # Reset key (R)
    teleop.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # TAB key to print camera pose and save scene
    def breakpoint_and_print_poses():
        print("=== TAB callback triggered ===", flush=True)
        
        # Viewer camera pose
        cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
        print("=== Viewer Camera Pose ===")
        print(f"World  pos: {cam_pos.tolist()}  quat: {cam_quat.tolist()}")

        # Save current scene config/state
        save_dir = "safe-manipulation-benchmark/unit_tests"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "unit_test_electrical_temp.json")
        og.sim.save([save_path])
        print(f"✅ Saved scene to: {save_path}")
        breakpoint()

    teleop.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Print camera pose and save scene",
        callback_fn=breakpoint_and_print_poses,
    )

    # Sink toggle key (Z)
    def toggle_sink():
        if sink is not None and hasattr(sink, 'states'):
            if ToggledOn in sink.states:
                current_state = sink.states[ToggledOn].get_value()
                new_state = not current_state
                try:
                    sink.states[ToggledOn].set_value(new_state)
                    status = "ON" if new_state else "OFF"
                    print(f"🚰 Sink turned {status}")
                    
                    # Check if water system is available and report particle count
                    if water_system is not None:
                        particle_count = water_system.n_particles
                        print(f"💧 Water particles in system: {particle_count}")
                    else:
                        print("⚠️ No water system detected - water may not flow")
                        
                except Exception as e:
                    print(f"⚠️ Failed to toggle sink: {e}")
            else:
                print("⚠️ Sink does not have ToggledOn state")
        else:
            print("⚠️ Sink not available for toggling")

    teleop.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.Z,
        description="Toggle sink on/off",
        callback_fn=toggle_sink,
    )


def main():
    """Main function: setup environment, run teleoperation, and generate videos."""
    # Easy integration knobs - Choose target object template key from TARGET_OBJECT_CONFIGS
    target_object_key = "laptop"  # Change this to switch between "laptop", "toy_car", or None for robot-only
    
    # Derived names and paths based on target object
    base_name = target_object_key if target_object_key is not None else "robot_only"
    
    # Create scene configuration
    cfg = create_scene_config(target_object_key)

    # Create environment and setup robot
    env = DamageableEnvironment(configs=cfg)
    robot = env.robots[0]
    setup_robot_controllers(robot)

    # Persist initial state and reset
    env.scene.update_initial_file()
    env.reset()
    robot.reset()

    # Set TIAGo head link poses at start (world frame) - slight tilt like mech_damage.py
    robot.set_joint_positions(th.tensor([0.0, -0.75]), indices=robot.camera_control_idx)

    # Get references to scene objects
    sink = env.scene.object_registry("name", "commercial_kitchen_sink")
    print("✅ Commercial kitchen sink loaded successfully")
    
    # Check sink states and abilities
    if hasattr(sink, 'states'):
        print(f"🔍 Sink states: {list(sink.states.keys())}")
        if hasattr(sink, 'abilities'):
            print(f"🔍 Sink abilities: {list(sink.abilities.keys())}")
    
    # Set initial sink state (OFF by default)
    if hasattr(sink, 'states') and ToggledOn in sink.states:
        sink.states[ToggledOn].set_value(False)
        print("🚰 Sink initialized to OFF state")
    else:
        print("⚠️ Sink does not have ToggledOn state")

    # Ensure sink is ON for always-on particle source
    if sink is not None and ToggledOn in sink.states:
        sink.states[ToggledOn].set_value(True)

    # Position robot near sink
    if sink is not None:
        sink_pos, _ = sink.get_position_orientation()
        new_pos = [float(sink_pos[0]), float(sink_pos[1]) + 1.0, 0.0]
        robot.set_position_orientation(
            position=th.tensor(new_pos), 
            orientation=th.tensor([0, 0, -0.7071068, 0.7071068])
        )

    # Get target object reference (only if target_object_key is not None)
    target_obj = None
    if target_object_key is not None:
        target_obj = env.scene.object_registry("name", "target_object")

    # Create electrical damage evaluators (per-link) for robot and target object
    r_params = PARAMS.get("tiago_robot", {}).get("electrical", {})
    robot_eval = ElectricalDamageEvaluator(
        entity=robot,
        damage_threshold=float(r_params.get("damage_threshold", 0.0)),
        scale=float(r_params.get("scale", 0.001)),
        water_system_name=str(r_params.get("water_system_name", "water")),
        link_thresholds=r_params.get("link_thresholds", {}),
    )
    target_eval = None
    if target_obj is not None:
        t_params = PARAMS.get(target_object_key, PARAMS["default"]).get("electrical", {})
        target_eval = ElectricalDamageEvaluator(
            entity=target_obj,
            damage_threshold=float(t_params.get("damage_threshold", 0.0)),
            scale=float(t_params.get("scale", 0.001)),
            water_system_name=str(t_params.get("water_system_name", "water")),
            link_thresholds=t_params.get("link_thresholds", {}),
        )

    # Find water particle system
    water_system = None
    possible_water_systems = ["water", "sludge", "fluid"]
    for system_name in possible_water_systems:
        if env.scene.is_physical_particle_system(system_name):
            water_system = env.scene.get_system(system_name)
            print(f"✅ Found water system: {system_name}")
            break
    
    if water_system is None:
        print("⚠️ No water particle system found - water flow may not work")
    else:
        print(f"💧 Water system ready: {water_system.name}")

    # Setup controls
    teleop = KeyboardRobotController(robot=robot)
    camera_mover = setup_camera_controls()
    setup_keyboard_controls(teleop, env, sink, water_system)

    # Set starting camera pose
    start_cam_pos = th.tensor([1.6177639961242676, -0.034418269991874695, 1.1219725608825684])
    start_cam_quat = th.tensor([0.4702153503894806, 0.43835628032684326, 0.5223233699798584, 0.5602852702140808])
    og.sim.viewer_camera.set_position_orientation(position=start_cam_pos, orientation=start_cam_quat)

    # Print control information
    teleop.print_keyboard_teleop_info()
    print("Press ESC in the viewer to quit.")
    print("🚰 Press 'Z' to toggle the sink on/off")
    if sink is not None:
        if hasattr(sink, 'states') and ToggledOn in sink.states:
            initial_state = sink.states[ToggledOn].get_value()
            status = "ON" if initial_state else "OFF"
            print(f"🚰 Sink is currently {status}")
            if water_system is not None:
                print(f"💧 Water system: {water_system.name}")

    # Let physics settle before recording
    print("Letting physics settle for 50 steps...")
    for _ in range(50):
        og.sim.step()

    # Run teleoperation and collect data
    run_teleoperation(teleop, env, robot, target_obj, water_system, camera_mover, robot_eval, target_eval, target_object_key, base_name)


def run_teleoperation(teleop, env, robot, target_obj, water_system, camera_mover, robot_eval, target_eval, target_object_key, base_name):
    """Run the main teleoperation loop and collect data for video generation."""
    # Initialize data collection
    frames = []
    frames_ego = []
    ego_key = None
    fps = 10
    
    # Health and contact tracking
    robot_healths = []
    target_healths = []
    target_contact_counts = []
    robot_contact_counts = []

    steps = 500
    print(f"Starting {steps}-step teleoperation session...")
    
    for step in range(steps):
        print(f"\n\n\n\nStep {step + 1}/{steps}\n\n\n\n")
        action = teleop.get_teleop_action()
        env.step(action=action)

        # Capture viewer camera frame
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        rgb_np = cv2.resize(rgb_np, (1080, 720))  # Scale to HD
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Capture robot ego camera frame
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
                # Use target object damage status when available
                status = str(target_obj.damage_status).lower()
            else:
                # Use robot damage status when no target object (robot-only mode)
                status = str(robot.damage_status).lower()
            
            if status in ("major", "critical"):
                bgr = (0, 0, 255)  # Red border
            elif status in ("minor",):
                bgr = (0, 255, 255)  # Yellow border
            else:
                bgr = (0, 255, 0)  # Green border
            border = 30  # 30px at 1080x720 -> ~10px after scaling to 360x240
            ego_bgr = cv2.copyMakeBorder(ego_bgr, border, border, border, border, cv2.BORDER_CONSTANT, value=bgr)
            frames_ego.append(ego_bgr)

        # Track healths
        robot_healths.append(robot.health)
        if target_obj is not None:
            target_healths.append(target_obj.health)
        else:
            target_healths.append(100.0)

        # Track water particle contacts via ElectricalDamageEvaluator summaries (max over links)
        # Target object contacts
        target_contact_count = 0
        if target_eval is not None:
            summary_t = target_eval.get_contact_summary()
            details_t = summary_t.get("link_details", {})
            target_contact_count = max((v.get("particle_count", 0) for v in details_t.values()), default=0)
        target_contact_counts.append(int(target_contact_count))

        # Robot contacts
        robot_contact_count = 0
        if robot_eval is not None:
            summary_r = robot_eval.get_contact_summary()
            details_r = summary_r.get("link_details", {})
            robot_contact_count = max((v.get("particle_count", 0) for v in details_r.values()), default=0)
        robot_contact_counts.append(int(robot_contact_count))

        # Progress indicator
        if step % 100 == 0:
            if target_object_key is not None:
                print(f"Step {step}/{steps} - {target_object_key.capitalize()} contacts: {target_contact_count}, Robot contacts: {robot_contact_count}")
            else:
                print(f"Step {step}/{steps} - Robot contacts: {robot_contact_count}")

    print("Teleoperation complete. Generating videos...")
    
    # Generate all videos
    generate_videos(frames, frames_ego, robot_healths, target_healths, target_contact_counts, robot_contact_counts, fps, target_object_key, base_name)
    
    camera_mover.clear()
    og.shutdown()


def generate_videos(frames, frames_ego, robot_healths, target_healths, target_contact_counts, robot_contact_counts, fps, target_object_key, base_name):
    """Generate all video outputs including plots and combined videos."""
    # Create output directory
    out_dir = os.path.join("safe-manipulation-benchmark", "unit_tests", "videos")
    os.makedirs(out_dir, exist_ok=True)

    # Write main camera video (temporary)
    sim_mp4 = os.path.join(out_dir, f"{base_name}_camera_obs.mp4")
    if len(frames) > 0:
        h0, w0 = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(sim_mp4, fourcc, fps, (w0, h0))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()

    # Write ego camera video (temporary)
    ego_mp4 = os.path.join(out_dir, f"{base_name}_ego_obs.mp4")
    if len(frames_ego) > 0:
        he, we = frames_ego[0].shape[:2]
        vw_e = cv2.VideoWriter(ego_mp4, fourcc, fps, (we, he))
        for f in frames_ego:
            vw_e.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw_e.release()

    # Create camera with ego overlay (temporary)
    cam_with_ego_mp4 = os.path.join(out_dir, f"{base_name}_camera_with_ego.mp4")
    if os.path.exists(sim_mp4) and os.path.exists(ego_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', sim_mp4,
            '-i', ego_mp4,
            '-filter_complex',
            '[0:v]setsar=1[base];[1:v]scale=360:240,setsar=1[ego];[base][ego]overlay=W-w-20:20[out]',
            '-map','[out]',
            '-c:v','mpeg4','-q:v','5', cam_with_ego_mp4
        ], check=True)

    # Generate plot videos (temporary)
    water_mp4 = generate_water_contacts_plot(target_contact_counts, robot_contact_counts, fps, out_dir, target_object_key)
    health_mp4 = generate_health_plot(robot_healths, target_healths, fps, out_dir, target_object_key)

    # Create final combined videos with mech_damage.py naming convention
    create_final_videos(cam_with_ego_mp4, water_mp4, health_mp4, out_dir, base_name)
    
    # Clean up intermediate videos
    cleanup_intermediate_videos(out_dir, sim_mp4, ego_mp4, cam_with_ego_mp4, water_mp4, health_mp4, base_name)


def generate_water_contacts_plot(target_contact_counts, robot_contact_counts, fps, out_dir, target_object_key):
    """Generate animated plot of water particle contacts."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if target_object_key is not None:
        water_mp4 = os.path.join(out_dir, f'{target_object_key}_water_contacts.mp4')
    else:
        water_mp4 = os.path.join(out_dir, 'robot_only_water_contacts.mp4')
    
    # Prepare data series
    target_series = target_contact_counts if len(target_contact_counts) > 0 else [0]
    robot_series = robot_contact_counts if len(robot_contact_counts) > 0 else [0]
    max_len = max(len(target_series), len(robot_series))
    
    # Create figure with professional styling
    fig_w, ax_w = plt.subplots(figsize=(9.6, 5.4))
    
    if target_object_key is not None:
        line_target, = ax_w.plot([], [], lw=6, color='tab:cyan', label=f'{target_object_key.capitalize()} water contacts')
        line_robot, = ax_w.plot([], [], lw=6, color='tab:red', label='Robot water contacts')
    else:
        line_robot, = ax_w.plot([], [], lw=6, color='tab:red', label='Robot water contacts')
    
    # Configure axes
    ax_w.set_xlim(0, max(1, max_len) / fps)
    y_min_w = 0.0
    y_max_w = max(float(max(target_series + [0])), float(max(robot_series + [0])))
    if y_min_w == y_max_w:
        y_min_w, y_max_w = (0.0, 1.0)
    ax_w.set_ylim(y_min_w, y_max_w * 1.1)
    ax_w.set_xlabel('Time (s)', fontsize=20)
    ax_w.set_ylabel('Water particle contacts', fontsize=20)
    ax_w.set_title('Water Particle Contacts Over Time', fontsize=26)
    
    # Add threshold lines for both robot and target
    robot_damage_threshold = PARAMS["tiago_robot"]["electrical"]["damage_threshold"]
    ax_w.axhline(y=robot_damage_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Robot Damage Threshold ({robot_damage_threshold} particles)')
    
    if target_object_key is not None:
        target_params = PARAMS.get(target_object_key, PARAMS["default"])
        target_damage_threshold = target_params["electrical"]["damage_threshold"]
        ax_w.axhline(y=target_damage_threshold, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label=f'{target_object_key.capitalize()} Damage Threshold ({target_damage_threshold} particles)')
    
    ax_w.legend(loc='best', fontsize=16)
    ax_w.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    ax_w.grid(True, linewidth=1.0, alpha=0.3)
    plt.tight_layout()

    # Animation functions
    if target_object_key is not None:
        def init_w():
            line_target.set_data([], [])
            line_robot.set_data([], [])
            return line_target, line_robot

        def animate_w(i):
            x = [k / fps for k in range(1, i + 2)]
            y_target = target_series[: i + 1]
            y_robot = robot_series[: i + 1]
            line_target.set_data(x, y_target)
            line_robot.set_data(x, y_robot)
            return line_target, line_robot
    else:
        def init_w():
            line_robot.set_data([], [])
            return line_robot,

        def animate_w(i):
            x = [k / fps for k in range(1, i + 2)]
            y_robot = robot_series[: i + 1]
            line_robot.set_data(x, y_robot)
            return line_robot,

    # Create and save animation
    ani_w = animation.FuncAnimation(fig_w, animate_w, init_func=init_w, frames=max_len, interval=1000 / fps, blit=True)
    writer_w = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani_w.save(water_mp4, writer=writer_w)
    plt.close(fig_w)
    
    return water_mp4


def generate_health_plot(robot_healths, target_healths, fps, out_dir, target_object_key):
    """Generate animated plot of robot and target object healths."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if target_object_key is not None:
        health_mp4 = os.path.join(out_dir, f'{target_object_key}_healths.mp4')
    else:
        health_mp4 = os.path.join(out_dir, 'robot_only_healths.mp4')
    
    if len(robot_healths) > 0 or len(target_healths) > 0:
        T = max(len(robot_healths), len(target_healths), 1)
        fig_h, ax_h = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax_h.plot([], [], lw=6, color='tab:blue', label='Robot')
        
        if target_object_key is not None:
            line_t, = ax_h.plot([], [], lw=6, color='tab:orange', label=target_object_key.capitalize())
        
        # Configure axes
        ax_h.set_xlim(0, max(1, T) / fps)
        ax_h.set_ylim(0.0, 100.0)
        ax_h.set_xlabel('Time (s)', fontsize=20)
        ax_h.set_ylabel('Health', fontsize=20)
        ax_h.set_title('Health Over Time', fontsize=26)
        ax_h.legend(loc='best', fontsize=16)
        ax_h.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax_h.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        # Animation functions
        if target_object_key is not None:
            def init_h():
                line_r.set_data([], [])
                line_t.set_data([], [])
                return line_r, line_t
            
            def animate_h(i):
                x = [k / fps for k in range(1, i + 2)]
                y_r = robot_healths[: i + 1] if len(robot_healths) > 0 else []
                y_t = target_healths[: i + 1] if len(target_healths) > 0 else []
                line_r.set_data(x[: len(y_r)], y_r)
                line_t.set_data(x[: len(y_t)], y_t)
                return line_r, line_t
        else:
            def init_h():
                line_r.set_data([], [])
                return line_r,
            
            def animate_h(i):
                x = [k / fps for k in range(1, i + 2)]
                y_r = robot_healths[: i + 1] if len(robot_healths) > 0 else []
                line_r.set_data(x[: len(y_r)], y_r)
                return line_r,
        
        # Create and save animation
        ani_h = animation.FuncAnimation(fig_h, animate_h, init_func=init_h, frames=T, interval=1000 / fps, blit=True)
        writer_h = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani_h.save(health_mp4, writer=writer_h)
        plt.close(fig_h)
    
    return health_mp4


def create_final_videos(cam_with_ego_mp4, water_mp4, health_mp4, out_dir, base_name):
    """Create final combined videos with mech_damage.py naming convention."""
    # Water contacts combined video - using mech_damage.py naming
    combined_water_mp4 = os.path.join(out_dir, f'{base_name}_with_water_contacts.mp4')
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(water_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', cam_with_ego_mp4,
            '-i', water_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_water_mp4
        ], check=True)

    # Health combined video - using mech_damage.py naming
    combined_health_mp4 = os.path.join(out_dir, f'{base_name}_with_health.mp4')
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


def cleanup_intermediate_videos(out_dir, sim_mp4, ego_mp4, cam_with_ego_mp4, water_mp4, health_mp4, base_name):
    """Clean up intermediate videos, keeping only the final combined videos."""
    print("Cleaning up intermediate videos...")
    videos_to_cleanup = [
        sim_mp4,  # {base_name}_camera_obs.mp4
        ego_mp4,   # {base_name}_ego_obs.mp4
        cam_with_ego_mp4,  # {base_name}_camera_with_ego.mp4
        water_mp4,  # {base_name}_water_contacts.mp4
        health_mp4,  # {base_name}_healths.mp4
    ]
    
    for video_path in videos_to_cleanup:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Deleted: {os.path.basename(video_path)}")
            except OSError as e:
                print(f"Warning: Could not delete {video_path}: {e}")
    
    print("Final videos kept:")
    final_water = os.path.join(out_dir, f'{base_name}_with_water_contacts.mp4')
    final_health = os.path.join(out_dir, f'{base_name}_with_health.mp4')
    
    if os.path.exists(final_water):
        print(f"  - {os.path.basename(final_water)}")
    if os.path.exists(final_health):
        print(f"  - {os.path.basename(final_health)}")


if __name__ == "__main__":
    main()