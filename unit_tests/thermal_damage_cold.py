# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')

import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import os
import cv2
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from safety_benchmark.damageable_env import DamageableEnvironment

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True

# Damage params (minimal) for DamageableEnvironment
PARAMS = {
    "tiago_robot": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "impact_threshold": 30.0,
            "impact_scale": 0.001,
            "crushing_threshold": 10.0,
            "crushing_scale": 0.001,
        },
    },
    "soda": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "heating_threshold": 100.0,
            "cooling_threshold": -10.0,
            "scale": 100.0,
        },
    },
    "apple": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "heating_threshold": 100.0,
            "cooling_threshold": -5.0,
            "scale": 0.02,
        },
    },
    "milk": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "heating_threshold": 100.0,
            "cooling_threshold": -2.0,
            "scale": 0.05,
        },
    },
    "ice_cream": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "heating_threshold": 100.0,
            "cooling_threshold": -8.0,
            "scale": 0.03,
        },
    },
    "bowl": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "heating_threshold": 100.0,
            "cooling_threshold": -30.0,
            "scale": 1e-10,
        },
    },
    "fridge": {
        "damage_evaluators": [],
        "health_thresholds": [90.0, 60.0, 30.0],
    },
}

# Target object templates for easy integration of new targets
TARGET_OBJECT_CONFIGS = {
    # Default soda used in current unit test
    "soda": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "can_of_soda",
        "model": "itolcg",
        "bounding_box": [0.08, 0.08, 0.13],
        "position": [0, 0, 5.0],
        "abilities": {
            "heatable": {}  # Ensure Temperature state exists
        },
        "damage_params": PARAMS["soda"],
    },
    # Apple alternative
    "apple": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "apple",
        "model": "agveuv",
        "bounding_box": [0.065, 0.065, 0.077],
        "position": [0, 0, 5.0],
        "abilities": {
            "heatable": {}
        },
        "damage_params": PARAMS["apple"],
    },
    # Milk carton alternative
    "milk": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "milk",
        "model": "brkitw",  # Using generic model for now
        "bounding_box": [0.10, 0.06, 0.20],
        "position": [0, 0, 5.0],
        "abilities": {
            "heatable": {}
        },
        "damage_params": PARAMS["milk"],
    },
    # Ice cream container alternative
    "ice_cream": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "ice_cream",
        "model": "brkitw",  # Using generic model for now
        "bounding_box": [0.12, 0.12, 0.08],
        "position": [0, 0, 5.0],
        "abilities": {
            "heatable": {}
        },
        "damage_params": PARAMS["ice_cream"],
    },
    "bowl": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "bowl",
        "model": "jpvcjv",
        "bounding_box": [0.20, 0.20, 0.07],
        "position": [0, 0, 5.0],
        "abilities": {
            "heatable": {}
        },
        "damage_params": PARAMS["bowl"],
    },
}


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of temperature change
    Loads a fridge and one target object (soda, apple, milk, ice_cream)
    The user can see the object change temperature inside the fridge
    This demo also shows how to set the initial temperature of an object
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Easy integration knobs - Choose target object template key from TARGET_OBJECT_CONFIGS
    target_object_key = "soda"  # Change this to switch between "soda", "apple", "milk", "ice_cream", or None for robot-only
    
    # Derived names and paths based on target object
    base_name = target_object_key if target_object_key is not None else "robot_only"

    # Define specific objects we want to load in with the scene directly
    obj_configs = []

    # Light
    obj_configs.append(
        dict(
            type="LightObject",
            light_type="Sphere",
            name="light",
            radius=0.01,
            intensity=1e8,
            position=[-2.0, -2.0, 1.0],
        )
    )

    # Fridge
    obj_configs.append(
        dict(
            type="DatasetObject",
            name="fridge",
            category="fridge",
            model="lleghp",
            bounding_box=[1.065, 1.149, 1.528],
            abilities={
                "coldSource": {
                    "temperature": -300.0,
                    "requires_inside": True,
                }
            },
            position=[0.0, -1.0, 0.81],
            orientation=[0, 0, -0.7071068, 0.7071068],  # 90 degree rotation around Z-axis
            damage_params=PARAMS["fridge"],
        )
    )

    # Add target object if specified
    if target_object_key is not None:
        if target_object_key not in TARGET_OBJECT_CONFIGS:
            print(f"⚠️ Unknown target object '{target_object_key}', using 'soda'")
            target_object_key = "soda"
        
        target_obj = TARGET_OBJECT_CONFIGS[target_object_key].copy()
        obj_configs.append(target_obj)

    # Robot configuration (needed to register TAB key callback, like mech_damage.py / thermal_damage.py)
    robot_cfg = {
        "type": "Tiago",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, -1, 1],
        "grasping_mode": "assisted",
        "damage_params": PARAMS["tiago_robot"],
    }

    # Create the scene config to load -- Rs_int background with desired objects
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_task_relevant_only": True,
        },
        "robots": [robot_cfg],
        "objects": obj_configs,
    }

    # Create the environment
    env = DamageableEnvironment(configs=cfg)

    # Get reference to relevant objects
    fridge = env.scene.object_registry("name", "fridge")
    
    # Get target object reference (only if target_object_key is not None)
    target_obj = None
    if target_object_key is not None:
        target_obj = env.scene.object_registry("name", "target_object")

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-0.005026338156312704, -3.8500046730041504, 1.1448171138763428]),
        orientation=th.tensor([0.6756195425987244, -0.004750031977891922, -0.0051832040771842, 0.7372169494628906]),
    )

    # Setup TAB breakpoint callback (like thermal_damage.py)
    robot = env.robots[0]

    def breakpoint_and_print_poses():
        print("=== TAB callback triggered ===", flush=True)
        cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
        print("=== Viewer Camera Pose ===")
        print(f"World  pos: {cam_pos.tolist()}  quat: {cam_quat.tolist()}")
        breakpoint()

    action_generator = KeyboardRobotController(robot=robot)
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Print camera pose and breakpoint",
        callback_fn=breakpoint_and_print_poses,
    )

    # Setup camera controls (like mech_damage.py and thermal_damage.py)
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

    print("Running cold thermal damage simulation. Press ESC to quit.")
    print("Press TAB to print camera pose and breakpoint.")
    print("Use WASD to move camera, G for vertical movement.")

    # Let objects settle
    for _ in range(25):
        action = action_generator.get_teleop_action()
        env.step(action=action)

    # Set initial temperature of the target object to 20 degrees Celsius, and move it to the fridge
    if target_obj is not None:
        target_obj.states[object_states.Temperature].set_value(20.0)
        target_obj.states[object_states.Inside].set_value(fridge, True)

    # Ensure the fridge door is closed (like thermal_damage.py / place_pan_on_stove_load.py)
    if object_states.Open in fridge.states:
        fridge.states[object_states.Open].set_value(False)
    else:
        # Fallback: explicitly set any door / fridge joints to zero position
        if hasattr(fridge, "joints"):
            for jname, joint in fridge.joints.items():
                name_l = jname.lower()
                if ("door" in name_l) or ("fridge" in name_l):
                    try:
                        # Some OG joints expose set_joint_position, others set_pos; try both safely.
                        if hasattr(joint, "set_joint_position"):
                            joint.set_joint_position(0.0)
                        elif hasattr(joint, "set_pos"):
                            joint.set_pos(0.0)
                    except Exception:
                        continue

    steps = 0
    max_steps = 200
    fps = 10
    frames = []
    
    # Health, status, and temperature tracking - only target object
    target_healths = []
    target_temperatures = []
    target_statuses = []

    # Main recording loop
    locations = [f"{loc:>20}" for loc in ["Inside fridge"]]
    print()
    if target_object_key is not None:
        print(f"{target_object_key.capitalize()} location:<20", *locations)
    else:
        print("No target object")
    while steps != max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        
        # Capture viewer camera RGB
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        rgb_np = cv2.resize(rgb_np, (1080, 720))
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        
        # Record health, damage status, and temperature - only target object
        if target_obj is not None:
            target_health = float(getattr(target_obj, "health", 100.0))
            target_temp = float(target_obj.damage_evaluators[0].get_temperature())
            target_status = str(getattr(target_obj, "damage_status", "none")).lower()
            
            target_healths.append(target_health)
            target_temperatures.append(target_temp)
            target_statuses.append(target_status)
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"Step {steps}: {target_object_key.capitalize()} temp={target_temp:.1f}°C, health={target_health:.1f}%")
        
        steps += 1

    # Create videos directory
    videos_dir = os.path.join("safe-manipulation-benchmark", "unit_tests", "videos")
    os.makedirs(videos_dir, exist_ok=True)

    # Write main camera video
    avi_path = os.path.join(videos_dir, "cold_camera_obs.avi")
    mp4_path = os.path.join(videos_dir, "cold_camera_obs.mp4")
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

        # Create a status-border sim video, like simple_task_load / rl_test:
        # - Border color encodes target health status (none/negligible -> green, minor -> yellow, major/critical -> red)
        # - Top-left text shows current target temperature
        border_mp4 = os.path.join(videos_dir, "cold_camera_with_status_border.mp4")
        border_width = 30
        fourcc_border = cv2.VideoWriter_fourcc(*"mp4v")
        vw_border = cv2.VideoWriter(
            border_mp4,
            fourcc_border,
            fps,
            (w + 2 * border_width, h + 2 * border_width),
        )

        # If for some reason we have no recorded temperatures/statuses, fall back to defaults
        if len(target_temperatures) == 0:
            target_temperatures = [0.0] * len(frames)
        if len(target_statuses) == 0:
            target_statuses = ["none"] * len(frames)

        for f, status, temp in zip(frames, target_statuses, target_temperatures):
            s = (status or "none").lower()
            if s in ("major", "critical"):
                bgr = (0, 0, 255)  # Red
            elif s == "minor":
                bgr = (0, 255, 255)  # Yellow
            else:
                bgr = (0, 255, 0)  # Green

            bordered = cv2.copyMakeBorder(
                f,
                border_width,
                border_width,
                border_width,
                border_width,
                cv2.BORDER_CONSTANT,
                value=bgr,
            )

            # Temperature text at top-left inside the sim image (not on the border)
            label = base_name.capitalize() if target_object_key is not None else "Target"
            text = f"{label} Temp: {float(temp):.2f} C"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = border_width + 20
            y = border_width + text_h + 20
            cv2.putText(
                bordered,
                text,
                (x, y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
            vw_border.write(np.ascontiguousarray(bordered, dtype=np.uint8))

        vw_border.release()
        print(f"✅ Saved status-border camera video: {border_mp4}")

    # Build stacked plots video (temperature top, health bottom) - like mech_damage.py forces
    plots_mp4 = os.path.join(videos_dir, f'{base_name}_plots.mp4')
    if len(target_temperatures) > 0 and len(target_healths) > 0:
        T = max(len(target_temperatures), len(target_healths))
        
        # Temperature figure (top)
        fig_temp, ax_temp = plt.subplots(figsize=(9.6, 5.4))
        line_temp, = ax_temp.plot([], [], lw=8, color='tab:orange', label=target_object_key.capitalize() if target_object_key is not None else 'Target')
        ax_temp.set_xlim(0, max(1, T) / fps)
        all_temps = target_temperatures
        y_min_temp = min(all_temps)
        y_max_temp = max(all_temps)
        ax_temp.set_ylim(y_min_temp, y_max_temp)
        
        # Add cooling threshold line
        if target_object_key is not None and target_object_key in PARAMS:
            thr = PARAMS[target_object_key].get("thermal", {}).get("cooling_threshold", None)
            if thr is not None:
                ax_temp.axhline(y=thr, color='red', linestyle='--', linewidth=4, alpha=0.8, label=f'Cooling Threshold ({thr}°C)')
        
        ax_temp.set_xlabel('Time (s)', fontsize=24, fontweight='bold')
        ax_temp.set_ylabel('Temperature (°C)', fontsize=24, fontweight='bold')
        ax_temp.set_title('Temperature Over Time', fontsize=28, fontweight='bold', pad=20)
        ax_temp.legend(loc='best', fontsize=20, frameon=True, fancybox=True, shadow=True)
        ax_temp.tick_params(axis='both', which='major', labelsize=20, width=2, length=8)
        ax_temp.grid(True, linewidth=2, alpha=0.3, color='gray')
        ax_temp.spines['top'].set_linewidth(2)
        ax_temp.spines['right'].set_linewidth(2)
        ax_temp.spines['bottom'].set_linewidth(2)
        ax_temp.spines['left'].set_linewidth(2)
        plt.tight_layout()

        def init_temp():
            line_temp.set_data([], [])
            return line_temp,

        def animate_temp(i):
            x = [k / fps for k in range(1, i + 2)]
            y_temp = target_temperatures[: i + 1]
            line_temp.set_data(x, y_temp)
            return line_temp,

        ani_temp = animation.FuncAnimation(fig_temp, animate_temp, init_func=init_temp, frames=T, interval=1000 / fps, blit=True)

        # Health figure (bottom)
        fig_health, ax_health = plt.subplots(figsize=(9.6, 5.4))
        line_health, = ax_health.plot([], [], lw=8, color='tab:green', label=target_object_key.capitalize() if target_object_key is not None else 'Target')
        ax_health.set_xlim(0, max(1, T) / fps)
        ax_health.set_ylim(0.0, 100.0)
        
        ax_health.set_xlabel('Time (s)', fontsize=24, fontweight='bold')
        ax_health.set_ylabel('Health', fontsize=24, fontweight='bold')
        ax_health.set_title('Health Over Time', fontsize=28, fontweight='bold', pad=20)
        ax_health.legend(loc='best', fontsize=20, frameon=True, fancybox=True, shadow=True)
        ax_health.tick_params(axis='both', which='major', labelsize=20, width=2, length=8)
        ax_health.grid(True, linewidth=2, alpha=0.3, color='gray')
        ax_health.spines['top'].set_linewidth(2)
        ax_health.spines['right'].set_linewidth(2)
        ax_health.spines['bottom'].set_linewidth(2)
        ax_health.spines['left'].set_linewidth(2)
        plt.tight_layout()

        def init_health():
            line_health.set_data([], [])
            return line_health,

        def animate_health(i):
            x = [k / fps for k in range(1, i + 2)]
            y_health = target_healths[: i + 1]
            line_health.set_data(x, y_health)
            return line_health,

        ani_health = animation.FuncAnimation(fig_health, animate_health, init_func=init_health, frames=T, interval=1000 / fps, blit=True)

        # Save and stack
        temp_mp4 = os.path.join(videos_dir, f'{base_name}_temp_plot.mp4')
        health_mp4 = os.path.join(videos_dir, f'{base_name}_health_plot.mp4')
        writer_f = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani_temp.save(temp_mp4, writer=writer_f)
        plt.close(fig_temp)
        ani_health.save(health_mp4, writer=writer_f)
        plt.close(fig_health)

        # Stack vertically like mech_damage.py
        subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_mp4,
            '-i', health_mp4,
            '-filter_complex',
            '[0:v]scale=1080:360,setsar=1[top];[1:v]scale=1080:360,setsar=1[bot];[top][bot]vstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            plots_mp4
        ], check=True)

    # Create combined video (camera + stacked plots) - like mech_damage.py
    combined_mp4 = os.path.join(videos_dir, f'{base_name}_with_plots.mp4')
    if os.path.exists(mp4_path) and os.path.exists(plots_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', mp4_path,
            '-i', plots_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_mp4
        ], check=True)

    # Clean up intermediate videos
    print("Cleaning up intermediate videos...")
    videos_to_cleanup = [
        mp4_path,  # cold_camera_obs.mp4
        plots_mp4,  # {base_name}_plots.mp4
        temp_mp4,  # {base_name}_temp_plot.mp4
        health_mp4,  # {base_name}_health_plot.mp4
    ]
    
    for video_path in videos_to_cleanup:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Deleted: {os.path.basename(video_path)}")
            except OSError as e:
                print(f"Warning: Could not delete {video_path}: {e}")
    
    print(f"Final videos kept:")
    if os.path.exists(combined_mp4):
        print(f"  - {os.path.basename(combined_mp4)}")

    # Clean shutdown
    camera_mover.clear()

    og.shutdown()


if __name__ == "__main__":
    main()
