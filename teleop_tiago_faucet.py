"""
Example script demo'ing robot control with a faucet.

Options for random actions, as well as selection of robot action space
"""

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')

import torch as th
import json
import os
import cv2
from datetime import datetime
import numpy as np
import subprocess

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.object_states import OnTop, ToggledOn, ContactParticles
from omnigibson.utils.ui_utils import KeyboardRobotController
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS

# Enable GPU dynamics for water particle systems (required for faucet water flow)
# The sludge system that handles water particles from the faucet requires GPU dynamics
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_HQ_RENDERING = False

OBJECT_CONFIGS = {
    "faucet": {
        "type": "DatasetObject",
        "name": "faucet",
        "category": "beer_tap",
        "model": "zcrgvq",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 1, 0],
        "scale": [1.0, 1.0, 1.0],
        "initial_state": {
            "toggleable": True,
        },
    },
    "sink": {
        "type": "DatasetObject",    
        "name": "sink",
        "category": "furniture_sink",
        "model": "zexzrc",
        "position": [0.0, -0.5, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 0.7],
    },
    # Add requested BEHAVIOR bowl model jblalf
    "bowl": {
        "type": "DatasetObject",
        "name": "bowl",
        "category": "bowl",
        "model": "jpvcjv",
        "position": [-0.2, -0.5, 3.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 1.0, 1.0],
    },
}

# Alternative configurations if the primary ones fail
ALTERNATIVE_CONFIGS = {
    "sink_czyfhq": {
        "type": "DatasetObject",    
        "name": "sink",
        "category": "furniture_sink",
        "model": "czyfhq",
        "position": [0.0, -0.5, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [0.7, 0.7, 0.7],
    },
    "faucet_alt": {
        "type": "DatasetObject",
        "name": "faucet",
        "category": "beer_tap",
        "model": "vgaluf",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 1, 0],
        "scale": [1.0, 1.0, 1.0],
        "initial_state": {
            "toggleable": True,
        },
    },
    "faucet_alt2": {
        "type": "DatasetObject",
        "name": "faucet",
        "category": "beer_tap",
        "model": "bebcmz",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 1, 0],
        "scale": [1.0, 1.0, 1.0],
        "initial_state": {
            "toggleable": True,
        },
    },
}


def try_load_objects_with_fallback():
    """
    Try to load objects with fallback options if primary ones fail.
    """
    # Try primary configuration first
    try:
        objects = [OBJECT_CONFIGS["sink"], OBJECT_CONFIGS["faucet"]]
        print("✅ Using primary object configuration")
        return objects
    except Exception as e:
        print(f"⚠️ Primary configuration failed: {e}")
        print("🔄 Trying alternative configuration...")
        
        # Try alternative sink
        try:
            objects = [ALTERNATIVE_CONFIGS["sink_czyfhq"], OBJECT_CONFIGS["faucet"]]
            print("✅ Using alternative sink configuration")
            return objects
        except Exception as e2:
            print(f"⚠️ Alternative sink failed: {e2}")
            print("🔄 Trying alternative faucet...")
            
            # Try alternative faucet
            try:
                objects = [OBJECT_CONFIGS["sink"], ALTERNATIVE_CONFIGS["faucet_alt"]]
                print("✅ Using alternative faucet configuration")
                return objects
            except Exception as e3:
                print(f"⚠️ Alternative faucet failed: {e3}")
                print("🔄 Trying second alternative faucet...")
                
                # Try second alternative faucet
                try:
                    objects = [OBJECT_CONFIGS["sink"], ALTERNATIVE_CONFIGS["faucet_alt2"]]
                    print("✅ Using second alternative faucet configuration")
                    return objects
                except Exception as e4:
                    print(f"⚠️ Second alternative faucet failed: {e4}")
                    print("🔄 Trying minimal configuration...")
                    
                    # Try minimal configuration with just primary faucet
                    try:
                        objects = [OBJECT_CONFIGS["faucet"]]
                        print("✅ Using minimal configuration (faucet only)")
                        return objects
                    except Exception as e5:
                        print(f"❌ All configurations failed: {e5}")
                        print("🚨 Falling back to empty scene with just robot")
                        return []


def main():
    """
    Minimal teleop demo with Tiago in an empty scene with a faucet.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Always use empty scene and Tiago robot
    scene_cfg = {"type": "Scene", "scene_id": "empty"}
    
    # Do not load from saved state; build scene from OBJECT_CONFIGS
    # scene_cfg["scene_file"] = None
    
    robot0_cfg = dict()
    robot0_cfg["type"] = "Tiago"
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True
    robot0_cfg["position"] = [0., -1.3, 0.0]
    robot0_cfg["orientation"] = [0, 0, 1, 1]
    robot0_cfg["grasping_mode"] = "assisted"
    robot0_cfg["damage_params"] = PARAMS["tiago_robot"]

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
    cfg["rendering_frequency"] = 60

    # CRITICAL: When loading from scene_file, the Scene constructor will automatically
    # populate _init_objs from the JSON, so we must NOT add objects to cfg["objects"]
    # or the Environment will try to load them again, causing duplicate name errors.
    # Fresh scene - add objects manually
    objects = try_load_objects_with_fallback()
    if objects:
        cfg["objects"] = objects

    # Create the environment
    env = DamageableEnvironment(configs=cfg)

    # Choose robot controller to use
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

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    
    # Fix gripper controller configuration for Tiago
    controller_config["gripper_left"]["inverted"] = True
    controller_config["gripper_right"]["inverted"] = True
    
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_file()
    
    # Reset environment and robot
    env.reset()
    robot.reset()

    # Get references to objects (with error handling)
    try:
        sink = env.scene.object_registry("name", "sink")
        print("✅ Sink loaded successfully")
    except:
        sink = None
        print("⚠️ Sink not available")
    
    try:
        faucet = env.scene.object_registry("name", "faucet")
        print("✅ Faucet loaded successfully")
    except:
        faucet = None
        print("⚠️ Faucet not available")
    
    # No bowl in the scene

    # Let physics settle
    for _ in range(20):
        og.sim.step()

    # Turn faucet ON at start if available (guard for missing state)
    if faucet is not None and ToggledOn in faucet.states:
        try:
            if not faucet.states[ToggledOn].get_value():
                faucet.states[ToggledOn].set_value(True)
            print("🚰 Faucet turned ON at start")
        except Exception as e:
            print(f"⚠️ Could not turn faucet on at start: {e}")
    else:
        # Fallback: try any object with ToggledOn
        try:
            toggle_targets = [obj for obj in env.scene.objects if ToggledOn in obj.states]
            for obj in toggle_targets:
                if not obj.states[ToggledOn].get_value():
                    obj.states[ToggledOn].set_value(True)
            if toggle_targets:
                print(f"🚰 Turned ON {len(toggle_targets)} toggleable object(s) at start")
        except Exception as e:
            print(f"⚠️ Fallback toggle failed: {e}")

    # Default visuals; no prototype-hiding edits
    
    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    
    # Enable camera teleoperation with custom key bindings
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
    
    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )
    
    # Function to toggle faucet on/off
    def toggle_faucet():
        if faucet is None:
            print("❌ Faucet not available")
            return
        try:
            current_state = faucet.states[ToggledOn].get_value()
            new_state = not current_state
            faucet.states[ToggledOn].set_value(new_state)
            status = "ON" if new_state else "OFF"
            print(f"🚰 Faucet turned {status}")
        except Exception as e:
            print(f"❌ Failed to toggle faucet: {e}")
    
    # Register F key for faucet toggle
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.F,
        description="Toggle faucet on/off",
        callback_fn=toggle_faucet,
    )
    
    # Print out relevant keyboard info
    action_generator.print_keyboard_teleop_info()
    
    # Other helpful user info
    print("Running faucet demo.")
    print("Press F to toggle faucet on/off")
    print("Press ESC to quit")

    # Initialize water contact tracking
    water_contact_counts = []
    water_system = None
    
    # Initialize robot health tracking (like other scripts)
    robot_healths = []
    robot_link_healths = []
    robot_damage_statuses = []
    
    # Try to get the water/sludge system for contact tracking
    # Use existing faucet fluid system
    water_system = None
    possible_water_systems = ["sludge", "water", "fluid"]
    for system_name in possible_water_systems:
        if env.scene.is_physical_particle_system(system_name):
            water_system = env.scene.get_system(system_name)
            print(f"✅ Found water system: {system_name}")
            break
    assert water_system is not None, "No water system found"
    print(f"✅ Water contact tracking enabled with system: {water_system.name}")

    # Print water contact tracking status after initialization
    if water_system is not None:
        print(f"💧 Water contact tracking enabled - monitoring {water_system.name} system")
        print(f"💧 Note: Will detect water particles even when faucet is OFF (existing water in sink)")
    else:
        print("⚠️ Water contact tracking disabled - no water system found")

    # Loop control until user quits
    max_steps = 250
    step = 0
    fps = 10

    images = []

    while step < max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1
        # Ensure faucet remains ON every step
        if faucet is not None and ToggledOn in faucet.states:
            if not faucet.states[ToggledOn].get_value():
                faucet.states[ToggledOn].set_value(True)

        rgb_img = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_img = rgb_img.cpu().numpy()[:, :, :3]
        rgb_img = cv2.resize(rgb_img, (512, 512))
        images.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # Track water contact with robot (supported path via ContactParticles)
        if water_system is not None:
            # Get water particles in contact with the robot using ContactParticles state
            contacting_particles = robot.states[ContactParticles].get_value(system=water_system)
            water_contact_count = len(contacting_particles)
            water_contact_counts.append(water_contact_count)

            if step % 30 == 0:  # Print every 30 steps
                faucet_state = "ON" if faucet and ToggledOn in faucet.states and faucet.states[ToggledOn].get_value() else "OFF"
                print(f"Step {step}: Faucet {faucet_state}, Contact={water_contact_count}")
        else:
            water_contact_counts.append(0)
        
        # Track robot health (like other scripts)
        robot_healths.append(robot.health)
        robot_damage_statuses.append(robot.damage_status)
        robot_link_healths.append(robot.link_healths.copy())

    # Clean up camera mover
    camera_mover.clear()

    # Save video
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create videos_and_images directory if it doesn't exist
    os.makedirs('videos_and_images', exist_ok=True)
    
    avi_path = 'videos_and_images/faucet_teleop.avi'
    mp4_path = 'videos_and_images/faucet_teleop.mp4'
    out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    for i, image in enumerate(images):
        # Add captions
        frame_copy = image.copy()
        y_pos = 30
        
        # Add robot health (printed on video only)
        robot_health = robot_healths[i] if i < len(robot_healths) else 100.0
        cv2.putText(frame_copy, f"Robot Health: {robot_health:.2f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add water contact count
        y_pos += 30
        water_contact_count = water_contact_counts[i] if i < len(water_contact_counts) else 0
        cv2.putText(frame_copy, f"Contact Water Particles: {water_contact_count}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add damage status
        y_pos += 30
        if i < len(robot_damage_statuses):
            damage_status = robot_damage_statuses[i]
            cv2.putText(frame_copy, f"Damage Status: {damage_status}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(np.ascontiguousarray(frame_copy, dtype=np.uint8))
    out.release()

    # Convert AVI to MP4 using ffmpeg
    subprocess.run([
        'ffmpeg', '-y', '-i', avi_path,
        '-c:v', 'mpeg4', mp4_path
    ], check=True)
    
    # Clean up AVI file
    os.remove(avi_path)

    # Create faucet state animation
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Ensure all data arrays have the same length
    min_length = min(len(water_contact_counts), len(robot_healths))
    if min_length == 0:
        print("⚠️ No data to plot - skipping animation")
        faucet_mp4 = 'videos_and_images/faucet_state_plot.mp4'
        # Create empty plot file
        fig, ax = plt.subplots(figsize=(6.83, 6.83))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('No Data Available')
        plt.savefig(faucet_mp4.replace('.mp4', '.png'))
        plt.close()
    else:
        # Truncate arrays to the same length
        water_contact_counts = water_contact_counts[:min_length]
        robot_healths = robot_healths[:min_length]
        
        print(f"📊 Creating animation with {min_length} data points")
        
        # Set up the figure and axis with 1 subplot
        fig, ax1 = plt.subplots(1, 1, figsize=(6.83, 6))
        
        # Water contact plot
        line1, = ax1.plot([], [], lw=2, color='red')
        ax1.set_xlim(1, min_length)
        ax1.set_ylim(-1, max(max(water_contact_counts) + 5, 10))
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Water Particles in Contact')
        ax1.set_title('Robot Water Contact Over Time')
        ax1.grid(True)
        
        plt.tight_layout()
        
        # Initialization function
        def init():
            line1.set_data([], [])
            return [line1]
        
        # Animation function which updates the figure
        def animate(i):
            x = list(range(1, i + 2))
            y1 = water_contact_counts[:i + 1]
            line1.set_data(x, y1)
            return [line1]
        
        # Create an animation object
        ani = animation.FuncAnimation(
            fig, animate, 
            init_func=init,
            frames=min_length,
            interval=1000/fps,
            blit=True
        )
        # Save animation and set faucet_mp4 for downstream combine step
        faucet_mp4 = 'videos_and_images/faucet_state_plot.mp4'
        writer = animation.FFMpegWriter(
            fps=fps,
            codec='mpeg4',
            extra_args=['-vcodec', 'mpeg4', '-qscale', '5']
        )
        ani.save(faucet_mp4, writer=writer)
        plt.close(fig)

    # Combine videos side by side
    combined_mp4 = 'videos_and_images/faucet_combined_view.mp4'
    
    # Check if we have a valid faucet plot file
    if os.path.exists(faucet_mp4):
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', mp4_path,
                '-i', faucet_mp4,
                '-filter_complex',
                '[0:v][1:v]scale2ref=oh*dar:ih[v0][v1];[v0][v1]hstack=inputs=2[v]',
                '-map', '[v]',
                '-vcodec', 'mpeg4',
                '-q:v', '5',
                combined_mp4
            ], check=True)
            print(f"✅ Combined video saved to: {combined_mp4}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to combine videos: {e}")
            print("🔄 Using main video only...")
            # Copy main video as fallback
            import shutil
            shutil.copy2(mp4_path, combined_mp4)
            print(f"✅ Main video copied to: {combined_mp4}")
    else:
        print("⚠️ Faucet plot file not found, using main video only...")
        # Copy main video as fallback
        import shutil
        shutil.copy2(mp4_path, combined_mp4)
        print(f"✅ Main video copied to: {combined_mp4}")

    # Always shut down the environment cleanly at the end
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main() 