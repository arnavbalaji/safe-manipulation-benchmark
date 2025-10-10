"""
Minimal teleop script demo'ing robot control with a faucet, using simplified
object loading and without setting any OnTop or Toggled states.

Allows teleoperation and camera movement; recording and plotting behavior is kept
similar to teleop_tiago_faucet.py.
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
from omnigibson.object_states import ContactParticles, OnTop
from omnigibson.utils.ui_utils import KeyboardRobotController
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS

# Enable GPU dynamics for water particle systems (required for faucet water flow)
# The sludge system that handles water particles from the faucet requires GPU dynamics
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_HQ_RENDERING = False

# Minimal object configuration: faucet + sink only
OBJECT_CONFIGS = {
    "faucet": {
        "type": "DatasetObject",
        "name": "faucet",
        "category": "beer_tap",
        "model": "zcrgvq",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 1, 0],
        "scale": [1.0, 1.0, 1.0],
        # Keep toggleable in initial state but do not set ToggledOn anywhere in this script
        "initial_state": {
            "toggleable": True,
        },
        # Add explicit particle source configuration with slower speed
        "abilities": {
            "particleSource": {
                "conditions": {
                    "water": [
                        ("TOGGLEDON", True)  # Must be toggled on for water source to be active
                    ]
                },
                "initial_speed": 0.5  # Slower water flow speed
            },
            "particleSink": {
                "conditions": {
                    "water": []  # No conditions, always sinking nearby particles
                }
            }
        },
    },
    "sink": {
        "type": "DatasetObject",
        "name": "sink",
        "category": "furniture_sink",
        "model": "zexzrc",
        "position": [0.05, -0.55, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 0.7],
    },
    "phone": {
        "type": "DatasetObject",
        "name": "phone",
        "category": "cell_phone",
        "model": "dbhfuh",
        # "position": [-0.25, -0.6, 1.5],
        "position": [0.0, -0.5, 1.5],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 1.0, 1.0],
        "mass": 3.0,
        "density": 1000.0,
    },
}


def main():
    """
    Minimal teleop demo with Tiago in an empty scene with a faucet, simplified loading.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Always use empty scene and Tiago robot
    scene_cfg = {"type": "Scene", "scene_id": "empty"}

    # Configure robot
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

    # Simplified: directly add objects (no alternative configs or fallback logic)
    cfg["objects"] = [OBJECT_CONFIGS["sink"], OBJECT_CONFIGS["faucet"], OBJECT_CONFIGS["phone"]]

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

    try:
        phone = env.scene.object_registry("name", "phone")
        print("✅ Phone loaded successfully")
    except:
        phone = None
        print("⚠️ Phone not available")

    # Resolve a stable visual link for the phone to use visual AABB during particle queries
    phone_link = None
    phone_link_name = "base_link"
    if phone is not None:
        try:
            if hasattr(phone, "links") and isinstance(phone.links, dict) and phone_link_name in phone.links:
                phone_link = phone.links[phone_link_name]
                print("✅ Using phone link: base_link")
            elif hasattr(phone, "root_link") and phone.root_link is not None:
                phone_link = phone.root_link
                print("✅ Using phone link: root_link")
            elif hasattr(phone, "links") and phone.links:
                phone_link = list(phone.links.values())[0] if isinstance(phone.links, dict) else phone.links[0]
                print("✅ Using phone link: first link")
        except Exception as e:
            print(f"⚠️ Could not resolve phone link: {e}")

    # If both phone and sink exist, set phone OnTop of sink
    # if 'phone' in locals() and phone is not None and sink is not None:
    #     try:
    #         assert phone.states[OnTop].set_value(sink, True), "Failed to set phone OnTop of sink"
    #         print("📱 Phone placed OnTop of sink")
    #     except Exception as e:
    #         print(f"⚠️ Could not set phone OnTop of sink: {e}")

    # Let physics settle
    for _ in range(20):
        og.sim.step()

    # Increase phone damping and ensure gravity is enabled to reduce fluid-induced launch
    if phone is not None:
        try:
            # Set mass/density already via config; here ensure gravity and optionally adjust damping via prim properties
            for link in phone.links.values():
                try:
                    link.enable_gravity()
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Could not adjust phone physics props: {e}")

    # Create teleop controller (no state toggling)
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

    # Register faucet toggle key (Q)
    def toggle_faucet():
        if faucet is not None and hasattr(faucet, 'states'):
            from omnigibson.object_states import ToggledOn
            if ToggledOn in faucet.states:
                current_state = faucet.states[ToggledOn].get_value()
                new_state = not current_state
                try:
                    faucet.states[ToggledOn].set_value(new_state)
                    status = "ON" if new_state else "OFF"
                    print(f"🚰 Faucet turned {status}")
                    
                    # Check if water system is available and report particle count
                    if water_system is not None:
                        particle_count = water_system.n_particles
                        print(f"💧 Water particles in system: {particle_count}")
                    else:
                        print("⚠️ No water system detected - water may not flow")
                        
                except Exception as e:
                    print(f"⚠️ Failed to toggle faucet: {e}")
            else:
                print("⚠️ Faucet does not have ToggledOn state")
        else:
            print("⚠️ Faucet not available for toggling")

    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.Q,
        description="Toggle faucet on/off",
        callback_fn=toggle_faucet,
    )

    # Print out relevant keyboard info
    action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running faucet demo (no auto-toggling).")
    print("🚰 Press 'Q' to toggle the faucet on/off")
    print("Press ESC to quit")

    # Initialize water contact tracking
    water_contact_counts = []
    water_system = None

    # Initialize robot health tracking (like other scripts)
    robot_healths = []
    robot_link_healths = []
    robot_damage_statuses = []

    # Try to get the water/sludge system for contact tracking
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

        rgb_img = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_img = rgb_img.cpu().numpy()[:, :, :3]
        rgb_img = cv2.resize(rgb_img, (512, 512))
        images.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # Stabilize phone in basin: bounds guard + keep-still once inside basin
        if phone is not None:
            try:
                p, q = phone.get_position_orientation()
                if sink is not None:
                    sink_pos, _ = sink.get_position_orientation()
                else:
                    sink_pos = th.tensor([0.0, -0.5, 0.0], dtype=th.float32)
                # Out-of-bounds guard: if z too low/high or far from sink XY, reposition above basin and zero velocity
                if p[-1].item() < -0.5 or p[-1].item() > 4.0 or th.norm(p[:2] - sink_pos[:2]).item() > 2.0:
                    target = th.tensor([sink_pos[0].item(), sink_pos[1].item(), 1.2], dtype=th.float32)
                    phone.set_position_orientation(position=target, orientation=q)
                    phone.keep_still()
                # If roughly within basin height, keep still to prevent fluid-induced launch
                elif p[-1].item() < 0.9:
                    phone.keep_still()
            except Exception:
                pass

        # Track water contact with phone base_link using physics queries directly (avoid AABB prim issues)
        if water_system is not None and phone is not None:
            # Ensure the phone link is valid during runtime (prim can be recreated)
            if phone_link is None or (hasattr(phone_link, "is_valid") and not phone_link.is_valid()):
                try:
                    if hasattr(phone, "links") and isinstance(phone.links, dict) and phone_link_name in phone.links:
                        phone_link = phone.links[phone_link_name]
                    elif hasattr(phone, "root_link") and phone.root_link is not None:
                        phone_link = phone.root_link
                    elif hasattr(phone, "links") and phone.links:
                        phone_link = list(phone.links.values())[0] if isinstance(phone.links, dict) else phone.links[0]
                except Exception:
                    phone_link = None

            # Prepare report callback to match only the desired phone link
            link_name = phone_link.prim_path.split("/")[-1] if phone_link is not None else phone_link_name
            def report_hit(hit):
                base = "/".join(hit.rigid_body.split("/")[:-1])
                body = hit.rigid_body.split("/")[-1]
                # Match only the target link on the phone
                if body == link_name and base == phone.prim_path:
                    # Signal a hit and stop traversal
                    return False
                return True

            # Iterate all particles and check overlap with the phone link
            positions = water_system.get_particles_position_orientation()[0]
            dist = water_system.particle_contact_radius + 5e-3
            contact_count = 0
            for i in range(positions.shape[0]):
                hit_detected = False
                def hit_cb(hit):
                    nonlocal hit_detected
                    base = "/".join(hit.rigid_body.split("/")[:-1])
                    body = hit.rigid_body.split("/")[-1]
                    if body == link_name and base == phone.prim_path:
                        hit_detected = True
                        return False
                    return True
                og.sim.psqi.overlap_sphere(dist, positions[i].cpu().numpy(), hit_cb, False)
                if hit_detected:
                    contact_count += 1

            water_contact_counts.append(contact_count)
            if step % 30 == 0:  # Print every 30 steps
                print(f"Step {step}: Phone contact={contact_count}")
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

    avi_path = 'videos_and_images/faucet_teleop2.avi'
    mp4_path = 'videos_and_images/faucet_teleop2.mp4'
    out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    for i, image in enumerate(images):
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
        faucet_mp4 = 'videos_and_images/faucet_state_plot2.mp4'
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
        ax1.set_title('Phone Water Contact Over Time')
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
        faucet_mp4 = 'videos_and_images/faucet_state_plot2.mp4'
        writer = animation.FFMpegWriter(
            fps=fps,
            codec='mpeg4',
            extra_args=['-vcodec', 'mpeg4', '-qscale', '5']
        )
        ani.save(faucet_mp4, writer=writer)
        plt.close(fig)

    # Combine videos side by side
    combined_mp4 = 'videos_and_images/faucet_combined_view2.mp4'

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