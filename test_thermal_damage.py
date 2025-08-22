import torch as th
import cv2
import numpy as np
import os
import subprocess

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.object_states import OnTop
from omnigibson.utils.ui_utils import KeyboardRobotController
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


def main():
    """
    Minimal teleop demo with Tiago in an empty scene with stove and apple.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Always use empty scene and Tiago robot
    scene_cfg = {"type": "Scene"}
    robot0_cfg = dict()
    robot0_cfg["type"] = "Tiago"
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True
    robot0_cfg["position"] = [0., 0.0, 0.0]  # Safe distance to reach stove without physics explosions
    robot0_cfg["orientation"] = [0, 0, -1, 1]  # Facing the stove (same as original working script)
    robot0_cfg["grasping_mode"] = "assisted"
    robot0_cfg["damage_params"] = PARAMS["tiago_robot"]

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Define objects - stove in same position as coffee table, apple positioned away from heat
    objects = [
        {
            "type": "DatasetObject",
            "name": "stove",
            "category": "stove",
            "model": "yhjzwg",
            "position": [0.0, -1.0, 0.0],  # Move to first row for easier robot access
            "bounding_box": [0.8, 0.66, 0.65],  # Even smaller bbox to match desired scale
            "orientation": [0, 0, 0.7071068, 0.7071068],  # Same orientation as coffee table
            "initial_state": {
                "joints": {"door": 0.0},  # 0.0 = closed, 1.0 = open
                "toggleable": True,  # Allow stove to be turned on/off
                "temperature": 200.0  # Set initial temperature for heating
            },
        },
        {
            "type": "DatasetObject",
            "name": "apple",
            "category": "apple",
            "model": "agveuv",
            "bounding_box": [0.1, 0.1, 0.12],  # Made apple a bit smaller
            "position": [0.25, -0.8, 0.8],  # Positioned on top of the stove surface
            "initial_state": {
                "OnTop": "stove"  # Ensure apple is on top of the stove
            },
            "damage_params": PARAMS["apple"],
        }
    ]
    cfg["objects"] = objects

    # Create the environment
    env = DamageableEnvironment(configs=cfg)

    # Choose robot controller to use
    robot = env.robots[0]
    controller_choices = {
        "base": "HolonomicBaseJointController",  # Correct for TIAGo's omnidirectional base
        "arm_left": "InverseKinematicsController",
        "arm_right": "InverseKinematicsController",
        "gripper_left": "MultiFingerGripperController",
        "gripper_right": "MultiFingerGripperController",
        "camera": "JointController",
        "trunk": "JointController",  # Added trunk controller which was missing
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

    # Get references to objects
    stove = env.scene.object_registry("name", "stove")
    apple = env.scene.object_registry("name", "apple")

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

    # Set initial temperature of the apple to room temperature (20°C)
        apple.states[object_states.Temperature].set_value(20.0)

    # Let physics settle
    for _ in range(20):  # Increased settling steps
        og.sim.step()

    for _ in range(20):  # Increased settling steps
        og.sim.step()

    # Add physics damping to prevent objects from flying away
    # Set solver iterations on objects to reduce bouncing and explosions
    for obj in env.scene.objects:
        if hasattr(obj, 'solver_position_iteration_count'):
            obj.solver_position_iteration_count = 8  # Increase position iterations for stability
        if hasattr(obj, 'solver_velocity_iteration_count'):
            obj.solver_velocity_iteration_count = 1  # Reduce velocity iterations to prevent explosions

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    
    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Enable camera teleoperation with custom key bindings
    from omnigibson.utils.ui_utils import CameraMover
    class CustomCameraMover(CameraMover):
        @property
        def input_to_command(self):
            """
            Returns:
                dict: Mapping from relevant keypresses to corresponding delta command to apply to the camera pose
            """
            return {
                lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
                lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }
    
    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()

    # Print out relevant keyboard info
    action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running thermal damage demo.")
    print("Use robot controls to push the apple onto the hot stove")
    print("Press ESC to quit")
    print()
    print("Base Control:")
    print("  1, 2: Switch between base joints (x, y, rotation)")
    print("  [, ]: Move selected joint backward/forward")
    print()
    print("Arm Control:")
    print("  Arrow keys: Move arm end-effector")
    print("  P, ;: Move arm up/down")
    print("  N, B: Rotate arm")
    print("  O, U: Rotate arm")
    print("  V, C: Rotate arm")
    print()
    print("Gripper Control:")
    print("  T: Toggle gripper open/close")

    # Loop control until user quits
    max_steps = 2000
    step = 0

    images = []
    healths = []
    temperatures = []

    while step != max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1

        rgb_img = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_img = rgb_img.cpu().numpy()[:, :, :3]
        rgb_img = cv2.resize(rgb_img, (512, 512))
        images.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # Get apple health and temperature
        apple_temp = apple.states[object_states.Temperature].get_value()
        apple_health = apple.health
        
        healths.append(apple_health)
        temperatures.append(apple_temp)

    # Clean up camera mover
    camera_mover.clear()

    # Save video
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create videos_and_images directory if it doesn't exist
    os.makedirs('videos_and_images', exist_ok=True)

    avi_path = 'videos_and_images/thermal_damage_teleop.avi'
    mp4_path = 'videos_and_images/thermal_damage_teleop.mp4'
    out = cv2.VideoWriter(avi_path, fourcc, 30, (width, height))

    for i, image in enumerate(images):
        # Add health captions
        frame_copy = image.copy()
        y_pos = 30
        cv2.putText(frame_copy, f"Apple Health: {healths[i]:.2f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add temperature
        y_pos += 30
        cv2.putText(frame_copy, f"Apple Temp: {temperatures[i]:.1f}", (10, y_pos),
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

    # Create temperature value animation
    # Set up the figure and axis with matching dimensions
    fig, ax = plt.subplots(figsize=(6.83, 6.83))  # Makes it match 512x512 with default DPI of 75
    line, = ax.plot([], [], lw=2)

    # Set the limits of the plot
    ax.set_xlim(1, len(temperatures))
    ax.set_ylim(min(temperatures), max(temperatures) * 1.1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Apple Temperature Over Time')
    plt.tight_layout()  # Adjust layout to fit in figure

    # Initialization function
    def init():
        line.set_data([], [])
        return line,

    # Animation function which updates the figure
    def animate(i):
        x = list(range(1, i + 2))
        y = temperatures[:i + 1]
        line.set_data(x, y)
        return line,

    # Create an animation object
    ani = animation.FuncAnimation(
        fig, animate, 
        init_func=init,
        frames=len(temperatures),
        interval=1000/30,
        blit=True
    )

    # Save the animation as a video file - using exact working configuration
    temp_mp4 = 'videos_and_images/thermal_damage_temp_plot.mp4'
    # Save animation using working configuration from teleop_tiago_bowl.py
    writer = animation.FFMpegWriter(
        fps=30,
        codec='mpeg4',
        extra_args=['-vcodec', 'mpeg4', '-qscale', '5']
    )
    ani.save(temp_mp4, writer=writer)
    plt.close()

    # Combine videos side by side using mpeg4 codec
    combined_mp4 = 'videos_and_images/thermal_damage_combined_view.mp4'
    subprocess.run([
        'ffmpeg', '-y',
        '-i', mp4_path,
        '-i', temp_mp4,
        '-filter_complex',
        '[0:v][1:v]scale2ref=oh*dar:ih[v0][v1];[v0][v1]hstack=inputs=2[v]',  # Scale videos to match height
        '-map', '[v]',
        '-vcodec', 'mpeg4',  # Use mpeg4 codec
        '-q:v', '5',         # Quality scale (fixed ambiguous -qscale)
        combined_mp4
    ], check=True)

    # Always shut down the environment cleanly at the end
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main() 