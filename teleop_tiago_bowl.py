"""
Example script demo'ing robot control.

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
from omnigibson.object_states import OnTop
from omnigibson.utils.ui_utils import KeyboardRobotController
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

OBJECT_CONFIGS = {
    "baseball": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "baseball",
        "model": "zanmar",
        "position": [0.1, 0.0, 3.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.5, 1.5, 1.5],
        "damage_params": PARAMS["baseball"]
    },
    "bowl": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "wineglass",
        "model": "cmdagy",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],  # Identity quaternion for upright orientation
        # "scale": [0.5, 0.5, 0.4],  # Reduced scale to make it lighter and easier to grasp
        # "scale": [0.75, 0.75, 0.65],
        "scale": [1.5, 1.5, 1.5],
        "damage_params": PARAMS["bowl"],
        "mass": 0.005,  # Very light mass for easier manipulation
        "friction": 1.0,  # Higher friction for better grip
        "restitution": 0.1,  # Low bounciness
    },
    "coffee_table": {
        "type": "DatasetObject",    
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.0],
        "damage_params": PARAMS["coffee_table"]
    },
    "coffee_table2": {
        "type": "DatasetObject",    
        "name": "coffee_table2",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [1.0, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 0.2],
        "damage_params": PARAMS["coffee_table"]
    },
    "coffee_table3": {
        "type": "DatasetObject",    
        "name": "coffee_table3",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [-1.0, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 0.2],
        "damage_params": PARAMS["coffee_table"]
    },
    "coffee_table4": {
        "type": "DatasetObject",    
        "name": "coffee_table4",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [0.0, -0.5, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 0.2],
        "damage_params": PARAMS["coffee_table"]
    }
}


def main():
    """
    Minimal teleop demo with Tiago in an empty scene.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Always use empty scene and Tiago robot
    scene_cfg = {"type": "Scene"}
    robot0_cfg = dict()
    robot0_cfg["type"] = "Tiago"
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True
    robot0_cfg["position"] = [0., 0.65, 0.0]  # Position next to the table
    robot0_cfg["orientation"] = [0, 0, -1, 1]  # Facing the table
    robot0_cfg["grasping_mode"] = "assisted"
    robot0_cfg["damage_params"] = PARAMS["tiago_robot"]

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    chosen_object = "baseball"

    objects = [OBJECT_CONFIGS["coffee_table"], OBJECT_CONFIGS[chosen_object]]
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
    

    # Update the simulator's viewer camera's pose so it points towards the robot
    # og.sim.viewer_camera.set_position_orientation(
    #     position=th.tensor([1.46949, -3.97358, 2.21529]),
    #     orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    # )

    # Reset environment and robot
    env.reset()
    robot.reset()

    coffee_table = env.scene.object_registry("name", "coffee_table")
    obj = env.scene.object_registry("name", "target_object")
    
    # First set the OnTop state
    assert obj.states[OnTop].set_value(coffee_table, True), "Failed to set OnTop state"
    
    # Get table top position and place bowl there
    table_pos = coffee_table.get_position()

    # Let physics settle
    for _ in range(20):  # Increased settling steps
        og.sim.step()

    # save_state_path = "safe-manipulation-benchmark/grasp_save_state_bowl2.json"
    # print(f"🔄 Loading saved simulation state from: {save_state_path}")
    # og.sim.restore(scene_files=[save_state_path])
    # env.inialize_damageable_objects()
    # print("✅ Successfully loaded saved simulation state")
    # if os.path.exists(save_state_path):
    #     print(f"🔄 Loading saved simulation state from: {save_state_path}")
    #     og.sim.restore(scene_files=[save_state_path])
    #     env.inialize_damageable_objects()
    #     print("✅ Successfully loaded saved simulation state")
    # else:
    #     print(f"📝 No saved state found at: {save_state_path}")
    #     print("Starting with fresh environment...")


    for _ in range(20):  # Increased settling steps
        og.sim.step()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    
    # Enable camera teleoperation with custom key bindings (excluding 'T' key)
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
                # Removed T key to avoid conflict with gripper control
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
    
    # Function to save simulation state with breakpoint
    def save_sim_state():
        filepath = f"safe-manipulation-benchmark/grasp_save_state_bowl2.json"
        og.sim.save(json_paths=[filepath])
        print(f"✅ Simulation state saved to: {filepath}")
        breakpoint()
    
    # Register TAB key for breakpoint and state saving
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Breakpoint and save simulation state",
        callback_fn=save_sim_state,
    )

    # Print out relevant keyboard info
    action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = 200
    step = 0
    fps = 10

    images = []
    healths = []
    link_healths = []
    damage_statuses = []

    while step != max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1

        rgb_img = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_img = rgb_img.cpu().numpy()[:, :, :3]
        rgb_img = cv2.resize(rgb_img, (512, 512))
        images.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        obj = env.scene.object_registry("name", "target_object")
        # obj = env.robots[0]
        healths.append(obj.health)
        damage_statuses.append(obj.damage_status)
        link_healths.append(obj.link_healths.copy())
        # print(obj.link_healths)
        # breakpoint()
        # print(obj.link_healths)



    # Clean up camera mover
    camera_mover.clear()

    force_values = obj.damage_evaluators[0].force_values

    # Save video
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create videos_and_images directory if it doesn't exist
    os.makedirs('videos_and_images', exist_ok=True)
    
    avi_path = f'videos_and_images/{chosen_object}_grasp_teleop.avi'
    mp4_path = f'videos_and_images/{chosen_object}_grasp_teleop.mp4'
    out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    for i, image in enumerate(images):
        # Add health captions
        frame_copy = image.copy()
        y_pos = 30
        cv2.putText(frame_copy, f"Robot Health: {healths[i]:.2f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Handle link health wrapping
        y_pos += 30
        # link_health_text = f"Link Health: {', '.join([f'{key}: {value:.2f}' for key, value in link_healths[i].items() if 'arm' in key or 'gripper' in key])}"
        # words = link_health_text.split()
        # current_line = ""
        # for word in words:
        #     test_line = current_line + " " + word if current_line else word
        #     if len(test_line) * 10 < width - 20:  # Approximate character width
        #         current_line = test_line
        #     else:
        #         cv2.putText(frame_copy, current_line, (10, y_pos),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        #         y_pos += 25
        #         current_line = word
        # if current_line:
        #     cv2.putText(frame_copy, current_line, (10, y_pos),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        #     y_pos += 25

        # Add damage status
        cv2.putText(frame_copy, f"Robot Damage Status: {damage_statuses[i]}", (10, y_pos),
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

    # Create force value animation
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Set up the figure and axis with matching dimensions
    fig, ax = plt.subplots(figsize=(6.83, 6.83))  # Makes it match 512x512 with default DPI of 75
    line, = ax.plot([], [], lw=2)

    # Set the limits of the plot
    ax.set_xlim(1, len(force_values))
    ax.set_ylim(min(force_values), max(force_values) * 1.1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Force')
    ax.set_title('Force Values Over Time')
    plt.tight_layout()  # Adjust layout to fit in figure

    # Initialization function
    def init():
        line.set_data([], [])
        return line,

    # Animation function which updates the figure
    def animate(i):
        x = list(range(1, i + 2))
        y = force_values[:i + 1]
        line.set_data(x, y)
        return line,

    # Create an animation object
    ani = animation.FuncAnimation(
        fig, animate, 
        init_func=init,
        frames=len(force_values),
        interval=1000/fps,
        blit=True
    )

    # Save the animation as a video file - using exact working configuration
    force_mp4 = f'videos_and_images/{chosen_object}_force_plot.mp4'
    # Save animation using working configuration from animate_values.py
    writer = animation.FFMpegWriter(
        fps=fps,
        codec='mpeg4',
        extra_args=['-vcodec', 'mpeg4', '-qscale', '5']
    )
    ani.save(force_mp4, writer=writer)
    plt.close()

    # Combine videos side by side using mpeg4 codec
    combined_mp4 = f'videos_and_images/{chosen_object}_combined_view.mp4'
    subprocess.run([
        'ffmpeg', '-y',
        '-i', mp4_path,
        '-i', force_mp4,
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