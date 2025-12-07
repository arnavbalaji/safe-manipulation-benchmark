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
from omnigibson.object_states import Open
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
    "laptop": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "laptop",
        "model": "nvulcs",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 1, 0],
        "scale": [1., 1., 1.],
        "damage_params": PARAMS["default"]
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
        "scale": [1.0, 1.0, 1.3],
        "damage_params": PARAMS["coffee_table"]
    },
    "coffee_table3": {
        "type": "DatasetObject",    
        "name": "coffee_table3",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [-1.0, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.3],
        "damage_params": PARAMS["coffee_table"]
    },
    "coffee_table4": {
        "type": "DatasetObject",    
        "name": "coffee_table4",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [0.0, -0.5, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.3],
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

    objects = [OBJECT_CONFIGS["coffee_table"], OBJECT_CONFIGS[chosen_object], OBJECT_CONFIGS["coffee_table2"], OBJECT_CONFIGS["coffee_table3"], OBJECT_CONFIGS["coffee_table4"]]
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

    # If the chosen object is a laptop, open it at start if supported
    if chosen_object == "laptop":
        try:
            if Open in obj.states:
                # Try ratio openness first, then boolean
                try:
                    obj.states[Open].set_value(1.0)
                except Exception:
                    obj.states[Open].set_value(True)
                print("✅ Laptop opened at start")
        except Exception as e:
            print(f"⚠️ Could not open laptop at start: {e}")
    
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
    max_steps = 300
    step = 0
    fps = 10

    images = []
    robot_healths = []
    obj_healths = []
    object_speeds = []
    robot_speeds = []
    # New series for object plots
    obj_impact_forces = []
    obj_net_impulses = []
    obj_sustained_curr = []

    while step != max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1

        rgb_img = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_img = rgb_img.cpu().numpy()[:, :, :3]
        rgb_img = cv2.resize(rgb_img, (512, 512))
        images.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        obj = env.scene.object_registry("name", "target_object")
        robot_healths.append(robot.health)
        obj_healths.append(obj.health)

        # Keep laptop open every step if selected
        if chosen_object == "laptop":
            try:
                if Open in obj.states:
                    val = obj.states[Open].get_value()
                    needs_open = False
                    try:
                        # Numeric openness (0..1)
                        needs_open = float(val) < 1.0
                    except Exception:
                        needs_open = not bool(val)
                    if needs_open:
                        try:
                            obj.states[Open].set_value(1.0)
                        except Exception:
                            obj.states[Open].set_value(True)
            except Exception:
                pass
        # print(obj.link_healths)
        # breakpoint()
        # print(obj.link_healths)

        # Record object velocity magnitude from damage evaluator (max over links)
        try:
            dmg_eval = obj.damage_evaluators[0]
            if hasattr(dmg_eval, 'prev_link_velocities') and dmg_eval.prev_link_velocities:
                speeds = [th.norm(v).item() for v in dmg_eval.prev_link_velocities.values()]
                object_speeds.append(max(speeds) if len(speeds) > 0 else 0.0)
            else:
                object_speeds.append(0.0)
            # Collect object impact and sustained (average within env step over physics substeps, then clear)
            if hasattr(dmg_eval, 'impact_forces_by_link') and dmg_eval.impact_forces_by_link:
                per_link_avgs = [sum(vals) / len(vals) for vals in dmg_eval.impact_forces_by_link.values() if vals]
                obj_impact_forces.append(max(per_link_avgs) if per_link_avgs else 0.0)
                dmg_eval.impact_forces_by_link = {}
            else:
                obj_impact_forces.append(0.0)

            # Current sustained force (pre-memory): use sustained_forces_by_link within this env step
            sustained_fallback_value = 0.0
            if hasattr(dmg_eval, 'sustained_forces_by_link') and dmg_eval.sustained_forces_by_link:
                per_link_avgs_s = [sum(vals) / len(vals) for vals in dmg_eval.sustained_forces_by_link.values() if vals]
                curr_sustained = max(per_link_avgs_s) if per_link_avgs_s else 0.0
                obj_sustained_curr.append(curr_sustained)
                sustained_fallback_value = curr_sustained
                dmg_eval.sustained_forces_by_link = {}
            else:
                obj_sustained_curr.append(0.0)
            # Use sustained running total if available; fallback to per-step sustained force average
            if hasattr(dmg_eval, '_sustained_running_total') and dmg_eval._sustained_running_total:
                obj_net_impulses.append(max(dmg_eval._sustained_running_total.values()))
            else:
                obj_net_impulses.append(sustained_fallback_value)
        except Exception:
            object_speeds.append(0.0)
            obj_impact_forces.append(0.0)
            obj_net_impulses.append(0.0)

        # Record robot velocity magnitude from damage evaluator (max over links)
        try:
            r_dmg_eval = robot.damage_evaluators[0]
            if hasattr(r_dmg_eval, 'prev_link_velocities') and r_dmg_eval.prev_link_velocities:
                r_speeds = [th.norm(v).item() for v in r_dmg_eval.prev_link_velocities.values()]
                robot_speeds.append(max(r_speeds) if len(r_speeds) > 0 else 0.0)
            else:
                robot_speeds.append(0.0)
        except Exception:
            robot_speeds.append(0.0)


    # Clean up camera mover
    camera_mover.clear()

    robot_force_values = robot.damage_evaluators[0].force_values
    object_force_values = obj.damage_evaluators[0].force_values

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
        # cv2.putText(frame_copy, f"Robot Health: {robot_healths[i]:.2f}", (10, y_pos),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add object health
        y_pos += 30
        # cv2.putText(frame_copy, f"Object Health: {obj_healths[i]:.2f}", (10, y_pos),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        out.write(np.ascontiguousarray(frame_copy, dtype=np.uint8))
    out.release()

    # Convert AVI to MP4 using ffmpeg
    subprocess.run([
        'ffmpeg', '-y', '-i', avi_path,
        '-c:v', 'mpeg4', mp4_path
    ], check=True)
    
    # Clean up AVI file
    os.remove(avi_path)

    # Create force value animation for ROBOT
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Set up the figure and axis with matching dimensions
    fig, ax = plt.subplots(figsize=(6.83, 6.83))  # Makes it match 512x512 with default DPI of 75
    line, = ax.plot([], [], lw=2)

    # Set the limits of the plot
    ax.set_xlim(1, len(robot_force_values))
    ax.set_ylim(min(robot_force_values) if robot_force_values else 0,
                (max(robot_force_values) * 1.1) if robot_force_values else 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Force')
    ax.set_title('Robot Forces Over Time')
    plt.tight_layout()  # Adjust layout to fit in figure

    # Initialization function
    def init():
        line.set_data([], [])
        return line,

    # Animation function which updates the figure
    def animate(i):
        x = list(range(1, i + 2))
        y = robot_force_values[:i + 1]
        line.set_data(x, y)
        return line,

    # Create an animation object
    ani = animation.FuncAnimation(
        fig, animate, 
        init_func=init,
        frames=len(robot_force_values),
        interval=1000/fps,
        blit=True
    )

    # Save the animation as a video file - using exact working configuration
    robot_force_mp4 = f'videos_and_images/{chosen_object}_force_plot_robot.mp4'
    # Save animation using working configuration from animate_values.py
    writer = animation.FFMpegWriter(
        fps=fps,
        codec='mpeg4',
        extra_args=['-vcodec', 'mpeg4', '-qscale', '5']
    )
    ani.save(robot_force_mp4, writer=writer)
    plt.close()

    # Create force value animation for OBJECT
    fig, ax = plt.subplots(figsize=(6.83, 6.83))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(1, len(object_force_values))
    ax.set_ylim(min(object_force_values) if object_force_values else 0,
                (max(object_force_values) * 1.1) if object_force_values else 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Force')
    ax.set_title('Object Forces Over Time')
    plt.tight_layout()

    def init_obj():
        line.set_data([], [])
        return line,

    def animate_obj(i):
        x = list(range(1, i + 2))
        y = object_force_values[:i + 1]
        line.set_data(x, y)
        return line,

    ani_obj = animation.FuncAnimation(
        fig, animate_obj,
        init_func=init_obj,
        frames=len(object_force_values),
        interval=1000/fps,
        blit=True
    )

    object_force_mp4 = f'videos_and_images/{chosen_object}_force_plot_object.mp4'
    ani_obj.save(object_force_mp4, writer=writer)
    plt.close()

    # Object IMPACT force plot (new)
    fig, ax = plt.subplots(figsize=(9.6, 9.6))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(1, len(obj_impact_forces))
    ax.set_ylim(min(obj_impact_forces) if obj_impact_forces else 0,
                (max(obj_impact_forces) * 1.1) if obj_impact_forces else 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Impact Force')
    ax.set_title('Object Impact Force (per step)')
    plt.tight_layout()

    def init_obj_imp():
        line.set_data([], [])
        return line,

    def animate_obj_imp(i):
        x = list(range(1, i + 2))
        y = obj_impact_forces[:i + 1]
        line.set_data(x, y)
        return line,

    ani_obj_imp = animation.FuncAnimation(
        fig, animate_obj_imp,
        init_func=init_obj_imp,
        frames=len(obj_impact_forces),
        interval=1000/fps,
        blit=True
    )

    object_impact_mp4 = f'videos_and_images/{chosen_object}_impact_plot_object.mp4'
    ani_obj_imp.save(object_impact_mp4, writer=writer)
    plt.close()

    # # Object SUSTAINED running total (deprecated for this final video)
    # fig, ax = plt.subplots(figsize=(9.6, 9.6))
    # line, = ax.plot([], [], lw=2)
    # ax.set_xlim(1, len(obj_net_impulses))
    # ax.set_ylim(min(obj_net_impulses) if obj_net_impulses else 0,
    #             (max(obj_net_impulses) * 1.1) if obj_net_impulses else 1)
    # ax.set_xlabel('Timestep')
    # ax.set_ylabel('Sustained Force (running total)')
    # ax.set_title('Object Sustained Forces (running total)')
    # plt.tight_layout()
    # def init_obj_impulse():
    #     line.set_data([], [])
    #     return line,
    # def animate_obj_impulse(i):
    #     x = list(range(1, i + 2))
    #     y = obj_net_impulses[:i + 1]
    #     line.set_data(x, y)
    #     return line,
    # ani_obj_impulse = animation.FuncAnimation(
    #     fig, animate_obj_impulse,
    #     init_func=init_obj_impulse,
    #     frames=len(obj_net_impulses),
    #     interval=1000/fps,
    #     blit=True
    # )
    # object_net_impulse_mp4 = f'videos_and_images/{chosen_object}_net_impulse_plot_object.mp4'
    # ani_obj_impulse.save(object_net_impulse_mp4, writer=writer)
    # plt.close()

    # Object SUSTAINED force per step (pre-memory)
    fig, ax = plt.subplots(figsize=(9.6, 9.6))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(1, len(obj_sustained_curr))
    ax.set_ylim(min(obj_sustained_curr) if obj_sustained_curr else 0,
                (max(obj_sustained_curr) * 1.1) if obj_sustained_curr else 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Sustained Force (per step)')
    ax.set_title('Object Sustained Force (per step)')
    plt.tight_layout()

    def init_obj_sus():
        line.set_data([], [])
        return line,

    def animate_obj_sus(i):
        x = list(range(1, i + 2))
        y = obj_sustained_curr[:i + 1]
        line.set_data(x, y)
        return line,

    ani_obj_sus = animation.FuncAnimation(
        fig, animate_obj_sus,
        init_func=init_obj_sus,
        frames=len(obj_sustained_curr),
        interval=1000/fps,
        blit=True
    )

    object_sustained_mp4 = f'videos_and_images/{chosen_object}_sustained_per_step_plot_object.mp4'
    ani_obj_sus.save(object_sustained_mp4, writer=writer)
    plt.close()

    # Create velocity magnitude animation for OBJECT
    fig, ax = plt.subplots(figsize=(6.83, 6.83))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(1, len(object_speeds))
    ax.set_ylim(min(object_speeds) if object_speeds else 0,
                (max(object_speeds) * 1.1) if object_speeds else 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity Magnitude (units/step)')
    ax.set_title('Object Velocity Over Time')
    plt.tight_layout()

    def init_vel():
        line.set_data([], [])
        return line,

    def animate_vel(i):
        x = list(range(1, i + 2))
        y = object_speeds[:i + 1]
        line.set_data(x, y)
        return line,

    ani_vel = animation.FuncAnimation(
        fig, animate_vel,
        init_func=init_vel,
        frames=len(object_speeds),
        interval=1000/fps,
        blit=True
    )

    object_velocity_mp4 = f'videos_and_images/{chosen_object}_velocity_plot_object.mp4'
    ani_vel.save(object_velocity_mp4, writer=writer)
    plt.close()

    # Create velocity magnitude animation for ROBOT
    fig, ax = plt.subplots(figsize=(6.83, 6.83))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(1, len(robot_speeds))
    ax.set_ylim(min(robot_speeds) if robot_speeds else 0,
                (max(robot_speeds) * 1.1) if robot_speeds else 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity Magnitude (units/step)')
    ax.set_title('Robot Velocity Over Time')
    plt.tight_layout()

    def init_vel_robot():
        line.set_data([], [])
        return line,

    def animate_vel_robot(i):
        x = list(range(1, i + 2))
        y = robot_speeds[:i + 1]
        line.set_data(x, y)
        return line,

    ani_vel_robot = animation.FuncAnimation(
        fig, animate_vel_robot,
        init_func=init_vel_robot,
        frames=len(robot_speeds),
        interval=1000/fps,
        blit=True
    )

    robot_velocity_mp4 = f'videos_and_images/{chosen_object}_velocity_plot_robot.mp4'
    ani_vel_robot.save(robot_velocity_mp4, writer=writer)
    plt.close()

    # Build plots on the right: impact (top) + sustained per-step (bottom)
    two_plots_stacked_mp4 = f'videos_and_images/{chosen_object}_two_plots_stacked.mp4'
    subprocess.run([
        'ffmpeg', '-y',
        '-i', object_impact_mp4,
        '-i', object_sustained_mp4,
        '-filter_complex',
        '[0:v]scale=1440:720,setsar=1[v0];'
        '[1:v]scale=1440:720,setsar=1[v1];'
        '[v0][v1]vstack=inputs=2[right]',
        '-map', '[right]',
        '-vcodec', 'mpeg4',
        '-q:v', '5',
        two_plots_stacked_mp4
    ], check=True)

    # Final combined video: teleop on left + two plots (impact, sustained per-step) on right
    mech_damage_mp4 = f'videos_and_images/{chosen_object}_mech_damage.mp4'
    subprocess.run([
        'ffmpeg', '-y',
        '-i', mp4_path,
        '-i', two_plots_stacked_mp4,
        '-filter_complex',
        '[0:v]scale=-2:1440,setsar=1,pad=1440:1440:(1440-iw)/2:(1440-ih)/2[left];'
        '[1:v]scale=1440:1440,setsar=1[right];'
        '[left][right]hstack=inputs=2[v]',
        '-map', '[v]',
        '-vcodec', 'mpeg4',
        '-q:v', '5',
        mech_damage_mp4
    ], check=True)


    # Always shut down the environment cleanly at the end
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main()