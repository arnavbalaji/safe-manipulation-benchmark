"""
Teleop demo with Tiago and a bottom cabinet (bamfsz). Tracks drawer damage.
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
from omnigibson.utils.ui_utils import KeyboardRobotController
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

OBJECT_CONFIGS = {
    "bottom_cabinet": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "bottom_cabinet",
        "model": "pkdnbu",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.5],
        "damage_params": PARAMS.get("drawer", PARAMS["drawer"])  # fallback-safe
    },
    "bottom_cabinet2": {
        "type": "DatasetObject",
        "name": "bottom_cabinet2",
        "category": "bottom_cabinet",
        "model": "pkdnbu",
        "position": [1.0, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.3],
        "damage_params": PARAMS.get("drawer", PARAMS["drawer"])  # fallback-safe
    },
    "bottom_cabinet3": {
        "type": "DatasetObject",
        "name": "bottom_cabinet3",
        "category": "bottom_cabinet",
        "model": "pkdnbu",
        "position": [-1.0, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.3],
        "damage_params": PARAMS.get("drawer", PARAMS["drawer"])  # fallback-safe
    },
    "bottom_cabinet4": {
        "type": "DatasetObject",
        "name": "bottom_cabinet4",
        "category": "bottom_cabinet",
        "model": "pkdnbu",
        "position": [0.0, -0.5, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.3],
        "damage_params": PARAMS.get("drawer", PARAMS["drawer"])  # fallback-safe
    }
}


def main():
    """
    Minimal teleop demo with Tiago and a bottom cabinet.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Always use empty scene and Tiago robot
    scene_cfg = {"type": "Scene"}
    robot0_cfg = dict()
    robot0_cfg["type"] = "Tiago"
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True
    robot0_cfg["position"] = [0., 0.65, 0.0]
    robot0_cfg["orientation"] = [0, 0, -1, 1]
    robot0_cfg["grasping_mode"] = "assisted"
    robot0_cfg["damage_params"] = PARAMS["tiago_robot"]

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    chosen_object = "drawer"

    objects = [OBJECT_CONFIGS["bottom_cabinet"]]
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
    controller_config["gripper_left"]["inverted"] = True
    controller_config["gripper_right"]["inverted"] = True
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_file()

    # Reset environment and robot
    env.reset()
    robot.reset()

    # Determine link_1 name once for plotting
    obj_ref = env.scene.object_registry("name", "target_object")
    link_names = list(obj_ref.links.keys())
    if "link_3" in link_names:
        target_link_name = "link_3"
    else:
        # Try to pick the fourth link if it exists; otherwise fall back to first available
        target_link_name = link_names[3] if len(link_names) > 3 else (link_names[0] if len(link_names) > 0 else "")

    # Let physics settle
    for _ in range(20):
        og.sim.step()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Enable camera teleoperation with custom key bindings (excluding 'T' key)
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

    # Function to save simulation state with breakpoint
    def save_sim_state():
        filepath = f"safe-manipulation-benchmark/grasp_save_state_drawer.json"
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

    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = 1000
    step = 0
    fps = 10

    images = []
    robot_healths = []
    obj_healths = []
    object_speeds = []
    robot_speeds = []
    # Series for plots (max of per-link averages across all links)
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

        # Record object velocity magnitude from damage evaluator (max over links)
        try:
            dmg_eval = obj.damage_evaluators[0]
            if hasattr(dmg_eval, 'prev_link_velocities') and dmg_eval.prev_link_velocities:
                speeds = [th.norm(v).item() for v in dmg_eval.prev_link_velocities.values()]
                object_speeds.append(max(speeds) if len(speeds) > 0 else 0.0)
            else:
                object_speeds.append(0.0)

            # Compute and print per-link damages (using latest impact and sustained running total)
            try:
                all_link_names = set(list(getattr(dmg_eval, 'impact_forces_by_link', {}).keys()) + list(getattr(dmg_eval, '_sustained_running_total', {}).keys()))
                per_link_damage = {}
                for lname in all_link_names:
                    # Latest impact value for this link (if present)
                    imp_list = dmg_eval.impact_forces_by_link.get(lname, []) if hasattr(dmg_eval, 'impact_forces_by_link') else []
                    imp_last = imp_list[-1] if len(imp_list) > 0 else 0.0
                    # Sustained running total C(t)
                    c_curr = 0.0
                    if hasattr(dmg_eval, '_sustained_running_total'):
                        c_curr = float(dmg_eval._sustained_running_total.get(lname, 0.0))
                    # Damage reconstruction
                    d_imp = max(0.0, (imp_last - float(dmg_eval.impact_threshold))) * float(dmg_eval.impact_scale)
                    d_sus = max(0.0, (c_curr - float(dmg_eval.sustained_threshold))) * float(dmg_eval.sustained_scale)
                    per_link_damage[lname] = d_imp + d_sus
                if per_link_damage:
                    clean = lambda s: s.split('/')[-1]
                    print({clean(k): round(v, 4) for k, v in per_link_damage.items()})
            except Exception:
                pass

            # Collect IMPACT: max of per-link averages across all links
            impact_val = 0.0
            if hasattr(dmg_eval, 'impact_forces_by_link') and dmg_eval.impact_forces_by_link:
                per_link_avgs = [sum(vals) / len(vals) for vals in dmg_eval.impact_forces_by_link.values() if vals]
                impact_val = float(max(per_link_avgs)) if per_link_avgs else 0.0
                # Clear after reading
                dmg_eval.impact_forces_by_link = {}
            obj_impact_forces.append(impact_val)

            # Current SUSTAINED (pre-memory): max of per-link averages across all links
            if hasattr(dmg_eval, 'sustained_forces_by_link') and dmg_eval.sustained_forces_by_link:
                per_link_avgs_s = [sum(vals) / len(vals) for vals in dmg_eval.sustained_forces_by_link.values() if vals]
                curr_sustained = float(max(per_link_avgs_s)) if per_link_avgs_s else 0.0
                obj_sustained_curr.append(curr_sustained)
                dmg_eval.sustained_forces_by_link = {}
            else:
                obj_sustained_curr.append(0.0)

            # Collect link_3 sustained running total (kept for compatibility)
            link3_sus = 0.0
            if hasattr(dmg_eval, '_sustained_running_total') and dmg_eval._sustained_running_total:
                link3_sus = float(dmg_eval._sustained_running_total.get(target_link_name, 0.0))
            obj_net_impulses.append(link3_sus)
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

    os.makedirs('videos_and_images', exist_ok=True)

    avi_path = f'videos_and_images/{chosen_object}_grasp_teleop.avi'
    mp4_path = f'videos_and_images/{chosen_object}_grasp_teleop.mp4'
    out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    for i, image in enumerate(images):
        frame_copy = image.copy()
        y_pos = 30
        # cv2.putText(frame_copy, f"Robot Health: {robot_healths[i]:.2f}", (10, y_pos),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
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
    os.remove(avi_path)

    # Create force value animation for ROBOT
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(6.83, 6.83))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(1, len(robot_force_values))
    ax.set_ylim(min(robot_force_values) if robot_force_values else 0,
                (max(robot_force_values) * 1.1) if robot_force_values else 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Force')
    ax.set_title('Robot Forces Over Time')
    plt.tight_layout()

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = list(range(1, i + 2))
        y = robot_force_values[:i + 1]
        line.set_data(x, y)
        return line,

    ani = animation.FuncAnimation(
        fig, animate,
        init_func=init,
        frames=len(robot_force_values),
        interval=1000/fps,
        blit=True
    )

    robot_force_mp4 = f'videos_and_images/{chosen_object}_force_plot_robot.mp4'
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

    # Object IMPACT force plot (max over links)
    fig, ax = plt.subplots(figsize=(9.6, 9.6))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(1, len(obj_impact_forces))
    ax.set_ylim(min(obj_impact_forces) if obj_impact_forces else 0,
                (max(obj_impact_forces) * 1.1) if obj_impact_forces else 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Impact Force (max over links)')
    ax.set_title('Impact Force (per step)')
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

    # SUSTAINED force per step (pre-memory, max over links)
    fig, ax = plt.subplots(figsize=(9.6, 9.6))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(1, len(obj_sustained_curr))
    ax.set_ylim(min(obj_sustained_curr) if obj_sustained_curr else 0,
                (max(obj_sustained_curr) * 1.1) if obj_sustained_curr else 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Sustained Force (max over links, per step)')
    ax.set_title('Sustained Force (per step)')
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

    # # Link_3 SUSTAINED running total plot (deprecated for final video)
    # fig, ax = plt.subplots(figsize=(9.6, 9.6))
    # line, = ax.plot([], [], lw=2)
    # ax.set_xlim(1, len(obj_net_impulses))
    # ax.set_ylim(min(obj_net_impulses) if obj_net_impulses else 0,
    #             (max(obj_net_impulses) * 1.1) if obj_net_impulses else 1)
    # ax.set_xlabel('Timestep')
    # ax.set_ylabel('Sustained Force (link_3, running total)')
    # ax.set_title('Link 3 Sustained Forces (running total)')
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

    # Combined right-side video with two plots (impact + sustained per-step)
    two_plots_stacked_mp4 = f'videos_and_images/{chosen_object}_two_plots_stacked.mp4'
    subprocess.run([
        'ffmpeg', '-y',
        '-i', object_impact_mp4,
        '-i', object_sustained_mp4,
        '-filter_complex',
        '[0:v]scale=1440:720,setsar=1[v0];[1:v]scale=1440:720,setsar=1[v1];[v0][v1]vstack=inputs=2[right]',
        '-map', '[right]',
        '-vcodec', 'mpeg4',
        '-q:v', '5',
        two_plots_stacked_mp4
    ], check=True)

    # Final combined video
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

    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main()
