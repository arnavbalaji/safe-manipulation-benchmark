import os
import sys

import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.object_states import OnTop, Inside
from omnigibson.macros import gm

# Extra imports for video / plotting
import cv2
import numpy as np
import subprocess

# Force headless plotting to avoid Qt/xcb issues
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")
import yaml
import json

from omnigibson.object_states import ToggledOn
from omnigibson.utils.constants import ParticleModifyCondition
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_OBJECT_STATES = True


def main():
    """
    Load a simple damageable environment and allow keyboard teleoperation.
    """
    # Resolve config paths next to this script
    base_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(base_dir, "task1.yaml")
    json_path = os.path.join(base_dir, "task1_new.json")

    # Prefer JSON scene file; if present, edit it in-place to set bowl damage params on mugs
    cfg = None
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                scene_dict = json.load(f)
            # Hardcode bowl damage params onto all mugs explicitly (no loops over registry)
            init_info = scene_dict.get("objects_info", {}).get("init_info", {})
            for mug_key in [
                "mug",
                "mug_ppzttc",
                "mug_ntgftr_1",
                "mug_ntgftr_2",
                "mug_ntgftr_3",
            ]:
                if mug_key in init_info and "args" in init_info[mug_key]:
                    init_info[mug_key]["args"]["params"] = PARAMS["mug"]
            # Print objects so you can edit others manually if desire
            init_info["box_of_crackers"]["args"]["params"] = PARAMS["box_of_crackers"]
            init_info["dish_rack"]["args"]["params"] = PARAMS["default"]
            init_info["robot_nttbgn"]["args"]["params"] = PARAMS["tiago_robot"]
            
            # Add toggleable ability and particle abilities to the commercial kitchen sink
            if "commercial_kitchen_sink" in init_info and "args" in init_info["commercial_kitchen_sink"]:
                sink_args = init_info["commercial_kitchen_sink"]["args"]
                if "abilities" not in sink_args:
                    sink_args["abilities"] = {}
                
                # Add toggleable ability
                sink_args["abilities"]["toggleable"] = {}
                
                # Add particle source ability (water flows when toggled on)
                sink_args["abilities"]["particleSource"] = {
                    "conditions": {
                        "water": [
                            (ParticleModifyCondition.TOGGLEDON, True)  # Must be toggled on for water source to be active
                        ]
                    },
                    "initial_speed": 5.0  # Increased water flow speed for more visible effect
                }
                
                # Add particle sink ability (drains water particles)
                sink_args["abilities"]["particleSink"] = {
                    "conditions": {
                        "water": []  # No conditions, always sinking nearby particles
                    }
                }
                
                print("✅ Added toggleable, particleSource, and particleSink abilities to commercial_kitchen_sink")
            
            # Write back the modified JSON before loading
            with open(json_path, "w") as f:
                json.dump(scene_dict, f)
            # Build config to load from scene_file (do NOT add objects here)
            cfg = {"scene": {"type": "Scene", "scene_file": json_path}}
        except Exception:
            # Fallback to YAML if JSON editing fails
            cfg = yaml_path
    else:
        # Fallback: YAML path (will not edit mugs if JSON absent)
        cfg = yaml_path

    # Create environment
    env = DamageableEnvironment(configs=cfg)

    # Grab the single robot (Tiago) and set controller configuration suitable for teleop
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
    # Tiago grippers are inverted
    controller_config["gripper_left"]["inverted"] = True
    controller_config["gripper_right"]["inverted"] = True
    robot.reload_controllers(controller_config=controller_config)

    # Persist initial state after controller reload
    env.scene.update_initial_file()

    # Reset before teleoperation
    env.reset()
    robot.reset()

    # Get reference to the sink for toggle control
    try:
        sink = env.scene.object_registry("name", "commercial_kitchen_sink")
        print("✅ Commercial kitchen sink loaded successfully")
        
        # Check sink states and abilities
        if hasattr(sink, 'states'):
            print(f"🔍 Sink states: {list(sink.states.keys())}")
            if hasattr(sink, 'abilities'):
                print(f"🔍 Sink abilities: {list(sink.abilities.keys())}")
        
        # Set initial sink state (OFF by default)
        from omnigibson.object_states import ToggledOn
        if hasattr(sink, 'states') and ToggledOn in sink.states:
            sink.states[ToggledOn].set_value(False)
            print("🚰 Sink initialized to OFF state")
        else:
            print("⚠️ Sink does not have ToggledOn state")
    except:
        sink = None
        print("⚠️ Commercial kitchen sink not available")
    
    # Check for water particle system
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

    # Keyboard teleoperation
    teleop = KeyboardRobotController(robot=robot)

    # Camera teleoperation (custom key bindings)
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

    # Register reset key (R)
    teleop.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Register sink toggle key (A)
    def toggle_sink():
        if sink is not None and hasattr(sink, 'states'):
            from omnigibson.object_states import ToggledOn
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
        key=lazy.carb.input.KeyboardInput.A,
        description="Toggle sink on/off",
        callback_fn=toggle_sink,
    )

    teleop.print_keyboard_teleop_info()
    print("Press ESC in the viewer to quit.")
    print("🚰 Press 'A' to toggle the sink on/off")
    if sink is not None:
        from omnigibson.object_states import ToggledOn
        if hasattr(sink, 'states') and ToggledOn in sink.states:
            initial_state = sink.states[ToggledOn].get_value()
            status = "ON" if initial_state else "OFF"
            print(f"🚰 Sink is currently {status}")
            if water_system is not None:
                print(f"💧 Water system: {water_system.name}")

    # Hardcoded objects of interest and descriptive names
    tracked_objects = {
        # "mug": "red_mug_in_sink",
        # "mug_ppzttc": "yellow_mug_in_sink",
        # "mug_ntgftr_1": "white_mug_in_sink",
        # "mug_ntgftr_2": "mug_on_dish_rack",
        # "mug_ntgftr_3": "mug_on_counter",
        # "dish_rack": "dish_rack",
        "box_of_crackers": "box_of_crackers",
        "commercial_kitchen_sink": "commercial_kitchen_sink",
        # "robot_nttbgn": "tiago_robot",
    }
    # Metric buffers per object
    metrics = {name: {"impact": [], "sustained": [], "health": []} for name in tracked_objects.keys()}

    # Frame buffer for the sim video
    frames = []

    try:
        steps = 600
        for _ in range(steps):
            action = teleop.get_teleop_action()
            env.step(action=action)

            # Capture viewer frame (RGB)
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
            rgb_np = rgb.cpu().numpy()[:, :, :3]
            rgb_np = cv2.resize(rgb_np, (512, 512))
            frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

            # Collect per-object metrics
            for obj_name in list(metrics.keys()):
                try:
                    obj = env.scene.object_registry("name", obj_name)
                    # breakpoint()
                    if obj is None:
                        # If missing in scene, drop tracking to avoid repeated lookups
                        del metrics[obj_name]
                        continue
                    # Health
                    try:
                        metrics[obj_name]["health"].append(float(getattr(obj, "health", 100.0)))
                    except Exception:
                        metrics[obj_name]["health"].append(100.0)
                    # Damage evaluator (first one if present)
                    impact_val = 0.0
                    sustained_val = 0.0
                    try:
                        if hasattr(obj, "damage_evaluators") and obj.damage_evaluators:
                            dmg_eval = obj.damage_evaluators[0]
                            # Impact force from base_link only (per step sum)
                            if hasattr(dmg_eval, "impact_forces_by_link") and dmg_eval.impact_forces_by_link:
                                base_vals = dmg_eval.impact_forces_by_link.get("base_link", [])
                                impact_val = float(max(base_vals)) if base_vals else 0.0
                                dmg_eval.impact_forces_by_link = {}
                            # Sustained running total (max over links)
                            # if hasattr(dmg_eval, "_sustained_running_total") and getattr(dmg_eval, "_sustained_running_total"):
                            #     sustained_val = float(max(dmg_eval._sustained_running_total.values()))
                            # else:
                            #     sustained_val = 0.0
                            # Per-timestep sustained force from base_link only (per step sum)
                            if hasattr(dmg_eval, "sustained_forces_by_link") and dmg_eval.sustained_forces_by_link:
                                base_vals_s = dmg_eval.sustained_forces_by_link.get("base_link", [])
                                sustained_val = float(sum(base_vals_s)) if base_vals_s else 0.0
                                dmg_eval.sustained_forces_by_link = {}
                            else:
                                sustained_val = 0.0
                    except Exception:
                        pass
                    metrics[obj_name]["impact"].append(impact_val)
                    metrics[obj_name]["sustained"].append(sustained_val)
                except Exception:
                    pass

        # Write sim video once
        out_dir = os.path.join(base_dir, "task1_videos")
        os.makedirs(out_dir, exist_ok=True)
        sim_mp4 = os.path.join(out_dir, "sim.mp4")
        # Efficient sim encoding: 512x512 @ 10fps
        h0, w0 = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(sim_mp4, fourcc, 10, (w0, h0))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()

        # Build per-object plots and composite videos
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        for obj_name, desc in tracked_objects.items():
            if obj_name not in metrics:
                continue
            data = metrics[obj_name]
            # Generate three plots mp4s
            plot_specs = [
                ("impact", "Impact Force (per step)", os.path.join(out_dir, f"{desc}_impact.mp4")),
                ("sustained", "Sustained Force (per step)", os.path.join(out_dir, f"{desc}_sustained.mp4")),
                ("health", "Health (per step)", os.path.join(out_dir, f"{desc}_health.mp4")),
            ]
            plot_videos = []
            for key, title, path in plot_specs:
                series = data[key]
                if len(series) == 0:
                    series = [0.0]
                fig, ax = plt.subplots(figsize=(6.4, 4.8))  # ~640x480 for faster renders
                line, = ax.plot([], [], lw=2)
                ax.set_xlim(1, max(1, len(series)))
                y_min = 0.0
                y_max = max(series) if key != "health" else 100.0
                if y_min == y_max:
                    y_min, y_max = (0.0, y_max if y_max != 0 else 1.0)
                ax.set_ylim(y_min, y_max * 1.1)
                ax.set_xlabel("Timestep")
                ax.set_ylabel(title)
                plt.tight_layout()

                def init():
                    line.set_data([], [])
                    return line,

                def animate(i):
                    x = list(range(1, i + 2))
                    y = series[: i + 1]
                    line.set_data(x, y)
                    return line,

                ani = animation.FuncAnimation(
                    fig, animate, init_func=init, frames=len(series), interval=100, blit=True
                )
                writer = animation.FFMpegWriter(fps=10, codec="mpeg4", extra_args=["-qscale", "5"])
                ani.save(path, writer=writer)
                plt.close(fig)
                plot_videos.append(path)

            # Stack the three plots vertically (right column), then hstack with sim video
            right_stack = os.path.join(out_dir, f"{desc}_plots_stacked.mp4")
            cmd_stack = [
                "ffmpeg", "-y",
                "-i", plot_videos[0],
                "-i", plot_videos[1],
                "-i", plot_videos[2],
                "-filter_complex",
                "[0:v]scale=720:170,setsar=1[v0];[1:v]scale=720:170,setsar=1[v1];[2:v]scale=720:170,setsar=1[v2];[v0][v1]vstack=inputs=2[top];[top][v2]vstack=inputs=2[right]",
                "-map", "[right]",
                "-vcodec", "mpeg4",
                "-q:v", "4",
                right_stack,
            ]
            subprocess.run(cmd_stack, check=True)

            # Final composite: left sim scaled to match right height (1440), hstack
            final_mp4 = os.path.join(out_dir, f"{desc}.mp4")
            cmd_final = [
                "ffmpeg", "-y",
                "-i", sim_mp4,
                "-i", right_stack,
                "-filter_complex",
                "[0:v]scale=576:512,setsar=1[left];[1:v]scale=512:512,setsar=1[right];[left][right]hstack=inputs=2[v]",
                "-map", "[v]",
                "-vcodec", "mpeg4",
                "-q:v", "4",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                final_mp4,
            ]
            subprocess.run(cmd_final, check=True)

            # Remove auxiliary videos (keep only the final per-object composite)
            try:
                for p in plot_videos:
                    if os.path.exists(p):
                        os.remove(p)
                if os.path.exists(right_stack):
                    os.remove(right_stack)
            except Exception:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        camera_mover.clear()
        og.shutdown()


if __name__ == "__main__":
    main()
