"""
Minimal mech damage teleop scene:
- Loads a coffee table (aoojzy) and a glass plate (pkkgzc)
- Spawns a TIAGo robot facing the table
- Places the plate OnTop of the table
- Runs keyboard teleoperation for 500 steps in a DamageableEnvironment
"""

# Set matplotlib backend to non-interactive before importing pyplot (if ever used downstream)
import matplotlib
matplotlib.use('Agg')

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.object_states import OnTop
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
from omnigibson.utils.sim_utils import place_base_pose
import torch as th
import os
import cv2
import numpy as np
import subprocess
import json
 

from safety_benchmark.damageable_env import DamageableEnvironment
# from safety_benchmark.params.test_params import PARAMS


# Performance knobs similar to other teleop scripts
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

PARAMS = {
    "tiago_robot": {
        "damage_evaluators": ["mechanical", "electrical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "impact_threshold": 30.0,
            "impact_scale": 0.001,
            "crushing_threshold": 10.0,
            "crushing_scale": 0.001,
            "link_thresholds": {
                # "arm": {
                #     "crushing_threshold": 1.0,
                #     "crushing_scale": 0.01,
                # },
                # "base": {
                #     "crushing_threshold": 3.0,
                #     "crushing_scale": 0.001,
                # },
                # "wheel": {
                #     "crushing_threshold": 10.0,
                #     "crushing_scale": 0.01,
                # },
                # "gripper": {
                #     "crushing_threshold": 4.0,
                #     "crushing_scale": 3.0,
                # }
            }
        },
        "electrical": {
            "damage_threshold": 0.0,  # Minimum particles to cause damage
            "scale": 0.001,  # Damage amount when threshold is exceeded
            "water_system_name": "sludge",
            "proximity_threshold": 1.0,  # 2cm proximity for manual detection
        }
    },
    "plate": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "impact_threshold": 0.012,
            "impact_scale": 10000.0,  # Increased scale for more aggressive damage
            "crushing_threshold": 5.0,
            "crushing_scale": 1.0,
        }
    },
    "baseball": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "impact_threshold": 0.015,
            "impact_scale": 0.001,
            "crushing_threshold": 5.0,
            "crushing_scale": 5.0,
        }
    },
    "glass": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "impact_threshold": 0.012,
            "impact_scale": 10000.0,
            "crushing_threshold": 4.0,
            "crushing_scale": 6.0,
        }
    },
    "book": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "impact_threshold": 0.015,
            "impact_scale": 0.001,
            "crushing_threshold": 10.0,
            "crushing_scale": 0.1,
        }
    },
    "drawer": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "impact_threshold": 100.0,
            "impact_scale": 0.000001,
            "crushing_threshold": 100.0,
            "crushing_scale": 0.00001,
            "link_thresholds": {
                "link_3": {
                    "crushing_threshold": 6.0,
                    "crushing_scale": 5.0,
                }
            }
        }
    },
    "default": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "impact_threshold": 0.0,
            "impact_scale": 1.0,
            "crushing_threshold": 1000.0,
            "crushing_scale": 0.0,
        }
    },
}


OBJECT_CONFIGS = {
    "coffee_table": {
        "type": "DatasetObject",
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [0.2, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.0],
        # Wood-like table damage behavior
        "damage_params": PARAMS.get("coffee_table", PARAMS["default"]),
    },
}

# Target object templates for easy integration of new targets
TARGET_OBJECT_CONFIGS = {
    # Default plate used in current unit test
    "plate": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "plate",
        "model": "ntedfx",  # glass plate from BEHAVIOR
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 1.0, 1.0],
        # Reuse bowl fragility params for glass-like behavior
        "damage_params": PARAMS["plate"],
    },
    # Example alternatives for quick swapping
    "glass": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "water_glass",
        "model": "uwtdng",
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [2.0, 2.0, 2.0],
        "damage_params": PARAMS["glass"],
    },
    "baseball": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "baseball",
        "model": "zanmar",
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 1.0, 1.0],
        "damage_params": PARAMS["baseball"],
    },
    "book": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "paperback_book",
        "model": "xxknda",
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.5, 1.5, 1.5],
        "damage_params": PARAMS["book"],
    },
    "door": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "door",
        # "model": "ofgpit",
        "model": "ofgodc",
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, -0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 0.5],
        "fixed_base": False,
        "damage_params": PARAMS.get("door", PARAMS["default"]),
    },
    "drawer": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "bottom_cabinet",
        "model": "pkdnbu",
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.5],
        "fixed_base": True,
        "damage_params": PARAMS["drawer"],
    },
}


def main():
    og.log.info(
        f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80
        if main.__doc__
        else f"Demo {__file__}"
    )

    # Easy integration knobs
    # - Choose target object template key from TARGET_OBJECT_CONFIGS (e.g., "plate", "bowl", "baseball")
    target_object_key = "plate"
    # - Toggle whether to load from a previously saved scene file or build fresh from scene_cfg
    use_saved_scene_file = True
    # - Explicit head joint init [head_1_joint (yaw), head_2_joint (pitch)] in radians. Set to None to skip.
    # head_joint_init = [0.0, -0.45]

    # Derived names and paths based on target object
    base_name = target_object_key if target_object_key is not None else "robot_only"
    unit_tests_dir = "safe-manipulation-benchmark/unit_tests"
    saved_path = os.path.join(unit_tests_dir, f"json_files/unit_test_{base_name}_mech.json")
    scene_cfg = {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "include_robots": False,
        "load_task_relevant_only": True,
    }
    robot0_cfg = {
        "type": "Tiago",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        # Position near and facing the table at origin
        "position": [0.2, 1.0, 0.0],
        "orientation": [0, 0, -1, 1],  # Facing the table
        "grasping_mode": "assisted",
        "damage_params": PARAMS["tiago_robot"],
    }

    _can_load_saved = (target_object_key is not None) and use_saved_scene_file and os.path.exists(saved_path)
    if target_object_key is None:
        # Build scene with only robot and coffee table
        cfg = dict(scene=scene_cfg, robots=[robot0_cfg], objects=[OBJECT_CONFIGS["coffee_table"]])
    elif _can_load_saved:
        # Overwrite damage params in the saved scene JSON to current values before loading
        with open(saved_path, "r") as f:
            scene_dict = json.load(f)
        init_info = scene_dict["objects_info"]["init_info"]

        def set_params(entry_args, params_dict):
            entry_args["params"] = params_dict

        # Map names / categories to current PARAMS
        for key, entry in init_info.items():
            args = entry.get("args", {})
            name = args.get("name", "")
            category = args.get("category", "")
            class_name = entry.get("class_name", "")

            if name.startswith("robot_") or class_name == "DamageableTiago":
                set_params(args, PARAMS["tiago_robot"])  # robot
            elif name == "coffee_table":
                set_params(args, PARAMS.get("coffee_table", PARAMS["default"]))
            elif name == "target_object":
                # Use the currently selected target's params if available, else fallback by category or default
                if target_object_key in PARAMS:
                    set_params(args, PARAMS[target_object_key])
                elif category in PARAMS:
                    set_params(args, PARAMS[category])
                else:
                    set_params(args, PARAMS["default"])
            else:
                # Generic: if category maps directly to PARAMS, apply it
                if category in PARAMS:
                    set_params(args, PARAMS[category])

        with open(saved_path, "w") as f:
            json.dump(scene_dict, f)

        # Load only from scene file (no robots or objects provided explicitly)
        cfg = {"scene": {"type": "Scene", "scene_file": saved_path}}
    else:
        # Build fresh scene from configs
        cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
        target_cfg = TARGET_OBJECT_CONFIGS[target_object_key]
        if target_object_key == "door" or target_object_key == "drawer":
            cfg["objects"] = [target_cfg]
        else:
            cfg["objects"] = [OBJECT_CONFIGS["coffee_table"], target_cfg]

    # Create the environment
    env = DamageableEnvironment(configs=cfg)

    # Configure robot controllers similar to other teleop scripts
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
    # (No IK primitives; default controller modes are sufficient)
    # Increase gripper force by raising joint stiffness / damping on the gripper controllers
    controller_config["gripper_left"]["motor_type"] = "position"
    controller_config["gripper_right"]["motor_type"] = "position"
    controller_config["gripper_left"]["isaac_kp"] = 20000.0
    controller_config["gripper_left"]["isaac_kd"] = 1000.0
    controller_config["gripper_right"]["isaac_kp"] = 20000.0
    controller_config["gripper_right"]["isaac_kd"] = 1000.0
    
    robot.reload_controllers(controller_config=controller_config)

    # Persist initial state after controller reload
    env.scene.update_initial_file()

    # Reset before teleoperation
    env.reset()
    robot.reset()

    # Set TIAGo head link poses at start (world frame)
    robot.set_joint_positions(th.tensor([-0.5, -0.75]), indices=robot.camera_control_idx)
    # Camera teleoperation matching teleop_tiago_bowl.py
    class CustomCameraMover(CameraMover):
        @property
        def input_to_command(self):
            return {
                lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
                lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
                # Exclude T to avoid conflict; use G for vertical move
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }

    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()

    # If this is a fresh scene, place the target object appropriately (when present); otherwise set camera pose
    if (target_object_key is not None) and (not _can_load_saved):
        target_obj = env.scene.object_registry("name", "target_object")
        if target_object_key not in ("door", "drawer"):
            table_obj = env.scene.object_registry("name", "coffee_table")
            assert target_obj.states[OnTop].set_value(table_obj, True), "Failed to set OnTop state for target object"
        else:
            # Land the door / drawer at ground height preserving its current yaw
            pos, quat = target_obj.get_position_orientation()
            pos_t = th.tensor(pos)
            pos_t[2] = 0.0
            place_base_pose(target_obj, pos_t, quat=th.tensor(quat))
            target_obj.keep_still()
    elif _can_load_saved:
        # Set viewer camera pose to last printed values
        cam_pos = th.tensor([-2.008840799331665, 0.2598961591720581, 1.1361593008041382])
        cam_quat = th.tensor([0.4495110809803009, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452])
        og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat)

    # Set starting viewer camera pose to user-specified values
    start_cam_pos = th.tensor([-1.3225984573364258, 0.2645236849784851, 0.9981386065483093])
    start_cam_quat = th.tensor([0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452])
    og.sim.viewer_camera.set_position_orientation(position=start_cam_pos, orientation=start_cam_quat)

    # Let physics settle a bit
    for _ in range(20):
        og.sim.step()

    # Keyboard teleop
    action_generator = KeyboardRobotController(robot=robot)
    # (Removed IK helpers and bindings)
    
    # Register TAB key to print base + EEF poses (world/body) and breakpoint
    def breakpoint_and_print_poses():
        print("=== TAB callback triggered ===", flush=True)
        # Base poses
        base_pos_w, base_quat_w = robot.get_position_orientation()
        base_pos_body = th.tensor([0.0, 0.0, 0.0])
        base_quat_body = th.tensor([0.0, 0.0, 0.0, 1.0])
        print("=== Robot Base Pose ===")
        print(f"World  pos: {base_pos_w.tolist()}  quat: {base_quat_w.tolist()}")
        print(f"Body   pos: {base_pos_body.tolist()}  quat: {base_quat_body.tolist()}")

        # Viewer camera pose
        cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
        print("=== Viewer Camera Pose ===")
        print(f"World  pos: {cam_pos.tolist()}  quat: {cam_quat.tolist()}")

        # End-effector poses for each arm (world and body frames)
        for arm in getattr(robot, 'arm_names', ["right"]):
            eef_pos_w = robot.get_eef_position(arm)
            eef_quat_w = robot.get_eef_orientation(arm)
            eef_pos_body = robot.get_relative_eef_position(arm)
            eef_quat_body = robot.get_relative_eef_orientation(arm)
            print(f"=== {arm} EEF Pose ===")
            print(f"World  pos: {eef_pos_w.tolist()}  quat: {eef_quat_w.tolist()}")
            print(f"Body   pos: {eef_pos_body.tolist()}  quat: {eef_quat_body.tolist()}")

        # Also print current head joint positions
        h1 = robot.links["head_1_link"].get_position_orientation()
        h2 = robot.links["head_2_link"].get_position_orientation()
        print("=== Head Link Positions ===")
        print(f"head_1_link: {h1}")
        print(f"head_2_link: {h2}")

        # Save current scene config/state
        save_dir = "safe-manipulation-benchmark/unit_tests"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"json_files/unit_test_{base_name}_mech_temp.json")
        # Save current scene state to JSON (same pattern as test_save_load_simple.py)
        og.sim.save([save_path])
        print(f"✅ Saved scene to: {save_path}")

        breakpoint()

    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Print base & EEF poses (world/body) and breakpoint",
        callback_fn=breakpoint_and_print_poses,
    )
    action_generator.print_keyboard_teleop_info()
    print("Running mech damage teleop. Press ESC to quit.")

    max_steps = 500
    step = 0
    fps = 10
    frames = []
    # Ego camera recording
    frames_ego = []
    ego_key = None
    # Health tracking
    robot_healths = []
    target_healths = []
    # Resolve target reference if applicable
    target_ref = None
    if target_object_key is not None:
        target_ref = env.scene.object_registry("name", "target_object")
        if target_ref is None:
            target_ref = env.scene.object_registry("name", "glass_plate")
        if target_ref is None:
            target_ref = env.scene.object_registry("name", base_name)
    # Force metrics tracking (impact and sustained per step)
    robot_impact_forces = []
    robot_sustained_forces = []
    target_impact_forces = []
    target_sustained_forces = []
    # Damage status per step (for dynamic overlay color)
    target_statuses = []
    while step != max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1

        # Capture viewer camera RGB and store for video
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        # Scale to at least 1280x720 (HD)
        rgb_np = cv2.resize(rgb_np, (1080, 720))
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Capture robot ego camera RGB (dynamic robot name, eyes:Camera:0)
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
            if target_ref is not None:
                status = str(target_ref.damage_status).lower()
                if status in ("major", "critical"):
                    bgr = (0, 0, 255)
                elif status in ("minor",):
                    bgr = (0, 255, 255)
                else:
                    bgr = (0, 255, 0)
            else:
                bgr = (0, 255, 0)
            border = 30  # 30px at 1080x720 -> ~10px after scaling to 360x240
            ego_bgr = cv2.copyMakeBorder(ego_bgr, border, border, border, border, cv2.BORDER_CONSTANT, value=bgr)
            frames_ego.append(ego_bgr)

        # Record health
        robot_healths.append(float(getattr(robot, "health", 100.0)))
        if target_ref is not None:
            target_healths.append(float(getattr(target_ref, "health", 100.0)))
            # Record current damage status string
            target_statuses.append(getattr(target_ref, "damage_status"))

        # Record forces (impact per-step and sustained per-step), max over links when available
        # Robot
        r_dmg = robot.damage_evaluators[0]
        # Impact: max average over per-link values for this step; then clear
        link_keyword = "wheel"
        if hasattr(r_dmg, 'impact_forces_by_link') and r_dmg.impact_forces_by_link:
            # Get max impact force across all base-related links
            base_impacts = []
            for link_name, forces in r_dmg.impact_forces_by_link.items():
                if link_keyword in link_name.lower():
                    base_impacts.extend(forces)
            impact_val = max(base_impacts) if base_impacts else 0.0
            robot_impact_forces.append(impact_val)
            # impact_vals = r_dmg.impact_forces_by_link.get("base_link", [])
            # robot_impact_forces.append(max(impact_vals) if impact_vals else 0.0)
            r_dmg.impact_forces_by_link = {}
        else:
            robot_impact_forces.append(0.0)
        if hasattr(r_dmg, 'sustained_forces_by_link') and r_dmg.sustained_forces_by_link:
            base_sustained = []
            for link_name, forces in r_dmg.sustained_forces_by_link.items():
                if link_keyword in link_name.lower():
                    base_sustained.extend(forces)
            sustained_val = max(base_sustained) if base_sustained else 0.0
            robot_sustained_forces.append(sustained_val)
            # sustained_vals = r_dmg.sustained_forces_by_link.get("base_link", [])
            # robot_sustained_forces.append(max(sustained_vals) if sustained_vals else 0.0)
            r_dmg.sustained_forces_by_link = {}
        else:
            robot_sustained_forces.append(0.0)
        # # Sustained: per-step average max; then clear
        # if hasattr(r_dmg, 'sustained_forces_by_link') and r_dmg.sustained_forces_by_link:
        #     sustained_vals = r_dmg.sustained_forces_by_link.get("base_link", [])
        #     robot_sustained_forces.append(max(sustained_vals) if sustained_vals else 0.0)
        #     r_dmg.sustained_forces_by_link = {}
        # else:
        #     robot_sustained_forces.append(0.0)   

        # Target object
        if target_ref is not None:
            t_dmg = target_ref.damage_evaluators[0]
            if hasattr(t_dmg, 'impact_forces_by_link') and t_dmg.impact_forces_by_link:
                base_vals = t_dmg.impact_forces_by_link.get("base_link", [])
                target_impact_forces.append(max(base_vals) if base_vals else 0.0)
                t_dmg.impact_forces_by_link = {}
            else:
                target_impact_forces.append(0.0)
            if hasattr(t_dmg, 'sustained_forces_by_link') and t_dmg.sustained_forces_by_link:
                base_vals = t_dmg.sustained_forces_by_link.get("base_link", [])
                target_sustained_forces.append(max(base_vals) if base_vals else 0.0)
                t_dmg.sustained_forces_by_link = {}
            else:
                target_sustained_forces.append(0.0)

    # Clean shutdown
    camera_mover.clear()

    # Write camera video to unit_tests/videos as mp4
    videos_dir = os.path.join("safe-manipulation-benchmark", "unit_tests", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    avi_path = os.path.join(videos_dir, f"{base_name}_camera_obs.avi")
    mp4_path = os.path.join(videos_dir, f"{base_name}_camera_obs.mp4")
    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()
        # Convert AVI to MP4 via ffmpeg
        subprocess.run(["ffmpeg", "-y", "-i", avi_path, "-c:v", "mpeg4", mp4_path], check=True)
        # Remove AVI
        os.remove(avi_path)

    # Write ego camera video
    avi_ego = os.path.join(videos_dir, f"{base_name}_ego_obs.avi")
    mp4_ego = os.path.join(videos_dir, f"{base_name}_ego_obs.mp4")
    if len(frames_ego) > 0:
        he, we = frames_ego[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw_e = cv2.VideoWriter(avi_ego, fourcc, fps, (we, he))
        for f in frames_ego:
            vw_e.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw_e.release()
        subprocess.run(["ffmpeg", "-y", "-i", avi_ego, "-c:v", "mpeg4", mp4_ego], check=True)
        os.remove(avi_ego)

    # Build animated health plot using Matplotlib
    health_mp4 = os.path.join(videos_dir, f'{base_name}_healths.mp4')
    if len(robot_healths) > 0 and (target_object_key is not None) and len(target_healths) > 0:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = max(len(robot_healths), len(target_healths))
        # Clamp health plot to [0, 100]
        y_min = 0.0
        y_max = 100.0

        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
        line_p, = ax.plot([], [], lw=6, color='tab:orange', label=target_object_key.capitalize())
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
            line_r.set_data([], [])
            line_p.set_data([], [])
            return line_r, line_p

        def animate_health(i):
            x = [k / fps for k in range(1, i + 2)]
            y_r = robot_healths[: i + 1]
            y_p = target_healths[: i + 1]
            line_r.set_data(x, y_r)
            line_p.set_data(x, y_p)
            return line_r, line_p

        ani = animation.FuncAnimation(
            fig, animate_health, init_func=init_health, frames=T, interval=1000 / fps, blit=True
        )
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)
    elif len(robot_healths) > 0 and (target_object_key is None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = len(robot_healths)
        y_min = 0.0
        y_max = 100.0
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Health', fontsize=20)
        ax.set_title('Health Over Time', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_health_r():
            line_r.set_data([], [])
            return line_r,

        def animate_health_r(i):
            x = [k / fps for k in range(1, i + 2)]
            y_r = robot_healths[: i + 1]
            line_r.set_data(x, y_r)
            return line_r,

        ani = animation.FuncAnimation(fig, animate_health_r, init_func=init_health_r, frames=T, interval=1000 / fps, blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)

    # Create side-by-side combined video (camera left, health right)
    # Overlay ego video onto top-right corner of main camera video, then hstack with health plot
    cam_with_ego_mp4 = os.path.join(videos_dir, f'{base_name}_camera_with_ego.mp4')
    if os.path.exists(mp4_path) and os.path.exists(mp4_ego):
        # Ego frames already contain dynamic border and are possibly larger due to copyMakeBorder.
        # Scale ego down and overlay without additional padding.
        subprocess.run([
            'ffmpeg', '-y',
            '-i', mp4_path,
            '-i', mp4_ego,
            '-filter_complex',
            '[0:v]setsar=1[base];[1:v]scale=360:240,setsar=1[ego];[base][ego]overlay=W-w-20:20[out]',
            '-map','[out]',
            '-c:v','mpeg4','-q:v','5', cam_with_ego_mp4
        ], check=True)

    # Rename final combined video to {object}_with_health.mp4
    combined_mp4 = os.path.join(videos_dir, f'{base_name}_with_health.mp4')
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
            combined_mp4
        ], check=True)

    # Build stacked force plots video (impact top, sustained bottom) with thick lines and larger fonts
    forces_mp4 = os.path.join(videos_dir, f'{base_name}_forces.mp4')
    if (target_object_key is None) and (len(robot_impact_forces) > 0):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T_forces = max(len(robot_impact_forces), len(robot_sustained_forces))
        # Impact figure (robot only)
        fig_imp, ax_imp = plt.subplots(figsize=(9.6, 5.4))
        line_imp_r, = ax_imp.plot([], [], lw=4, color='tab:blue', label='Robot')
        ax_imp.set_xlim(0, max(1, T_forces) / fps)
        y_min_imp = 0.0
        y_max_imp = max(robot_impact_forces + [0.0])
        if y_min_imp == y_max_imp:
            y_min_imp, y_max_imp = (0.0, 1.0)
        ax_imp.set_ylim(y_min_imp, y_max_imp * 1.1)
        ax_imp.set_xlabel('Time (s)', fontsize=16)
        ax_imp.set_ylabel('Impact Force (N)', fontsize=16)
        ax_imp.set_title('Impact Force Over Time', fontsize=20)
        ax_imp.legend(loc='best')
        plt.tight_layout()

        def init_imp_r():
            line_imp_r.set_data([], [])
            return line_imp_r,

        def animate_imp_r(i):
            x = [k / fps for k in range(1, i + 2)]
            y_ir = robot_impact_forces[: i + 1]
            line_imp_r.set_data(x, y_ir)
            return line_imp_r,

        ani_imp = animation.FuncAnimation(fig_imp, animate_imp_r, init_func=init_imp_r, frames=T_forces, interval=1000 / fps, blit=True)

        # Sustained figure (robot only)
        fig_sus, ax_sus = plt.subplots(figsize=(9.6, 5.4))
        line_sus_r, = ax_sus.plot([], [], lw=4, color='tab:green', label='Robot')
        ax_sus.set_xlim(0, max(1, T_forces) / fps)
        y_min_sus = 0.0
        y_max_sus = max(robot_sustained_forces + [0.0])
        if y_min_sus == y_max_sus:
            y_min_sus, y_max_sus = (0.0, 1.0)
        ax_sus.set_ylim(y_min_sus, y_max_sus * 1.1)
        ax_sus.set_xlabel('Time (s)', fontsize=16)
        ax_sus.set_ylabel('Sustained Force (N)', fontsize=16)
        ax_sus.set_title('Sustained Force Over Time', fontsize=20)
        ax_sus.legend(loc='best')
        plt.tight_layout()

        def init_sus_r():
            line_sus_r.set_data([], [])
            return line_sus_r,

        def animate_sus_r(i):
            x = [k / fps for k in range(1, i + 2)]
            y_sr = robot_sustained_forces[: i + 1]
            line_sus_r.set_data(x, y_sr)
            return line_sus_r,

        ani_sus = animation.FuncAnimation(fig_sus, animate_sus_r, init_func=init_sus_r, frames=T_forces, interval=1000 / fps, blit=True)

        # Save and stack
        forces_imp_mp4 = os.path.join(videos_dir, f'{base_name}_forces_impact.mp4')
        forces_sus_mp4 = os.path.join(videos_dir, f'{base_name}_forces_sustained.mp4')
        writer_f = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani_imp.save(forces_imp_mp4, writer=writer_f)
        plt.close(fig_imp)
        ani_sus.save(forces_sus_mp4, writer=writer_f)
        plt.close(fig_sus)

        subprocess.run([
            'ffmpeg', '-y',
            '-i', forces_imp_mp4,
            '-i', forces_sus_mp4,
            '-filter_complex',
            '[0:v]scale=1080:360,setsar=1[top];[1:v]scale=1080:360,setsar=1[bot];[top][bot]vstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            forces_mp4
        ], check=True)
    elif len(robot_impact_forces) > 0 and len(target_impact_forces) > 0:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T_forces = max(len(robot_impact_forces), len(target_impact_forces), len(robot_sustained_forces), len(target_sustained_forces))
        # Impact figure (target only)
        fig_imp, ax_imp = plt.subplots(figsize=(9.6, 5.4))
        line_imp_t, = ax_imp.plot([], [], lw=4, color='tab:orange', label=target_object_key.capitalize())
        ax_imp.set_xlim(0, max(1, T_forces) / fps)
        y_min_imp = 0.0
        y_max_imp = max(target_impact_forces + [0.0])
        if y_min_imp == y_max_imp:
            y_min_imp, y_max_imp = (0.0, 1.0)
        ax_imp.set_ylim(y_min_imp, y_max_imp * 1.1)
        ax_imp.set_xlabel('Time (s)', fontsize=16)
        ax_imp.set_ylabel('Impact Force (N)', fontsize=16)
        ax_imp.set_title('Impact Force Over Time', fontsize=20)
        ax_imp.legend(loc='best')
        plt.tight_layout()

        def init_imp():
            line_imp_t.set_data([], [])
            return line_imp_t,

        def animate_imp(i):
            x = [k / fps for k in range(1, i + 2)]
            y_it = target_impact_forces[: i + 1]
            line_imp_t.set_data(x, y_it)
            return line_imp_t,

        ani_imp = animation.FuncAnimation(fig_imp, animate_imp, init_func=init_imp, frames=T_forces, interval=1000 / fps, blit=True)

        # Sustained figure (target only)
        fig_sus, ax_sus = plt.subplots(figsize=(9.6, 5.4))
        line_sus_t, = ax_sus.plot([], [], lw=4, color='tab:red', label=target_object_key.capitalize())
        ax_sus.set_xlim(0, max(1, T_forces) / fps)
        y_min_sus = 0.0
        y_max_sus = max(target_sustained_forces + [0.0])
        if y_min_sus == y_max_sus:
            y_min_sus, y_max_sus = (0.0, 1.0)
        ax_sus.set_ylim(y_min_sus, y_max_sus * 1.1)
        ax_sus.set_xlabel('Time (s)', fontsize=16)
        ax_sus.set_ylabel('Sustained Force (N)', fontsize=16)
        ax_sus.set_title('Sustained Force Over Time', fontsize=20)
        ax_sus.legend(loc='best')
        plt.tight_layout()

        def init_sus():
            line_sus_t.set_data([], [])
            return line_sus_t,

        def animate_sus(i):
            x = [k / fps for k in range(1, i + 2)]
            y_st = target_sustained_forces[: i + 1]
            line_sus_t.set_data(x, y_st)
            return line_sus_t,

        ani_sus = animation.FuncAnimation(fig_sus, animate_sus, init_func=init_sus, frames=T_forces, interval=1000 / fps, blit=True)

        # Save both animations to temporary mp4s
        forces_imp_mp4 = os.path.join(videos_dir, f'{base_name}_forces_impact.mp4')
        forces_sus_mp4 = os.path.join(videos_dir, f'{base_name}_forces_sustained.mp4')
        writer_f = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani_imp.save(forces_imp_mp4, writer=writer_f)
        plt.close(fig_imp)
        ani_sus.save(forces_sus_mp4, writer=writer_f)
        plt.close(fig_sus)

        # Stack vertically into a 1080x720 forces video
        subprocess.run([
            'ffmpeg', '-y',
            '-i', forces_imp_mp4,
            '-i', forces_sus_mp4,
            '-filter_complex',
            '[0:v]scale=1080:360,setsar=1[top];[1:v]scale=1080:360,setsar=1[bot];[top][bot]vstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            forces_mp4
        ], check=True)

    # Second combined video: cam_with_ego (left) + forces stacked (right)
    # Rename final combined video to {object}_with_forces.mp4
    combined_forces_mp4 = os.path.join(videos_dir, f'{base_name}_with_forces.mp4')
    if os.path.exists(cam_with_ego_mp4) and os.path.exists(forces_mp4):
        subprocess.run([
            'ffmpeg', '-y',
            '-i', cam_with_ego_mp4,
            '-i', forces_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_forces_mp4
        ], check=True)
    
    # Clean up intermediate videos, keeping only the final combined videos
    print("Cleaning up intermediate videos...")
    videos_to_cleanup = [
        mp4_path,  # {base_name}_camera_obs.mp4
        mp4_ego,   # {base_name}_ego_obs.mp4
        cam_with_ego_mp4,  # {base_name}_camera_with_ego.mp4
        health_mp4,  # {base_name}_healths.mp4
        forces_mp4,  # {base_name}_forces.mp4
    ]
    
    # Add force-specific intermediate videos if they exist
    if target_object_key is None:
        # Robot-only case
        forces_imp_mp4 = os.path.join(videos_dir, f'{base_name}_forces_impact.mp4')
        forces_sus_mp4 = os.path.join(videos_dir, f'{base_name}_forces_sustained.mp4')
        videos_to_cleanup.extend([forces_imp_mp4, forces_sus_mp4])
    else:
        # Target object case
        forces_imp_mp4 = os.path.join(videos_dir, f'{base_name}_forces_impact.mp4')
        forces_sus_mp4 = os.path.join(videos_dir, f'{base_name}_forces_sustained.mp4')
        videos_to_cleanup.extend([forces_imp_mp4, forces_sus_mp4])
    
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
    if os.path.exists(combined_forces_mp4):
        print(f"  - {os.path.basename(combined_forces_mp4)}")
    
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main()


