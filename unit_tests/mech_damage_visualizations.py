"""
Visualization script for mechanical damage scenes:
- Loads a coffee table and selected target object
- Spawns a TIAGo robot facing the table
- Places the target object OnTop of the table
- Runs keyboard teleoperation and saves high-quality PNG frames every 10 steps
"""

# Set matplotlib backend to non-interactive before importing pyplot
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
import json
import subprocess

from safety_benchmark.damageable_env import DamageableEnvironment


# Performance knobs similar to other teleop scripts
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

PARAMS = {
    "tiago_robot": {
        "damage_evaluators": ["mechanical", "electrical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 1e10,
            "scale": 1e-10,
            "instant_coefficient": 1.0,
            "creep_coefficient": 1.0,
            "object_type": "brittle",
            "link_thresholds": {
                "arm": {
                    "damage_threshold": 200.0,
                    "scale": 0.0001,
                },
                "gripper": {
                    "damage_threshold": 200.0,
                    "scale": 0.0001,
                }
            }
        },
        "electrical": {
            "damage_threshold": 0.0,
            "scale": 0.001,
            "water_system_name": "sludge",
        }
    },
    "plate": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 200.0,
            "scale": 0.5,
            "instant_coefficient": 1.0,
            "creep_coefficient": 0.0,
            "object_type": "brittle",
        }
    },
    "baseball": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 1000,
            "scale": 0.01,
            "instant_coefficient": 1.0,
            "creep_coefficient": 1.0,
            "object_type": "ductile",
        }
    },
    "glass": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 200.0,
            "scale": 0.5,
            "instant_coefficient": 1.0,
            "creep_coefficient": 0.0,
            "object_type": "brittle",
        }
    },
    "book": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 2000,
            "scale": 0.01,
            "instant_coefficient": 1.0,
            "creep_coefficient": 0.0,
            "object_type": "brittle",
        }
    },
    "orange": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 200.0,
            "scale": 10.0,
            "instant_coefficient": 1.0,
            "creep_coefficient": 0.5,
            "object_type": "ductile",
        }
    },
    "default": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 1000.0,
            "scale": 1.0,
            "instant_coefficient": 1.0,
            "creep_coefficient": 1.0,
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
    "orange": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "orange",
        "model": "ucstpm",  # Orange from BEHAVIOR
        # Reference: https://behavior.stanford.edu/knowledgebase/objects/ucstpm.html
        # Bounding Box Size: 0.08, 0.09, 0.08 (we rely on scale only here)
        "position": [0.1, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.25, 1.25, 1.25],
        "damage_params": PARAMS["orange"],
    },
    "door": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "door",
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
        "damage_params": PARAMS.get("drawer", PARAMS["default"]),
    },
}


def main():
    og.log.info(
        f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80
        if main.__doc__
        else f"Demo {__file__}"
    )

    # Easy integration knobs
    # - Choose target object template key from TARGET_OBJECT_CONFIGS
    target_object_key = "drawer"
    # - Toggle whether to load from a previously saved scene file or build fresh from scene_cfg
    use_saved_scene_file = True
    # - Toggle whether to track robot health/strain in the plots
    track_robot_health = True

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
    # Special case: if target is plate, load the scene exactly like rl_test.py
    if target_object_key == "plate":
        save_path = "lift_test.json"
        if not os.path.exists(save_path):
            print(f"Warning: Saved state file '{save_path}' not found! Creating fresh scene.")
            _can_load_saved = False
        else:
            print(f"Loading simulation state from: {save_path}")
            # Overwrite damage params in lift_test.json to current PARAMS before loading
            try:
                with open(save_path, "r") as f:
                    scene_dict = json.load(f)
                init_info = scene_dict["objects_info"]["init_info"]

                def set_params(entry_args, params_dict):
                    entry_args["params"] = params_dict

                for key, entry in init_info.items():
                    args = entry.get("args", {})
                    name = args.get("name", "")
                    category = args.get("category", "")
                    class_name = entry.get("class_name", "")

                    if name.startswith("robot_") or class_name == "DamageableTiago":
                        set_params(args, PARAMS["tiago_robot"])
                    elif name == "coffee_table":
                        set_params(args, PARAMS.get("coffee_table", PARAMS["default"]))
                    elif name == "target_object" or name == "glass_plate" or category == "plate":
                        if target_object_key in PARAMS:
                            set_params(args, PARAMS[target_object_key])
                        elif category in PARAMS:
                            set_params(args, PARAMS[category])
                        else:
                            set_params(args, PARAMS["default"])
                    else:
                        if category in PARAMS:
                            set_params(args, PARAMS[category])

                with open(save_path, "w") as f:
                    json.dump(scene_dict, f)
            except Exception as e:
                print(f"Warning: Could not rewrite params in {save_path}: {e}")

            cfg = {"scene": {"type": "Scene", "scene_file": save_path}}
    elif target_object_key is None:
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
                set_params(args, PARAMS["tiago_robot"])
            elif name == "coffee_table":
                set_params(args, PARAMS.get("coffee_table", PARAMS["default"]))
            elif name == "target_object":
                if target_object_key in PARAMS:
                    set_params(args, PARAMS[target_object_key])
                elif category in PARAMS:
                    set_params(args, PARAMS[category])
                else:
                    set_params(args, PARAMS["default"])
            else:
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

    # Match rl_test.py: ensure a fresh simulator before creating the environment
    if og.sim is None:
        minimal_cfg = {
            "env": {"action_timestep": 1.0 / 60.0, "physics_timestep": 1.0 / 120.0},
            "scene": {"type": "Scene"},
            "robots": [],
        }
        _tmp_env = DamageableEnvironment(configs=minimal_cfg)
        _tmp_env.reset()
        og.clear()
    else:
        og.sim.stop()
        og.clear()

    # Create the environment
    env = DamageableEnvironment(configs=cfg)
    # Reset immediately after creation
    env.reset()

    # Resolve target object (for damage-based coloring in the viewer camera)
    target_obj = None
    for name in ["target_object", "glass_plate", base_name]:
        try:
            target_obj = env.scene.object_registry("name", name)
        except Exception:
            target_obj = None
        if target_obj is not None:
            break

    if target_obj is None:
        print("Warning: No target object found for damage visualization")

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
    controller_config["gripper_left"]["motor_type"] = "position"
    controller_config["gripper_right"]["motor_type"] = "position"
    controller_config["gripper_left"]["isaac_kp"] = 4000.0
    controller_config["gripper_left"]["isaac_kd"] = 2000.0
    controller_config["gripper_right"]["isaac_kp"] = 4000.0
    controller_config["gripper_right"]["isaac_kd"] = 2000.0
    
    robot.reload_controllers(controller_config=controller_config)

    # Increase friction on gripper fingers to reduce slip during grasp
    try:
        for _link_name, _link in robot.links.items():
            _n = _link_name.lower()
            if ("gripper" in _n) or ("finger" in _n):
                try:
                    _link.set_attribute("physxMaterial:staticFriction", 2.0)
                    _link.set_attribute("physxMaterial:dynamicFriction", 2.0)
                except Exception:
                    pass
    except Exception:
        pass

    # Persist initial state after controller reload
    env.scene.update_initial_file()

    # Reset before teleoperation
    robot.reset()

    # Set TIAGo head link poses at start
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
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }

    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()

    # If this is a fresh scene, place the target object appropriately (when present)
    if (target_object_key is not None) and (not _can_load_saved):
        target_obj = env.scene.object_registry("name", "target_object")
        if target_object_key not in ("door", "drawer"):
            table_obj = env.scene.object_registry("name", "coffee_table")
            assert target_obj.states[OnTop].set_value(table_obj, True), "Failed to set OnTop state for target object"
            # Keep the target still after placement (especially useful for the orange)
            if hasattr(target_obj, "keep_still"):
                target_obj.keep_still()
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

    # Try to enable instance segmentation on the viewer camera so we can recolor the target
    seg_instance_available = True
    target_seg_ids = None  # Will be populated lazily from seg_instance info
    try:
        og.sim.viewer_camera.add_modality("seg_instance")
    except Exception as e:
        print(f"Warning: could not add seg_instance modality to viewer camera: {e}")
        seg_instance_available = False

    # Keyboard teleop
    action_generator = KeyboardRobotController(robot=robot)
    
    # Register TAB key to save scene JSON
    def save_scene():
        print("=== TAB callback triggered ===", flush=True)
        save_dir = os.path.join(unit_tests_dir, "json_files")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"unit_test_{base_name}_mech.json")
        og.sim.save([save_path])
        print(f"✅ Saved scene to: {save_path}")
        # Print current viewer camera pose
        cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
        print(f"📷 Current camera position: {cam_pos.tolist() if hasattr(cam_pos, 'tolist') else cam_pos}")
        print(f"📷 Current camera orientation (quaternion): {cam_quat.tolist() if hasattr(cam_quat, 'tolist') else cam_quat}")
        breakpoint()

    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Save scene to JSON",
        callback_fn=save_scene,
    )
    action_generator.print_keyboard_teleop_info()
    print("Running visualization teleop. Press ESC to quit.")

    # Create output directory for PNG frames
    output_dir = os.path.join(unit_tests_dir, "visualizations", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Helper to convert RGB frame to uint8 (handle both 0-1 and 0-255 ranges)
    def prepare_frame_for_png(frame):
        frame_max = np.max(frame)
        if frame_max <= 1.0:
            # Values are in 0-1 range, convert to 0-255
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            # Values are already in 0-255 range
            frame_uint8 = frame.astype(np.uint8)
        return frame_uint8

    max_steps = 1000
    # Match rl_test.py: use 30 FPS and accumulate per-step frames
    fps = 30
    step = 0
    frame_count = 0
    # Recolored sim frames (with damage-based coloring)
    video_frames = []
    # Original sim frames (without recoloring), for side-by-side comparison
    video_frames_original = []
    # Segmentation-only frames for the target object (for debugging)
    seg_frames = []

    # Health / strain tracking similar to mech_damage.py
    robot_healths = []
    target_healths = []
    robot_strains = []
    target_strains = []
    # Per-step target damage status (for colored border video)
    target_statuses = []
    
    while step != max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1

        # Capture viewer camera RGB (and optional instance segmentation) every step,
        # and recolor the target object similarly to rl_test.py.
        cam_obs_raw = og.sim.viewer_camera.get_obs()

        # Handle different possible return formats from get_obs():
        # (obs, info) or obs only; obs may itself be a list (per-env) or a dict.
        cam_obs, cam_info = None, {}
        if isinstance(cam_obs_raw, tuple) and len(cam_obs_raw) == 2:
            cam_obs, cam_info = cam_obs_raw
        else:
            cam_obs = cam_obs_raw

        if isinstance(cam_obs, (list, tuple)):
            obs0 = cam_obs[0]
            info0 = cam_info[0] if isinstance(cam_info, (list, tuple)) and len(cam_info) > 0 else cam_info
        else:
            obs0 = cam_obs
            info0 = cam_info

        # Base RGB frame from viewer camera (match rl_test.py)
        rgb = obs0["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        rgb_np = cv2.resize(rgb_np, (1080, 720))
        rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        # Keep an untouched copy for "no-color" video / comparison
        rgb_bgr_original = rgb_bgr.copy()

        # Compute target damage status for this step
        status = str(getattr(target_obj, "damage_status", "none")).lower() if target_obj is not None else "none"
        target_statuses.append(status)

        # Recolor the target object in the main camera view based on damage status (same logic as rl_test.py)
        if seg_instance_available and target_obj is not None:
            try:
                seg_im = obs0.get("seg_instance", None)
            except Exception:
                seg_im = None

            if seg_im is not None:
                # Lazily infer segmentation IDs corresponding to the target object name
                if target_seg_ids is None and isinstance(info0, dict) and "seg_instance" in info0:
                    seg_info = info0["seg_instance"]
                    # seg_info is a dict {id: object_name}; be robust to naming differences
                    # between the saved scene and the current target configuration.
                    target_name = getattr(target_obj, "name", None)
                    target_category = getattr(target_obj, "category", None)
                    target_model = getattr(target_obj, "_model", None)

                    alias_candidates = [
                        target_name,
                        target_category,
                        target_model,
                        target_object_key,
                        base_name,
                    ]
                    alias_tokens = [a.lower() for a in alias_candidates if isinstance(a, str)]

                    target_seg_ids = []
                    for seg_id, label in seg_info.items():
                        if not isinstance(label, str):
                            continue
                        ll = label.lower()
                        if any(tok in ll for tok in alias_tokens):
                            target_seg_ids.append(int(seg_id))

                    target_seg_ids = target_seg_ids or []

                if target_seg_ids:
                    # Create a binary mask for all pixels belonging to the target object
                    seg_im_np = seg_im.cpu().numpy()
                    mask = np.zeros_like(seg_im_np, dtype=np.uint8)
                    for seg_id in target_seg_ids:
                        mask |= (seg_im_np == seg_id).astype(np.uint8)

                    # Resize mask to match RGB frame resolution
                    mask_resized = cv2.resize(
                        mask, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
                    )

                    # Build a segmentation-only visualization frame (black background + colored target)
                    seg_color = np.zeros_like(rgb_bgr, dtype=np.uint8)

                    # Only recolor when damage is non-zero (minor or worse)
                    if status in ("minor", "major", "critical"):
                        # Yellow for minor, red for major / critical (BGR)
                        if status == "minor":
                            color = (0, 255, 255)  # Yellow
                        else:
                            color = (0, 0, 255)  # Red

                        # Apply solid color to the target pixels in the camera frame
                        rgb_bgr[mask_resized == 1] = color
                        seg_color[mask_resized == 1] = color

                    seg_frames.append(seg_color)
                else:
                    # If we couldn't identify the target instance, append a black seg frame
                    seg_frames.append(np.zeros_like(rgb_bgr, dtype=np.uint8))
            else:
                # seg_instance missing – append black seg frame
                seg_frames.append(np.zeros_like(rgb_bgr, dtype=np.uint8))
        else:
            # No segmentation available or no target – append black seg frame
            seg_frames.append(np.zeros_like(rgb_bgr, dtype=np.uint8))


        # Record health (conditionally track robot based on track_robot_health)
        should_track_robot = track_robot_health or (target_object_key is None)
        if should_track_robot:
            robot_healths.append(float(getattr(robot, "health", 100.0)))
        if target_obj is not None:
            target_healths.append(float(getattr(target_obj, "health", 100.0)))

        # Record strain (per-step): take max over per-link last_strain_by_link after this env step
        # For robot, only include links with "arm" or "gripper" in their names
        if should_track_robot and hasattr(robot, "damage_evaluators") and len(getattr(robot, "damage_evaluators", [])) > 0:
            r_dmg = robot.damage_evaluators[0]
            if hasattr(r_dmg, "last_strain_by_link") and len(getattr(r_dmg, "last_strain_by_link", {})) > 0:
                arm_gripper_strains = {
                    link_name: strain
                    for link_name, strain in r_dmg.last_strain_by_link.items()
                    if "arm" in link_name.lower() or "gripper" in link_name.lower()
                }
                if arm_gripper_strains:
                    robot_strains.append(float(max(arm_gripper_strains.values())))
                else:
                    robot_strains.append(float(max(r_dmg.last_strain_by_link.values())))
            else:
                if hasattr(r_dmg, "get_current_env_step_strain"):
                    robot_strains.append(float(r_dmg.get_current_env_step_strain()))
                elif hasattr(r_dmg, "strain_values") and len(r_dmg.strain_values) > 0:
                    robot_strains.append(float(r_dmg.strain_values[-1]))
                else:
                    robot_strains.append(0.0)

        if target_obj is not None and hasattr(target_obj, "damage_evaluators") and len(getattr(target_obj, "damage_evaluators", [])) > 0:
            t_dmg = target_obj.damage_evaluators[0]
            if hasattr(t_dmg, "last_strain_by_link") and len(getattr(t_dmg, "last_strain_by_link", {})) > 0:
                if target_object_key == "drawer":
                    # Match mech_damage.py: only track strain from link_3 when target is a drawer
                    link3_val = None
                    for k, v in t_dmg.last_strain_by_link.items():
                        if k.lower() == "link_3" or "link_3" in k.lower():
                            link3_val = float(v)
                            break
                    target_strains.append(0.0 if link3_val is None else link3_val)
                else:
                    target_strains.append(float(max(t_dmg.last_strain_by_link.values())))
            else:
                if hasattr(t_dmg, "get_current_env_step_strain"):
                    target_strains.append(float(t_dmg.get_current_env_step_strain()))
                elif hasattr(t_dmg, "strain_values") and len(t_dmg.strain_values) > 0:
                    target_strains.append(float(t_dmg.strain_values[-1]))
                else:
                    target_strains.append(0.0)

        # Prepare frame for video and optional PNG export
        frame_bgr = rgb_bgr
        video_frames.append(frame_bgr)
        video_frames_original.append(rgb_bgr_original)

        # Downsample PNGs for disk by saving every 10th frame,
        # while the video keeps all frames (same structure as rl_test).
        if step % 10 == 0:
            frame_filename = f"frame_{frame_count:05d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame_bgr)
            frame_count += 1

            if step % 50 == 0:
                print(f"Step {step}/{max_steps} - Saved frame {frame_count}")

    print(f"Teleoperation complete. Saved {frame_count} frames to: {output_dir}")

    # Build a sim video from the recolored frames, similar to rl_test.py
    videos_dir = os.path.join(unit_tests_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    sim_mp4_path = None
    sim_nocolor_mp4_path = None
    comparison_mp4_path = None
    if len(video_frames) > 0:
        # Recolored sim video
        avi_path = os.path.join(videos_dir, f"{base_name}_sim.avi")
        sim_mp4_path = os.path.join(videos_dir, f"{base_name}_sim.mp4")

        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        for f in video_frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()

        subprocess.run(
            ["ffmpeg", "-y", "-i", avi_path, "-c:v", "mpeg4", "-q:v", "2", sim_mp4_path],
            check=True,
        )
        os.remove(avi_path)

        print(f"Sim video (with coloring) saved to: {sim_mp4_path}")

        # Original (no-color) sim video for comparison
        if len(video_frames_original) == len(video_frames):
            avi_nocolor = os.path.join(videos_dir, f"{base_name}_sim_nocolor.avi")
            sim_nocolor_mp4_path = os.path.join(videos_dir, f"{base_name}_sim_nocolor.mp4")

            vw_nc = cv2.VideoWriter(avi_nocolor, fourcc, fps, (w, h))
            for f in video_frames_original:
                vw_nc.write(np.ascontiguousarray(f, dtype=np.uint8))
            vw_nc.release()

            subprocess.run(
                ["ffmpeg", "-y", "-i", avi_nocolor, "-c:v", "mpeg4", "-q:v", "2", sim_nocolor_mp4_path],
                check=True,
            )
            os.remove(avi_nocolor)

            print(f"Sim video (no coloring) saved to: {sim_nocolor_mp4_path}")

            # Side-by-side comparison: left = no-color, right = colored
            comparison_mp4_path = os.path.join(videos_dir, f"{base_name}_sim_comparison.mp4")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", sim_nocolor_mp4_path,
                "-i", sim_mp4_path,
                "-filter_complex",
                "[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]",
                "-map", "[v]",
                "-c:v", "mpeg4",
                "-q:v", "5",
                comparison_mp4_path,
            ], check=True)

            print(f"Comparison video (no color vs. colored) saved to: {comparison_mp4_path}")

        # Also build a bordered sim video whose border encodes damage status (green / yellow / red),
        # matching the pattern in rl_test.py.
        if sim_mp4_path is not None and len(video_frames) == len(target_statuses):
            border_avi = os.path.join(videos_dir, f"{base_name}_sim_with_status_border.avi")
            border_mp4 = os.path.join(videos_dir, f"{base_name}_sim_with_status_border.mp4")
            vw_border = cv2.VideoWriter(border_avi, fourcc, fps, (w + 60, h + 60))
            for f, status in zip(video_frames, target_statuses):
                s = (status or "none").lower()
                if s in ("major", "critical"):
                    bgr = (0, 0, 255)  # Red
                elif s == "minor":
                    bgr = (0, 255, 255)  # Yellow
                else:
                    bgr = (0, 255, 0)  # Green for none/negligible
                bordered = cv2.copyMakeBorder(
                    f, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=bgr
                )
                vw_border.write(np.ascontiguousarray(bordered, dtype=np.uint8))
            vw_border.release()

            subprocess.run(
                ["ffmpeg", "-y", "-i", border_avi, "-c:v", "mpeg4", "-q:v", "2", border_mp4],
                check=True,
            )
            os.remove(border_avi)

            print(f"Bordered sim video (damage-status border) saved to: {border_mp4}")

        # Build segmentation-only video for the target object (colored by damage status),
        # using in-memory seg_frames (no PNGs), exactly like rl_test.py.
        if len(seg_frames) == len(video_frames):
            seg_avi = os.path.join(videos_dir, f"{base_name}_target_segmentation.avi")
            seg_mp4 = os.path.join(videos_dir, f"{base_name}_target_segmentation.mp4")
            hs, ws = seg_frames[0].shape[:2]
            vw_seg = cv2.VideoWriter(seg_avi, fourcc, fps, (ws, hs))
            for f in seg_frames:
                vw_seg.write(np.ascontiguousarray(f, dtype=np.uint8))
            vw_seg.release()

            subprocess.run(
                ["ffmpeg", "-y", "-i", seg_avi, "-c:v", "mpeg4", "-q:v", "2", seg_mp4],
                check=True,
            )
            os.remove(seg_avi)

            print(f"Segmentation-only video (target colored by damage) saved to: {seg_mp4}")

    # Build animated health plot video (robot + target) similar to mech_damage.py
    health_mp4 = os.path.join(videos_dir, f"{base_name}_healths.mp4")
    should_track_robot = track_robot_health or (target_object_key is None)

    if should_track_robot and len(robot_healths) > 0 and (target_object_key is not None) and len(target_healths) > 0:
        # Both robot and target
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = max(len(robot_healths), len(target_healths))
        y_min = 0.0
        y_max = 100.0

        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax.plot([], [], lw=6, color='tab:blue', label='Robot')
        line_t, = ax.plot([], [], lw=6, color='tab:orange', label=target_object_key.capitalize())
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
            line_t.set_data([], [])
            return line_r, line_t

        def animate_health(i):
            x = [k / fps for k in range(1, i + 2)]
            y_r = robot_healths[: i + 1]
            y_t = target_healths[: i + 1]
            line_r.set_data(x, y_r)
            line_t.set_data(x, y_t)
            return line_r, line_t

        ani = animation.FuncAnimation(
            fig, animate_health, init_func=init_health, frames=T, interval=1000 / fps, blit=True
        )
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)
    elif should_track_robot and len(robot_healths) > 0 and (target_object_key is None):
        # Robot only
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
    elif not should_track_robot and (target_object_key is not None) and len(target_healths) > 0:
        # Target only (when track_robot_health is False)
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = len(target_healths)
        y_min = 0.0
        y_max = 100.0
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_t, = ax.plot([], [], lw=6, color='tab:orange', label=target_object_key.capitalize())
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Health', fontsize=20)
        ax.set_title('Health Over Time', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_health_t():
            line_t.set_data([], [])
            return line_t,

        def animate_health_t(i):
            x = [k / fps for k in range(1, i + 2)]
            y_t = target_healths[: i + 1]
            line_t.set_data(x, y_t)
            return line_t,

        ani = animation.FuncAnimation(fig, animate_health_t, init_func=init_health_t, frames=T, interval=1000 / fps, blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)

    # Build strain-over-time video (robot + target) similar to mech_damage.py
    strain_mp4 = os.path.join(videos_dir, f"{base_name}_strain.mp4")
    should_track_robot = track_robot_health or (target_object_key is None)

    if should_track_robot and len(robot_strains) > 0:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T_strain = max(len(robot_strains), len(target_strains) if target_obj is not None else 0)
        fig_s, ax_s = plt.subplots(figsize=(9.6, 5.4))

        line_sr, = ax_s.plot([], [], lw=4, color='tab:blue', label='Robot')
        line_st = None
        if target_obj is not None and len(target_strains) > 0:
            line_st, = ax_s.plot([], [], lw=4, color='tab:orange', label=target_object_key.capitalize())

        ax_s.set_xlim(0, max(1, T_strain) / fps)

        # Threshold lines for both robot and target
        robot_mech_params = PARAMS["tiago_robot"]["mechanical"]
        link_thresholds = robot_mech_params.get("link_thresholds", {})
        robot_thresh = None
        if "arm" in link_thresholds:
            robot_thresh = link_thresholds["arm"]["damage_threshold"]
        elif "gripper" in link_thresholds:
            robot_thresh = link_thresholds["gripper"]["damage_threshold"]
        else:
            robot_thresh = robot_mech_params.get("damage_threshold", 0.0)

        target_thresh = None
        if target_object_key is not None:
            target_params = PARAMS.get(target_object_key, PARAMS["default"])
            target_thresh = target_params["mechanical"]["damage_threshold"]

        extra = []
        if robot_thresh is not None:
            extra.append(robot_thresh)
        if target_thresh is not None:
            extra.append(target_thresh)
        all_strains = robot_strains + (target_strains if target_obj is not None else [])
        y_min = 0.0
        y_max = max(all_strains + extra + [0.0])
        if y_min == y_max:
            y_min, y_max = (0.0, 1.0)
        ax_s.set_ylim(y_min, y_max * 1.1)

        if robot_thresh is not None:
            ax_s.axhline(y=robot_thresh, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Robot Threshold ({robot_thresh})')
        if target_thresh is not None:
            ax_s.axhline(y=target_thresh, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'{target_object_key.capitalize()} Threshold ({target_thresh})')

        ax_s.set_xlabel('Time (s)', fontsize=16)
        ax_s.set_ylabel('Strain', fontsize=16)
        ax_s.set_title('Strain Over Time', fontsize=20)
        ax_s.legend(loc='best')
        plt.tight_layout()

        def init_strain():
            line_sr.set_data([], [])
            if line_st is not None:
                line_st.set_data([], [])
            return (line_sr, line_st) if line_st is not None else (line_sr,)

        def animate_strain(i):
            x = [k / fps for k in range(1, i + 2)]
            y_r = robot_strains[: i + 1]
            line_sr.set_data(x, y_r)
            if line_st is not None:
                y_t = target_strains[: i + 1]
                line_st.set_data(x, y_t)
                return line_sr, line_st
            return line_sr,

        ani_s = animation.FuncAnimation(fig_s, animate_strain, init_func=init_strain, frames=T_strain, interval=1000 / fps, blit=True)
        writer_s = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani_s.save(strain_mp4, writer=writer_s)
        plt.close(fig_s)
    elif not should_track_robot and target_obj is not None and len(target_strains) > 0:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T_strain = len(target_strains)
        fig_s, ax_s = plt.subplots(figsize=(9.6, 5.4))

        line_st, = ax_s.plot([], [], lw=4, color='tab:orange', label=target_object_key.capitalize())
        ax_s.set_xlim(0, max(1, T_strain) / fps)

        y_min = 0.0
        y_max = max(target_strains + [0.0])
        if y_min == y_max:
            y_min, y_max = (0.0, 1.0)
        ax_s.set_ylim(y_min, y_max * 1.1)

        target_params = PARAMS.get(target_object_key, PARAMS["default"])
        target_thresh = target_params["mechanical"]["damage_threshold"]
        ax_s.axhline(y=target_thresh, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'{target_object_key.capitalize()} Threshold ({target_thresh})')

        ax_s.set_xlabel('Time (s)', fontsize=16)
        ax_s.set_ylabel('Strain', fontsize=16)
        ax_s.set_title('Strain Over Time', fontsize=20)
        ax_s.legend(loc='best')
        plt.tight_layout()

        def init_strain_t():
            line_st.set_data([], [])
            return line_st,

        def animate_strain_t(i):
            x = [k / fps for k in range(1, i + 2)]
            y_t = target_strains[: i + 1]
            line_st.set_data(x, y_t)
            return line_st,

        ani_s = animation.FuncAnimation(fig_s, animate_strain_t, init_func=init_strain_t, frames=T_strain, interval=1000 / fps, blit=True)
        writer_s = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani_s.save(strain_mp4, writer=writer_s)
        plt.close(fig_s)

    # Clean shutdown
    camera_mover.clear()
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main()

