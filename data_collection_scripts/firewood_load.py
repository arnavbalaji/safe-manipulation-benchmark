"""
Load a saved firewood state and teleoperate. Run with --state_path pointing to a saved state pkl.
TAB: save 3840x2160 viewer camera PNG. WASD+G: move camera. R: reset. Teleop and capture images only.
"""
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import pickle
import yaml
from datetime import datetime
import torch as th
import numpy as np
import cv2

import omnigibson as og
from omnigibson.macros import gm
from omnigibson import object_states

from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import omnigibson.lazy as lazy
from omnigibson.controllers.controller_base import IsGraspingState

from safety_benchmark.damageable_env import DamageableEnvironment, load_sim_state_with_size_fallback

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False

# Task objects (same as firewood.py)
TASK_OBJECTS = {
    "fireplace": {
        "type": "DatasetObject",
        "name": "fireplace",
        "category": "wood_fireplace",
        "model": "gpnsij",
        "position": [-1.5, -2.0, 0.5],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 0.85, 0.85],
        "fixed_base": True,
        "abilities": {"heatSource": {"temperature": 100.0, "heating_rate": 0.1, "distance_threshold": 0.12, "requires_toggled_on": False}},
        "initial_state": {"temperature": 100.0},
    },
    "log_center": {
        "type": "DatasetObject",
        "name": "log_center",
        "category": "log",
        "model": "pepele",
        "position": [-1.65, -2.0, 0.15],
        "orientation": [0, 0, 0, 1],
        "scale": [0.8, 0.6, 0.6],
        "abilities": {"flammable": {}},
        "initial_state": {"onFire": True},
        "damage_params": {"damage_evaluators": ["mechanical"], "health_thresholds": [90.0, 60.0, 30.0], "mechanical": {"damage_threshold": 200.0, "scale": 0.5, "instant_coefficient": 1.0, "creep_coefficient": 0.0, "object_type": "brittle"}},
    },
    "log_left": {
        "type": "DatasetObject",
        "name": "log_left",
        "category": "log",
        "model": "pepele",
        "position": [-1.65, -2.15, 0.17],
        "orientation": [0, 0, 0, 1],
        "scale": [0.8, 0.6, 0.6],
        "abilities": {"flammable": {}},
        "initial_state": {"onFire": True},
        "damage_params": {"damage_evaluators": ["mechanical"], "health_thresholds": [90.0, 60.0, 30.0], "mechanical": {"damage_threshold": 200.0, "scale": 0.5, "instant_coefficient": 1.0, "creep_coefficient": 0.0, "object_type": "brittle"}},
    },
    "target_object": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "log",
        "model": "pepele",
        "position": [-1.0, -2.25, 0.1],
        "orientation": [0, 0, 0, 1],
        "scale": [0.7, 0.5, 0.5],
        "abilities": {"flammable": {}},
        "initial_state": {"onFire": False},
        "damage_params": {"damage_evaluators": ["mechanical"], "health_thresholds": [90.0, 60.0, 30.0], "mechanical": {"damage_threshold": 200.0, "scale": 0.5, "instant_coefficient": 1.0, "creep_coefficient": 0.0, "object_type": "brittle"}},
    },
}


def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", type=str, default="safe-manipulation-benchmark/resources/saved_states/firewood_init_state.pkl", help="Path to saved state pkl")
    args = parser.parse_args()

    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"] = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int", "include_robots": False, "load_task_relevant_only": True}
    cfg["robots"][0] = {
        "type": "FrankaPanda",
        "name": "franka0",
        "position": [-0.85, -2.0, 0.0],
        "orientation": [0.0, 0.0, 1.0, 0.0],
        "grasping_mode": "assisted",
        "obs_modalities": ["rgb", "depth"],
        "action_normalize": False,
        "self_collisions": True,
        "controller_config": {"arm_0": {"name": "InverseKinematicsController", "command_input_limits": None}, "gripper_0": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0), "mode": "smooth"}},
    }
    cfg["env"]["external_sensors"] = []
    cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

    env = DamageableEnvironment(configs=cfg)
    env.reset()
    robot = env.robots[0]
    for _ in range(10):
        og.sim.step()

    state_path = args.state_path
    if not os.path.exists(state_path):
        print(f"Error: State file not found at {state_path}")
        print("Provide --state_path to a saved state pkl from firewood (or similar) teleop.")
        return
    with open(state_path, "rb") as f:
        state_flat_array = pickle.load(f)
    load_sim_state_with_size_fallback(state_flat_array)
    print(f"Loaded state from {state_path}")

    robot_pos, robot_orn = robot.get_position_orientation()
    robot_joint_positions = robot.get_joint_positions()
    robot.keep_still()
    for _ in range(20):
        robot.keep_still()
        og.sim.step()
    robot.set_position_orientation(robot_pos, robot_orn)
    robot.set_joint_positions(robot_joint_positions)
    robot.set_joint_velocities(th.zeros(robot.n_dof))
    arm_controller = robot.controllers.get("arm_0")
    if arm_controller is not None:
        arm_controller.reset()
    gripper_controller = robot.controllers.get("gripper_0")
    if gripper_controller is not None:
        gripper_controller.reset()
    close_gripper_action = th.zeros(robot.action_dim)
    close_gripper_action[robot.gripper_action_idx[robot.default_arm]] = -1.0
    for _ in range(15):
        robot.apply_action(close_gripper_action)
        og.sim.step()
        robot.keep_still()

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-0.37351322174072266, -0.9105080366134644, 0.9984497427940369]),
        orientation=th.tensor([0.1866627037525177, 0.5293360948562622, 0.7805155515670776, 0.2752378284931183]),
    )

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
    action_generator = KeyboardRobotController(robot=robot)
    # Keep gripper closed by default during teleop
    for gripper_name in action_generator.binary_grippers:
        action_generator.persistent_gripper_action[gripper_name] = -1.0
        action_generator.gripper_direction[gripper_name] = -1.0

    TASK_NAME = "firewood"
    SCREENSHOT_DIR = "safe-manipulation-benchmark/resources/teleop_data/firewood_screenshots"
    RGB_SIZE = (3840, 2160)

    def breakpoint_and_print_poses():
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = np.asarray(rgb.cpu().numpy()[:, :, :3], dtype=np.uint8)
        frame = cv2.resize(rgb_np, RGB_SIZE)
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(SCREENSHOT_DIR, f"{TASK_NAME}_{timestamp}.png")
        cv2.imwrite(png_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Viewer camera screenshot saved: {png_path} ({RGB_SIZE[0]}x{RGB_SIZE[1]})")
        print("\n" + "=" * 60)
        print("POSE INFORMATION (TAB pressed)")
        print("=" * 60)
        eef_pos = robot.get_eef_position(arm="default")
        eef_orn = robot.get_eef_orientation(arm="default")
        print("Robot EEF Pose:")
        print(f"  Position: {eef_pos.tolist()}")
        print(f"  Orientation: {eef_orn.tolist()}")
        target_obj = env.scene.object_registry("name", "target_object")
        if target_obj is not None:
            p, o = target_obj.get_position_orientation()
            print("Target object Pose:")
            print(f"  Position: {p.tolist()}")
            print(f"  Orientation: {o.tolist()}")
        cam_pos, cam_orn = og.sim.viewer_camera.get_position_orientation()
        print("Viewer Camera Pose:")
        print(f"  Position: {cam_pos.tolist()}")
        print(f"  Orientation: {cam_orn.tolist()}")
        print("=" * 60 + "\n")
        breakpoint()

    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.R, description="Reset", callback_fn=lambda: env.reset())
    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.TAB, description="Save 3840x2160 PNG, print poses, breakpoint", callback_fn=breakpoint_and_print_poses)
    action_generator.print_keyboard_teleop_info()

    print("=" * 60)
    print("Firewood load: Teleoperate, WASD+G camera, TAB save image, ESC quit")
    print("=" * 60)

    prev_is_grasping = False
    while True:
        ret = action_generator.get_teleop_action()
        if isinstance(ret, tuple) and len(ret) == 2:
            action, keypress_str = ret
        else:
            action = ret
            keypress_str = None
        if keypress_str and keypress_str in {"W", "A", "S", "D", "G"}:
            env.step(th.zeros(robot.action_dim))
            continue
        try:
            is_grasping = robot.is_grasping().value == IsGraspingState.TRUE
            if is_grasping != prev_is_grasping and og.sim.is_playing():
                try:
                    og.sim.update_handles()
                except Exception:
                    pass
            prev_is_grasping = is_grasping
        except Exception:
            pass
        if og.sim.is_playing():
            try:
                og.sim.update_handles()
            except Exception:
                pass
        try:
            env.step(action)
        except AttributeError as e:
            if "'NoneType' object has no attribute" in str(e):
                if og.sim.is_playing():
                    try:
                        og.sim.update_handles()
                    except Exception:
                        pass
                env.step(action)
            else:
                raise
    camera_mover.clear()
    og.shutdown()


if __name__ == "__main__":
    __main__()
