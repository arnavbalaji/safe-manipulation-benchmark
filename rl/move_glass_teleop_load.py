"""
Load the existing move_glass pkl and teleoperate.

The pkl was saved from an env built from reset_saved.json (same scene as ppo_train_move_glass).
We must create the env from that same JSON so object UUIDs match; then load the pkl and run teleop.

Usage:
  python safe-manipulation-benchmark/rl/move_glass_teleop_load.py [--state_path <path>]
"""
import math
import os
import sys

os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import pickle
import torch as th
import numpy as np

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RL_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARK_DIR)

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import Filled
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import omnigibson.lazy as lazy

from safety_benchmark.damageable_env import DamageableEnvironment, load_sim_state_with_size_fallback
from safety_benchmark.lenient_registry_load import apply_lenient_registry_patch

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False

# Same scene file as ppo_train_move_glass — required so loaded pkl UUIDs match
RESET_SAVED_JSON = os.path.join(_RL_DIR, "saved_states", "reset_saved.json")
DEFAULT_STATE_PATH = os.path.join(_BENCHMARK_DIR, "resources", "saved_states", "move_glass_init_state_final.pkl")
CAMERA_POSITION = [-1.3225984573364258, 0.2645236849784851, 0.9981386065483093]
CAMERA_ORIENTATION = [0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452]
SAVE_STATE_PATH = os.path.join(_BENCHMARK_DIR, "resources", "saved_states", "move_glass_init_state_final.pkl")


def _resolve_state_path(path):
    if os.path.exists(path):
        return os.path.normpath(path)
    alt = os.path.join(_BENCHMARK_DIR, "resources", "saved_states", "move_glass_init_state_final.pkl")
    return os.path.normpath(alt)


def _execute_init_primitive(env, robot):
    """Run the same init primitive as ppo_train_move_glass (warm-up + one small EEF delta move) before water fill."""
    action_dim = int(robot.action_dim)
    right_arm = "right" if (hasattr(robot, "arm_names") and "right" in robot.arm_names) else getattr(robot, "default_arm", "right")
    right_arm_idx = robot.arm_action_idx[right_arm].cpu().numpy().astype(np.int64)
    right_grip_idx = robot.gripper_action_idx[right_arm].cpu().numpy().astype(np.int64)
    arm_action_scale = 0.05
    pos_dim = 3  # dx, dy, dz

    def _safe_env_step(action):
        try:
            env.step(th.from_numpy(action).to(dtype=th.float32))
            return True
        except (AttributeError, TypeError) as e:
            if "view" in str(e) or "NoneType" in str(e):
                return False
            raise

    zero_action = np.zeros((action_dim,), dtype=np.float32)
    zero_action[right_grip_idx] = -1.0  # gripper closed (same convention as rest of this script)
    for _ in range(80):
        if _safe_env_step(zero_action):
            break
        for _ in range(5):
            try:
                og.sim.step()
            except Exception:
                pass
    else:
        return

    def _exec_delta(delta, max_steps=50):
        dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
        step_delta = (dx / max_steps, dy / max_steps, dz / max_steps)
        arm_cmd = np.array([
            step_delta[0] / arm_action_scale,
            step_delta[1] / arm_action_scale,
            step_delta[2] / arm_action_scale,
        ], dtype=np.float32)
        arm_cmd = np.clip(arm_cmd, -1.0, 1.0)
        full = np.zeros((action_dim,), dtype=np.float32)
        full[right_arm_idx[:pos_dim]] = arm_cmd * arm_action_scale
        full[right_grip_idx] = -1.0
        for _ in range(max_steps):
            if not _safe_env_step(full):
                for _ in range(15):
                    try:
                        og.sim.step()
                    except Exception:
                        pass
                _safe_env_step(full)
                break

    x_delta = np.random.uniform(-0.05, 0.1)
    y_delta = np.random.uniform(-0.1, 0.05)
    _exec_delta(np.array([x_delta, y_delta, 0.0]))
    print("Executed init primitive (small EEF delta move).")


def _set_laptop_pose(env, target_deg: float = 120.0):
    """Keep laptop lid at target angle (same as ppo_train_move_glass.py)."""
    laptop = env.scene.object_registry("name", "laptop")
    if laptop is None:
        return
    target_rad = math.radians(float(target_deg))
    if hasattr(laptop, "joints"):
        for joint in laptop.joints.values():
            try:
                lo = joint.lower_limit
                hi = joint.upper_limit
            except Exception:
                continue
            if lo is None or hi is None:
                continue
            target = max(lo, min(hi, target_rad))
            try:
                joint.set_pos(target)
                if hasattr(joint, "keep_still"):
                    joint.keep_still()
            except Exception:
                continue
    if hasattr(laptop, "keep_still"):
        laptop.keep_still()
    og.sim.step_physics()


def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", type=str, default=DEFAULT_STATE_PATH, help="Path to pkl state")
    args = parser.parse_args()
    state_path = _resolve_state_path(args.state_path)

    if not os.path.exists(state_path):
        print(f"Error: State file not found: {state_path}")
        return
    if not os.path.exists(RESET_SAVED_JSON):
        print(f"Error: Scene file not found: {RESET_SAVED_JSON}")
        return

    try:
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_TRANSITION_RULES = False
    except Exception:
        pass

    if og.sim is not None:
        og.sim.stop()
        og.clear()

    # Lenient deserialize so pkls from scenes with extra objects can load (skip unknown UUIDs).
    apply_lenient_registry_patch()

    # Create env from same scene file that produced the pkl (so UUIDs match)
    cfg = {"scene": {"type": "Scene", "scene_file": RESET_SAVED_JSON}}
    env = DamageableEnvironment(configs=cfg)
    env.reset()
    robot = env.robots[0]

    for _ in range(10):
        og.sim.step()

    with open(state_path, "rb") as f:
        state_flat_array = pickle.load(f)
    load_sim_state_with_size_fallback(state_flat_array)
    print(f"Loaded state from {state_path}")

    for _ in range(15):
        og.sim.step()

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(CAMERA_POSITION, dtype=th.float32),
        orientation=th.tensor(CAMERA_ORIENTATION, dtype=th.float32),
    )

    # IK + pose_delta_ori so keyboard teleop works (same as move_glass_teleop / ppo_train_move_glass)
    controller_config = {
        "arm_left": {"name": "InverseKinematicsController", "command_input_limits": None, "mode": "pose_delta_ori", "subsume_controllers": ["trunk"]},
        "arm_right": {"name": "InverseKinematicsController", "command_input_limits": None, "mode": "pose_delta_ori"},
        "gripper_left": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0), "mode": "smooth"},
        "gripper_right": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0), "mode": "smooth"},
    }
    robot.reload_controllers(controller_config=controller_config)
    for ctrl_name in ["arm_left", "arm_right", "gripper_left", "gripper_right"]:
        ctrl = robot.controllers.get(ctrl_name)
        if ctrl is not None and hasattr(ctrl, "reset"):
            ctrl.reset()

    # Execute init primitive before water (same order as ppo_train_move_glass: _execute_init then fill mug).
    _execute_init_primitive(env, robot)

    # Keep gripper closed and settle.
    close_gripper_action = th.zeros(robot.action_dim)
    for arm_name in getattr(robot, "gripper_action_idx", {}):
        idx = robot.gripper_action_idx[arm_name]
        close_gripper_action[idx] = -1.0
    for _ in range(15):
        robot.apply_action(close_gripper_action)
        og.sim.step()

    # Generate water particles in the mug (same as ppo_train_move_glass reset).
    mug_obj = env.scene.object_registry("name", "mug")
    if mug_obj is not None:
        try:
            water_system = env.scene.get_system("water", force_init=True)
            if water_system is not None:
                if Filled in mug_obj.states:
                    mug_obj.states[Filled].set_value(water_system, True)
                mug_pos, _ = mug_obj.get_position_orientation()
                z_offset = 0.05
                if isinstance(mug_pos, th.Tensor):
                    mug_pos = mug_pos.cpu().numpy()
                for _ in range(230):
                    drop_pos = [float(mug_pos[0]), float(mug_pos[1]), float(mug_pos[2]) + z_offset]
                    water_system.generate_particles(positions=[drop_pos])
                    og.sim.step()
                print("Generated water in mug.")
        except Exception as e:
            print(f"Could not generate water in mug: {e}")

    for _ in range(10):
        _set_laptop_pose(env)
        og.sim.step()

    try:
        table = env.scene.object_registry("name", "breakfast_table")
        if table is not None:
            if getattr(table, "fixed_base", None) is not None:
                table.fixed_base = True
            if hasattr(table, "keep_still"):
                table.keep_still()
    except Exception:
        pass

    for _ in range(10):
        _set_laptop_pose(env)
        og.sim.step()

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
    # Gripper closed by default during teleop.
    for gripper_name in action_generator.binary_grippers:
        action_generator.persistent_gripper_action[gripper_name] = -1.0
        action_generator.gripper_direction[gripper_name] = -1.0

    def save_state_to_pkl():
        os.makedirs(os.path.dirname(SAVE_STATE_PATH), exist_ok=True)
        og.sim.update_handles()
        state = og.sim.dump_state(serialized=True)
        with open(SAVE_STATE_PATH, "wb") as f:
            pickle.dump(state, f)
        print(f"Simulation state saved to: {SAVE_STATE_PATH}")

    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the environment",
        callback_fn=lambda: env.reset(),
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.S,
        description="Save simulation state to pkl file",
        callback_fn=save_state_to_pkl,
    )
    action_generator.print_keyboard_teleop_info()

    print("Move-glass teleop (loaded from pkl). S = save state, R = reset, ESC = quit.")

    while True:
        _set_laptop_pose(env)
        ret = action_generator.get_teleop_action()
        if isinstance(ret, tuple) and len(ret) == 2:
            action, keypress_str = ret
        else:
            action = ret
            keypress_str = None

        camera_keys = {"W", "A", "S", "D", "G"}
        if keypress_str and keypress_str in camera_keys:
            env.step(th.zeros(robot.action_dim))
            continue

        if action is not None:
            env.step(action)
        else:
            og.sim.step()


if __name__ == "__main__":
    __main__()
