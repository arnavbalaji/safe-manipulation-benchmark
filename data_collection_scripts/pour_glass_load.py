"""
Load a saved pour_glass init state and teleoperate. Uses config from teleop_pour_glass.py,
loads init state, settles, generates water in the glass, then teleop (gripper closed by default).
Default state: resources/saved_states/pour_glass_init_state.pkl (path is relative to script). Override with --state_path.
TAB: save 3840x2160 viewer camera PNG. WASD+G: move camera. R: reset.
"""
import os
import sys

# Allow importing from teleop_scripts (same repo)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_benchmark_dir = os.path.dirname(_script_dir)
if _benchmark_dir not in sys.path:
    sys.path.insert(0, _benchmark_dir)

os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import math
import pickle
from datetime import datetime
import torch as th
import numpy as np
import cv2

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import Filled
from omnigibson.systems import FluidSystem

from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import omnigibson.lazy as lazy
from omnigibson.controllers.controller_base import IsGraspingState

from safety_benchmark.damageable_env import DamageableEnvironment, load_sim_state_with_size_fallback

from teleop_scripts.teleop_pour_glass import get_pour_glass_config

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False


def set_laptop_pose(env, target_deg: float = 130.0):
    """Open the laptop to a specified angle (from pour_glass.py)."""
    laptop = env.scene.object_registry("name", "laptop")
    if laptop is None:
        return
    target_rad = math.radians(float(target_deg))
    if hasattr(laptop, "joints"):
        for joint in laptop.joints.values():
            lo = joint.lower_limit
            hi = joint.upper_limit
            target = max(lo, min(hi, target_rad))
            joint.set_pos(target)
            joint.friction = 50000000.0
            if hasattr(joint, "keep_still"):
                joint.keep_still()
    if hasattr(laptop, "keep_still"):
        laptop.keep_still()


def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    _default_state = os.path.join(_benchmark_dir, "resources", "saved_states", "pour_glass_init_state.pkl")
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", type=str, default=_default_state, help="Path to saved state pkl")
    args = parser.parse_args()

    # Config from teleop_pour_glass (no external sensors for load script)
    cfg = get_pour_glass_config(external_sensors=False)

    env = DamageableEnvironment(configs=cfg)
    env.reset()
    robot = env.robots[0]
    for _ in range(10):
        og.sim.step()

    state_path = os.path.abspath(args.state_path)
    if not os.path.exists(state_path):
        print("Error: State file not found at", state_path)
        print("Default is resources/saved_states/pour_glass_init_state.pkl (save one with teleop_pour_glass.py S key).")
        print("Or pass --state_path /path/to/your_state.pkl")
        return
    with open(state_path, "rb") as f:
        state_flat_array = pickle.load(f)
    load_sim_state_with_size_fallback(state_flat_array)
    print("Loaded state from", state_path)

    # Capture poses immediately after load
    robot_pos, robot_orn = robot.get_position_orientation()
    robot_joint_positions = robot.get_joint_positions()
    close_gripper_action = th.zeros(robot.action_dim)
    close_gripper_action[robot.gripper_action_idx[robot.default_arm]] = -1.0

    # Settle: gripper closed the whole time
    robot.keep_still()
    for _ in range(20):
        robot.keep_still()
        robot.apply_action(close_gripper_action)
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
    for _ in range(15):
        robot.apply_action(close_gripper_action)
        og.sim.step()
        robot.keep_still()

    # Keep the laptop open (same as pour_glass.py)
    set_laptop_pose(env, target_deg=130.0)
    og.sim.update_handles()

    # Generate water in the glass (gripper closed every step)
    water_glass = env.scene.object_registry("name", "water_glass")
    if water_glass is not None:
        water_system = env.scene.get_system("water", force_init=True)
        if Filled in water_glass.states:
            water_glass.states[Filled].set_value(water_system, True)
        glass_pos, _ = water_glass.get_position_orientation()
        z_offset = 0.05
        for _ in range(100):
            if isinstance(glass_pos, th.Tensor):
                drop_pos = (glass_pos + th.tensor([0.0, 0.0, z_offset], dtype=th.float32)).tolist()
            else:
                drop_pos = [glass_pos[0], glass_pos[1], glass_pos[2] + z_offset]
            water_system.generate_particles(positions=[drop_pos])
            robot.set_joint_positions(robot_joint_positions)
            robot.set_joint_velocities(th.zeros(robot.n_dof))
            robot.keep_still()
            robot.apply_action(close_gripper_action)
            og.sim.step()
            robot.set_joint_positions(robot_joint_positions)
            robot.set_joint_velocities(th.zeros(robot.n_dof))
            robot.keep_still()
        print(f"Generated 100 water particles. Total particles: {water_system.n_particles}")

    # Settle again with gripper closed
    for _ in range(30):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        robot.apply_action(close_gripper_action)
        og.sim.step()
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([7.0659, -0.7141, 1.9185]),
        orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
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
    # Gripper closed the whole time during teleop
    for gripper_name in action_generator.binary_grippers:
        action_generator.gripper_direction[gripper_name] = -1.0
        action_generator.persistent_gripper_action[gripper_name] = -1.0

    TASK_NAME = "pour_glass"
    SCREENSHOT_DIR = "safe-manipulation-benchmark/resources/teleop_data/pour_glass_screenshots"
    RGB_SIZE = (3840, 2160)

    def breakpoint_and_print_poses():
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = np.asarray(rgb.cpu().numpy()[:, :, :3], dtype=np.uint8)
        frame = cv2.resize(rgb_np, RGB_SIZE)
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(SCREENSHOT_DIR, TASK_NAME + "_" + timestamp + ".png")
        cv2.imwrite(png_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("Viewer camera screenshot saved:", png_path, RGB_SIZE)
        print("\n" + "=" * 60)
        print("POSE INFORMATION (TAB pressed)")
        print("=" * 60)
        eef_pos = robot.get_eef_position(arm="default")
        eef_orn = robot.get_eef_orientation(arm="default")
        print("Robot EEF Pose:", eef_pos.tolist(), eef_orn.tolist())
        water_glass_obj = env.scene.object_registry("name", "water_glass")
        if water_glass_obj is not None:
            p, o = water_glass_obj.get_position_orientation()
            print("Water glass Pose:", p.tolist(), o.tolist())
        cam_pos, cam_orn = og.sim.viewer_camera.get_position_orientation()
        print("Viewer Camera Pose:", cam_pos.tolist(), cam_orn.tolist())
        print("=" * 60 + "\n")
        breakpoint()

    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.R, description="Reset", callback_fn=lambda: env.reset())
    action_generator.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.TAB, description="Save 3840x2160 PNG, print poses, breakpoint", callback_fn=breakpoint_and_print_poses)
    action_generator.print_keyboard_teleop_info()

    print("=" * 60)
    print("Pour glass load: Teleoperate, WASD+G camera, TAB save image, gripper closed by default, ESC quit")
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
