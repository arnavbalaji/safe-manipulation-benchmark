"""
Move-glass teleop: Tiago in Rs_int with breakfast_table, laptop, mug, plate, sink.
Config inspired by user YAML. Press S to save state to pkl, R to reset.
"""
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import pickle
import torch as th
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import omnigibson.lazy as lazy

from safety_benchmark.damageable_env import (
    DamageableEnvironment,
    DamageableDataCollectionWrapper,
    load_sim_state_with_size_fallback,
)

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False

# Scene: Rs_int, task-relevant only; robot pose and objects from user YAML
ROBOT_DAMAGE_PARAMS = {
    "damage_evaluators": ["mechanical", "electrical"],
    "health_thresholds": [90.0, 60.0, 30.0],
    "mechanical": {
        "damage_threshold": 10000000000.0,
        "scale": 1e-10,
        "instant_coefficient": 1.0,
        "creep_coefficient": 1.0,
        "object_type": "brittle",
        "link_thresholds": {
            "arm": {"damage_threshold": 200.0, "scale": 0.0001},
            "gripper": {"damage_threshold": 200.0, "scale": 0.0001},
        },
    },
    "electrical": {
        "damage_threshold": 0.0,
        "scale": 0.001,
        "water_system_name": "sludge",
    },
}

# Object poses from reset_saved.json so table is waist-high and mug/laptop/plate are OnTop of table
TASK_OBJECTS = [
    {
        "type": "DatasetObject",
        "name": "breakfast_table",
        "category": "breakfast_table",
        "model": "hggsao",
        "position": [-0.008, -0.001, 0.396],  # raised so tabletop ~0.5–0.65 (was [0,0,0] = on floor)
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.0],
        "fixed_base": True,
    },
    {
        "type": "DatasetObject",
        "name": "laptop",
        "category": "laptop",
        "model": "nvulcs",
        "position": [-0.18, 0.001, 0.506],  # on table (reset_saved)
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [0.9, 0.9, 0.9],
        "abilities": {"openable": {}},
        "damage_params": {
            "damage_evaluators": ["mechanical", "electrical"],
            "health_thresholds": [90.0, 60.0, 30.0],
            "mechanical": {
                "damage_threshold": 200.0,
                "scale": 0.1,
                "instant_coefficient": 1.0,
                "creep_coefficient": 0.0,
                "object_type": "brittle",
            },
            "electrical": {
                "damage_threshold": 100.0,
                "scale": 1000.0,
                "water_system_name": "water",
            },
        },
    },
    {
        "type": "DatasetObject",
        "name": "mug",
        "category": "water_glass",
        "model": "vcwsbm",
        "position": [-0.10, 0.25, 0.647],  # glass mug on table (reset_saved)
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.0],
        "damage_params": {
            "damage_evaluators": ["mechanical"],
            "health_thresholds": [90.0, 60.0, 30.0],
            "mechanical": {
                "damage_threshold": 200.0,
                "scale": 0.5,
                "instant_coefficient": 1.0,
                "creep_coefficient": 0.0,
                "object_type": "brittle",
            },
        },
    },
    {
        "type": "DatasetObject",
        "name": "plate",
        "category": "plate",
        "model": "ntedfx",
        "position": [-0.40, 0.20, 0.513],  # on table (reset_saved)
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [0.6, 0.6, 0.6],
    },
    {
        "type": "DatasetObject",
        "name": "furniture_sink_far",
        "category": "furniture_sink",
        "model": "czyfhq",
        "position": [0.0, 3.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 2.0, 0.6],
        "abilities": {
            "toggleable": {},
            "particleSource": {"conditions": {"water": []}, "initial_speed": 0.2},
            "particleSink": {"conditions": {"water": []}},
        },
    },
]

# Viewer camera (same as ppo_train_move_glass)
CAMERA_POSITION = [-1.3225984573364258, 0.2645236849784851, 0.9981386065483093]
CAMERA_ORIENTATION = [0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452]

SAVE_STATE_PATH = "safe-manipulation-benchmark/resources/saved_states/move_glass_init_state.pkl"


def get_move_glass_config():
    """Build config from tiago_primitives + user YAML (scene, robot pose/damage, objects)."""
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Scene: Rs_int, task-relevant only
    cfg["scene"]["scene_model"] = "Rs_int"
    cfg["scene"]["include_robots"] = False
    cfg["scene"]["load_task_relevant_only"] = True

    # Tiago: pose and damage_params from user YAML; IK with pose_delta_ori for teleop
    base_robot = cfg["robots"][0]
    # Trunk down: reset_joint_pos index 6 is torso_lift_joint (0 = down, 0.35 in yaml = too high)
    reset_joint_pos = list(base_robot.get("reset_joint_pos", []))
    if len(reset_joint_pos) > 6:
        reset_joint_pos[6] = 0.0

    # Inverse kinematics with pose_delta_ori so keyboard teleop controls EEF pose (like ppo_train_move_glass)
    controller_config = dict(base_robot.get("controller_config", {}))
    controller_config["arm_left"] = {
        "name": "InverseKinematicsController",
        "command_input_limits": None,
        "mode": "pose_delta_ori",
        "subsume_controllers": ["trunk"],
    }
    controller_config["arm_right"] = {
        "name": "InverseKinematicsController",
        "command_input_limits": None,
        "mode": "pose_delta_ori",
    }
    controller_config["gripper_left"] = {
        "name": "MultiFingerGripperController",
        "command_input_limits": (0.0, 1.0),
        "mode": "smooth",
    }
    controller_config["gripper_right"] = {
        "name": "MultiFingerGripperController",
        "command_input_limits": (0.0, 1.0),
        "mode": "smooth",
    }

    cfg["robots"][0] = {
        **base_robot,
        "position": [0.2, 1.0, 0.0],
        "orientation": [0, 0, -0.7071068, 0.7071068],  # -90° Z, normalized from [0,0,-1,1]
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        "grasping_mode": "assisted",
        "damage_params": ROBOT_DAMAGE_PARAMS,
        "reset_joint_pos": reset_joint_pos,
        "controller_config": controller_config,
    }

    cfg["objects"] = TASK_OBJECTS
    return cfg


def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_state", action="store_true", help="Load a state (legacy, use --state_path)")
    parser.add_argument(
        "--state_path",
        type=str,
        default=None,
        help="Path to pkl state from move_glass_teleop (S key). Load in-process so scene UUIDs match.",
    )
    args = parser.parse_args()

    collect_hdf5_path = "safe-manipulation-benchmark/resources/teleop_data/move_glass.hdf5"
    os.makedirs(os.path.dirname(collect_hdf5_path), exist_ok=True)

    cfg = get_move_glass_config()
    env = DamageableEnvironment(configs=cfg)
    env = DamageableDataCollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
        enable_dump_filters=False,
    )

    robot = env.robots[0]

    # Apply default pose so Tiago loads with trunk down and arms in (not stuck out)
    env.reset()
    for _ in range(5):
        og.sim.step()

    # Explicitly set trunk down and arms to default "diagonal15" pose so teleop works
    try:
        pos = robot.get_joint_positions().clone()
        # Trunk down (0 = lowest)
        if hasattr(robot, "trunk_control_idx"):
            pos[robot.trunk_control_idx] = 0.0
        # Arms in default diagonal15 pose (not horizontal/out)
        diagonal15_arm = th.tensor(
            [0.90522, -0.42811, 2.23505, 1.64627, 0.76867, -0.79464, -1.08908],
            device=pos.device,
            dtype=pos.dtype,
        )
        if hasattr(robot, "arm_control_idx"):
            for arm in robot.arm_names:
                if arm in robot.arm_control_idx:
                    pos[robot.arm_control_idx[arm]] = diagonal15_arm
        if hasattr(robot, "gripper_control_idx"):
            for arm in robot.arm_names:
                if arm in robot.gripper_control_idx:
                    pos[robot.gripper_control_idx[arm]] = th.tensor(
                        [0.045, 0.045], device=pos.device, dtype=pos.dtype
                    )
        robot.set_joint_positions(pos)
    except Exception as e:
        print(f"Warning: could not set Tiago default pose: {e}")

    for _ in range(15):
        og.sim.step()

    # Load saved state in-process (same env/scene so UUIDs match; avoids AssertionError in load script)
    if args.state_path is not None:
        if not os.path.exists(args.state_path):
            print(f"Error: --state_path not found: {args.state_path}")
            return
        try:
            with open(args.state_path, "rb") as f:
                state_flat_array = pickle.load(f)
            load_sim_state_with_size_fallback(state_flat_array)
            print(f"Loaded state from {args.state_path}")
        except Exception as e:
            print(f"Error loading state: {e}")
            raise
        for _ in range(15):
            og.sim.step()
        # Reset IK/gripper controllers so their internal state matches loaded pose (teleop works)
        for ctrl_name in ["arm_left", "arm_right", "gripper_left", "gripper_right"]:
            ctrl = robot.controllers.get(ctrl_name)
            if ctrl is not None and hasattr(ctrl, "reset"):
                ctrl.reset()

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(CAMERA_POSITION, dtype=th.float32),
        orientation=th.tensor(CAMERA_ORIENTATION, dtype=th.float32),
    )

    # Optional: fix table after settling
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
        og.sim.step()

    # Camera teleoperation (WASD + G)
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

    print("Move-glass teleop ready. S = save state, R = reset, ESC = quit.")

    n_episodes = 1
    for i in range(n_episodes):
        print(f"Episode {i} starts")
        if args.load_state:
            breakpoint()
            for _ in range(50):
                og.sim.step()

        while True:
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

    print("Data saved")
    env.save_data()
    camera_mover.clear()
    og.shutdown()


if __name__ == "__main__":
    __main__()
