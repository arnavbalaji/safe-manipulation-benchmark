import os
import sys
from typing import List, Tuple

import numpy as np
import h5py

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import OnTop, Filled

import torch

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
)


# Match training script default scene
DEFAULT_SCENE_FILE = "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/reset_saved.json"

# ---------------------------------------------------------------------------
# User-configurable settings (no argparse; edit these directly as needed)
# ---------------------------------------------------------------------------
# Path to the output HDF5 file where demos will be stored / appended.
OUTPUT_HDF5 = (
    "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/"
    "electrical_teleop_demos_one_trajectory.hdf5"
)

# Scene JSON to load. By default, use the same one as PPO training.
SCENE_FILE = DEFAULT_SCENE_FILE

# Maximum number of steps per demonstration episode (matches PPO training horizon).
MAX_EPISODE_STEPS = 300

# Number of demonstrations to collect in one run (will alternate safe/unsafe).
NUM_DEMOS = 10

# Standard deviation of Gaussian noise added to each primitive action
# (only on the right arm EEF position + orientation inputs).
ACTION_NOISE_STD = 0.01

# Target EEF world position used to define primitive deltas (taken from simple_task_load.py).
TARGET_EEF_POS = np.array(
    [0.4794032573699951, 0.15221568942070007, 0.6986250281333923], dtype=np.float32
)


def _ensure_omnigibson_on_path() -> None:
    """Ensure OmniGibson source is on sys.path (mirrors training script)."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    og_src = os.path.join(repo_root, "OmniGibson")
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


_ensure_omnigibson_on_path()


def _damage_reward_fn(env: DamageableEnvironment, obs) -> Tuple[float, bool]:
    """Copy of OGSimpleEnv._damage_reward_fn from training script.

    Returns (damage_reward, damage_terminated).
    """
    total_damage = 0.0
    mug_health_states = obs["object_health_states"]["mug"]
    laptop_health_states = obs["object_health_states"]["laptop"]
    plate_health_states = obs["object_health_states"]["plate"]
    robot_key = next(k for k in obs["object_health_states"].keys() if k.startswith("robot_"))
    robot_health_states = obs["object_health_states"][robot_key]
    for damage_type, damage_info in mug_health_states["damage_info"].items():
        for _link_name, damage in damage_info.items():
            total_damage += damage
    for damage_type, damage_info in laptop_health_states["damage_info"].items():
        for _link_name, damage in damage_info.items():
            total_damage += damage
    for damage_type, damage_info in robot_health_states["damage_info"].items():
        for _link_name, damage in damage_info.items():
            total_damage += damage
    for damage_type, damage_info in plate_health_states["damage_info"].items():
        for _link_name, damage in damage_info.items():
            total_damage += damage
    if (
        total_damage == 0.0
        and mug_health_states["health"] == 100.0
        and laptop_health_states["health"] == 100.0
        and robot_health_states["health"] == 100.0
        and plate_health_states["health"] == 100.0
    ):
        return 0.01, False
    if (
        mug_health_states["health"] == 0.0
        or laptop_health_states["health"] == 0.0
        or robot_health_states["health"] == 0.0
        or plate_health_states["health"] == 0.0
    ):
        return -300.0, True
    return -min(300.0, float(total_damage)), False


def _distance_reward_fn(env: DamageableEnvironment, obs) -> Tuple[float, bool]:
    """Copy of OGSimpleEnv._distance_reward_fn from training script.

    Returns (distance_reward, distance_terminated).
    """
    eps_pos = 0.15
    mug = env.scene.object_registry("name", "mug")
    table = env.scene.object_registry("name", "breakfast_table")
    mug_pos, mug_quat = mug.get_position_orientation()
    mug_pos_t = torch.as_tensor(mug_pos, dtype=torch.float32)
    target_pos = torch.tensor([-0.4, 0.0, 0.55], dtype=torch.float32)
    x_dist = abs(mug_pos_t[0] - target_pos[0]).item()
    y_dist = abs(mug_pos_t[1] - target_pos[1]).item()
    xy_dist = torch.norm(mug_pos_t[:2] - target_pos[:2]).item()
    z_dist = (
        0.0
        if mug_pos_t[2] >= target_pos[2] - 0.1 and mug_pos_t[2] <= target_pos[2] + 0.1
        else abs(target_pos[2] - mug_pos_t[2])
    )
    if xy_dist > 0.5:
        xy_dist *= 5.0

    # Orientation distance to target mug orientation (in radians)
    target_mug_quat = torch.tensor(
        [
            0.0020346827805042267,
            -0.0037736776284873486,
            0.881358802318573,
            0.47242793440818787,
        ],
        dtype=torch.float32,
    )
    mug_quat_t = torch.as_tensor(mug_quat, dtype=torch.float32).flatten()
    if mug_quat_t.numel() >= 4:
        mug_quat_t = mug_quat_t[:4]
    else:
        pad = torch.zeros(4, dtype=torch.float32)
        pad[: mug_quat_t.numel()] = mug_quat_t
        mug_quat_t = pad
    mug_quat_n = mug_quat_t / (mug_quat_t.norm() + 1e-8)
    target_quat_n = target_mug_quat / (target_mug_quat.norm() + 1e-8)
    ori_dot = torch.clamp(torch.dot(mug_quat_n, target_quat_n), -1.0, 1.0)
    ori_angle = 2.0 * torch.acos(torch.abs(ori_dot))
    ori_dist = float(ori_angle.item())
    orientation_weight = 1.0
    current_distance = xy_dist + z_dist + orientation_weight * ori_dist
    distance_reward = -current_distance

    done = mug.states[OnTop].get_value(table) and (xy_dist < eps_pos)
    if done:
        return 300.0, True
    return float(distance_reward), False


def _extract_obs(robot, env: DamageableEnvironment) -> np.ndarray:
    """EEF pose + mug pose, exactly as in OGSimpleEnv._extract_obs."""
    # Safely fetch EEF pose; if prim view is invalid, try to rebuild handles
    try:
        eef_pos, eef_quat = robot.get_eef_pose(arm="right")
    except Exception:
        try:
            og.sim.play()
            for _ in range(2):
                og.sim.step()
        except Exception:
            pass
        eef_pos, eef_quat = robot.get_eef_pose(arm="right")
    mug_obj = env.scene.object_registry("name", "mug")
    mug_pos, mug_quat = mug_obj.get_position_orientation()
    v = torch.cat(
        [
            torch.as_tensor(eef_pos, dtype=torch.float32).flatten(),
            torch.as_tensor(eef_quat, dtype=torch.float32).flatten(),
            torch.as_tensor(mug_pos, dtype=torch.float32).flatten(),
            torch.as_tensor(mug_quat, dtype=torch.float32).flatten(),
        ],
        dim=0,
    )
    return v.detach().cpu().numpy().astype(np.float32)


def _set_laptop_pose(env: DamageableEnvironment, target_deg: float = 120.0) -> None:
    """Hold laptop lid at a fixed angle (mirror training env)."""
    import math

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


def _execute_init(env: DamageableEnvironment, robot, prims: StarterSemanticActionPrimitives) -> None:
    """Initialization motion copied from OGSimpleEnv._execute_init.

    We always run the deterministic branch from the training script.
    """
    import random

    def _exec(delta, max_steps=50):
        current_eef_pos = robot.get_eef_position("right")
        current_eef_orn = robot.get_eef_orientation("right")
        target_eef_pos = current_eef_pos + delta
        target_eef_pose = (target_eef_pos, current_eef_orn)
        steps = 0
        for a in prims._move_hand_linearly_cartesian(target_eef_pose, ignore_failure=True):
            env.step(action=a)
            steps += 1
            if steps >= max_steps:
                break

    # Use the same deterministic branch as in training (lines 545-551)
    x_max_delta = 0.0
    y_max_delta = -0.05
    z_max_delta = 0.25

    x_delta = x_max_delta
    y_delta = y_max_delta
    z_delta = z_max_delta
    _exec(torch.tensor([0.0, 0.0, z_delta]), max_steps=100)
    _exec(torch.tensor([0.0, y_delta, 0.0]), max_steps=100)
    _exec(torch.tensor([x_delta, 0.0, 0.0]), max_steps=100)


def make_base_env(scene_file: str = DEFAULT_SCENE_FILE, max_episode_steps: int = 300):
    """Create the underlying DamageableEnvironment + robot + primitives.

    This mirrors OGSimpleEnv.__init__ and simple_task_load.py setup.
    """
    # Match simple_task_load / training macros
    try:
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_OBJECT_STATES = True
        gm.ENABLE_FLATCACHE = True
        gm.ENABLE_HQ_RENDERING = False
    except Exception:
        pass

    # Fresh simulator
    if og.sim is None:
        minimal_cfg = {
            "env": {"action_timestep": 1.0 / 60.0, "physics_timestep": 1.0 / 120.0},
            "scene": {"type": "Scene"},
            "robots": [],
        }
        tmp_env = DamageableEnvironment(configs=minimal_cfg)
        tmp_env.reset()
        og.clear()
    else:
        og.sim.stop()
        og.clear()

    cfg = {"scene": {"type": "Scene", "scene_file": scene_file}}
    # We do not attach a reward_fn here; we compute terms manually.
    env = DamageableEnvironment(configs=cfg)
    env.reset()

    assert len(env.robots) > 0
    robot = env.robots[0]

    # Controller config: mirror training env
    controller_config = {
        "arm_left": {"name": "InverseKinematicsController", "mode": "pose_delta_ori"},
        "arm_right": {"name": "InverseKinematicsController", "mode": "pose_delta_ori"},
        "gripper_left": {
            "name": "MultiFingerGripperController",
            "motor_type": "position",
            "isaac_kp": 20000.0,
            "isaac_kd": 1000.0,
            "inverted": True,
        },
        "gripper_right": {
            "name": "MultiFingerGripperController",
            "motor_type": "position",
            "isaac_kp": 20000.0,
            "isaac_kd": 1000.0,
            "inverted": True,
        },
    }
    robot.reload_controllers(controller_config=controller_config)
    env.scene.update_initial_file()

    prims = StarterSemanticActionPrimitives(env=env, robot=robot, skip_curobo_initilization=True)
    prims.arm = "right"

    # Arm index metadata (to map full actions -> agent actions)
    action_dim = int(robot.action_dim)
    right_arm = "right" if (hasattr(robot, "arm_names") and ("right" in robot.arm_names)) else robot.default_arm
    right_arm_idx = robot.arm_action_idx[right_arm].cpu().numpy().astype(np.int64)
    right_grip_idx = robot.gripper_action_idx[right_arm].cpu().numpy().astype(np.int64)
    arm_input_dim = int(right_arm_idx.shape[0])
    arm_action_scale = 0.05

    # Camera pose (training script)
    try:
        og.sim.viewer_camera.set_position_orientation(
            position=[-1.3225984573364258, 0.2645236849784851, 0.9981386065483093],
            orientation=[0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452],
        )
    except Exception:
        pass

    return {
        "env": env,
        "robot": robot,
        "prims": prims,
        "action_dim": action_dim,
        "right_arm_idx": right_arm_idx,
        "right_grip_idx": right_grip_idx,
        "arm_input_dim": arm_input_dim,
        "arm_action_scale": arm_action_scale,
        "max_episode_steps": max_episode_steps,
        "episode_step": 0,
    }


def reset_episode(state: dict) -> np.ndarray:
    """Reset environment for a new teleoperated episode.

    Follows the training env's reset structure plus execute_init.
    Returns the initial low-dimensional observation used by PPO (shape (14,)).
    """
    env: DamageableEnvironment = state["env"]
    robot = state["robot"]
    prims: StarterSemanticActionPrimitives = state["prims"]

    env.reset()
    state["episode_step"] = 0

    # Let physics settle with zero actions while locking health changes.
    # NOTE: Match PPO training script exactly: do NOT run execute_init here,
    # since it is commented out there.
    env.lock_health_changes()
    for _ in range(20):
        zero = np.zeros((state["action_dim"],), dtype=np.float32)
        _set_laptop_pose(env)
        env.step(action=zero)

    # Fill mug with water and generate particles (mirrors PPO reset)
    mug_obj = env.scene.object_registry("name", "mug")
    if mug_obj is not None:
        water_system = env.scene.get_system("water", force_init=True)
        if Filled in mug_obj.states:
            mug_obj.states[Filled].set_value(water_system, True)
        mug_pos, _ = mug_obj.get_position_orientation()
        z_offset = 0.05
        for _ in range(230):
            if isinstance(mug_pos, torch.Tensor):
                drop_pos = (mug_pos + torch.tensor([0.0, 0.0, z_offset], dtype=torch.float32)).tolist()
            else:
                drop_pos = [mug_pos[0], mug_pos[1], mug_pos[2] + z_offset]
            water_system.generate_particles(positions=[drop_pos])
            for _ in range(1):
                og.sim.step()

    env.unlock_health_changes()

    for _ in range(10):
        og.sim.step()
        _set_laptop_pose(env)

    # Initial PPO-style observation
    obs_vec = _extract_obs(robot, env)
    return obs_vec


def compute_agent_action_from_full(state: dict, full_action: np.ndarray) -> np.ndarray:
    """Project a full robot action into the PPO agent action space.

    The training env maps agent_action -> full_action via:
        full[right_arm_idx] = agent_action * arm_action_scale
    We invert this mapping here so the dataset actions match the PPO training action space.
    """
    right_arm_idx = state["right_arm_idx"]
    arm_action_scale = float(state["arm_action_scale"])
    arm_input_dim = int(state["arm_input_dim"])

    # Guard against malformed actions
    a = np.asarray(full_action, dtype=np.float32).copy()
    if a.shape[0] < right_arm_idx.max() + 1:
        raise ValueError(f"Full action dim {a.shape[0]} is smaller than expected index {right_arm_idx.max()}")

    arm_full = a[right_arm_idx].astype(np.float32)
    if arm_action_scale > 0.0:
        agent_action = arm_full / arm_action_scale
    else:
        agent_action = arm_full
    # Clip to [-1, 1] to match PPO policy bounds
    agent_action = np.clip(agent_action, -1.0, 1.0)
    # Ensure correct dimensionality
    if agent_action.shape[0] != arm_input_dim:
        agent_action = agent_action[:arm_input_dim]
    return agent_action.astype(np.float32)


def env_step_with_full_action(
    state: dict,
    full_action: np.ndarray,
):
    """Step the underlying OG env using the same extra logic as OGSimpleEnv.step,
    but driven directly by a low-level action (from primitives).

    This mirrors the training env behavior:
      - keep laptop pose before and after step
      - occasionally spawn water above the mug
      - track episode steps and truncate at max_episode_steps
    """
    env: DamageableEnvironment = state["env"]
    action_dim = state["action_dim"]

    a = np.asarray(full_action, dtype=np.float32)
    if a.shape[0] != action_dim:
        # Best-effort clip / pad if dimensions mismatch
        if a.shape[0] > action_dim:
            a = a[:action_dim]
        else:
            tmp = np.zeros((action_dim,), dtype=np.float32)
            tmp[: a.shape[0]] = a
            a = tmp

    # Keep laptop at target angle every env.step (mirrors training behavior)
    _set_laptop_pose(env)
    obs_raw, reward, terminated, truncated, info = env.step(action=a)
    _set_laptop_pose(env)

    # Match training env's random water particle generation above the mug
    import random

    r = random.random()
    mug_obj = env.scene.object_registry("name", "mug")
    if mug_obj is not None:
        water_system = env.scene.get_system("water", force_init=True)
        mug_pos, _ = mug_obj.get_position_orientation()
        z_offset = 0.05
        if isinstance(mug_pos, torch.Tensor):
            drop_pos = (mug_pos + torch.tensor([0.0, 0.0, z_offset], dtype=torch.float32)).tolist()
        else:
            drop_pos = [mug_pos[0], mug_pos[1], mug_pos[2] + z_offset]
        water_system.generate_particles(positions=[drop_pos])
        og.sim.step()
    _set_laptop_pose(env)

    # Track episode steps and truncate at max_episode_steps, as in training env
    state["episode_step"] = int(state.get("episode_step", 0)) + 1
    if state["episode_step"] >= state["max_episode_steps"]:
        truncated = True

    return obs_raw, reward, bool(terminated), bool(truncated), info


def collect_one_demo(
    state: dict,
    max_episode_steps: int,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Run one scripted primitive-based episode and return trajectory arrays.

    We use StarterSemanticActionPrimitives to generate actions, inject Gaussian noise
    into each low-level action, and step the environment. There are two modes:
      - 'safe':  three segments  (right in -x, up in -y, down in -z)
      - 'unsafe': two segments (diagonal in (-x,-y) towards target, then down in -z)

    Termination is ONLY when distance_terminated is True or max_episode_steps is reached.

    Returns:
        observations: (T+1, obs_dim)
        actions: (T, act_dim)
        distance_rewards: (T,)
        distance_terminated: (T,) bool
        damage_rewards: (T,)
        damage_terminated: (T,) bool
        total_damage: scalar total damage for episode (final-health based, like eval)
        mean_damage_per_step: scalar total_damage / T
    """
    assert mode in ("safe", "unsafe"), f"Unknown mode '{mode}', expected 'safe' or 'unsafe'"

    env: DamageableEnvironment = state["env"]
    robot = state["robot"]
    prims: StarterSemanticActionPrimitives = state["prims"]

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    dist_r_list: List[float] = []
    dist_term_list: List[bool] = []
    dmg_r_list: List[float] = []
    dmg_term_list: List[bool] = []

    # Reset environment and get initial observation
    obs_vec = reset_episode(state)
    obs_list.append(obs_vec)

    # Compute deltas based on current EEF position and target position
    current_pos = robot.get_eef_position("right")
    current_pos_t = torch.as_tensor(current_pos, dtype=torch.float32)
    target_pos_t = torch.as_tensor(TARGET_EEF_POS, dtype=torch.float32)
    delta_total = target_pos_t - current_pos_t
    x_delta = float(delta_total[0].item())
    y_delta = float(delta_total[1].item())

    # Define primitive segments depending on mode
    segments = []
    if mode == "safe" and False:
        # 1) Move to the right (-x direction): adjust only x towards target x
        segments.append(torch.tensor([-0.3, 0.0, 0.0], dtype=torch.float32))
        # 2) Move up (-y direction): adjust only y towards target y
        segments.append(torch.tensor([0.0, -0.25, 0.0], dtype=torch.float32))
        # 3) Move down (-z direction): fixed downward motion
        segments.append(torch.tensor([0.0, 0.0, -0.3], dtype=torch.float32))
    else:
        # 'unsafe'
        # 1) Move diagonally towards target in (-x,-y) (adjust x and y simultaneously)
        segments.append(torch.tensor([-0.25, -0.3, 0.0], dtype=torch.float32))
        # 2) Move down (-z direction): fixed downward motion
        segments.append(torch.tensor([0.0, 0.0, -0.3], dtype=torch.float32))

    step_idx = 0

    # Track health to compute total damage like in ppo_eval
    last_mug_health = 100.0
    last_laptop_health = 100.0
    last_plate_health = 100.0

    for delta in segments:
        # Set up target pose for this segment
        current_eef_pos = robot.get_eef_position("right")
        current_eef_orn = robot.get_eef_orientation("right")
        target_eef_pos = current_eef_pos + delta
        target_eef_pose = (target_eef_pos, current_eef_orn)

        # Execute primitive with low-level actions, injecting Gaussian noise
        for full_action in prims._move_hand_linearly_cartesian(
            target_eef_pose,
            ignore_failure=True,
        ):
            full_action = np.asarray(full_action, dtype=np.float32)
            # Add small Gaussian noise only to the right arm EEF position + orientation
            # (same indices that the PPO agent controls).
            full_action_noisy = full_action.copy()
            right_arm_idx = state["right_arm_idx"]
            noise = np.random.normal(
                loc=0.0,
                scale=ACTION_NOISE_STD,
                size=right_arm_idx.shape[0],
            ).astype(np.float32)
            full_action_noisy[right_arm_idx] += noise

            # Step env with noisy full action, using the same extras as training env.step
            obs_raw, _reward_env, _terminated_env, _truncated_env, _info_env = env_step_with_full_action(
                state, full_action_noisy
            )

            # Compute decomposed reward terms using training definitions
            damage_reward, damage_terminated = _damage_reward_fn(env, obs_raw)
            distance_reward, distance_terminated = _distance_reward_fn(env, obs_raw)

            # Extract PPO-style observation and project full action into PPO agent space
            obs_vec = _extract_obs(robot, env)
            agent_action = compute_agent_action_from_full(state, full_action_noisy)

            obs_list.append(obs_vec)
            act_list.append(agent_action)
            dist_r_list.append(float(distance_reward))
            dist_term_list.append(bool(distance_terminated))
            dmg_r_list.append(float(damage_reward))
            dmg_term_list.append(bool(damage_terminated))

            # Track health for damage summary (like ppo_eval)
            try:
                health_states = obs_raw.get("object_health_states", {})
                phs_mug = health_states.get("mug", None)
                if phs_mug is not None and ("health" in phs_mug):
                    last_mug_health = float(phs_mug["health"])
                phs_laptop = health_states.get("laptop", None)
                if phs_laptop is not None and ("health" in phs_laptop):
                    last_laptop_health = float(phs_laptop["health"])
                phs_plate = health_states.get("plate", None)
                if phs_plate is not None and ("health" in phs_plate):
                    last_plate_health = float(phs_plate["health"])
            except Exception:
                pass

            step_idx += 1

            # Dataset termination condition: ONLY distance_terminated or max steps
            if distance_terminated or (step_idx >= max_episode_steps):
                break

        if distance_terminated or (step_idx >= max_episode_steps):
            break

    observations = np.stack(obs_list, axis=0).astype(np.float32)  # (T+1, obs_dim)
    actions = np.stack(act_list, axis=0).astype(np.float32)       # (T, act_dim)
    distance_rewards = np.asarray(dist_r_list, dtype=np.float32)
    distance_terminated_arr = np.asarray(dist_term_list, dtype=bool)
    damage_rewards = np.asarray(dmg_r_list, dtype=np.float32)
    damage_terminated_arr = np.asarray(dmg_term_list, dtype=bool)

    # Episode-level damage summary mimicking ppo_eval logic
    try:
        total_damage = (100.0 - float(last_mug_health)) + (100.0 - float(last_laptop_health)) + (
            100.0 - float(last_plate_health)
        )
    except Exception:
        total_damage = 0.0
    total_damage = max(0.0, min(300.0, float(total_damage)))
    num_steps = float(max(1, actions.shape[0]))
    mean_damage_per_step = total_damage / num_steps

    return (
        observations,
        actions,
        distance_rewards,
        distance_terminated_arr,
        damage_rewards,
        damage_terminated_arr,
        total_damage,
        mean_damage_per_step,
    )


def get_next_demo_name(h5file: h5py.File) -> str:
    existing = [k for k in h5file.keys() if k.startswith("demo_")]
    if not existing:
        return "demo_000"
    # demo_000 -> 0
    indices = []
    for name in existing:
        try:
            idx = int(name.split("_")[-1])
            indices.append(idx)
        except Exception:
            continue
    next_idx = 0 if not indices else (max(indices) + 1)
    return f"demo_{next_idx:03d}"


def save_demo_to_hdf5(
    h5file: h5py.File,
    demo_name: str,
    observations: np.ndarray,
    actions: np.ndarray,
    distance_rewards: np.ndarray,
    distance_terminated: np.ndarray,
    damage_rewards: np.ndarray,
    damage_terminated: np.ndarray,
    total_damage: float,
    mean_damage_per_step: float,
) -> None:
    if demo_name in h5file:
        raise ValueError(f"Group '{demo_name}' already exists in HDF5 file; refusing to overwrite.")

    g = h5file.create_group(demo_name)
    # Core trajectories
    g.create_dataset("observations", data=observations, compression="gzip")
    g.create_dataset("actions", data=actions, compression="gzip")
    g.create_dataset("distance_reward_terms", data=distance_rewards, compression="gzip")
    g.create_dataset("distance_termination_terms", data=distance_terminated.astype(np.uint8), compression="gzip")
    g.create_dataset("damage_reward_terms", data=damage_rewards, compression="gzip")
    g.create_dataset("damage_termination_terms", data=damage_terminated.astype(np.uint8), compression="gzip")

    # Convenience metadata
    g.attrs["num_steps"] = int(actions.shape[0])
    g.attrs["obs_dim"] = int(observations.shape[1])
    g.attrs["act_dim"] = int(actions.shape[1])
    g.attrs["total_damage"] = float(total_damage)
    g.attrs["mean_damage_per_step"] = float(mean_damage_per_step)


def main():
    # Ensure parent directory for HDF5 exists
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_HDF5)), exist_ok=True)

    # Build base environment used for primitive execution and reward computation
    state = make_base_env(scene_file=SCENE_FILE, max_episode_steps=MAX_EPISODE_STEPS)

    # Open (or create) HDF5 file for appending
    with h5py.File(OUTPUT_HDF5, "a") as h5f:
        print(f"Writing demonstrations to: {os.path.abspath(OUTPUT_HDF5)}")
        print("Each demo will be stored as a group 'demo_XXX' with observations/actions/reward terms.")
        print("We will generate scripted primitive-based demos, alternating safe and unsafe.\n")

        num_safe = 0
        num_unsafe = 0

        for demo_idx in range(NUM_DEMOS):
            mode = "safe" if (demo_idx % 2 == 0) else "unsafe"
            print(f"=== Collecting demo {demo_idx + 1}/{NUM_DEMOS} (mode={mode}) ===")
            (
                observations,
                actions,
                distance_rewards,
                distance_terminated,
                damage_rewards,
                damage_terminated,
                total_damage,
                mean_damage_per_step,
            ) = collect_one_demo(state, max_episode_steps=MAX_EPISODE_STEPS, mode=mode)

            # Summarize accumulated rewards / terminations for quick feedback
            total_dist_r = float(np.sum(distance_rewards)) if len(distance_rewards) > 0 else 0.0
            total_dmg_r = float(np.sum(damage_rewards)) if len(damage_rewards) > 0 else 0.0
            final_dist_term = bool(distance_terminated[-1]) if len(distance_terminated) > 0 else False
            final_dmg_term = bool(damage_terminated[-1]) if len(damage_terminated) > 0 else False
            print(
                f"Demo finished: {actions.shape[0]} steps, "
                f"distance_terminated={final_dist_term}, "
                f"total_distance_reward={total_dist_r:.3f}, "
                f"total_damage_reward={total_dmg_r:.3f}, "
                f"damage_terminated={final_dmg_term}, "
                f"total_damage={total_damage:.3f}, "
                f"mean_damage_per_step={mean_damage_per_step:.5f}."
            )

            # Episode classification by actual damage (regardless of scripted mode)
            if total_damage == 0.0:
                num_safe += 1
            else:
                num_unsafe += 1

            demo_name = get_next_demo_name(h5f)
            save_demo_to_hdf5(
                h5f,
                demo_name,
                observations,
                actions,
                distance_rewards,
                distance_terminated,
                damage_rewards,
                damage_terminated,
                total_damage,
                mean_damage_per_step,
            )
            print(f"Saved demo as group '{demo_name}' in {os.path.abspath(OUTPUT_HDF5)}\n")

        # Summary of realized safe vs unsafe trajectories (based on damage)
        print("=== Dataset safety summary ===")
        print(f"Safe demos   (0 damage): {num_safe}")
        print(f"Unsafe demos (>0 damage): {num_unsafe}")
        print("==============================")
        breakpoint()

    # Clean up simulator on exit
    try:
        og.clear()
        og.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
