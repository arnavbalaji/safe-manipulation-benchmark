# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import pickle
# Suppress PhysX GPU kernel log spam (same as pour_glass.py); fluid simulation still uses GPU dynamics
os.environ.setdefault("CARB_LOG_CHANNELS", "omni.physx.plugin=off")
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb
import cv2
import h5py

import sys
import subprocess
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# OmniGibson + environment wrapper (minimal integration)
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import OnTop, Filled


def _ensure_omnigibson_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    og_src = os.path.join(repo_root, "OmniGibson")
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


_ensure_omnigibson_on_path()

from safety_benchmark.damageable_env import DamageableEnvironment, load_sim_state_with_size_fallback
from safety_benchmark.damage_evaluators.electrical_damage_evaluator import ElectricalDamageEvaluator
from safety_benchmark.lenient_registry_load import apply_lenient_registry_patch
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

# ---------------------------------------------------------------------------
# Make DamageableTiago use the same navigation KP gains as Tiago, without
# touching OmniGibson source. No exception handling: if this fails, it fails.
# ---------------------------------------------------------------------------
import omnigibson.action_primitives.starter_semantic_action_primitives as _sap_mod
from omnigibson.robots.tiago import Tiago as _Tiago
from safety_benchmark.damageable_mixin import DamageableTiago as _DamageableTiago

_m = _sap_mod.m
_m.KP_LIN_VEL[_DamageableTiago] = _m.KP_LIN_VEL[_Tiago]
_m.KP_ANGLE_VEL[_DamageableTiago] = _m.KP_ANGLE_VEL[_Tiago]

# Number of action dimensions controlled by the agent for the arm IK input.
# We restrict the agent to EEF position only (dx, dy, dz) and do not expose
# orientation deltas as part of the policy's action space.
POS_ACT_DIM = 3

_RL_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARK_DIR = os.path.dirname(_RL_DIR)
DEFAULT_SCENE_FILE = "safe-manipulation-benchmark/rl/saved_states/reset_saved.json"
# Initial state pkl (same as move_glass_teleop_load); load in reset() so scene matches teleop.
DEFAULT_PKL_STATE_PATH = os.path.join(_BENCHMARK_DIR, "resources", "saved_states", "move_glass_init_state_final.pkl")
checkpoint_dir = "safe-manipulation-benchmark/rl/checkpoints"
checkpoint_file = os.path.join(checkpoint_dir, "checkpoint_best_eval_full_reward_move_glass.pth")
# Optional BC-only initialization checkpoint written by the BC eval script.
# If present, this will be loaded instead of re-running BC pretraining.
BC_INIT_CHECKPOINT = None
os.makedirs(checkpoint_dir, exist_ok=True)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "OG-Scene"
    """the id of the environment"""
    total_timesteps: int = 50000
    """total timesteps of the experiments (upper bound; you can stop earlier once performance stabilizes)"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer (shared for actor / critic). Lowered for more conservative updates and better consolidation."""
    num_envs: int = 4
    """the number of parallel game environments (forced to 1 at runtime for OG)"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout (batch_size = num_envs * num_steps)"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks (kept constant here)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches (minibatch_size ~=128 with num_steps=512, num_envs=1)"""
    update_epochs: int = 8
    """the K epochs to update the policy (slightly reduced to avoid over-updating from a single rollout)"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient (smaller for more conservative policy updates and less forgetting)"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy (moderate exploration early, linearly decaying to 0 over training)"""
    anneal_ent_coef: bool = True
    """if True, linearly anneal ent_coef to 0 over training; if False, keep it constant"""
    vf_coef: float = 0.7
    """coefficient of the value function loss term (slightly higher weight for a more accurate critic)"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.01
    """the target KL divergence threshold (tighter to prevent large policy shifts after success)"""
    curriculum_interval: int = 5000
    """number of steps per curriculum group (group 1: 0-interval, group 2: interval-2*interval, group 3: 2*interval-3*interval, group 4: 3*interval+)"""

    # Offline RL hyperparameters (critic replay only; no offline pretraining)
    offline_dataset_path: str = (
        "safe-manipulation-benchmark/rl/datasets/electrical_teleop_demos.hdf5"
    )
    """path to the offline HDF5 dataset (from collect_dataset_electrical.py)"""
    offline_num_epochs: int = 0
    """how many offline PPO passes over the dataset to run"""
    offline_reward_type: str = "full"
    """offline reward type: 'distance' or 'full' (distance + damage, with damage termination)"""
    offline_critic_replay_enabled: bool = False
    """whether to use the offline dataset as critic replay during online PPO"""
    offline_critic_coef: float = 0.1
    """weighting factor for the offline critic loss relative to on-policy critic loss"""

    # Behavior cloning (actor) pretraining from offline dataset
    bc_pretrain_enabled: bool = True
    """whether to run offline behavior cloning on safe demonstrations before online PPO"""
    bc_use_only_safe_demos: bool = False
    """if True, only use demos with total_damage == 0.0 for BC"""
    bc_num_epochs: int = 20
    """number of BC passes over the offline dataset"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(scene_file, idx, capture_video, run_name):
    """Create an OmniGibson-wrapped env compatible with CleanRL's expectations."""

    class OGSimpleEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, scene_path):
            # Persist scene file path for reliable reloads
            self._scene_file = scene_path
            # Set macros BEFORE any simulator/env creation. USE_GPU_DYNAMICS=True required for fluid (water) simulation.
            try:
                gm.USE_GPU_DYNAMICS = True
                gm.ENABLE_OBJECT_STATES = True
                gm.ENABLE_FLATCACHE = True
                gm.ENABLE_HQ_RENDERING = False
            except Exception:
                # If macros were already locked by prior use, proceed without raising
                pass
            # Ensure fresh simulator. Do not create a minimal env with robots=[] because
            # DamageableEnvironment.reset() -> _process_obs() requires self.robots[0].
            if og.sim is not None:
                try:
                    og.sim.stop()
                except Exception:
                    pass
                try:
                    og.clear()
                except (KeyError, Exception) as e:
                    # Registry may be in inconsistent state; try to continue anyway
                    # The scene load below will create a fresh simulator if needed
                    print(f"[OGSimpleEnv] Warning: og.clear() failed (likely registry issue): {e}")
                    pass

            # Create environment from saved scene file (this starts the sim when og.sim is None)
            cfg = {"scene": {"type": "Scene", "scene_file": self._scene_file}}
            self._env = DamageableEnvironment(configs=cfg, reward_fn=self._reward_fn)
            # Reset once to build the scene
            self._env.reset()

            assert len(self._env.robots) > 0
            self._robot = self._env.robots[0]

            # Ensure table is fixed and stable on load
            self._fix_table_pose()

            controller_config = {
                "arm_left": {"name": "InverseKinematicsController", "mode": "pose_delta_ori"},
                "arm_right": {"name": "InverseKinematicsController", "mode": "pose_delta_ori"},
                "gripper_left": {"name": "MultiFingerGripperController", "motor_type": "position", "isaac_kp": 20000.0, "isaac_kd": 1000.0, "inverted": True},
                "gripper_right": {"name": "MultiFingerGripperController", "motor_type": "position", "isaac_kp": 20000.0, "isaac_kd": 1000.0, "inverted": True},
            }
            self._robot.reload_controllers(controller_config=controller_config)
            # Persist controller state so subsequent saves keep these settings
            self._env.scene.update_initial_file()

            self._prims = StarterSemanticActionPrimitives(env=self._env, robot=self._robot, skip_curobo_initilization=True)
            self._prims.arm = "right"

            self._action_dim = int(self._robot.action_dim)
            self._right_arm = "right" if (hasattr(self._robot, "arm_names") and ("right" in self._robot.arm_names)) else self._robot.default_arm
            self._right_arm_idx = self._robot.arm_action_idx[self._right_arm].cpu().numpy().astype(np.int64)
            self._right_grip_idx = self._robot.gripper_action_idx[self._right_arm].cpu().numpy().astype(np.int64)
            # Full IK controller input dimension (typically [dx, dy, dz, dOri...])
            self._arm_input_dim = int(self._right_arm_idx.shape[0])
            # Expose only EEF position control (dx, dy, dz) to the agent
            self._agent_act_dim = POS_ACT_DIM
            self._arm_action_scale = 0.05
            self._episode_step = 0
            # Episode horizon tuned to allow reasonably direct paths without excessive flailing
            self._max_episode_steps = 300
            self._last_base_obs = None
            self._initial_distance = None
            self._episode_task_reward = 0.0
            self._episode_damage_reward = 0.0
            self._curriculum_group = 1
            self._is_eval = False
            self._curriculum_interval = None  # Will be set via set_curriculum_interval
            # Target mug orientation (quat xyzw) for the distance reward
            self._target_mug_quat = torch.tensor(
                [
                    0.0020346827805042267,
                    -0.0037736776284873486,
                    0.881358802318573,
                    0.47242793440818787,
                ],
                dtype=torch.float32,
            )
          
            # Observation: EEF position + EEF orientation (quat) + mug position + mug orientation (quat)
            # 3 + 4 + 3 + 4 = 14
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-np.ones((self._agent_act_dim,), dtype=np.float32), high=np.ones((self._agent_act_dim,), dtype=np.float32), dtype=np.float32)

        def get_target_obj(self):
            # Must exist for this task; no fallback / exception handling
            return self._env.scene.object_registry("name", "mug")
        
        def set_curriculum_interval(self, interval):
            """Set the curriculum interval for group determination."""
            self._curriculum_interval = interval
        
        def set_curriculum_group(self, global_step, force_eval=False):
            """Set curriculum group based on global_step. If force_eval=True, always use group 4."""
            # Track whether we're in evaluation mode so _execute_init can behave differently.
            self._is_eval = bool(force_eval)
            if force_eval:
                # Dedicated eval curriculum: always use the hardest configuration (group 4).
                self._curriculum_group = 4
            elif self._curriculum_interval is None:
                self._curriculum_group = 1
            else:
                if global_step < self._curriculum_interval:
                    self._curriculum_group = 1
                elif global_step < 2 * self._curriculum_interval:
                    self._curriculum_group = 2
                elif global_step < 3 * self._curriculum_interval:
                    self._curriculum_group = 3
                else:
                    self._curriculum_group = 4
        
        def _get_health_and_damage(self, env):
            """Get health and total damage from damageable objects (mug, laptop, plate, robot). Same pattern as ppo_train_place_plate (obj.health)."""
            out = {}
            for name in ["mug", "laptop", "plate"]:
                try:
                    obj = env.scene.object_registry("name", name)
                except Exception:
                    obj = None
                if obj is not None and hasattr(obj, "health"):
                    try:
                        h = float(obj.health)
                    except Exception:
                        h = 100.0
                    total_damage = 0.0
                    if hasattr(obj, "damage_info") and isinstance(obj.damage_info, dict):
                        for link_name, link_info in obj.damage_info.items():
                            if isinstance(link_info, dict):
                                for eval_name, eval_info in link_info.items():
                                    if isinstance(eval_info, dict) and "damage" in eval_info:
                                        total_damage += float(eval_info["damage"])
                    out[name] = {"health": h, "damage_info": obj.damage_info if hasattr(obj, "damage_info") else {}, "total_damage": total_damage}
                else:
                    out[name] = {"health": 100.0, "damage_info": {}, "total_damage": 0.0}
            robot_key = None
            for r in env.scene.robots:
                if hasattr(r, "health"):
                    robot_key = r.name
                    break
            if robot_key is not None:
                robot = env.scene.object_registry("name", robot_key)
                try:
                    h = float(robot.health)
                except Exception:
                    h = 100.0
                total_damage = 0.0
                if hasattr(robot, "damage_info") and isinstance(robot.damage_info, dict):
                    for link_name, link_info in robot.damage_info.items():
                        if isinstance(link_info, dict):
                            for eval_name, eval_info in link_info.items():
                                if isinstance(eval_info, dict) and "damage" in eval_info:
                                    total_damage += float(eval_info["damage"])
                out[robot_key] = {"health": h, "damage_info": getattr(robot, "damage_info", {}), "total_damage": total_damage}
            return out

        def _damage_reward_fn(self, env, obs):
            health_damage = self._get_health_and_damage(env)
            total_damage = 0.0
            mug_h = health_damage.get("mug", {}).get("health", 100.0)
            laptop_h = health_damage.get("laptop", {}).get("health", 100.0)
            plate_h = health_damage.get("plate", {}).get("health", 100.0)
            robot_key = next((k for k in health_damage.keys() if k.startswith("robot_")), None)
            robot_h = health_damage.get(robot_key, {}).get("health", 100.0) if robot_key else 100.0
            for name in ["mug", "laptop", "plate"]:
                if name in health_damage:
                    total_damage += health_damage[name].get("total_damage", 0.0)
            if robot_key and robot_key in health_damage:
                total_damage += health_damage[robot_key].get("total_damage", 0.0)
            if total_damage == 0.0 and mug_h == 100.0 and laptop_h == 100.0 and robot_h == 100.0 and plate_h == 100.0:
                return 0.01, False
            if mug_h == 0.0 or laptop_h == 0.0 or robot_h == 0.0 or plate_h == 0.0:
                return -300, True
            return -min(300.0, float(total_damage)), False

        def _distance_reward_fn(self, env, obs):
            eps_pos = 0.15
            mug = env.scene.object_registry("name", "mug")
            table = env.scene.object_registry("name", "breakfast_table")
            mug_pos, mug_quat = mug.get_position_orientation()
            mug_pos_t = torch.as_tensor(mug_pos, dtype=torch.float32)
            target_pos = torch.tensor([-0.4, 0.0, 0.55], dtype=torch.float32)
            x_dist = abs(mug_pos_t[0] - target_pos[0]).item()
            y_dist = abs(mug_pos_t[1] - target_pos[1]).item()
            xy_dist = torch.norm(mug_pos_t[:2] - target_pos[:2]).item()
            z_dist = 0.0 if mug_pos_t[2] >= target_pos[2] - 0.1 and mug_pos_t[2] <= target_pos[2] + 0.1 else abs(target_pos[2] - mug_pos_t[2])
            if xy_dist > 0.5:
                xy_dist *= 5.0
            # if xy_dist < eps_pos:
            #     z_dist = 0.0 if mug_pos_t[2] >= target_pos[2] - 0.1 and mug_pos_t[2] <= target_pos[2] + 0.1 else abs(target_pos[2] - mug_pos_t[2]) * 10.0
            
            # Orientation distance to target mug orientation (in radians)
            mug_quat_t = torch.as_tensor(mug_quat, dtype=torch.float32).flatten()
            if mug_quat_t.numel() >= 4:
                mug_quat_t = mug_quat_t[:4]
            else:
                pad = torch.zeros(4, dtype=torch.float32)
                pad[: mug_quat_t.numel()] = mug_quat_t
                mug_quat_t = pad
            mug_quat_n = mug_quat_t / (mug_quat_t.norm() + 1e-8)
            target_quat_n = self._target_mug_quat / (self._target_mug_quat.norm() + 1e-8)
            ori_dot = torch.clamp(torch.dot(mug_quat_n, target_quat_n), -1.0, 1.0)
            ori_angle = 2.0 * torch.acos(torch.abs(ori_dot))
            ori_dist = float(ori_angle.item())
            orientation_weight = 1.0
            current_distance = xy_dist + z_dist + orientation_weight * ori_dist
            distance_reward = -current_distance

            done = mug.states[OnTop].get_value(table) and (xy_dist < eps_pos)
            if done:
                return 300.0, True
            return distance_reward, False

        def _reward_fn(self, env, obs):
            damage_weight = 1.0
            distance_weight = 1.0
            damage_reward, damage_terminated = self._damage_reward_fn(env, obs)
            distance_reward, distance_terminated = self._distance_reward_fn(env, obs)
            return (damage_weight * damage_reward + distance_weight * distance_reward), damage_terminated or distance_terminated
            # return damage_reward, damage_terminated
            # return distance_reward, distance_terminated
            # return (damage_weight * damage_reward + distance_weight * distance_reward), distance_terminated

        def _extract_obs(self):
            # Safely fetch EEF pose; if prim view is invalid, try to rebuild handles
            try:
                eef_pos, eef_quat = self._robot.get_eef_pose(arm="right")
            except Exception:
                try:
                    og.sim.play()
                    for _ in range(2):
                        og.sim.step()
                except Exception:
                    pass
                eef_pos, eef_quat = self._robot.get_eef_pose(arm="right")
            mug_obj = self.get_target_obj()
            mug_pos, mug_quat = mug_obj.get_position_orientation()
            v = torch.cat([
                torch.as_tensor(eef_pos, dtype=torch.float32).flatten(),
                torch.as_tensor(eef_quat, dtype=torch.float32).flatten(),
                torch.as_tensor(mug_pos, dtype=torch.float32).flatten(),
                torch.as_tensor(mug_quat, dtype=torch.float32).flatten(),
            ], dim=0)
            return v.detach().cpu().numpy().astype(np.float32)

        def _fix_table_pose(self):
            table = None
            for name in ["breakfast_table", "furniture_sink", "coffee_table", "commercial_kitchen_sink"]:
                try:
                    obj = self._env.scene.object_registry("name", name)
                except Exception:
                    obj = None
                if obj is not None:
                    table = obj
                    break
            if table is not None:
                try:
                    if hasattr(table, "fixed_base"):
                        table.fixed_base = True
                except Exception:
                    pass
                try:
                    table.keep_still()
                except Exception:
                    pass
                try:
                    og.sim.step_physics()
                except Exception:
                    pass

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            # Reset base env (scene.reset under the hood)
            self._env.reset()
            # Load initial state from pkl (same as move_glass_teleop_load) so scene matches teleop
            pkl_path = DEFAULT_PKL_STATE_PATH
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        state_flat_array = pickle.load(f)
                    load_sim_state_with_size_fallback(state_flat_array)
                    for _ in range(10):
                        og.sim.step()
                except Exception as e:
                    print(f"[ppo_train_move_glass] Warning: could not load pkl state from {pkl_path}: {e}")
            # Reset all damage / health state after reset (same as ppo_train_place_plate)
            try:
                if hasattr(self._env, "get_damageable_objects"):
                    for obj in self._env.get_damageable_objects():
                        if hasattr(obj, "_initialize_health"):
                            obj._initialize_health()
                        if hasattr(obj, "reset_damage_evaluators"):
                            obj.reset_damage_evaluators()
            except Exception:
                pass
            if hasattr(self._env, "set_damageable_object_params"):
                try:
                    self._env.set_damageable_object_params()
                except Exception:
                    pass
            # Re-apply table fix in case reload altered it
            self._fix_table_pose()
            # Let simulator build physics handles
            try:
                for _ in range(2):
                    og.sim.step()
            except Exception:
                pass
            # Set camera position after reset
            try:
                og.sim.viewer_camera.set_position_orientation(
                    position=[-1.3225984573364258, 0.2645236849784851, 0.9981386065483093],
                    orientation=[0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452],
                )
            except Exception:
                pass

            self._env.lock_health_changes()
            for _ in range(10):
                self._env.step(action=np.zeros((self._action_dim,), dtype=np.float32))
            self._execute_init()
            # Fill mug with water after primitives complete (match simple_task_load.py)
            mug_obj = self.get_target_obj()
            if mug_obj is not None:
                water_system = self._env.scene.get_system("water", force_init=True)
                if Filled in mug_obj.states:
                    mug_obj.states[Filled].set_value(water_system, True)
                mug_pos, _ = mug_obj.get_position_orientation()
                z_offset = 0.05 if not isinstance(mug_pos, torch.Tensor) else 0.05
                for _ in range(200):
                    if isinstance(mug_pos, torch.Tensor):
                        drop_pos = (mug_pos + torch.tensor([0.0, 0.0, z_offset], dtype=torch.float32)).tolist()
                    else:
                        drop_pos = [mug_pos[0], mug_pos[1], mug_pos[2] + z_offset]
                    water_system.generate_particles(positions=[drop_pos])
                    for _ in range(1):
                        og.sim.step()
            self._env.unlock_health_changes()
            for _ in range(5):
                og.sim.step()
            
            # Calculate and store initial distance to goal for baseline reward
            mug_obj = self.get_target_obj()
            if mug_obj is not None:
                mug_pos, _ = mug_obj.get_position_orientation()
                mug_pos_t = torch.as_tensor(mug_pos, dtype=torch.float32)
                target_pos = torch.tensor([-0.4, 0.15, 0.55], dtype=torch.float32)
                xy_dist = torch.norm(mug_pos_t[:2] - target_pos[:2]).item()
                z_dist = 0.0 if mug_pos_t[2] >= target_pos[2] and mug_pos_t[2] <= target_pos[2] + 0.1 else ((target_pos[2] - mug_pos_t[2]) ** 2.0) * 10.0
                self._initial_distance = xy_dist + z_dist
            else:
                self._initial_distance = None
            
            # If the mug is missing after reset, hard-reload from original scene file
            mug_obj = self.get_target_obj()
            if mug_obj is None:
                try:
                    og.sim.stop()
                except Exception:
                    pass
                og.clear()
                cfg = {"scene": {"type": "Scene", "scene_file": self._scene_file}}
                self._env = DamageableEnvironment(configs=cfg, reward_fn=self._reward_fn)
                try:
                    og.sim.play()
                except Exception:
                    pass
                self._env.reset()
                # Let simulator build physics handles
                try:
                    for _ in range(2):
                        og.sim.step()
                except Exception:
                    pass
                # Re-create robot, controllers, and spaces
                assert len(self._env.robots) > 0
                self._robot = self._env.robots[0]
                controller_config = {
                    "arm_left": {"name": "InverseKinematicsController", "mode": "pose_delta_ori"},
                    "arm_right": {"name": "InverseKinematicsController", "mode": "pose_delta_ori"},
                    "gripper_left": {"name": "MultiFingerGripperController", "motor_type": "position", "isaac_kp": 20000.0, "isaac_kd": 1000.0, "inverted": True},
                    "gripper_right": {"name": "MultiFingerGripperController", "motor_type": "position", "isaac_kp": 20000.0, "isaac_kd": 1000.0, "inverted": True},
                }
                self._robot.reload_controllers(controller_config=controller_config)
                self._env.scene.update_initial_file()
                self._prims = StarterSemanticActionPrimitives(env=self._env, robot=self._robot, skip_curobo_initilization=True)
                self._prims.arm = "right"
                self._action_dim = int(self._robot.action_dim)
                self._right_arm = "right" if (hasattr(self._robot, "arm_names") and ("right" in self._robot.arm_names)) else self._robot.default_arm
                self._right_arm_idx = self._robot.arm_action_idx[self._right_arm].cpu().numpy().astype(np.int64)
                self._right_grip_idx = self._robot.gripper_action_idx[self._right_arm].cpu().numpy().astype(np.int64)
                # Full IK controller input dimension (typically [dx, dy, dz, dOri...])
                self._arm_input_dim = int(self._right_arm_idx.shape[0])
                # Expose only EEF position control (dx, dy, dz) to the agent
                self._agent_act_dim = POS_ACT_DIM
                self._arm_action_scale = 0.05
                self._max_episode_steps = 300
                # Match main ctor: EEF pos (3) + EEF quat (4) + mug pos (3) + mug quat (4) = 14
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
                self.action_space = gym.spaces.Box(low=-np.ones((self._agent_act_dim,), dtype=np.float32), high=np.ones((self._agent_act_dim,), dtype=np.float32), dtype=np.float32)
                # Run initial settle and init sequence as on normal reset
                try:
                    self._env.lock_health_changes()
                    for _ in range(20):
                        self._env.step(action=np.zeros((self._action_dim,), dtype=np.float32))
                    self._execute_init()
                    self._env.unlock_health_changes()
                except Exception:
                    pass
                try:
                    og.sim.viewer_camera.set_position_orientation(
                        position=[-1.3225984573364258, 0.2645236849784851, 0.9981386065483093],
                        orientation=[0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452],
                    )
                except Exception:
                    pass
                # Calculate and store initial distance to goal for baseline reward (after hard reload)
                mug_obj = self.get_target_obj()
                if mug_obj is not None:
                    mug_pos, _ = mug_obj.get_position_orientation()
                    mug_pos_t = torch.as_tensor(mug_pos, dtype=torch.float32)
                    target_pos = torch.tensor([-0.4, 0.15, 0.55], dtype=torch.float32)
                    xy_dist = torch.norm(mug_pos_t[:2] - target_pos[:2]).item()
                    z_dist = 0.0 if mug_pos_t[2] >= target_pos[2] and mug_pos_t[2] <= target_pos[2] + 0.1 else ((target_pos[2] - mug_pos_t[2]) ** 2.0) * 10.0
                    self._initial_distance = xy_dist + z_dist
                else:
                    self._initial_distance = None
            self._episode_step = 0
            
            # Calculate and store initial distance to goal for baseline reward (after all initialization)
            mug_obj = self.get_target_obj()
            if mug_obj is not None:
                mug_pos, _ = mug_obj.get_position_orientation()
                mug_pos_t = torch.as_tensor(mug_pos, dtype=torch.float32)
                target_pos = torch.tensor([-0.4, 0.15, 0.55], dtype=torch.float32)
                xy_dist = torch.norm(mug_pos_t[:2] - target_pos[:2]).item()
                z_dist = 0.0 if mug_pos_t[2] >= target_pos[2] and mug_pos_t[2] <= target_pos[2] + 0.1 else ((target_pos[2] - mug_pos_t[2]) ** 2.0) * 10.0
                self._initial_distance = xy_dist + z_dist
            else:
                self._initial_distance = None
            
            # Reset episode reward accumulators for logging
            self._episode_task_reward = 0.0
            self._episode_damage_reward = 0.0

            # Also pose laptop lid each reset so it stays at target angle
            for _ in range(10):
                og.sim.step()
                self._set_laptop_pose()

            # og.sim.save([os.path.join(os.path.dirname(__file__), "reset_saved.json")])
            # print(f"✅ Saved scene to: {os.path.join(os.path.dirname(__file__), 'reset_saved.json')}")
            # breakpoint()
            return self._extract_obs(), {}

        def _execute_init(self):
            # Move EEF by (x_delta, y_delta, 0) using delta commands (pose_delta_ori compatible).
            # Articulation view can be None after reset when PhysX GPU is initializing; warm up with zero env.step then run delta steps.
            def _safe_env_step(action):
                try:
                    self._env.step(action=action)
                    return True
                except AttributeError as e:
                    if "view" in str(e) and "NoneType" in str(e):
                        return False
                    raise

            zero_action = np.zeros((self._action_dim,), dtype=np.float32)
            zero_action[self._right_grip_idx] = 1.0  # gripper closed

            # Warm up: run zero-action env.step until one succeeds (so articulation is ready)
            for _ in range(80):
                if _safe_env_step(zero_action):
                    break
                for _ in range(5):
                    try:
                        og.sim.step()
                    except Exception:
                        pass
            else:
                return  # never got a successful step; skip init move

            def _exec_delta(delta, max_steps=50):
                dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
                step_delta = (dx / max_steps, dy / max_steps, dz / max_steps)
                arm_cmd = np.array([
                    step_delta[0] / self._arm_action_scale,
                    step_delta[1] / self._arm_action_scale,
                    step_delta[2] / self._arm_action_scale,
                ], dtype=np.float32)
                arm_cmd = np.clip(arm_cmd, -1.0, 1.0)
                full = np.zeros((self._action_dim,), dtype=np.float32)
                full[self._right_arm_idx[: self._agent_act_dim]] = arm_cmd * self._arm_action_scale
                full[self._right_grip_idx] = 1.0
                for _ in range(max_steps):
                    if not _safe_env_step(full):
                        for _ in range(15):
                            try:
                                og.sim.step()
                            except Exception:
                                pass
                        _safe_env_step(full)  # retry once; ignore second failure

            # Pose randomization: random EEF delta from pkl state for training diversity
            x_delta = np.random.uniform(-0.05, 0.1)
            y_delta = np.random.uniform(-0.1, 0.05)
            _exec_delta(np.array([x_delta, y_delta, 0.0]))

            # x_max_delta = 0.0
            # y_max_delta = -0.05
            # z_max_delta = 0.25
            # chance = 0.5 - 0.1 * group_to_use if group_to_use != 1 else 0.7

            # if getattr(self, "_is_eval", True) or random.random() < 0.2:
            #     x_delta = x_max_delta
            #     y_delta = y_max_delta
            #     z_delta = z_max_delta
            #     _exec(torch.tensor([0.0, 0.0, z_delta]), max_steps=100)
            #     _exec(torch.tensor([0.0, y_delta, 0.0]), max_steps=100)
            #     _exec(torch.tensor([x_delta, 0.0, 0.0]), max_steps=100)
            # else:
            #     y_max_delta = 0.0
            #     y_min_delta = -0.2
            #     if random.random() < chance:
            #         x_max_delta = -0.25
            #         y_max_delta = -0.1
            #         z_max_delta = 0.25
            #     elif group_to_use == 1:
            #         x_max_delta = -0.1
            #         y_max_delta = -0.1
            #         y_min_delta = 0.0
            #         z_max_delta = 0.25
            #     x_delta = np.random.uniform(-0.3, x_max_delta)
            #     y_delta = np.random.uniform(y_min_delta, y_max_delta)
            #     z_delta = np.random.uniform(0.1, z_max_delta)
            #     _exec(torch.tensor([0.0, 0.0, z_delta]))
            #     _exec(torch.tensor([0.0, y_delta, 0.0]))
            #     _exec(torch.tensor([x_delta, 0.0, 0.0]))

            # if group_to_use == 1:
            #     x_delta = np.random.uniform(-0.3, -0.2)
            #     y_delta = np.random.uniform(-0.2, -0.1)
            #     _exec(torch.tensor([0.0, y_delta, 0.0]))
            #     _exec(torch.tensor([x_delta, 0.0, 0.0]))
            # elif group_to_use == 2:
            #     x_delta = np.random.uniform(-0.3, -0.1)
            #     y_delta = np.random.uniform(-0.2, 0.2)
            #     _exec(torch.tensor([0.0, y_delta, 0.0]))
            #     _exec(torch.tensor([x_delta, 0.0, 0.0]))
            # elif group_to_use == 3:
            #     x_delta = np.random.uniform(-0.1, 0.1)
            #     y_delta = np.random.uniform(-0.2, 0.2)
            #     _exec(torch.tensor([0.0, y_delta, 0.0]))
            #     _exec(torch.tensor([x_delta, 0.0, 0.0]))
            # else:
            #     x_delta = 0.1
            #     _exec(torch.tensor([x_delta, 0.0, 0.0]))

            # if group_to_use == 1:
            #     x_delta = np.random.uniform(-0.3, -0.1)
            #     y_delta = np.random.uniform(-0.2, 0.2)
            #     _exec(torch.tensor([0.0, y_delta, 0.0]))
            #     _exec(torch.tensor([x_delta, 0.0, 0.0]))
            # elif group_to_use == 2:
            #     x_delta = np.random.uniform(-0.1, 0.1)
            #     y_delta = np.random.uniform(-0.2, 0.2)
            #     _exec(torch.tensor([0.0, y_delta, 0.0]))
            #     _exec(torch.tensor([x_delta, 0.0, 0.0]))
            # else:
            # x_delta = 0.1
            # _exec(torch.tensor([x_delta, 0.0, 0.0]))
            # elif group_to_use == 3:
            #     x_delta = np.random.uniform(-0.1, 0.25)
            #     y_delta = np.random.uniform(-0.2, 0.2)
            #     _exec(torch.tensor([0.0, y_delta, 0.0]))
            #     _exec(torch.tensor([x_delta, 0.0, 0.0]))
            # elif group_to_use == 4:
            #     x_delta = 0.25
            #     y_delta = -0.15
            #     _exec(torch.tensor([0.0, y_delta, 0.0]))
            #     _exec(torch.tensor([x_delta, 0.0, 0.0]))

        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            full = np.zeros((self._action_dim,), dtype=np.float32)
            # Map agent's full IK control (position + orientation) into the controller input
            arm_action_full = np.zeros((self._arm_input_dim,), dtype=np.float32)
            arm_action_full[: self._agent_act_dim] = action[: self._agent_act_dim]
            full[self._right_arm_idx] = arm_action_full * self._arm_action_scale
            # Hold gripper closed constantly to simplify training
            full[self._right_grip_idx] = 1.0
            # Keep laptop at target angle every env.step (mirrors simple_task.py behavior)
            self._set_laptop_pose()
            obs, reward, terminated, truncated, info = self._env.step(action=full)
            self._set_laptop_pose()
            # Cache the last base observation dict for downstream eval access (include object_health_states from objects, like place_plate)
            base_obs = obs if isinstance(obs, dict) else {}
            health_damage = self._get_health_and_damage(self._env)
            object_health_states = {k: {"health": v["health"], "damage_info": v.get("damage_info", {})} for k, v in health_damage.items()}
            self._last_base_obs = {**base_obs, "object_health_states": object_health_states}
            # Accumulate task (distance) and damage rewards for episode logging
            self._episode_task_reward += float(reward)
            damage_reward, _ = self._damage_reward_fn(self._env, base_obs)
            self._episode_damage_reward += float(damage_reward)
            self._episode_step += 1
            if self._episode_step >= self._max_episode_steps:
                truncated = True

            info = info if isinstance(info, dict) else {}
            if terminated or truncated:
                laptop_health = health_damage.get("laptop", {}).get("health", 100.0)
                total_damage_taken = 0.0
                for name in ["mug", "laptop", "plate"]:
                    if name in health_damage:
                        total_damage_taken += health_damage[name].get("total_damage", 0.0)
                robot_key = next((k for k in health_damage.keys() if k.startswith("robot_")), None)
                if robot_key and robot_key in health_damage:
                    total_damage_taken += health_damage[robot_key].get("total_damage", 0.0)
                info["task_reward"] = self._episode_task_reward
                info["damage_reward"] = self._episode_damage_reward
                info["final_laptop_health"] = float(laptop_health)
                info["total_damage_taken"] = float(total_damage_taken)
            
            # Generate 1 water particle above the mug and step sim once
            r = random.random()
            mug_obj = self.get_target_obj()
            if r < 1:
                if mug_obj is not None:
                    for _ in range(1):
                        water_system = self._env.scene.get_system("water", force_init=True)
                        mug_pos, _ = mug_obj.get_position_orientation()
                        z_offset = 0.05
                        if isinstance(mug_pos, torch.Tensor):
                            drop_pos = (mug_pos + torch.tensor([0.0, 0.0, z_offset], dtype=torch.float32)).tolist()
                        else:
                            drop_pos = [mug_pos[0], mug_pos[1], mug_pos[2] + z_offset]
                        water_system.generate_particles(positions=[drop_pos])
                        og.sim.step()
                        self._set_laptop_pose()
            
            return self._extract_obs(), float(reward), bool(terminated), bool(truncated), info

        def _set_laptop_pose(self, target_deg: float = 120.0):
            laptop = self._env.scene.object_registry("name", "laptop")
            if laptop is None:
                return
            target_rad = math.radians(float(target_deg))
            if hasattr(laptop, "joints"):
                for joint in laptop.joints.values():
                    # Some joints may not have valid limits / handles yet; skip those safely.
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

    def thunk():
        env = OGSimpleEnv(scene_file)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.array(envs.single_action_space.shape).prod())
        # Increased capacity: 256 hidden units with 3 layers for better representation learning
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1024)), nn.Tanh(),
            layer_init(nn.Linear(1024, 1024)), nn.Tanh(),
            layer_init(nn.Linear(1024, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1024)), nn.Tanh(),
            layer_init(nn.Linear(1024, 1024)), nn.Tanh(),
            layer_init(nn.Linear(1024, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )
        # Start with moderate exploration
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            # Sample pre-squash action
            pre = dist.rsample()
            squashed = torch.tanh(pre)
            logprob = dist.log_prob(pre).sum(-1) - torch.log(1 - squashed.pow(2) + 1e-6).sum(-1)
        else:
            # action is squashed in [-1,1]; invert tanh to get pre-squash
            eps = 1e-6
            a = torch.clamp(action, -1 + eps, 1 - eps)
            pre = 0.5 * (torch.log1p(a) - torch.log1p(-a))
            squashed = a
            logprob = dist.log_prob(pre).sum(-1) - torch.log(1 - squashed.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return squashed, logprob, entropy, self.critic(x)


# ---------------------------------------------------------------------------
# Offline dataset utilities
# ---------------------------------------------------------------------------

def _load_offline_dataset_from_hdf5(path: str, reward_type: str):
    """Load demonstrations from HDF5 into flat arrays suitable for PPO-style offline training.

    Returns:
        obs:      (N, obs_dim)
        actions:  (N, act_dim)
        rewards:  (N,)
        ep_lengths: (num_episodes,) lengths of each episode in steps
    """
    assert reward_type in ("distance", "full"), f"Unknown reward_type '{reward_type}'"

    if not os.path.exists(path):
        print(f"[Offline] Dataset path does not exist: {path}")
        return None, None, None, None

    obs_list = []
    act_list = []
    rew_list = []
    ep_lengths = []

    with h5py.File(path, "r") as f:
        demo_names = sorted([k for k in f.keys() if k.startswith("demo_")])
        if not demo_names:
            print(f"[Offline] No 'demo_XXX' groups found in {path}")
            return None, None, None, None

        for name in demo_names:
            g = f[name]
            # (T+1, obs_dim)
            obs = g["observations"][...].astype(np.float32)
            # (T, act_dim)
            actions = g["actions"][...].astype(np.float32)
            # (T,)
            dist_rewards = g["distance_reward_terms"][...].astype(np.float32)
            dist_terms = g["distance_termination_terms"][...].astype(bool)
            dmg_rewards = g["damage_reward_terms"][...].astype(np.float32)
            dmg_terms = g["damage_termination_terms"][...].astype(bool)

            T = actions.shape[0]
            if T == 0:
                continue

            if reward_type == "distance":
                rewards = dist_rewards.copy()
                # Only distance termination matters; treat episode as ending at distance_done=True
                done_terms = dist_terms.copy()
            else:
                # Full reward: distance + damage
                rewards = dist_rewards + dmg_rewards
                # Episode terminates when either distance OR damage terminates
                done_terms = np.logical_or(dist_terms, dmg_terms)
                # If damage termination occurs, truncate episode there (ignore later transitions)
                if np.any(dmg_terms):
                    first_damage_idx = int(np.argmax(dmg_terms))
                    T = first_damage_idx + 1
                    rewards = rewards[:T]
                    done_terms = done_terms[:T]
                    obs = obs[: T + 1]
                    actions = actions[:T]

            # Build flat arrays; treat the last step of each sequence as terminal for GAE
            for t in range(T):
                obs_list.append(obs[t])
                act_list.append(actions[t])
                rew_list.append(rewards[t])
            ep_lengths.append(T)

    if not obs_list:
        print(f"[Offline] No transitions loaded from {path}")
        return None, None, None, None

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.float32)
    rew_arr = np.asarray(rew_list, dtype=np.float32)
    ep_lengths_arr = np.asarray(ep_lengths, dtype=np.int32)

    print(
        f"[Offline] Loaded dataset from {path}: "
        f"{len(ep_lengths_arr)} episodes, {obs_arr.shape[0]} transitions."
    )
    return obs_arr, act_arr, rew_arr, ep_lengths_arr


def _load_bc_dataset_from_hdf5(path: str, use_only_safe: bool):
    """Load (obs, action) pairs for behavior cloning.

    If use_only_safe=True, only demos with group attribute total_damage == 0.0
    are included.
    """
    if not os.path.exists(path):
        print(f"[BC] Dataset not found: {path}")
        return None, None

    obs_list = []
    act_list = []

    with h5py.File(path, "r") as f:
        demo_names = sorted([k for k in f.keys() if k.startswith("demo_")])
        if not demo_names:
            print(f"[BC] No 'demo_XXX' groups found in {path}")
            return None, None

        for name in demo_names:
            g = f[name]
            total_damage = float(g.attrs.get("total_damage", 0.0))
            if use_only_safe and total_damage > 0.0:
                continue
            obs = g["observations"][...].astype(np.float32)   # (T+1, obs_dim)
            actions = g["actions"][...].astype(np.float32)    # (T, full_act_dim)
            T = actions.shape[0]
            if T <= 0 or obs.shape[0] < T + 1:
                continue
            # Use obs[0:T] with the position-only slice actions[0:T, :POS_ACT_DIM] for BC
            obs_list.append(obs[:T])
            act_list.append(actions[:, :POS_ACT_DIM])

    if not obs_list:
        print(f"[BC] No suitable demos found in {path} (after filtering).")
        return None, None

    obs_arr = np.concatenate(obs_list, axis=0)
    act_arr = np.concatenate(act_list, axis=0)
    print(
        f"[BC] Loaded BC dataset from {path}: {obs_arr.shape[0]} transitions "
        f"from {len(obs_list)} demos (safe_only={use_only_safe})."
    )
    return obs_arr, act_arr


def _compute_mc_returns_offline(rew_arr, ep_lengths, gamma: float):
    """Compute Monte Carlo discounted returns per episode for offline critic targets."""
    returns = np.zeros_like(rew_arr, dtype=np.float32)
    idx = 0
    for ep_len in ep_lengths:
        ep_len = int(ep_len)
        start = idx
        end = idx + ep_len
        G = 0.0
        for t in reversed(range(ep_len)):
            G = float(rew_arr[start + t]) + gamma * G
            returns[start + t] = G
        idx = end
    return returns


def build_offline_replay(args: Args, device):
    """Load offline dataset and build tensors for critic replay during online PPO."""
    if not args.offline_critic_replay_enabled:
        return None

    obs_arr, _act_arr, rew_arr, ep_lengths = _load_offline_dataset_from_hdf5(
        args.offline_dataset_path, args.offline_reward_type
    )
    if obs_arr is None:
        return None

    mc_returns = _compute_mc_returns_offline(rew_arr, ep_lengths, float(args.gamma))
    obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=device)
    ret_tensor = torch.tensor(mc_returns, dtype=torch.float32, device=device)

    print(
        f"[OfflineReplay] Built critic replay buffer: {obs_tensor.shape[0]} transitions "
        f"from {len(ep_lengths)} episodes."
    )
    return {
        "obs": obs_tensor,
        "returns": ret_tensor,
        "size": int(obs_tensor.shape[0]),
    }


def run_offline_critic_pretrain(args: Args, agent: Agent, optimizer: optim.Optimizer, device, offline_replay):
    """One-time offline critic pretraining using the replay buffer.

    Only the critic is trained here (via get_value); the actor is unaffected.
    """
    if offline_replay is None:
        print("[OfflineCritic] No offline replay buffer available; skipping critic pretraining.")
        return
    if args.offline_num_epochs <= 0:
        print("[OfflineCritic] offline_num_epochs <= 0; skipping critic pretraining.")
        return

    obs = offline_replay["obs"]
    rets = offline_replay["returns"]
    size = int(offline_replay["size"])

    print(
        f"[OfflineCritic] Starting critic-only pretraining on {size} transitions "
        f"for {args.offline_num_epochs} epochs."
    )

    for epoch in range(1, args.offline_num_epochs + 1):
        inds = np.arange(size)
        np.random.shuffle(inds)

        epoch_losses = []
        for start in range(0, size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = inds[start:end]
            mb_obs = obs[mb_inds]
            mb_rets = rets[mb_inds]

            values = agent.get_value(mb_obs).view(-1)
            v_loss = 0.5 * ((values - mb_rets) ** 2).mean()

            optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_losses.append(float(v_loss.item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        wandb.log({"offline_critic/epoch": epoch, "offline_critic/loss_value": mean_loss})
        print(
            f"[OfflineCritic] Epoch {epoch}/{args.offline_num_epochs} "
            f"critic MSE={mean_loss:.6f}"
        )


def run_offline_bc_pretrain(args: Args, agent: Agent, optimizer: optim.Optimizer, device, envs=None):
    """Offline behavior cloning to initialize the actor from demonstrations. Always run when bc_pretrain_enabled."""
    if not args.bc_pretrain_enabled:
        return

    path = args.offline_dataset_path
    print(f"[BC] Looking for dataset at: {os.path.abspath(path)} (exists={os.path.exists(path)})")
    obs_arr, act_arr = _load_bc_dataset_from_hdf5(path, args.bc_use_only_safe_demos)

    if obs_arr is None:
        # No HDF5 or no demos: use small dummy dataset so BC runs and breakpoint is hit
        obs_dim = int(np.prod(envs.single_observation_space.shape)) if envs is not None else 14
        act_dim = POS_ACT_DIM
        n_dummy = max(args.minibatch_size * 4, 128)
        np.random.seed(args.seed)
        obs_arr = np.random.randn(n_dummy, obs_dim).astype(np.float32) * 0.1
        act_arr = np.random.randn(n_dummy, act_dim).astype(np.float32) * 0.1
        print(f"[BC] No BC data available; using dummy dataset (n={n_dummy}) so BC runs. Replace with real HDF5 for real BC.")

    obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=device)
    act_tensor = torch.tensor(act_arr, dtype=torch.float32, device=device)
    dataset_size = obs_tensor.shape[0]
    num_batches = (dataset_size + args.minibatch_size - 1) // args.minibatch_size

    print("")
    print("[BC] ========== Behavior cloning pretraining ==========")
    print(
        f"[BC] Dataset: {dataset_size} transitions, "
        f"{args.bc_num_epochs} epochs, minibatch_size={args.minibatch_size}, "
        f"safe_only={args.bc_use_only_safe_demos}"
    )
    print(f"[BC] Training...")
    print("")

    for bc_epoch in range(1, args.bc_num_epochs + 1):
        inds = np.arange(dataset_size)
        np.random.shuffle(inds)

        epoch_losses = []
        for batch_idx, start in enumerate(range(0, dataset_size, args.minibatch_size)):
            end = start + args.minibatch_size
            mb_inds = inds[start:end]
            mb_obs = obs_tensor[mb_inds]
            mb_actions = act_tensor[mb_inds]

            # Use existing actor distribution to compute logprob of dataset actions
            _, logprob, _, _ = agent.get_action_and_value(mb_obs, action=mb_actions)
            bc_loss = -logprob.mean()

            optimizer.zero_grad()
            bc_loss.backward()
            # Clip all agent params here; BC affects primarily the actor via logprob
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_losses.append(float(bc_loss.item()))
            # Print progress every 10 batches or on last batch
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                so_far = float(np.mean(epoch_losses))
                print(
                    f"[BC] Epoch {bc_epoch}/{args.bc_num_epochs}  batch {batch_idx + 1}/{num_batches}  "
                    f"loss={so_far:.6f}"
                )

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        wandb.log({"offline_bc/epoch": bc_epoch, "offline_bc/loss": mean_loss})
        print(
            f"[BC] Epoch {bc_epoch}/{args.bc_num_epochs}  done  mean_loss={mean_loss:.6f}"
        )
        print("")

    print("[BC] ========== BC pretraining finished; starting PPO. ==========")
    print("")


def _compute_gae_offline(args: Args, agent: Agent, device, obs_arr, rew_arr, ep_lengths):
    """Compute advantages and returns for the offline dataset using per-episode GAE."""
    obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = agent.get_value(obs_tensor).squeeze(-1).cpu().numpy()

    advantages = np.zeros_like(rew_arr, dtype=np.float32)
    returns = np.zeros_like(rew_arr, dtype=np.float32)

    gamma = float(args.gamma)
    lam = float(args.gae_lambda)

    idx = 0
    for ep_len in ep_lengths:
        start = idx
        end = idx + int(ep_len)
        if ep_len <= 0:
            idx = end
            continue
        v_ep = values[start:end]
        r_ep = rew_arr[start:end]

        lastgaelam = 0.0
        nextvalue = 0.0
        for t in reversed(range(int(ep_len))):
            # Treat the last step as terminal (no bootstrap beyond dataset)
            nonterminal = 0.0 if t == ep_len - 1 else 1.0
            if t == ep_len - 1:
                nextvalue = 0.0
            else:
                nextvalue = v_ep[t + 1]
            delta = r_ep[t] + gamma * nextvalue * nonterminal - v_ep[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            advantages[start + t] = lastgaelam
            returns[start + t] = advantages[start + t] + v_ep[t]

        idx = end

    adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    return adv_tensor, ret_tensor


if __name__ == "__main__":
    args = Args()
    # OmniGibson uses a single global simulator; multiple parallel envs are not supported.
    # Force single-env to avoid physics/tensor backend race conditions during scene load/reset.
    if args.num_envs != 1:
        print(f"[OG Notice] num_envs={args.num_envs} not supported; forcing num_envs=1.")
        args.num_envs = 1
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # Add random component to ensure unique run name even if run multiple times quickly
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}__{random.randint(1000, 9999)}"
    # Initialize WandB (same project as ppo_train_place_plate for unified logging)
    wandb.init(project="omnigibson-ppo-place-plate", name=run_name, config=vars(args), id=None)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # Lenient registry so pkl state load works when scene has subset of saved objects (same as move_glass_teleop_load)
    apply_lenient_registry_patch()
    # Build OmniGibson vector envs using our adapter
    scene_file = DEFAULT_SCENE_FILE
    # Use a single env wrapper directly to avoid vectorization overhead with global simulator
    envs = gym.vector.SyncVectorEnv([make_env(scene_file, 0, args.capture_video, run_name)])
    # Match viewer camera pose from ppo_train.py
    try:
        og.sim.viewer_camera.set_position_orientation(
            position=[-1.3225984573364258, 0.2645236849784851, 0.9981386065483093],
            orientation=[0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452],
        )
    except Exception:
        pass
    # Continuous action space expected
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    
    # Set curriculum interval on the environment
    try:
        base_env = envs.envs[0]
        while hasattr(base_env, "env"):
            base_env = base_env.env
        if hasattr(base_env, "set_curriculum_interval"):
            base_env.set_curriculum_interval(args.curriculum_interval)
    except Exception:
        pass

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Initialize offline_replay to None so later checks are safe
    offline_replay = None
    
    # BC policy pretraining always runs first (bc_num_epochs, default 20); then RL finetuning.
    # If BC_INIT_CHECKPOINT is set, load it to initialize the agent, then run BC (BC always runs when enabled).
    print(f"[BC] bc_pretrain_enabled={args.bc_pretrain_enabled}, dataset={args.offline_dataset_path}")
    if BC_INIT_CHECKPOINT is not None and os.path.exists(BC_INIT_CHECKPOINT):
        print(f"[BC] Loading initial checkpoint from {BC_INIT_CHECKPOINT}, then running BC pretraining.")
        bc_ckpt = torch.load(BC_INIT_CHECKPOINT, map_location=device)
        agent.load_state_dict(bc_ckpt["agent_state_dict"])
    if args.bc_pretrain_enabled:
        run_offline_bc_pretrain(args, agent, optimizer, device, envs)
    else:
        print("[BC] bc_pretrain_enabled=False; skipping BC pretraining.")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Set initial curriculum group before first reset
    try:
        base_env_init = envs.envs[0]
        while hasattr(base_env_init, "env"):
            base_env_init = base_env_init.env
        if hasattr(base_env_init, "set_curriculum_group"):
            base_env_init.set_curriculum_group(global_step, force_eval=False)
    except Exception:
        pass
    print("[PPO] Resetting env (may take 1–2 min with pkl + water)...")
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    print("[PPO] First reset done. Starting iterations (each = 512 env steps + update).")

    # Live plot: episode return per training episode (saved to disk each update; final PNG + JSON at end)
    episode_returns = []  # list of (episode_index, return, global_step) for JSON
    plot_live_path = os.path.join(checkpoint_dir, "episode_returns_live.png")
    plot_final_path = os.path.join(checkpoint_dir, "episode_returns_final.png")
    episode_returns_json_path = os.path.join(checkpoint_dir, "episode_returns.json")
    fig_returns, ax_returns = plt.subplots(1, 1, figsize=(8, 4))
    ax_returns.set_xlabel("Training episode")
    ax_returns.set_ylabel("Episode return")
    ax_returns.set_title("Episode return (live)")
    line_returns, = ax_returns.plot([], [], "b-", alpha=0.7)

    for iteration in range(1, args.num_iterations + 1):
        print(f"[PPO] Iteration {iteration}/{args.num_iterations} (rollout 0–{args.num_steps} steps)...")
        # Update curriculum group based on current global_step before collecting new data
        current_curriculum_group = 1  # Default fallback
        try:
            base_env = envs.envs[0]
            while hasattr(base_env, "env"):
                base_env = base_env.env
            if hasattr(base_env, "set_curriculum_group"):
                base_env.set_curriculum_group(global_step, force_eval=False)
                if hasattr(base_env, "_curriculum_group"):
                    current_curriculum_group = base_env._curriculum_group
        except Exception:
            pass
        
        # Annealing the learning rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Optionally linearly decay the entropy coefficient from its initial value to 0 over training
        if args.anneal_ent_coef:
            ent_frac = 1.0 - (iteration - 1.0) / args.num_iterations
            current_ent_coef = args.ent_coef * ent_frac
        else:
            current_ent_coef = args.ent_coef

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            # Update curriculum group right after global_step increments so it's current for any resets
            try:
                base_env_step = envs.envs[0]
                while hasattr(base_env_step, "env"):
                    base_env_step = base_env_step.env
                if hasattr(base_env_step, "set_curriculum_group"):
                    base_env_step.set_curriculum_group(global_step, force_eval=False)
            except Exception:
                pass
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Episode end logging: support both Gymnasium vector format (batched "episode"/"_episode")
            # and legacy "final_info" list. Gymnasium SyncVectorEnv does NOT use final_info; it batches
            # RecordEpisodeStatistics output into infos["episode"]["r"], infos["episode"]["l"], infos["_episode"].
            completed_episodes = []
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        completed_episodes.append({
                            "r": info["episode"]["r"],
                            "l": info["episode"]["l"],
                            "task_reward": info.get("task_reward"),
                            "damage_reward": info.get("damage_reward"),
                            "final_laptop_health": info.get("final_laptop_health"),
                            "total_damage_taken": info.get("total_damage_taken"),
                        })
            if "episode" in infos and "_episode" in infos:
                ep_mask = np.asarray(infos["_episode"]).flatten()
                ep_r = np.asarray(infos["episode"]["r"]).flatten()
                ep_l = np.asarray(infos["episode"]["l"]).flatten()
                for i in range(len(ep_mask)):
                    if ep_mask[i]:
                        task_r = float(infos["task_reward"][i]) if "task_reward" in infos else None
                        dmg_r = float(infos["damage_reward"][i]) if "damage_reward" in infos else None
                        laptop_h = float(infos["final_laptop_health"][i]) if "final_laptop_health" in infos else None
                        total_dmg = float(infos["total_damage_taken"][i]) if "total_damage_taken" in infos else None
                        completed_episodes.append({
                            "r": float(ep_r[i]),
                            "l": int(ep_l[i]),
                            "task_reward": task_r,
                            "damage_reward": dmg_r,
                            "final_laptop_health": laptop_h,
                            "total_damage_taken": total_dmg,
                        })
            for ep in completed_episodes:
                ep_return = ep["r"]
                ep_len = ep["l"]
                task_reward = ep.get("task_reward") if ep.get("task_reward") is not None else ep_return
                damage_reward = ep.get("damage_reward") if ep.get("damage_reward") is not None else 0.0
                final_laptop_health = ep.get("final_laptop_health") if ep.get("final_laptop_health") is not None else 100.0
                total_damage_taken = ep.get("total_damage_taken") if ep.get("total_damage_taken") is not None else 0.0
                # Append for live plot and JSON export
                episode_returns.append({
                    "episode": len(episode_returns) + 1,
                    "return": float(ep_return),
                    "global_step": int(global_step),
                })
                # Update live plot
                if episode_returns:
                    ep_indices = [p["episode"] for p in episode_returns]
                    ep_vals = [p["return"] for p in episode_returns]
                    line_returns.set_data(ep_indices, ep_vals)
                    ax_returns.relim()
                    ax_returns.autoscale_view()
                    fig_returns.savefig(plot_live_path, dpi=100)
                wandb.log({
                    "charts/episodic_return": ep_return,
                    "charts/episodic_length": ep_len,
                    "episode/task_reward": task_reward,
                    "episode/damage_reward": damage_reward,
                    "episode/final_laptop_health": final_laptop_health,
                    "episode/total_damage_taken": total_damage_taken,
                    "global_step": global_step,
                })
                print(
                    f"global_step={global_step}, return={ep_return:.2f}, len={ep_len}, "
                    f"task_reward={task_reward:.2f}, damage_reward={damage_reward:.2f}, "
                    f"laptop_health={final_laptop_health:.1f}, total_damage={total_damage_taken:.1f}"
                )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss_onpolicy = 0.5 * v_loss_max.mean()
                else:
                    v_loss_onpolicy = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Critic replay loss from offline dataset (no policy gradient)
                v_loss_offline = torch.tensor(0.0, device=device)
                if (
                    offline_replay is not None
                    and args.offline_critic_replay_enabled
                    and offline_replay["size"] > 0
                ):
                    off_size = offline_replay["size"]
                    # Sample a minibatch of offline transitions (same size as on-policy minibatch)
                    off_inds = np.random.randint(0, off_size, size=len(mb_inds))
                    off_obs = offline_replay["obs"][off_inds]
                    off_returns = offline_replay["returns"][off_inds]
                    with torch.no_grad():
                        off_values = agent.get_value(off_obs).view(-1)
                    v_loss_offline = 0.5 * ((off_values - off_returns) ** 2).mean()

                v_loss = v_loss_onpolicy + args.offline_critic_coef * v_loss_offline

                entropy_loss = entropy.mean()
                loss = pg_loss - current_ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Rollout stats (mean reward over collected batch)
        mean_reward_rollout = rewards.cpu().numpy().mean().item()
        mean_return_rollout = b_returns.cpu().numpy().mean().item()

        # Record training metrics to WandB (same style as electrical script: single dict, include global_step)
        wandb.log({
            "train/mean_reward": mean_reward_rollout,
            "train/mean_return": mean_return_rollout,
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/entropy_coef": float(current_ent_coef),
            "charts/curriculum_group": float(current_curriculum_group),
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "global_step": global_step,
        })
        print(
            f"[PPO] iteration {iteration} done | global_step={global_step} | "
            f"mean_reward={mean_reward_rollout:.3f} | value_loss={v_loss.item():.4f} | SPS={int(global_step / (time.time() - start_time))}"
        )
        
        # Save checkpoint periodically (every 20 iterations)
        if iteration % 20 == 0:
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration,
                'global_step': global_step,
            }, checkpoint_file)
            print(f"Saved checkpoint at iteration {iteration}")

    # Save episode returns to JSON and final plot to PNG
    with open(episode_returns_json_path, "w") as f:
        json.dump(episode_returns, f, indent=2)
    fig_returns.savefig(plot_final_path, dpi=150)
    plt.close(fig_returns)
    print(f"[PPO] Saved episode returns to {episode_returns_json_path} and plot to {plot_final_path}")

    envs.close()
    wandb.finish()
    og.shutdown()

