# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
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

from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.damage_evaluators.electrical_damage_evaluator import ElectricalDamageEvaluator
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

DEFAULT_SCENE_FILE = "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/reset_saved.json"
# Match training episode horizon for evaluation
EVAL_MAX_STEPS = 300
# Number of eval episodes for which we emit a damage-status-border sim video
# (can be overridden by external scripts, e.g., BC eval wants all episodes).
STATUS_BORDER_EPISODES = 2
checkpoint_dir = "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/checkpoints"
checkpoint_file = os.path.join(checkpoint_dir, "checkpoint_bootstrap_distance.pth")
# Optional BC-only initialization checkpoint written by the BC eval script.
# If present, this will be loaded instead of re-running BC pretraining.
BC_INIT_CHECKPOINT = os.path.join(checkpoint_dir, "checkpoint_all_demos.pth")
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
    total_timesteps: int = 30000
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
        "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/"
        "electrical_teleop_demos.hdf5"
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
            # Match simple_task_load.py: set macros BEFORE any simulator/env creation
            try:
                gm.USE_GPU_DYNAMICS = True
                gm.ENABLE_OBJECT_STATES = True
                gm.ENABLE_FLATCACHE = True
                gm.ENABLE_HQ_RENDERING = False
            except Exception:
                # If macros were already locked by prior use, proceed without raising
                pass
            # Ensure fresh simulator (match ppo_train.py)
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

            # Create environment from saved scene file
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
        
        def _damage_reward_fn(self,env, obs):
            total_damage = 0.0
            mug_health_states = obs["object_health_states"]["mug"]
            laptop_health_states = obs["object_health_states"]["laptop"]
            plate_health_states = obs["object_health_states"]["plate"]
            robot_key = next(k for k in obs["object_health_states"].keys() if k.startswith("robot_"))
            robot_health_states = obs["object_health_states"][robot_key]
            for damage_type, damage_info in mug_health_states["damage_info"].items():
                for link_name, damage in damage_info.items():
                    total_damage += damage
            for damage_type, damage_info in laptop_health_states["damage_info"].items():
                for link_name, damage in damage_info.items():
                    total_damage += damage
            for damage_type, damage_info in robot_health_states["damage_info"].items():
                for link_name, damage in damage_info.items():
                    total_damage += damage
            for damage_type, damage_info in plate_health_states["damage_info"].items():
                for link_name, damage in damage_info.items():
                    total_damage += damage
            if total_damage == 0.0 and mug_health_states["health"] == 100.0 and laptop_health_states["health"] == 100.0 and robot_health_states["health"] == 100.0 and plate_health_states["health"] == 100.0:
                return 0.01, False
            if mug_health_states["health"] == 0.0 or laptop_health_states["health"] == 0.0 or robot_health_states["health"] == 0.0 or plate_health_states["health"] == 0.0:
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
            # return (damage_weight * damage_reward + distance_weight * distance_reward), damage_terminated or distance_terminated
            # return damage_reward, damage_terminated
            return distance_reward, distance_terminated
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
            for _ in range(20):
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
                for _ in range(230):
                    if isinstance(mug_pos, torch.Tensor):
                        drop_pos = (mug_pos + torch.tensor([0.0, 0.0, z_offset], dtype=torch.float32)).tolist()
                    else:
                        drop_pos = [mug_pos[0], mug_pos[1], mug_pos[2] + z_offset]
                    water_system.generate_particles(positions=[drop_pos])
                    for _ in range(1):
                        og.sim.step()
            self._env.unlock_health_changes()
            for _ in range(10):
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
            
            # Also pose laptop lid each reset so it stays at target angle
            for _ in range(10):
                og.sim.step()
                self._set_laptop_pose()

            # og.sim.save([os.path.join(os.path.dirname(__file__), "reset_saved.json")])
            # print(f"✅ Saved scene to: {os.path.join(os.path.dirname(__file__), 'reset_saved.json')}")
            # breakpoint()
            return self._extract_obs(), {}

        def _execute_init(self):
            def _exec(delta, max_steps=50):
                current_eef_pos = self._robot.get_eef_position("right")
                current_eef_orn = self._robot.get_eef_orientation("right")
                target_eef_pos = current_eef_pos + delta
                target_eef_pose = (target_eef_pos, current_eef_orn)
                steps = 0
                for a in self._prims._move_hand_linearly_cartesian(target_eef_pose, ignore_failure=True):
                    self._env.step(action=a)
                    steps += 1
                    if steps >= max_steps:
                        break
            
            # Always do the first z movement regardless of group
            
            group_to_use = self._curriculum_group
            # if not getattr(self, "_is_eval", False) and group_to_use in (3, 4):
            #     r = random.random()
            #     if r < 0.05:
            #         group_to_use = 1
            #     elif r < 0.10:
            #         group_to_use = 2

            x_delta = np.random.uniform(-0.05, 0.1)
            y_delta = np.random.uniform(-0.1, 0.05)
            _exec(torch.tensor([x_delta, y_delta, 0.0]))

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
            # Cache the last base observation dict for downstream eval access
            self._last_base_obs = obs if isinstance(obs, dict) else None
            self._episode_step += 1
            if self._episode_step >= self._max_episode_steps:
                truncated = True
            
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
        wandb.log(
            {
                "offline_critic/epoch": epoch,
                "offline_critic/loss_value": mean_loss,
            }
        )
        print(
            f"[OfflineCritic] Epoch {epoch}/{args.offline_num_epochs} "
            f"critic MSE={mean_loss:.6f}"
        )


def run_offline_bc_pretrain(args: Args, agent: Agent, optimizer: optim.Optimizer, device):
    """Offline behavior cloning to initialize the actor from demonstrations."""
    if not args.bc_pretrain_enabled:
        print("[BC] bc_pretrain_enabled=False; skipping BC pretraining.")
        return

    obs_arr, act_arr = _load_bc_dataset_from_hdf5(
        args.offline_dataset_path, args.bc_use_only_safe_demos
    )
    if obs_arr is None:
        print("[BC] No BC data available; skipping BC pretraining.")
        return

    obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=device)
    act_tensor = torch.tensor(act_arr, dtype=torch.float32, device=device)
    dataset_size = obs_tensor.shape[0]

    print(
        f"[BC] Starting behavior cloning pretraining on {dataset_size} transitions, "
        f"epochs={args.bc_num_epochs}, safe_only={args.bc_use_only_safe_demos}."
    )

    for bc_epoch in range(1, args.bc_num_epochs + 1):
        inds = np.arange(dataset_size)
        np.random.shuffle(inds)

        epoch_losses = []
        for start in range(0, dataset_size, args.minibatch_size):
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

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        wandb.log(
            {
                "offline_bc/epoch": bc_epoch,
                "offline_bc/loss": mean_loss,
            }
        )
        print(
            f"[BC] Epoch {bc_epoch}/{args.bc_num_epochs} "
            f"BC loss={mean_loss:.6f}"
        )


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


def run_eval_episode(env, agent, device, eval_videos_dir, iteration_num, capture_video=True, episode_idx=None):
    """Run a single evaluation episode and optionally save an AVI with on-frame cumulative reward text.
    
    Returns (total_reward, terminated, truncated, health_avi_path_or_None, water_avi_path_or_None, reward_avi_path_or_None, final_mug_health, final_laptop_health, final_plate_health, steps_taken)
    """
    fps = 30
    
    # Get base OGSimpleEnv and force curriculum into eval group before reset
    base = env
    while hasattr(base, "env"):
        base = base.env
    try:
        if hasattr(base, "set_curriculum_group"):
            # force_eval=True ensures curriculum group 4 (eval) regardless of step
            base.set_curriculum_group(0, force_eval=True)
    except Exception:
        pass

    # Reset environment after setting eval curriculum group
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    
    # Get mug / laptop / plate objects and laptop's electrical damage evaluator for water contact tracking
    mug = None
    laptop = None
    plate = None
    laptop_eval = None
    if hasattr(base, "_env"):
        try:
            scene = base._env.scene
            mug = scene.object_registry("name", "mug")
            laptop = scene.object_registry("name", "laptop")
            plate = scene.object_registry("name", "plate")
            if laptop is not None:
                existing_evals = getattr(laptop, "damage_evaluators", [])
                for ev in existing_evals:
                    if isinstance(ev, ElectricalDamageEvaluator):
                        laptop_eval = ev
                        break
        except Exception:
            pass
    
    vw = None
    avi_path = None
    # Always capture a video for this episode; we'll decide later whether to keep it
    suffix = f"_ep{int(episode_idx)+1}" if episode_idx is not None else ""
    avi_path = os.path.join(eval_videos_dir, f"eval_iteration_{iteration_num}{suffix}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(avi_path, fourcc, fps, (1080, 720))
    
    # Store original camera pose and define alternate camera pose (as plain Python lists)
    orig_cam_pos, orig_cam_quat = og.sim.viewer_camera.get_position_orientation()
    if hasattr(orig_cam_pos, "tolist"):
        orig_cam_pos = orig_cam_pos.tolist()
    if hasattr(orig_cam_quat, "tolist"):
        orig_cam_quat = orig_cam_quat.tolist()
    alt_cam_pos = [0.12231605052948, -1.3176746368408203, 1.1935482025146484]
    alt_cam_quat = [0.5858199596405029, -1.0838084563147277e-05, -1.502305985923158e-05, 0.810441255569458]
    # Decide which camera to use this episode:
    # odd-numbered episodes (1,3,5 -> episode_idx 0,2,4) use original camera,
    # even-numbered episodes (2,4 -> episode_idx 1,3) use alternate camera.
    use_alt_camera = False
    if episode_idx is not None:
        use_alt_camera = (int(episode_idx) % 2 == 1)
    main_cam_pos, main_cam_quat = (alt_cam_pos, alt_cam_quat) if use_alt_camera else (orig_cam_pos, orig_cam_quat)
    try:
        og.sim.viewer_camera.set_position_orientation(position=main_cam_pos, orientation=main_cam_quat)
        og.sim.render()
    except Exception:
        pass
    
    total_reward = 0.0
    final_terminated = False
    final_truncated = False
    final_mug_health = None
    final_laptop_health = None
    final_plate_health = None
    steps_taken = 0
    # Track mug and laptop health over time for plotting alongside sim video
    mug_health_series = []
    laptop_health_series = []
    last_mug_health = 100.0
    last_laptop_health = 100.0
    last_plate_health = 100.0
    # Track laptop water contact counts over time
    laptop_contact_counts = []
    # Track cumulative reward over time for plotting
    reward_series = []
    
    # For the first episode, also capture a sim-only sequence (no overlays) for a high-quality video
    sim_only_frames = [] if (episode_idx is not None and int(episode_idx) == 0) else None
    # For the first few episodes (controlled by STATUS_BORDER_EPISODES), also capture a
    # sim-only sequence for a damage-status-border video.
    record_status_border = episode_idx is not None and int(episode_idx) < STATUS_BORDER_EPISODES
    border_frames = [] if record_status_border else None
    damage_status_series = [] if record_status_border else None
    
    # Extra visualization-only rollout after first done (policy steps only, for video)
    extra_policy_steps_after_done = 10
    done_reached = False
    remaining_extra_policy_steps = 0
    max_total_steps = EVAL_MAX_STEPS + extra_policy_steps_after_done

    # Run evaluation episode
    for step in range(max_total_steps):
        with torch.no_grad():
            # Use deterministic mean action for evaluation (no sampling)
            mean = agent.actor_mean(obs_tensor.unsqueeze(0))
            action = torch.tanh(mean)  # Squash to [-1, 1] like in training
        action_np = action.squeeze(0).cpu().numpy()

        if not done_reached:
            # Normal evaluation step: count reward and episode steps
            obs, reward, terminated, truncated, info = env.step(action_np)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            steps_taken += 1

            total_reward += float(reward)
            reward_series.append(total_reward)
        else:
            # Extra visualization-only steps after done: do not accumulate reward or steps_taken
            if remaining_extra_policy_steps <= 0:
                break
            obs, _, _, _, _ = env.step(action_np)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            remaining_extra_policy_steps -= 1
        # Record mug and laptop health per step from the base env's observation dict
        base = env
        while hasattr(base, "env"):
            base = base.env
        base_obs = getattr(base, "_last_base_obs", None)
        if isinstance(base_obs, dict):
            health_states = base_obs.get("object_health_states", {})
            # Mug health
            phs_mug = health_states.get("mug", None)
            if phs_mug is not None and ("health" in phs_mug):
                last_mug_health = float(phs_mug["health"])
                final_mug_health = last_mug_health
            # Laptop health
            phs_laptop = health_states.get("laptop", None)
            if phs_laptop is not None and ("health" in phs_laptop):
                last_laptop_health = float(phs_laptop["health"])
                final_laptop_health = last_laptop_health
            # Plate health
            phs_plate = health_states.get("plate", None)
            if phs_plate is not None and ("health" in phs_plate):
                last_plate_health = float(phs_plate["health"])
                final_plate_health = last_plate_health
        mug_health_series.append(last_mug_health)
        laptop_health_series.append(last_laptop_health)
        
        # Track laptop water contact counts
        laptop_contact_count = 0
        if laptop_eval is not None:
            try:
                summary = laptop_eval.get_contact_summary()
                details = summary.get("link_details", {})
                laptop_contact_count = max((v.get("particle_count", 0) for v in details.values()), default=0)
            except Exception:
                pass
        laptop_contact_counts.append(int(laptop_contact_count))
        
        if vw is not None:
            # Capture from the chosen camera angle for this episode
            try:
                og.sim.viewer_camera.set_position_orientation(position=main_cam_pos, orientation=main_cam_quat)
                og.sim.render()
            except Exception:
                pass
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
            rgb_np = rgb.cpu().numpy()[:, :, :3]
            frame = cv2.resize(rgb_np, (1080, 720))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # For sim-only recording (first episode), store raw BGR frames before adding text
            if sim_only_frames is not None:
                sim_only_frames.append(np.ascontiguousarray(frame_bgr.copy(), dtype=np.uint8))
            # For status-border recording (first two episodes), store raw BGR frames and combined damage status
            if record_status_border and border_frames is not None and damage_status_series is not None:
                border_frames.append(np.ascontiguousarray(frame_bgr.copy(), dtype=np.uint8))
                # Combine damage statuses from mug, laptop, and plate (worst-case)
                statuses = []
                for obj in (mug, laptop, plate):
                    if obj is not None and hasattr(obj, "damage_status"):
                        statuses.append(str(obj.damage_status).lower())
                if any(s in ("critical", "major") for s in statuses):
                    combined_status = "major"
                elif any(s == "minor" for s in statuses):
                    combined_status = "minor"
                else:
                    combined_status = "none"
                damage_status_series.append(combined_status)
            # Overlay reward text for the standard eval video
            text = f"Reward: {total_reward:.2f}"
            cv2.putText(frame_bgr, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            vw.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8))
        
        if not done_reached and (terminated or truncated):
            final_terminated = bool(terminated)
            final_truncated = bool(truncated)
            done_reached = True
            remaining_extra_policy_steps = extra_policy_steps_after_done
    
    # Ensure final health values are set (use last tracked values if not found in observations)
    if final_mug_health is None:
        final_mug_health = last_mug_health
    if final_laptop_health is None:
        final_laptop_health = last_laptop_health
    if final_plate_health is None:
        final_plate_health = last_plate_health
    
    # Temporary debugging: only save video if reward >= 0
    should_save_video = (total_reward >= 0.0) or capture_video
    
    if vw is not None and should_save_video:
        # Play the last frame with final reward 60 more times (2 seconds at 30fps)
        for _ in range(60):
            try:
                og.sim.viewer_camera.set_position_orientation(position=main_cam_pos, orientation=main_cam_quat)
                og.sim.render()
            except Exception:
                pass
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
            rgb_np = rgb.cpu().numpy()[:, :, :3]
            frame = cv2.resize(rgb_np, (1080, 720))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            text = f"Final Reward: {total_reward:.2f}"
            cv2.putText(frame_bgr, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            vw.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8))

        vw.release()
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        print(f"Evaluation video saved: {os.path.basename(avi_path)}")

        # For the first episode, also save a high-quality sim-only MP4 without overlays (like rl_test.py)
        if sim_only_frames is not None and len(sim_only_frames) > 0:
            sim_only_avi = os.path.join(
                eval_videos_dir, f"eval_iteration_{iteration_num}{suffix}_sim_only.avi"
            )
            sim_only_mp4 = os.path.join(
                eval_videos_dir, f"eval_iteration_{iteration_num}{suffix}_sim_only.mp4"
            )
            h, w = sim_only_frames[0].shape[:2]
            fourcc_sim = cv2.VideoWriter_fourcc(*"XVID")
            vw_sim = cv2.VideoWriter(sim_only_avi, fourcc_sim, fps, (w, h))
            for f in sim_only_frames:
                vw_sim.write(np.ascontiguousarray(f, dtype=np.uint8))
            vw_sim.release()
            # High-quality MPEG-4 encoding with low quantizer (same as rl_test.py)
            subprocess.run(
                ["ffmpeg", "-y", "-i", sim_only_avi, "-c:v", "mpeg4", "-q:v", "2", sim_only_mp4],
                check=True,
            )
            try:
                os.remove(sim_only_avi)
            except OSError:
                pass
        sim_path = avi_path
        # For the first few episodes (controlled by STATUS_BORDER_EPISODES), also save
        # a damage-status-border sim MP4 without overlays
        border_mp4 = None
        if record_status_border and border_frames is not None and len(border_frames) > 0:
            border_mp4 = os.path.join(
                eval_videos_dir,
                f"eval_iteration_{iteration_num}{suffix}_status_border.mp4",
            )
            h_b, w_b = border_frames[0].shape[:2]
            border_width = 30
            fourcc_border = cv2.VideoWriter_fourcc(*"mp4v")
            vw_border = cv2.VideoWriter(
                border_mp4,
                fourcc_border,
                fps,
                (w_b + 2 * border_width, h_b + 2 * border_width),
            )
            # Pad statuses if needed
            if len(damage_status_series) < len(border_frames):
                damage_status_series.extend(
                    ["none"] * (len(border_frames) - len(damage_status_series))
                )
            last_bordered = None
            for f, status in zip(border_frames, damage_status_series):
                s = (status or "none").lower()
                if s in ("major", "critical"):
                    bgr = (0, 0, 255)  # Red
                elif s == "minor":
                    bgr = (0, 255, 255)  # Yellow
                else:
                    bgr = (0, 255, 0)  # Green
                bordered = cv2.copyMakeBorder(
                    f,
                    border_width,
                    border_width,
                    border_width,
                    border_width,
                    cv2.BORDER_CONSTANT,
                    value=bgr,
                )
                last_bordered = np.ascontiguousarray(bordered, dtype=np.uint8)
                vw_border.write(last_bordered)

            # Replay the last bordered frame for ~2 seconds (60 frames at 30 FPS)
            if last_bordered is not None:
                for _ in range(60):
                    vw_border.write(last_bordered)
            vw_border.release()
            print(f"Saved status-border sim video: {os.path.basename(border_mp4)}")
        # Build health-over-time MP4 using Matplotlib (match mech_damage.py style)
        health_mp4 = os.path.join(
            eval_videos_dir,
            f"eval_iteration_{iteration_num}{suffix}_health.mp4",
        )
        # Extend by 2s hold
        if len(mug_health_series) > 0:
            mug_health_series_ext = mug_health_series + [mug_health_series[-1]] * 60
        else:
            mug_health_series_ext = [100.0] * 60
        if len(laptop_health_series) > 0:
            laptop_health_series_ext = laptop_health_series + [laptop_health_series[-1]] * 60
        else:
            laptop_health_series_ext = [100.0] * 60
        T = max(len(mug_health_series_ext), len(laptop_health_series_ext))
        y_min, y_max = 0.0, 100.0
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_mug, = ax.plot([], [], lw=6, color='tab:orange', label='Mug')
        line_laptop, = ax.plot([], [], lw=6, color='tab:blue', label='Laptop')
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
            line_mug.set_data([], [])
            line_laptop.set_data([], [])
            return line_mug, line_laptop

        def animate_health(i):
            x = [k / fps for k in range(1, i + 2)]
            y_mug = mug_health_series_ext[: i + 1]
            y_laptop = laptop_health_series_ext[: i + 1]
            line_mug.set_data(x[: len(y_mug)], y_mug)
            line_laptop.set_data(x[: len(y_laptop)], y_laptop)
            return line_mug, line_laptop

        ani = animation.FuncAnimation(fig, animate_health, init_func=init_health, frames=T, interval=1000 / fps, blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(health_mp4, writer=writer)
        plt.close(fig)

        # Side-by-side: sim video + health plot using ffmpeg like mech_damage.py
        with_health_path = os.path.join(
            eval_videos_dir,
            f"eval_iteration_{iteration_num}{suffix}_with_health.mp4",
        )
        subprocess.run([
            'ffmpeg', '-y',
            '-i', sim_path,
            '-i', health_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            with_health_path
        ], check=True)
        # Keep health_mp4 for alt camera videos, will clean up later
        # Build reward-over-time MP4 using Matplotlib (cumulative episode reward)
        reward_mp4 = os.path.join(
            eval_videos_dir,
            f"eval_iteration_{iteration_num}{suffix}_reward.mp4",
        )
        if len(reward_series) > 0:
            reward_series_ext = reward_series + [reward_series[-1]] * 60
        else:
            reward_series_ext = [0.0] * 60
        T_r = len(reward_series_ext)
        # Determine y-limits robustly
        min_r = float(np.min(reward_series_ext)) if len(reward_series_ext) > 0 else 0.0
        max_r = float(np.max(reward_series_ext)) if len(reward_series_ext) > 0 else 1.0
        rng = max(1e-5, max_r - min_r)
        y_min_r = min_r - 0.1 * rng
        y_max_r = max_r + 0.1 * rng
        fig_r, ax_r = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax_r.plot([], [], lw=6, color='tab:green', label='Cumulative Reward')
        ax_r.set_xlim(0, max(1, T_r) / fps)
        ax_r.set_ylim(y_min_r, y_max_r)
        ax_r.set_xlabel('Time (s)', fontsize=20)
        ax_r.set_ylabel('Cumulative Reward', fontsize=20)
        ax_r.set_title('Total Episode Reward Over Time', fontsize=26)
        ax_r.legend(loc='best', fontsize=16)
        ax_r.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax_r.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_reward():
            line_r.set_data([], [])
            return line_r,

        def animate_reward(i):
            x = [k / fps for k in range(1, i + 2)]
            y = reward_series_ext[: i + 1]
            line_r.set_data(x, y)
            return line_r,

        ani_r = animation.FuncAnimation(fig_r, animate_reward, init_func=init_reward, frames=T_r, interval=1000 / fps, blit=True)
        writer_r = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani_r.save(reward_mp4, writer=writer_r)
        plt.close(fig_r)

        # Side-by-side: sim video + reward plot
        with_reward_path = os.path.join(
            eval_videos_dir,
            f"eval_iteration_{iteration_num}{suffix}_with_reward.mp4",
        )
        subprocess.run([
            'ffmpeg', '-y',
            '-i', sim_path,
            '-i', reward_mp4,
            '-filter_complex',
            '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            with_reward_path
        ], check=True)
        # Keep the reward plot video (clean up intermediate reward_mp4 only)
        try:
            os.remove(reward_mp4)
        except Exception:
            pass
        # Keep with_reward_path for final output
        # Generate water contacts plot
        water_mp4 = generate_water_contacts_plot(laptop_contact_counts, fps, eval_videos_dir, iteration_num, suffix)
        
        # Side-by-side: sim video + water contacts plot
        with_water_path = os.path.join(
            eval_videos_dir,
            f"eval_iteration_{iteration_num}{suffix}_with_water.mp4",
        )
        if os.path.exists(sim_path) and os.path.exists(water_mp4):
            subprocess.run([
                'ffmpeg', '-y',
                '-i', sim_path,
                '-i', water_mp4,
                '-filter_complex',
                '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
                '-map', '[v]',
                '-c:v', 'mpeg4',
                '-q:v', '5',
                with_water_path
            ], check=True)
            try:
                os.remove(water_mp4)
            except Exception:
                pass
        else:
            with_water_path = None
        
        # Clean up intermediate files
        try:
            os.remove(health_mp4)
        except Exception:
            pass
        # Now remove the raw sim AVI after all composites are handled
        try:
            os.remove(sim_path)
        except Exception:
            pass
        health_avi_path = with_health_path
        water_avi_path = with_water_path
        reward_avi_path = with_reward_path
        print(f"Evaluation video with health saved: {os.path.basename(health_avi_path)}")
        if water_avi_path is not None:
            print(f"Evaluation video with water contacts saved: {os.path.basename(water_avi_path)}")
        if reward_avi_path is not None:
            print(f"Evaluation video with reward plot saved: {os.path.basename(reward_avi_path)}")
        return total_reward, final_terminated, final_truncated, health_avi_path, water_avi_path, reward_avi_path, final_mug_health, final_laptop_health, final_plate_health, steps_taken
    elif vw is not None and not should_save_video:
        # Clean up video writers if we're not saving
        vw.release()
        try:
            os.remove(avi_path)
        except Exception:
            pass
        return total_reward, final_terminated, final_truncated, None, None, None, final_mug_health, final_laptop_health, final_plate_health, steps_taken
    else:
        return total_reward, final_terminated, final_truncated, None, None, None, final_mug_health, final_laptop_health, final_plate_health, steps_taken


def generate_water_contacts_plot(laptop_contact_counts, fps, out_dir, iteration_num, suffix):
    """Generate animated plot of water particle contacts for laptop."""
    water_mp4 = os.path.join(
        out_dir,
        f"eval_iteration_{iteration_num}{suffix}_water.mp4",
    )
    
    # Prepare data series
    contact_series = laptop_contact_counts if len(laptop_contact_counts) > 0 else [0]
    max_len = len(contact_series)
    
    # Extend by 2s hold (60 frames at 30fps)
    if len(contact_series) > 0:
        contact_series_ext = contact_series + [contact_series[-1]] * 60
    else:
        contact_series_ext = [0] * 60
    
    T = len(contact_series_ext)
    
    # Create figure with professional styling
    fig_w, ax_w = plt.subplots(figsize=(9.6, 5.4))
    
    line_laptop, = ax_w.plot([], [], lw=6, color='tab:cyan', label='Laptop water contacts')
    
    # Configure axes
    ax_w.set_xlim(0, max(1, T) / fps)
    y_min_w = 0.0
    y_max_w = max(float(max(contact_series_ext + [0])), 1.0)
    if y_min_w == y_max_w:
        y_min_w, y_max_w = (0.0, 1.0)
    ax_w.set_ylim(y_min_w, y_max_w * 1.1)
    ax_w.set_xlabel('Time (s)', fontsize=20)
    ax_w.set_ylabel('Water particle contacts', fontsize=20)
    ax_w.set_title('Laptop Water Particle Contacts Over Time', fontsize=26)
    
    # Add damage threshold line (using typical laptop threshold from electrical_damage.py)
    damage_threshold = 30.0
    ax_w.axhline(y=damage_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                 label=f'Damage Threshold ({damage_threshold} particles)')
    
    ax_w.legend(loc='best', fontsize=16)
    ax_w.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    ax_w.grid(True, linewidth=1.0, alpha=0.3)
    plt.tight_layout()

    # Animation functions
    def init_w():
        line_laptop.set_data([], [])
        return line_laptop,

    def animate_w(i):
        x = [k / fps for k in range(1, i + 2)]
        y_laptop = contact_series_ext[: i + 1]
        line_laptop.set_data(x[: len(y_laptop)], y_laptop)
        return line_laptop,

    # Create and save animation
    ani_w = animation.FuncAnimation(fig_w, animate_w, init_func=init_w, frames=T, 
                                    interval=1000 / fps, blit=True)
    writer_w = animation.FFMpegWriter(fps=fps, codec='mpeg4', 
                                       extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
    ani_w.save(water_mp4, writer=writer_w)
    plt.close(fig_w)
    
    return water_mp4


def run_eval(env, agent, device, eval_videos_dir, iteration_num, num_episodes=5):
    task_successes = 0
    damage_failures = 0
    safe_successes = 0
    total_returns = []
    successful_steps = []  # Steps for episodes that succeed the task
    saved_videos_health = []
    saved_videos_water = []
    saved_videos_reward = []
    episode_damages = []
    for i in range(num_episodes):
        capture = (i < 5)  # Save videos for all episodes
        ep_ret, terminated, truncated, health_avi_path, water_avi_path, reward_avi_path, mug_health, laptop_health, plate_health, steps_taken = run_eval_episode(
            env, agent, device, eval_videos_dir, iteration_num, capture_video=capture, episode_idx=i
        )
        total_returns.append(ep_ret)
        if capture and health_avi_path is not None:
            saved_videos_health.append(health_avi_path)
        if capture and water_avi_path is not None:
            saved_videos_water.append(water_avi_path)
        if capture and reward_avi_path is not None:
            saved_videos_reward.append(reward_avi_path)
        
        # Task success: terminated is True AND reward is positive (distance reward gives +300 on success)
        task_success = bool(terminated) and float(ep_ret) > 0
        if task_success:
            task_successes += 1
            successful_steps.append(steps_taken)
        
        # Per-episode total damage (max 300 if all three go to 0 health)
        try:
            dm = (100.0 - float(mug_health)) + (100.0 - float(laptop_health)) + (100.0 - float(plate_health))
        except Exception:
            dm = 0.0
        dm = max(0.0, min(300.0, dm))
        episode_damages.append(dm)

        # Damage failure: any object's health is 0 (would trigger damage termination)
        if float(mug_health) < 100.0 or float(laptop_health) < 100.0 or float(plate_health) < 100.0:
            damage_failures += 1
        # Safe success: task success AND no damage to any object (all healths exactly 100)
        try:
            if (
                task_success
                and float(mug_health) == 100.0
                and float(laptop_health) == 100.0
                and float(plate_health) == 100.0
            ):
                safe_successes += 1
        except Exception:
            pass
    
    task_success_rate = task_successes / float(max(1, num_episodes))
    damage_failure_rate = damage_failures / float(max(1, num_episodes))
    safe_success_rate = safe_successes / float(max(1, num_episodes))
    
    # Average steps to solve: only count successful runs, use max episode steps (300) for failed runs
    if len(successful_steps) > 0:
        avg_steps_to_solve = float(np.mean(successful_steps))
    else:
        avg_steps_to_solve = float(EVAL_MAX_STEPS)  # All failed, use max steps
    
    # Keep old success_rate variable for backward compatibility (same as task_success_rate)
    success_rate = task_success_rate
    avg_return = float(np.mean(total_returns)) if len(total_returns) > 0 else 0.0
    mean_total_damage = float(np.mean(episode_damages)) if len(episode_damages) > 0 else 0.0
    # Combine all captured episodes into a single video (health plot version)
    combined_health_path = None
    if len(saved_videos_health) >= 1:
        try:
            import cv2
            # Get video properties from first video
            cap_first = cv2.VideoCapture(saved_videos_health[0])
            fps = cap_first.get(cv2.CAP_PROP_FPS) or 15
            w = int(cap_first.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(cap_first.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
            cap_first.release()
            
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            combined_health_path = os.path.join(eval_videos_dir, f"eval_iteration_{iteration_num}_health.avi")
            out = cv2.VideoWriter(combined_health_path, fourcc, fps, (w, h))
            
            # Write all episodes sequentially
            for video_path in saved_videos_health:
                cap = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                cap.release()
            
            out.release()
            
            # Cleanup individual episode videos
            for p in saved_videos_health:
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: Failed to combine eval health videos: {e}")
            pass
    
    # Combine all captured episodes into a single video (water contacts plot version)
    combined_water_path = None
    if len(saved_videos_water) >= 1:
        try:
            import cv2
            # Get video properties from first video
            cap_first = cv2.VideoCapture(saved_videos_water[0])
            fps = cap_first.get(cv2.CAP_PROP_FPS) or 15
            w = int(cap_first.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(cap_first.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
            cap_first.release()
            
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            combined_water_path = os.path.join(eval_videos_dir, f"eval_iteration_{iteration_num}_water.avi")
            out = cv2.VideoWriter(combined_water_path, fourcc, fps, (w, h))
            
            # Write all episodes sequentially
            for video_path in saved_videos_water:
                cap = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                cap.release()
            
            out.release()
            
            # Cleanup individual episode videos
            for p in saved_videos_water:
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: Failed to combine eval water videos: {e}")
            pass
    
    # Combine all captured episodes into a single video (reward plot version)
    combined_reward_path = None
    if len(saved_videos_reward) >= 1:
        try:
            import cv2
            # Get video properties from first video
            cap_first = cv2.VideoCapture(saved_videos_reward[0])
            fps = cap_first.get(cv2.CAP_PROP_FPS) or 15
            w = int(cap_first.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(cap_first.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
            cap_first.release()
            
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            combined_reward_path = os.path.join(eval_videos_dir, f"eval_iteration_{iteration_num}_reward.avi")
            out = cv2.VideoWriter(combined_reward_path, fourcc, fps, (w, h))
            
            # Write all episodes sequentially
            for video_path in saved_videos_reward:
                cap = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                cap.release()
            
            out.release()
            
            # Cleanup individual episode videos
            for p in saved_videos_reward:
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: Failed to combine eval reward videos: {e}")
            pass
    
    saved_videos = []
    if combined_health_path is not None:
        saved_videos.append(combined_health_path)
    if combined_water_path is not None:
        saved_videos.append(combined_water_path)
    if combined_reward_path is not None:
        saved_videos.append(combined_reward_path)
    
    return success_rate, avg_return, saved_videos, task_success_rate, damage_failure_rate, avg_steps_to_solve, mean_total_damage, safe_success_rate


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
    # Initialize WandB (always on) - let WandB generate a new unique ID automatically
    wandb.init(project="omnigibson-ppo", name=run_name, config=vars(args), id=None)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create eval videos directory
    eval_videos_dir = "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/eval_videos"
    os.makedirs(eval_videos_dir, exist_ok=True)
    
    # env setup
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
    
    # Track best eval average return for checkpoint saving
    best_eval_avg_return = None
    
    # Initialize offline_replay to None so later checks are safe
    offline_replay = None
    
    # Offline pretraining from dataset:
    # (A) If a BC-only initialization checkpoint exists (from the BC eval script),
    #     load it and skip BC pretraining / critic replay.
    # (B) Otherwise, optionally run BC pretraining as usual (no offline critic replay).
    if os.path.exists(BC_INIT_CHECKPOINT):
        print(f"[Offline] Loading BC init checkpoint from {BC_INIT_CHECKPOINT} (skipping BC pretraining).")
        bc_ckpt = torch.load(BC_INIT_CHECKPOINT, map_location=device)
        agent.load_state_dict(bc_ckpt["agent_state_dict"])
    else:
        if args.bc_pretrain_enabled:
            run_offline_bc_pretrain(args, agent, optimizer, device)

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
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
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

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_return = info["episode"]["r"]
                        print(f"global_step={global_step}, episodic_return={ep_return}")
                        wandb.log({
                            "charts/episodic_return": ep_return,
                            "charts/episodic_length": info["episode"]["l"],
                            "global_step": global_step,
                        })

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

        # Record metrics to WandB
        wandb.log({
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        
        # Evaluation: run every 20 iterations
        if iteration % 5 == 0:
            print(f"Running evaluation after iteration {iteration}...")
            # IMPORTANT: Reuse the SAME underlying env to avoid multiple global simulators
            try:
                base_env = envs.envs[0]
                while hasattr(base_env, "env"):
                    base_env = base_env.env
            except Exception:
                base_env = None
            if base_env is not None:
                # Set curriculum group to 4 (eval) before evaluation
                if hasattr(base_env, "set_curriculum_group"):
                    base_env.set_curriculum_group(global_step, force_eval=True)
                # Extra safety resets around evaluation to prevent stale simulator state
                base_env.reset()
                (
                    success_rate,
                    eval_avg_return,
                    _,
                    task_success_rate,
                    damage_failure_rate,
                    avg_steps_to_solve,
                    mean_total_damage,
                    safe_success_rate,
                ) = run_eval(base_env, agent, device, eval_videos_dir, iteration, num_episodes=5)
                
                # Save checkpoint if first eval or better than previous best
                if best_eval_avg_return is None or eval_avg_return > best_eval_avg_return:
                    best_eval_avg_return = eval_avg_return
                    torch.save({
                        'agent_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_eval_avg_return': best_eval_avg_return,
                        'iteration': iteration,
                        'global_step': global_step,
                    }, checkpoint_file)
                    print(f"Saved new best checkpoint with avg return: {best_eval_avg_return:.3f}")
                else:
                    print(f"Eval avg return {eval_avg_return:.3f} not better than best {best_eval_avg_return:.3f}, not saving checkpoint")
                
                # Log evaluation metrics
                wandb.log({
                    "eval_iteration": iteration,
                    "eval_success_rate": success_rate,
                    "eval_avg_return": eval_avg_return,
                    "eval_task_success_rate": task_success_rate,
                    "eval_damage_failure_rate": damage_failure_rate,
                    "eval_avg_steps_to_solve": avg_steps_to_solve,
                    "eval_mean_total_damage": mean_total_damage,
                    "eval_safe_success_rate": safe_success_rate,
                })
                print(
                    f"Eval - success rate: {success_rate:.3f}, avg return: {eval_avg_return:.3f}, "
                    f"task success: {task_success_rate:.3f}, damage failure: {damage_failure_rate:.3f}, "
                    f"avg steps: {avg_steps_to_solve:.1f}, mean total damage: {mean_total_damage:.3f}, "
                    f"safe success rate: {safe_success_rate:.3f}"
                )
                # Reset curriculum group back to training group after evaluation
                # (envs.reset() below will call reset() on base_env, which will use the training group)
                if hasattr(base_env, "set_curriculum_group"):
                    base_env.set_curriculum_group(global_step, force_eval=False)
                # Resync vector env state after direct base env usage
                # Note: envs.reset() will call reset() on base_env, which will use training group with potential random fallback
                next_obs, _ = envs.reset(seed=None)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)
            
            # After each evaluation, restart simulator and environments to mitigate host memory / GPU growth.
            # IMPORTANT: We intentionally DO NOT call og.shutdown() here, because that tears down
            # the entire Simulation App and exits the process. Instead we close the gym envs and
            # rely on make_env (which internally uses og.clear()/og.sim.stop()) to refresh sim state.
            if iteration < args.num_iterations:
                print(f"Restarting simulator and environments after evaluation at iteration {iteration} to free memory...", flush=True)
                # Best-effort close of current vector envs
                try:
                    envs.close()
                except Exception:
                    pass

                # Recreate environments with the same configuration
                envs = gym.vector.SyncVectorEnv([make_env(scene_file, 0, args.capture_video, run_name)])
                assert isinstance(envs.single_action_space, gym.spaces.Box)

                # Re-apply curriculum interval and current training group on the new base env
                try:
                    base_env = envs.envs[0]
                    while hasattr(base_env, "env"):
                        base_env = base_env.env
                    if hasattr(base_env, "set_curriculum_interval"):
                        base_env.set_curriculum_interval(args.curriculum_interval)
                    if hasattr(base_env, "set_curriculum_group"):
                        base_env.set_curriculum_group(global_step, force_eval=False)
                except Exception:
                    pass

                # Reset new envs and resync rollout state tensors
                next_obs, _ = envs.reset(seed=None)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)

    # Final evaluation at end of training - load best checkpoint
    base_env = envs.envs[0]
    while hasattr(base_env, "env"):
        base_env = base_env.env
    
    # Load best checkpoint for final evaluation
    if os.path.exists(checkpoint_file):
        print(f"Loading best checkpoint from {checkpoint_file} for final evaluation...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        print(f"Loaded checkpoint from iteration {checkpoint.get('iteration', 'unknown')} with best avg return: {checkpoint.get('best_eval_avg_return', 'unknown')}")
    else:
        print("No checkpoint found, using current agent for final evaluation...")
    
    print("Running final evaluation at end of training...")
    # Set curriculum group to 4 (eval) for final evaluation
    if hasattr(base_env, "set_curriculum_group"):
        base_env.set_curriculum_group(global_step, force_eval=True)
    base_env.reset()
    (
        success_rate,
        eval_avg_return,
        _,
        task_success_rate,
        damage_failure_rate,
        avg_steps_to_solve,
        mean_total_damage,
        safe_success_rate,
    ) = run_eval(base_env, agent, device, eval_videos_dir, args.num_iterations, num_episodes=10)
    wandb.log({
        "eval_iteration": args.num_iterations,
        "eval_success_rate": success_rate,
        "eval_avg_return": eval_avg_return,
        "eval_task_success_rate": task_success_rate,
        "eval_damage_failure_rate": damage_failure_rate,
        "eval_avg_steps_to_solve": avg_steps_to_solve,
        "eval_mean_total_damage": mean_total_damage,
        "eval_safe_success_rate": safe_success_rate,
        "final_eval": True,
    })
    print(
        f"Final Eval - success rate: {success_rate:.3f}, avg return: {eval_avg_return:.3f}, "
        f"task success: {task_success_rate:.3f}, damage failure: {damage_failure_rate:.3f}, "
        f"avg steps: {avg_steps_to_solve:.1f}, mean total damage: {mean_total_damage:.3f}, "
        f"safe success rate: {safe_success_rate:.3f}"
    )

    envs.close()
    wandb.finish()
    og.shutdown()


