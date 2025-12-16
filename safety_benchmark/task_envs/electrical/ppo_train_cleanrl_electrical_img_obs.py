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

DEFAULT_SCENE_FILE = "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/curriculum_learning_2.json"
# Match training episode horizon for evaluation
EVAL_MAX_STEPS = 300


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
    total_timesteps: int = 100000
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
    update_epochs: int = 4
    """the K epochs to update the policy (slightly reduced to avoid over-updating from a single rollout)"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient (smaller for more conservative policy updates and less forgetting)"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.005
    """coefficient of the entropy (moderate exploration early, linearly decaying to 0 over training)"""
    vf_coef: float = 0.7
    """coefficient of the value function loss term (slightly higher weight for a more accurate critic)"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.01
    """the target KL divergence threshold (tighter to prevent large policy shifts after success)"""
    curriculum_interval: int = 10000
    """number of steps per curriculum group (group 1: 0-interval, group 2: interval-2*interval, group 3: 2*interval-3*interval, group 4: 3*interval+)"""

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
            # Expose only 3D positional deltas to the agent; keep orientation fixed by zeroing those DOFs
            self._agent_act_dim = min(3, self._arm_input_dim)
            self._arm_action_scale = 0.05
            self._episode_step = 0
            # Episode horizon tuned to allow reasonably direct paths without excessive flailing
            self._max_episode_steps = 300
            self._last_base_obs = None
            self._initial_distance = None
            self._curriculum_group = 1
            self._is_eval = False
            self._curriculum_interval = None  # Will be set via set_curriculum_interval
            self._ego_key = None  # Cache ego camera key

            # Image observation space: 256x256 RGB images, normalized to [0, 1]
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(256, 256, 3), dtype=np.float32)
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
            for damage_type, damage_info in mug_health_states["damage_info"].items():
                for link_name, damage in damage_info.items():
                    total_damage += damage
            for damage_type, damage_info in laptop_health_states["damage_info"].items():
                for link_name, damage in damage_info.items():
                    total_damage += damage
            if total_damage == 0.0 and mug_health_states["health"] == 100.0 and laptop_health_states["health"] == 100.0:
                return 0.01, False
            if mug_health_states["health"] == 0.0 or laptop_health_states["health"] == 0.0:
                return -300, True
            return -total_damage, False

        def _distance_reward_fn(self, env, obs):
            eps_pos = 0.15
            mug = env.scene.object_registry("name", "mug")
            table = env.scene.object_registry("name", "breakfast_table")
            mug_pos, mug_quat = mug.get_position_orientation()
            mug_pos_t = torch.as_tensor(mug_pos, dtype=torch.float32)
            target_pos = torch.tensor([-0.4, 0.0, 0.55], dtype=torch.float32)
            xy_dist = torch.norm(mug_pos_t[:2] - target_pos[:2]).item()
            done = mug.states[OnTop].get_value(table) and (xy_dist < eps_pos)
            z_dist = 0.0 if mug_pos_t[2] >= target_pos[2] - 0.1 and mug_pos_t[2] <= target_pos[2] + 0.1 else abs(target_pos[2] - mug_pos_t[2]) * 10.0
            if xy_dist > 0.5:
                xy_dist *= 3.0
            current_distance = xy_dist + z_dist
            distance_reward = -current_distance
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
            # Get ego camera RGB observation from robot
            rob_obs = self._robot.get_obs()[0]
            
            # Find ego camera key if not cached
            if self._ego_key is None:
                for k in rob_obs.keys():
                    if ":eyes:Camera:0" in k and "rgb" in rob_obs[k]:
                        self._ego_key = k
                        break
            
            # Extract RGB image
            if self._ego_key is not None and self._ego_key in rob_obs:
                ego_rgb = rob_obs[self._ego_key]["rgb"]
                # Convert to numpy and extract RGB channels (H, W, 3)
                ego_np = ego_rgb.cpu().numpy()[:, :, :3]
                # Resize to 256x256 using cv2
                ego_resized = cv2.resize(ego_np, (256, 256))
                # Normalize to [0, 1] range (assuming input is already in [0, 1] or [0, 255])
                if ego_resized.max() > 1.0:
                    ego_resized = ego_resized / 255.0
                return ego_resized.astype(np.float32)
            else:
                # Fallback: return black image if camera not found
                return np.zeros((256, 256, 3), dtype=np.float32)

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
            # Set robot head / camera joints to look down at the workspace (match mech_damage.py)
            try:
                if hasattr(self._robot, "camera_control_idx"):
                    self._robot.set_joint_positions(
                        torch.tensor([-0.5, -0.75], dtype=torch.float32),
                        indices=self._robot.camera_control_idx,
                    )
            except Exception:
                # Fail fast policy elsewhere; here we just skip if camera joints are unavailable
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
                # Expose only 3D positional deltas to the agent; keep orientation fixed by zeroing those DOFs
                self._agent_act_dim = min(3, self._arm_input_dim)
                self._arm_action_scale = 0.05
                self._max_episode_steps = 300
                self._ego_key = None  # Reset ego camera key cache
                self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(256, 256, 3), dtype=np.float32)
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

            if getattr(self, "_is_eval", True):
                x_delta = 0.0
                z_delta = 0.25
                _exec(torch.tensor([0.0, 0.0, z_delta]))
                _exec(torch.tensor([x_delta, 0.0, 0.0]))
            else:
                x_delta = np.random.uniform(-0.3, 0.0)
                y_delta = np.random.uniform(-0.2, 0.2)
                z_delta = np.random.uniform(0.0, 0.25)
                _exec(torch.tensor([0.0, 0.0, z_delta]))
                _exec(torch.tensor([0.0, y_delta, 0.0]))
                _exec(torch.tensor([x_delta, 0.0, 0.0]))

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
            # Map agent's 3D positional control into the full IK input; zero orientation DOFs
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
            mug_obj = self.get_target_obj()
            if mug_obj is not None:
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
        act_dim = int(np.array(envs.single_action_space.shape).prod())
        
        # CNN encoder for image observations (256x256x3)
        # Input: (batch, 3, 256, 256) - need to permute from (H, W, C) to (C, H, W)
        self.cnn = nn.Sequential(
            # First conv block: 256x256 -> 128x128
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Second conv block: 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Third conv block: 64x64 -> 32x32
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Fourth conv block: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Fifth conv block: 16x16 -> 8x8
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Adaptive pooling to get fixed-size feature vector
            nn.AdaptiveAvgPool2d((4, 4)),
            # Flatten: 128 * 4 * 4 = 2048
            nn.Flatten(),
        )
        
        # Feature dimension after CNN
        feature_dim = 128 * 4 * 4  # 2048
        
        # Critic head: CNN features -> value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        
        # Actor head: CNN features -> action mean
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )
        
        # Start with moderate exploration
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def _encode_image(self, x):
        """Encode image observation through CNN.
        
        Args:
            x: Image tensor of shape (batch, H, W, C) or (H, W, C)
        
        Returns:
            Feature vector of shape (batch, feature_dim)
        """
        # Convert from (H, W, C) to (C, H, W) if needed
        if x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
        elif x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # (batch, H, W, C) -> (batch, C, H, W)
        # x should now be (batch, C, H, W)
        return self.cnn(x)

    def get_value(self, x):
        features = self._encode_image(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self._encode_image(x)
        mean = self.actor_mean(features)
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
        return squashed, logprob, entropy, self.critic(features)


def run_eval_episode(env, agent, device, eval_videos_dir, iteration_num, capture_video=True, episode_idx=None):
    """Run a single evaluation episode and optionally save an AVI with on-frame cumulative reward text.
    
    Returns (total_reward, terminated, truncated, health_avi_path_or_None, water_avi_path_or_None, reward_avi_path_or_None, final_mug_health, final_laptop_health, final_plate_health)
    """
    fps = 30
    
    # Reset environment
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    
    # Get laptop and its electrical damage evaluator for water contact tracking
    base = env
    while hasattr(base, "env"):
        base = base.env
    laptop = None
    laptop_eval = None
    robot = None
    ego_key = None
    if hasattr(base, "_env"):
        try:
            laptop = base._env.scene.object_registry("name", "laptop")
            if laptop is not None:
                existing_evals = getattr(laptop, "damage_evaluators", [])
                for ev in existing_evals:
                    if isinstance(ev, ElectricalDamageEvaluator):
                        laptop_eval = ev
                        break
        except Exception:
            pass
    # Try to get the robot for ego camera rendering
    if hasattr(base, "_robot"):
        robot = base._robot
    
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
    
    # Run evaluation episode
    for step in range(EVAL_MAX_STEPS):
        with torch.no_grad():
            # Use deterministic mean action for evaluation (no sampling).
            # For image observations, encode through the CNN first.
            features = agent._encode_image(obs_tensor.unsqueeze(0))
            mean = agent.actor_mean(features)
            action = torch.tanh(mean)  # Squash to [-1, 1] like in training
        action_np = action.squeeze(0).cpu().numpy()
        
        obs, reward, terminated, truncated, info = env.step(action_np)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        
        total_reward += float(reward)
        reward_series.append(total_reward)
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
            # Capture from the chosen main viewer camera angle for this episode (no ego overlay)
            try:
                og.sim.viewer_camera.set_position_orientation(position=main_cam_pos, orientation=main_cam_quat)
                og.sim.render()
            except Exception:
                pass
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
            rgb_np = rgb.cpu().numpy()[:, :, :3]
            frame = cv2.resize(rgb_np, (1080, 720))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            text = f"Reward: {total_reward:.2f}"
            cv2.putText(frame_bgr, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            vw.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8))
        
        if terminated or truncated:
            final_terminated = bool(terminated)
            final_truncated = bool(truncated)
            break
    
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
        sim_path = avi_path
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
        return total_reward, final_terminated, final_truncated, health_avi_path, water_avi_path, reward_avi_path, final_mug_health, final_laptop_health, final_plate_health
    elif vw is not None and not should_save_video:
        # Clean up video writers if we're not saving
        vw.release()
        try:
            os.remove(avi_path)
        except Exception:
            pass
        return total_reward, final_terminated, final_truncated, None, None, None, final_mug_health, final_laptop_health, final_plate_health
    else:
        return total_reward, final_terminated, final_truncated, None, None, None, final_mug_health, final_laptop_health, final_plate_health


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
    successes = 0
    total_returns = []
    saved_videos_health = []
    saved_videos_water = []
    saved_videos_reward = []
    for i in range(num_episodes):
        capture = (i < 5)  # Save videos for all episodes
        ep_ret, terminated, truncated, health_avi_path, water_avi_path, reward_avi_path, mug_health, laptop_health, plate_health = run_eval_episode(
            env, agent, device, eval_videos_dir, iteration_num, capture_video=capture, episode_idx=i
        )
        total_returns.append(ep_ret)
        if capture and health_avi_path is not None:
            saved_videos_health.append(health_avi_path)
        if capture and water_avi_path is not None:
            saved_videos_water.append(water_avi_path)
        if capture and reward_avi_path is not None:
            saved_videos_reward.append(reward_avi_path)
        # Success criterion: terminated and reward > 10 and mug health >= 90
        # if bool(terminated):
        #     successes += 1
        if bool(terminated) and float(mug_health) == 100 and float(laptop_health) == 100 and float(plate_health) == 100:
            successes += 1
        # if float(mug_health) == 100.0:
        #     successes += 1
    success_rate = successes / float(max(1, num_episodes))
    avg_return = float(np.mean(total_returns)) if len(total_returns) > 0 else 0.0
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
    
    return success_rate, avg_return, saved_videos


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

        # Linearly decay the entropy coefficient from its initial value to 0 over training
        ent_frac = 1.0 - (iteration - 1.0) / args.num_iterations
        current_ent_coef = args.ent_coef * ent_frac

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
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

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
        if iteration % 20 == 0:
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
                success_rate, eval_avg_return, _ = run_eval(base_env, agent, device, eval_videos_dir, iteration, num_episodes=5)
                # Log evaluation metrics
                wandb.log({
                    "eval_iteration": iteration,
                    "eval_success_rate": success_rate,
                    "eval_avg_return": eval_avg_return,
                })
                print(f"Eval success rate: {success_rate:.3f}, avg return: {eval_avg_return:.3f}")
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

    # Final evaluation at end of training
    base_env = envs.envs[0]
    while hasattr(base_env, "env"):
        base_env = base_env.env
    print("Running final evaluation at end of training...")
    # Set curriculum group to 4 (eval) for final evaluation
    if hasattr(base_env, "set_curriculum_group"):
        base_env.set_curriculum_group(global_step, force_eval=True)
    base_env.reset()
    success_rate, eval_avg_return, _ = run_eval(base_env, agent, device, eval_videos_dir, args.num_iterations, num_episodes=10)
    wandb.log({
        "eval_iteration": args.num_iterations,
        "eval_success_rate": success_rate,
        "eval_avg_return": eval_avg_return,
    })

    envs.close()
    wandb.finish()