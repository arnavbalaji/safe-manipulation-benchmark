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

# OmniGibson + environment wrapper (minimal integration)
import omnigibson as og


def _ensure_omnigibson_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    og_src = os.path.join(repo_root, "OmniGibson")
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


_ensure_omnigibson_on_path()

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.object_states import OnTop

DEFAULT_SCENE_FILE = "lift_test.json"
EVAL_MAX_STEPS = 100


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
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 500
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 10
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.01
    """the target KL divergence threshold"""

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
            self._agent_arm_dim = int(self._right_arm_idx.shape[0])
            self._agent_act_dim = int(self._agent_arm_dim + 1)
            self._arm_action_scale = 0.05
            self._episode_step = 0
            self._max_episode_steps = 100
            self._last_base_obs = None

            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-np.ones((self._agent_act_dim,), dtype=np.float32), high=np.ones((self._agent_act_dim,), dtype=np.float32), dtype=np.float32)

        def _get_plate_obj(self):
            # Robust lookup across possible names
            for name in ["glass_plate", "target_object", "plate"]:
                try:
                    obj = self._env.scene.object_registry("name", name)
                except Exception:
                    obj = None
                if obj is not None:
                    return obj
            return None

        # def _reward_fn(self, env, obs):
        #     # Try multiple possible names for robustness
        #     plate_obj = None
        #     for name in ["glass_plate", "target_object", "plate"]:
        #         try:
        #             plate_obj = env.scene.object_registry("name", name)
        #         except Exception:
        #             plate_obj = None
        #         if plate_obj is not None:
        #             break
        #     if plate_obj is not None:
        #         plate_pos, _ = plate_obj.get_position_orientation()
        #     else:
        #         plate_pos = torch.zeros(3)
        #     target_pos = torch.tensor([-0.0966, 0.0843, 0.4593])
        #     distance = torch.norm(plate_pos - target_pos).item()
        #     if distance < 0.1:
        #         return 100.0, True
        #     return -distance, False
        def _damage_reward_fn(self,env, obs):
            terminated = False
            plate_health_states = obs["object_health_states"]["glass_plate"]
            total_damage = 0.0
            for damage_type, damage_info in plate_health_states["damage_info"].items():
                for link_name, damage in damage_info.items():
                    total_damage += damage
            if total_damage == 0.0 and plate_health_states["health"] == 100.0:
                return 0.01, terminated
            if plate_health_states["health"] == 0.0:
                terminated = True
            return -total_damage, terminated

        def _distance_reward_fn(self, env, obs):
            eps = 0.2
            plate_obj = env.scene.object_registry("name", "glass_plate")
            table_obj = env.scene.object_registry("name", "coffee_table")
            plate_pos, plate_orn = plate_obj.get_position_orientation()
            target_pos = torch.tensor([-0.0966, 0.0843, 0.4593])
            distance = torch.norm(plate_pos - target_pos).item()
            if (distance < eps) and plate_obj.states[OnTop].get_value(table_obj):
                return 10.0, True
            return -distance, False

        def _reward_fn(self, env, obs):
            damage_weight = 2.0
            distance_weight = 1.0
            damage_reward, damage_terminated = self._damage_reward_fn(env, obs)
            distance_reward, distance_terminated = self._distance_reward_fn(env, obs)
            # return (damage_weight * damage_reward + distance_weight * distance_reward), damage_terminated or distance_terminated
            return damage_reward, damage_terminated

        def _extract_obs(self):
            # Safely fetch EEF position; if prim view is invalid, try to rebuild handles
            try:
                eef_pos = self._robot.get_eef_position("right")
            except Exception:
                try:
                    og.sim.play()
                except Exception:
                    pass
                try:
                    for _ in range(2):
                        og.sim.step()
                    eef_pos = self._robot.get_eef_position("right")
                except Exception:
                    eef_pos = torch.zeros(3)
            plate = self._get_plate_obj()
            if plate is not None:
                plate_pos, _ = plate.get_position_orientation()
            else:
                plate_pos = torch.zeros(3)
            v = torch.cat([torch.as_tensor(eef_pos, dtype=torch.float32).flatten(), torch.as_tensor(plate_pos, dtype=torch.float32).flatten()], dim=0)
            return v.detach().cpu().numpy().astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            # Reset base env (scene.reset under the hood)
            self._env.reset()
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
            self._env.unlock_health_changes()
            # If the plate is missing after reset, hard-reload from original scene file
            try:
                plate = self._get_plate_obj()
            except Exception:
                plate = None
            if plate is None:
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
                self._agent_arm_dim = int(self._right_arm_idx.shape[0])
                self._agent_act_dim = int(self._agent_arm_dim + 1)
                self._arm_action_scale = 0.05
                self._max_episode_steps = 100
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
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
            self._episode_step = 0
            return self._extract_obs(), {}

        def _execute_init(self):
            def _exec(delta):
                current_eef_pos = self._robot.get_eef_position("right")
                current_eef_orn = self._robot.get_eef_orientation("right")
                target_eef_pos = current_eef_pos + delta
                target_eef_pose = (target_eef_pos, current_eef_orn)
                steps = 0
                for a in self._prims._move_hand_linearly_cartesian(target_eef_pose, ignore_failure=True):
                    self._env.step(action=a)
                    steps += 1
                    if steps >= 300:
                        break
            _exec(torch.tensor([0.0, 0.0, 0.45]))
            _exec(torch.tensor([0.0, -0.15, 0.0]))

        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            full = np.zeros((self._action_dim,), dtype=np.float32)
            arm_action = action[: self._agent_arm_dim]
            grip_scalar = action[self._agent_arm_dim]
            full[self._right_arm_idx] = arm_action * self._arm_action_scale
            grip_binary = 1.0 if float(grip_scalar) >= 0.0 else -1.0
            full[self._right_grip_idx] = grip_binary
            obs, reward, terminated, truncated, info = self._env.step(action=full)
            # Cache the last base observation dict for downstream eval access
            self._last_base_obs = obs if isinstance(obs, dict) else None
            self._episode_step += 1
            if self._episode_step >= self._max_episode_steps:
                truncated = True
            return self._extract_obs(), float(reward), bool(terminated), bool(truncated), info

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
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
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


def run_eval_episode(env, agent, device, eval_videos_dir, iteration_num, capture_video=True, episode_idx=None):
    """Run a single evaluation episode and optionally save an AVI with on-frame cumulative reward text.
    
    Returns (total_reward, terminated, truncated, avi_path_or_None)
    """
    fps = 15
    
    # Reset environment
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    
    vw = None
    avi_path = None
    if capture_video:
        suffix = f"_ep{int(episode_idx)+1}" if episode_idx is not None else ""
        avi_path = os.path.join(eval_videos_dir, f"eval_iteration_{iteration_num}{suffix}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(avi_path, fourcc, fps, (640, 360))
    
    total_reward = 0.0
    final_terminated = False
    final_truncated = False
    final_plate_health = None
    # Track plate health over time for plotting alongside sim video
    health_series = []
    last_health = 100.0
    # Track cumulative reward over time for plotting
    reward_series = []
    
    # Run evaluation episode
    for step in range(EVAL_MAX_STEPS):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor.unsqueeze(0))
        action_np = action.squeeze(0).cpu().numpy()
        
        obs, reward, terminated, truncated, info = env.step(action_np)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        
        total_reward += float(reward)
        reward_series.append(total_reward)
        # Record plate health per step from the base env's observation dict
        base = env
        while hasattr(base, "env"):
            base = base.env
        base_obs = getattr(base, "_last_base_obs", None)
        if isinstance(base_obs, dict):
            phs = base_obs.get("object_health_states", {}).get("glass_plate", None)
            if phs is not None and ("health" in phs):
                last_health = float(phs["health"])
                final_plate_health = last_health
        health_series.append(last_health)
        
        if vw is not None:
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
            rgb_np = rgb.cpu().numpy()[:, :, :3]
            frame = cv2.resize(rgb_np, (640, 360))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            text = f"Reward: {total_reward:.2f}"
            cv2.putText(frame_bgr, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            vw.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8))
        
        if terminated or truncated:
            final_terminated = bool(terminated)
            final_truncated = bool(truncated)
            break
    
    if vw is not None:
        # Play the last frame with final reward 30 more times (2 seconds at 15fps)
        for _ in range(30):
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
            rgb_np = rgb.cpu().numpy()[:, :, :3]
            frame = cv2.resize(rgb_np, (640, 360))
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
        if len(health_series) > 0:
            health_series_ext = health_series + [health_series[-1]] * 30
        else:
            health_series_ext = [100.0] * 30
        T = len(health_series_ext)
        y_min, y_max = 0.0, 100.0
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_h, = ax.plot([], [], lw=6, color='tab:orange', label='Plate')
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
            line_h.set_data([], [])
            return line_h,

        def animate_health(i):
            x = [k / fps for k in range(1, i + 2)]
            y = health_series_ext[: i + 1]
            line_h.set_data(x, y)
            return line_h,

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
        # Optionally cleanup sources
        try:
            os.remove(health_mp4)
        except Exception:
            pass
        # Build reward-over-time MP4 using Matplotlib (cumulative episode reward)
        reward_mp4 = os.path.join(
            eval_videos_dir,
            f"eval_iteration_{iteration_num}{suffix}_reward.mp4",
        )
        if len(reward_series) > 0:
            reward_series_ext = reward_series + [reward_series[-1]] * 30
        else:
            reward_series_ext = [0.0] * 30
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
        try:
            os.remove(reward_mp4)
        except Exception:
            pass
        # Delete the reward side-by-side video; not needed in final outputs
        try:
            os.remove(with_reward_path)
        except Exception:
            pass
        # Now remove the raw sim AVI after both composites are handled
        try:
            os.remove(sim_path)
        except Exception:
            pass
        avi_path = with_health_path
        print(f"Evaluation video with health saved: {os.path.basename(avi_path)}")
        return total_reward, final_terminated, final_truncated, avi_path, final_plate_health
    else:
        return total_reward, final_terminated, final_truncated, None, final_plate_health


def run_eval(env, agent, device, eval_videos_dir, iteration_num, num_episodes=5):
    successes = 0
    total_returns = []
    saved_videos = []
    for i in range(num_episodes):
        capture = (i < 2)  # Save videos for the first two episodes
        ep_ret, terminated, truncated, avi_path, plate_health = run_eval_episode(
            env, agent, device, eval_videos_dir, iteration_num, capture_video=capture, episode_idx=i
        )
        total_returns.append(ep_ret)
        if capture and avi_path is not None:
            saved_videos.append(avi_path)
        # Success criterion: terminated and reward > 10 and plate health >= 90
        # if bool(terminated) and (float(ep_ret) > 10.0) and (plate_health is not None and float(plate_health) >= 90.0):
        #     successes += 1
        # if bool(terminated) and float(plate_health) == 100.0:
        #     successes += 1
        if float(plate_health) == 100.0:
            successes += 1
    success_rate = successes / float(max(1, num_episodes))
    avg_return = float(np.mean(total_returns)) if len(total_returns) > 0 else 0.0
    # If we captured two episodes, combine into a single AVI (ep1 then ep2)
    if len(saved_videos) >= 2:
        try:
            import cv2
            cap1 = cv2.VideoCapture(saved_videos[0])
            cap2 = cv2.VideoCapture(saved_videos[1])
            fps = cap1.get(cv2.CAP_PROP_FPS) or 15
            w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            combined_path = os.path.join(eval_videos_dir, f"eval_iteration_{iteration_num}.avi")
            out = cv2.VideoWriter(combined_path, fourcc, fps, (w, h))
            # write ep1
            while True:
                ret, frame = cap1.read()
                if not ret:
                    break
                out.write(frame)
            # write ep2
            while True:
                ret, frame = cap2.read()
                if not ret:
                    break
                out.write(frame)
            cap1.release()
            cap2.release()
            out.release()
            # cleanup individual epis videos
            for p in saved_videos[:2]:
                try:
                    os.remove(p)
                except Exception:
                    pass
            saved_videos = [combined_path]
        except Exception:
            pass
    elif len(saved_videos) == 1:
        # Rename single episode video to the final combined name and delete source
        try:
            combined_path = os.path.join(eval_videos_dir, f"eval_iteration_{iteration_num}.avi")
            # Move/rename
            try:
                os.replace(saved_videos[0], combined_path)
            except Exception:
                # Fallback: copy then delete
                import shutil
                shutil.copyfile(saved_videos[0], combined_path)
                try:
                    os.remove(saved_videos[0])
                except Exception:
                    pass
            saved_videos = [combined_path]
        except Exception:
            pass
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
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # Initialize WandB (always on)
    wandb.init(project="omnigibson-ppo", name=run_name, config=vars(args))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create eval videos directory
    eval_videos_dir = "safe-manipulation-benchmark/rl/eval_videos"
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
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
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
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        wandb.log({
                            "charts/episodic_return": info["episode"]["r"],
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

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
        
        # Evaluation: first iteration and then every 10 iterations
        if (iteration == 1) or (iteration % 10 == 0):
            print(f"Running evaluation after iteration {iteration}...")
            # IMPORTANT: Reuse the SAME underlying env to avoid multiple global simulators
            try:
                base_env = envs.envs[0]
            except Exception:
                base_env = None
            if base_env is not None:
                success_rate, eval_avg_return, _ = run_eval(base_env, agent, device, eval_videos_dir, iteration, num_episodes=5)
                # Log evaluation metrics
                wandb.log({
                    "eval_iteration": iteration,
                    "eval_success_rate": success_rate,
                    "eval_avg_return": eval_avg_return,
                })
                print(f"Eval success rate: {success_rate:.3f}, avg return: {eval_avg_return:.3f}")
                # Resync vector env state after direct base env usage
                try:
                    next_obs, _ = envs.reset(seed=None)
                    next_obs = torch.Tensor(next_obs).to(device)
                    next_done = torch.zeros(args.num_envs).to(device)
                except Exception:
                    pass

    envs.close()
    wandb.finish()


