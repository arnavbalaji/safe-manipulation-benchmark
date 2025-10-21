import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import random
import omnigibson as og
import wandb
import matplotlib
# Force non-GUI backend for headless runs to avoid Qt/xcb errors during eval plotting
matplotlib.use('Agg')


def _ensure_omnigibson_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    og_src = os.path.join(repo_root, "OmniGibson")
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


_ensure_omnigibson_on_path()


from safety_benchmark.damageable_env import DamageableEnvironment

def damage_reward_fn(env, obs):
    plate_health_states = obs["object_health_states"]["glass_plate"]
    total_damage = 0.0
    for damage_type, damage_info in plate_health_states["damage_info"].items():
        for link_name, damage in damage_info.items():
            total_damage += damage
    if total_damage == 0.0 and plate_health_states["health"] == 100.0:
        return 0.0001
    return -total_damage


def distance_reward_fn(env, obs, eps=0.01):
    plate_obj = env.scene.object_registry("name", "glass_plate")
    plate_pos, plate_orn = plate_obj.get_position_orientation()
    target_pos = th.tensor([-0.0966, 0.0843, 0.4593])
    distance = th.norm(plate_pos - target_pos).item()
    if distance < eps:
        return 100.0
    return -distance

def reward_fn(env, obs):
    return damage_reward_fn(env, obs) + distance_reward_fn(env, obs)

def _execute_controller_sequence(env, controller, robot, delta_pose, ignore_failure=True, max_steps=300):
    current_eef_pos = robot.get_eef_position("right")
    current_eef_orn = robot.get_eef_orientation("right")
    target_eef_pos = current_eef_pos + delta_pose
    target_eef_pose = (target_eef_pos, current_eef_orn)

    steps = 0
    for action in controller._move_hand_linearly_cartesian(target_eef_pose, ignore_failure=ignore_failure):
        env.step(action=action)
        steps += 1
        if steps >= max_steps:
            break


class InitializedDamageableEnv:
    """
    Minimal Gym-like wrapper around DamageableEnvironment that:
    - Constrains actions to right arm and gripper. Agent action = [right_arm_continuous..., right_gripper_binary]
      where the final gripper entry is binary in {-1, +1} (open / close). The single gripper
      command is broadcast to all underlying gripper joints. No primitives during learning.
    - Runs the action primitives during reset to establish the start state (up then forward).
    - Returns a compact vector observation suitable for MLP policies.

    Note: This wrapper intentionally does NOT implement vectorized env or full Gym API, but
    exposes reset() / step() compatible with common PPO loops.
    """

    def __init__(self, scene_file: str):
        import omnigibson as og  # local import after setting path
        self.og = og

        # Create environment from saved scene
        cfg = {"scene": {"type": "Scene", "scene_file": scene_file}}
        self.env = DamageableEnvironment(configs=cfg, reward_fn=reward_fn)
        self.env.reset()

        # Grab robot
        assert len(self.env.robots) > 0, "No robots found after loading scene!"
        self.robot = self.env.robots[0]

        # Configure controllers to match rl_test.py
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
        self.robot.reload_controllers(controller_config=controller_config)
        self.env.scene.update_initial_file()

        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            StarterSemanticActionPrimitives,
        )
        self.prims = StarterSemanticActionPrimitives(env=self.env, robot=self.robot, skip_curobo_initilization=True)
        self.prims.arm = "right"

        self.plate_obj = None
        for obj_name in ["glass_plate", "target_object", "plate"]:
            self.plate_obj = self.env.scene.object_registry("name", obj_name)
            if self.plate_obj is not None:
                break

        # Use OmniGibson default flattened action space; we will mask to right arm + gripper
        self._action_dim = int(self.robot.action_dim)
        self._right_arm = "right" if (hasattr(self.robot, "arm_names") and ("right" in self.robot.arm_names)) else self.robot.default_arm
        self._right_arm_idx = self.robot.arm_action_idx[self._right_arm].cpu().numpy().astype(np.int64)
        self._right_grip_idx = self.robot.gripper_action_idx[self._right_arm].cpu().numpy().astype(np.int64)
        self._right_indices = np.concatenate([self._right_arm_idx, self._right_grip_idx], axis=0)
        # Agent-facing action space uses a single binary gripper dimension
        self._agent_arm_dim = int(self._right_arm_idx.shape[0])
        self._agent_act_dim = int(self._agent_arm_dim + 1)  # +1 for binary gripper

        # Target kept for reward computation; observations will be proprioceptive
        self.target_pos = th.tensor([-0.0966, 0.0843, 0.4593])

    def reset(self) -> np.ndarray:
        # Reset base env
        self.env.reset()
        self.env.lock_health_changes()

        for _ in range(20):
            self.env.step(action=np.zeros((self._action_dim,), dtype=np.float32))

        # Run initialization primitives: up then forward
        _execute_controller_sequence(self.env, self.prims, self.robot, th.tensor([0.0, 0.0, 0.45]))
        _execute_controller_sequence(self.env, self.prims, self.robot, th.tensor([0.0, -0.15, 0.0]))
        self.env.unlock_health_changes()
        # Return compact observation
        return self._extract_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Accept agent action [arm_continuous..., gripper_binary], expand to full robot action vector
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        full = np.zeros((self._action_dim,), dtype=np.float32)

        # Split arm and gripper
        assert action.shape[0] >= self._agent_act_dim, \
            f"Expected action with dim >= {self._agent_act_dim}, got {action.shape[0]}"
        arm_action = action[: self._agent_arm_dim]
        grip_scalar = action[self._agent_arm_dim]

        # Broadcast to underlying indices
        full[self._right_arm_idx] = arm_action
        # Binary gripper: map to {-1, +1}
        grip_binary = 1.0 if float(grip_scalar) >= 0.0 else -1.0
        full[self._right_grip_idx] = grip_binary

        obs, reward, terminated, truncated, info = self.env.step(action=full)
        return self._extract_obs(), float(reward), bool(terminated), bool(truncated), info

    def _extract_obs(self) -> np.ndarray:
        # Build observation = [proprioception, right_eef_position(3), plate_position(3)]
        proprio, _ = self.robot.get_proprioception()
        # Right EEF position
        eef_pos = self.robot.get_eef_position("right")
        # Plate position (zeros if not found)
        if self.plate_obj is not None:
            plate_pos, _ = self.plate_obj.get_position_orientation()
        else:
            plate_pos = th.zeros(3)
        obs_vec = th.cat([
            proprio.float().flatten(),
            th.as_tensor(eef_pos, dtype=th.float32).flatten(),
            th.as_tensor(plate_pos, dtype=th.float32).flatten(),
        ], dim=0)
        return obs_vec.detach().cpu().numpy().astype(np.float32)

    @property
    def observation_space(self):
        # Lazy import gym to avoid hard dependency elsewhere
        import gym
        # Dimension matches robot.proprioception_dim
        obs_dim = int(self.robot.proprioception_dim) + 6  # +3 right EEF position, +3 plate position
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    @property
    def action_space(self):
        import gym
        # Agent-facing: right arm continuous dims + 1 binary gripper dim (bounded in [-1, 1])
        low = -np.ones((self._agent_act_dim,), dtype=np.float32)
        high = np.ones((self._agent_act_dim,), dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 256

        # Mixed-action policy: Gaussian for arm (act_dim - 1), Bernoulli for gripper (1)
        assert act_dim >= 2, "Expected at least 2 actions (>=1 arm dim + 1 gripper)"
        self.arm_dim = act_dim - 1

        # Shared actor body
        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Policy heads
        self.arm_mean = nn.Linear(hidden, self.arm_dim)
        self.log_std = nn.Parameter(th.full((self.arm_dim,), -0.5))
        self.grip_logits = nn.Linear(hidden, 1)

        # Critic
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Orthogonal initialization similar to CleanRL
        def orthogonal_init(layer, gain=th.nn.init.calculate_gain('tanh')):
            if isinstance(layer, nn.Linear):
                th.nn.init.orthogonal_(layer.weight, gain=gain)
                th.nn.init.constant_(layer.bias, 0.0)

        # Initialize actor body
        orthogonal_init(self.actor_body[0])
        orthogonal_init(self.actor_body[2])
        # Smaller init for policy outputs
        th.nn.init.orthogonal_(self.arm_mean.weight, gain=0.01)
        th.nn.init.constant_(self.arm_mean.bias, 0.0)
        th.nn.init.orthogonal_(self.grip_logits.weight, gain=0.01)
        th.nn.init.constant_(self.grip_logits.bias, 0.0)

        # Initialize critic
        orthogonal_init(self.v[0])
        orthogonal_init(self.v[2])
        th.nn.init.orthogonal_(self.v[4].weight, gain=1.0)
        th.nn.init.constant_(self.v[4].bias, 0.0)

    def forward(self, x):
        raise NotImplementedError

    def get_action_and_value(self, obs, action=None):
        # Forward actor
        h = self.actor_body(obs)
        arm_mean = self.arm_mean(h)
        arm_std = th.exp(self.log_std)
        arm_normal = th.distributions.Normal(arm_mean, arm_std)
        grip_dist = th.distributions.Bernoulli(logits=self.grip_logits(h))

        if action is None:
            # Sample arm and gripper
            arm_pre = arm_normal.rsample()
            arm_squashed = th.tanh(arm_pre)
            arm_logp_pre = arm_normal.log_prob(arm_pre).sum(-1)
            arm_logp_corr = th.log(1 - arm_squashed.pow(2) + 1e-6).sum(-1)
            arm_logp = arm_logp_pre - arm_logp_corr

            grip_sample01 = grip_dist.sample()  # in {0,1}
            grip_action = grip_sample01 * 2.0 - 1.0  # map to {-1, +1}
            grip_logp = grip_dist.log_prob(grip_sample01).squeeze(-1)

            # Compose outputs
            action_out = th.cat([arm_squashed, grip_action], dim=-1)
            logprob = arm_logp + grip_logp
            entropy = arm_normal.entropy().sum(-1) + grip_dist.entropy().squeeze(-1)
        else:
            # Given action: split and compute logprob
            eps = 1e-6
            action = action.clamp(-1.0 + eps, 1.0 - eps)
            arm_action = action[..., : self.arm_dim]
            grip_action = action[..., self.arm_dim]

            # Arm inverse tanh correction
            arm_atanh = 0.5 * (th.log1p(arm_action) - th.log1p(-arm_action))
            arm_pre = arm_atanh.clamp(-10.0, 10.0)
            arm_logp_pre = arm_normal.log_prob(arm_pre).sum(-1)
            arm_logp_corr = th.log(1 - arm_action.pow(2) + 1e-6).sum(-1)
            arm_logp = arm_logp_pre - arm_logp_corr

            # Gripper: map {-1,+1} -> {0,1} and discretize
            grip01 = (grip_action + 1.0) * 0.5
            grip01 = (grip01 >= 0.5).float().unsqueeze(-1)
            grip_logp = grip_dist.log_prob(grip01).squeeze(-1)

            logprob = arm_logp + grip_logp
            entropy = arm_normal.entropy().sum(-1) + grip_dist.entropy().squeeze(-1)
            action_out = action

        value = self.v(obs).squeeze(-1)
        return action_out, logprob, entropy, value


@dataclass
class PPOConfig:
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_steps: int = 2000
    update_epochs: int = 10
    num_minibatches: int = 40
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    device: str = "cuda" if th.cuda.is_available() else "cpu"
    seed: int = 1


def _run_eval_episode(env, policy, device, eval_videos_dir, update_num):
    """Run a single evaluation episode and save video with reward plot"""
    import cv2
    import subprocess
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import omnigibson as og
    
    # Reset environment
    obs = env.reset()
    obs_tensor = th.tensor(obs, dtype=th.float32, device=device)
    
    frames = []
    total_rewards = []
    total_reward = 0.0
    fps = 15
    
    # Run evaluation episode
    for step in range(100):
        with th.no_grad():
            action, _, _, _ = policy.get_action_and_value(obs_tensor.unsqueeze(0))
        action_np = action.squeeze(0).cpu().numpy()
        
        obs, reward, terminated, truncated, info = env.step(action_np)
        obs_tensor = th.tensor(obs, dtype=th.float32, device=device)
        
        total_reward += float(reward)
        total_rewards.append(total_reward)
        
        # Capture frame (downsampled)
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        rgb_np = cv2.resize(rgb_np, (640, 360))
        frames.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        
        if terminated or truncated:
            break
    
    # Generate videos
    if len(frames) > 0:
        # Camera video
        avi_path = os.path.join(eval_videos_dir, f"eval_update_{update_num}_camera.avi")
        mp4_path = os.path.join(eval_videos_dir, f"eval_update_{update_num}_camera.mp4")
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        for f in frames:
            vw.write(np.ascontiguousarray(f, dtype=np.uint8))
        vw.release()
        # Convert AVI to MP4 with mpeg4 (match rl_test.py)
        subprocess.run(["ffmpeg", "-y", "-i", avi_path, "-c:v", "mpeg4", mp4_path], check=True)
        os.remove(avi_path)
        
        # Reward plot video
        reward_mp4 = os.path.join(eval_videos_dir, f"eval_update_{update_num}_reward.mp4")
        if len(total_rewards) > 0:
            T = len(total_rewards)
            min_r = min(total_rewards)
            max_r = max(total_rewards)
            rng = max(1e-5, max_r - min_r)
            y_min = min(-150, min_r)
            y_max = 0
            
            # Handle NaN and infinity values
            if not (np.isfinite(y_min) and np.isfinite(y_max)):
                y_min, y_max = -1000.0, 1000.0

            fig, ax = plt.subplots(figsize=(9.6, 5.4))
            line_r, = ax.plot([], [], lw=6, color='tab:green', label='Cumulative Reward')
            ax.set_xlim(0, max(1, T) / fps)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('Time (s)', fontsize=20)
            ax.set_ylabel('Cumulative Reward', fontsize=20)
            ax.set_title(f'Evaluation Episode Reward (Update {update_num})', fontsize=26)
            ax.legend(loc='best', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
            ax.grid(True, linewidth=1.0, alpha=0.3)
            plt.tight_layout()

            def init_reward():
                line_r.set_data([], [])
                return line_r,

            def animate_reward(i):
                x = [k / fps for k in range(1, i + 2)]
                y = total_rewards[: i + 1]
                line_r.set_data(x, y)
                return line_r,

            ani = animation.FuncAnimation(
                fig, animate_reward, init_func=init_reward, frames=T, interval=1000 / fps, blit=True
            )
            writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
            ani.save(reward_mp4, writer=writer)
            plt.close(fig)
        
        # Combined video (match rl_test.py exact codec and command)
        combined_mp4 = os.path.join(eval_videos_dir, f"eval_update_{update_num}_combined.mp4")
        subprocess.run([
            'ffmpeg', '-y',
            '-i', mp4_path,
            '-i', reward_mp4,
            '-filter_complex',
            '[0:v]scale=640:360,setsar=1[left];[1:v]scale=640:360,setsar=1[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'mpeg4',
            '-q:v', '5',
            combined_mp4
        ], check=True)

        # Clean up intermediate files, keep only combined
        os.remove(mp4_path)
        os.remove(reward_mp4)

        print(f"Evaluation video saved: {os.path.basename(combined_mp4)}")
        # Clear frames and force GC to reduce memory
        try:
            frames.clear()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
        return combined_mp4
    else:
        print("No frames captured for evaluation video")
        return None
    
    # Always reset environment after evaluation
    env.reset()


def _save_learning_plots(learning_stats, eval_videos_dir, update_num):
    """Save learning progress plots"""
    import matplotlib.pyplot as plt
    
    if len(learning_stats['episode_returns']) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode returns
    axes[0, 0].plot(learning_stats['episode_returns'], alpha=0.7)
    axes[0, 0].set_title('Episode Returns Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Policy loss
    axes[0, 1].plot(learning_stats['policy_losses'], alpha=0.7)
    axes[0, 1].set_title('Policy Loss Over Time')
    axes[0, 1].set_xlabel('Update')
    axes[0, 1].set_ylabel('Policy Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Value loss
    axes[1, 0].plot(learning_stats['value_losses'], alpha=0.7)
    axes[1, 0].set_title('Value Loss Over Time')
    axes[1, 0].set_xlabel('Update')
    axes[1, 0].set_ylabel('Value Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 1].plot(learning_stats['entropies'], alpha=0.7)
    axes[1, 1].set_title('Policy Entropy Over Time')
    axes[1, 1].set_xlabel('Update')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(eval_videos_dir, f"learning_progress_update_{update_num}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Learning progress plot saved: {os.path.basename(plot_path)}")
    return plot_path


def _save_training_debug_video(frames, total_rewards, eval_videos_dir, tag, fps=15):
    """Save a combined camera + cumulative reward plot video for debugging training episodes.
    Matches the eval video pipeline (mpeg4, no error handling).
    """
    import cv2
    import subprocess
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if len(frames) == 0:
        return None

    # Camera video
    avi_path = os.path.join(eval_videos_dir, f"train_debug_{tag}_camera.avi")
    mp4_path = os.path.join(eval_videos_dir, f"train_debug_{tag}_camera.mp4")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
    for f in frames:
        vw.write(np.ascontiguousarray(f, dtype=np.uint8))
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", avi_path, "-c:v", "mpeg4", mp4_path], check=True)
    os.remove(avi_path)

    # Reward plot video
    reward_mp4 = os.path.join(eval_videos_dir, f"train_debug_{tag}_reward.mp4")
    if len(total_rewards) > 0:
        T = len(total_rewards)
        min_r = min(total_rewards)
        max_r = max(total_rewards)
        rng = max(1e-5, max_r - min_r)
        y_min = min(-150, min_r)
        y_max = 0
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            y_min, y_max = -1000.0, 1000.0

        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_r, = ax.plot([], [], lw=6, color='tab:green', label='Cumulative Reward')
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Cumulative Reward', fontsize=20)
        ax.set_title(f'Training Episode Reward (tag {tag})', fontsize=26)
        ax.legend(loc='best', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_reward():
            line_r.set_data([], [])
            return line_r,

        def animate_reward(i):
            x = [k / fps for k in range(1, i + 2)]
            y = total_rewards[: i + 1]
            line_r.set_data(x, y)
            return line_r,

        ani = animation.FuncAnimation(
            fig, animate_reward, init_func=init_reward, frames=T, interval=1000 / fps, blit=True
        )
        writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', extra_args=['-vcodec', 'mpeg4', '-qscale', '5'])
        ani.save(reward_mp4, writer=writer)
        plt.close(fig)

    # Combine camera and reward videos side by side
    combined_mp4 = os.path.join(eval_videos_dir, f"train_debug_{tag}_combined.mp4")
    subprocess.run([
        'ffmpeg', '-y',
        '-i', mp4_path,
        '-i', reward_mp4,
        '-filter_complex',
        '[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]',
        '-map', '[v]',
        '-c:v', 'mpeg4',
        '-q:v', '5',
        combined_mp4
    ], check=True)

    # Clean up intermediate videos
    try:
        os.remove(mp4_path)
        os.remove(reward_mp4)
    except OSError:
        pass

    print(f"Training debug video saved: {os.path.basename(combined_mp4)}")
    return combined_mp4

def train(scene_file: str, cfg: PPOConfig):
    import omnigibson as og

    # Initialize WandB
    wandb.init(
        project="omnigibson-ppo",
        name=f"ppo_training_{int(time.time())}",
        config={
            "learning_rate": cfg.learning_rate,
            "num_steps": cfg.num_steps,
            "total_timesteps": cfg.total_timesteps,
            "gamma": cfg.gamma,
            "gae_lambda": cfg.gae_lambda,
            "num_minibatches": cfg.num_minibatches,
            "update_epochs": cfg.update_epochs,
            "clip_coef": cfg.clip_coef,
            "ent_coef": cfg.ent_coef,
            "vf_coef": cfg.vf_coef,
            "max_grad_norm": cfg.max_grad_norm,
            "target_kl": cfg.target_kl,
            "device": cfg.device,
            "seed": cfg.seed,
            "max_episode_steps": 100,
            "eval_interval": "every_update",
        }
    )

    # Seeding for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(cfg.seed)
    try:
        og.sim.set_random_seed(cfg.seed)  # if supported
    except Exception:
        pass

    # Ensure a fresh sim with GUI enabled
    if og.sim is None:
        minimal_cfg = {
            "env": {"action_timestep": 1.0 / 60.0, "physics_timestep": 1.0 / 120.0},
            "scene": {"type": "Scene"},
            "robots": [],
        }
        temp_env = DamageableEnvironment(configs=minimal_cfg)
        temp_env.reset()
        og.clear()
    else:
        og.sim.stop()
        og.clear()

    env = InitializedDamageableEnv(scene_file)

    # Match viewer camera pose from rl_test.py
    try:
        og.sim.viewer_camera.set_position_orientation(
            position=[-1.3225984573364258, 0.2645236849784851, 0.9981386065483093],
            orientation=[0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452],
        )
    except Exception:
        pass

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    policy = ActorCritic(obs_dim, act_dim).to(cfg.device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Storage
    num_updates = math.ceil(cfg.total_timesteps / cfg.num_steps)
    obs_buf = th.zeros((cfg.num_steps, obs_dim), dtype=th.float32, device=cfg.device)
    actions_buf = th.zeros((cfg.num_steps, act_dim), dtype=th.float32, device=cfg.device)
    logprobs_buf = th.zeros((cfg.num_steps,), dtype=th.float32, device=cfg.device)
    rewards_buf = th.zeros((cfg.num_steps,), dtype=th.float32, device=cfg.device)
    dones_buf = th.zeros((cfg.num_steps,), dtype=th.float32, device=cfg.device)
    values_buf = th.zeros((cfg.num_steps,), dtype=th.float32, device=cfg.device)

    next_obs = th.tensor(env.reset(), dtype=th.float32, device=cfg.device)
    next_done = th.tensor(0.0, dtype=th.float32, device=cfg.device)

    episode_return = 0.0
    episode_length = 0
    episode_step = 0
    max_episode_steps = 100
    start_time = time.time()
    
    # Learning progress tracking
    recent_returns = []
    best_return = float('-inf')
    learning_stats = {
        'episode_returns': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': []
    }
    # No training-time frame capture to avoid memory spikes
    episode_frames = []
    episode_cum_rewards = []
    
    # Evaluation tracking
    total_steps = 0

    print("Starting PPO training. Logging to WandB. Press ESC in the viewer to stop.")

    # Create eval videos directory
    eval_videos_dir = "safe-manipulation-benchmark/rl/eval_videos"
    os.makedirs(eval_videos_dir, exist_ok=True)

    just_evaluated = False

    for update in range(1, num_updates + 1):
        # Anneal LR (optional)
        frac = 1.0 - (update - 1.0) / num_updates
        lr_now = frac * cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        # If we just evaluated, force a clean reset before collecting more data
        if just_evaluated:
            next_obs = th.tensor(env.reset(), dtype=th.float32, device=cfg.device)
            next_done = th.tensor(0.0, dtype=th.float32, device=cfg.device)
            episode_return = 0.0
            episode_length = 0
            episode_step = 0
            episode_frames = []
            episode_cum_rewards = []
            just_evaluated = False

        for step in range(cfg.num_steps):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with th.no_grad():
                action, logprob, _, value = policy.get_action_and_value(next_obs.unsqueeze(0))
            actions = action.squeeze(0)
            logprob = logprob.squeeze(0)
            value = value.squeeze(0)

            # Step env
            np_action = actions.detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(np_action)

            next_obs = th.tensor(obs, dtype=th.float32, device=cfg.device)
            reward_t = float(reward)
            episode_step += 1
            reached_limit = episode_step >= max_episode_steps
            done = bool(terminated or truncated or reached_limit)
            next_done = th.tensor(1.0 if done else 0.0, dtype=th.float32, device=cfg.device)

            rewards_buf[step] = reward_t
            values_buf[step] = value
            logprobs_buf[step] = logprob
            actions_buf[step] = actions

            # GUI-friendly printing
            episode_return += reward_t
            episode_length += 1

            # No training-time video capture
            if done:
                # Record episode statistics
                learning_stats['episode_returns'].append(episode_return)
                learning_stats['episode_lengths'].append(episode_length)
                recent_returns.append(episode_return)
                if len(recent_returns) > 10:  # Keep only last 10 episodes
                    recent_returns.pop(0)
                
                if episode_return > best_return:
                    best_return = episode_return
                
                # Update total steps and episodes
                total_steps += episode_length
                total_episodes = len(learning_stats['episode_returns'])

                # Do not record training videos even if return==0 (per request)
                
                # Log episode completion to WandB
                wandb.log({
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "episode": total_episodes,
                    "total_steps": total_steps
                })
                next_obs = th.tensor(env.reset(), dtype=th.float32, device=cfg.device)
                next_done = th.tensor(0.0, dtype=th.float32, device=cfg.device)
                episode_return = 0.0
                episode_length = 0
                episode_step = 0
                # No training buffers to clear

        with th.no_grad():
            next_value = policy.v(next_obs.unsqueeze(0)).squeeze(0)

        # GAE-Lambda advantage
        advantages = th.zeros_like(rewards_buf, device=cfg.device)
        lastgaelam = 0
        for t in reversed(range(cfg.num_steps)):
            if t == cfg.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + cfg.gamma * nextvalues * nextnonterminal - values_buf[t]
            advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values_buf

        # Flatten the batch
        b_obs = obs_buf
        b_logprobs = logprobs_buf
        b_actions = actions_buf
        b_advantages = advantages
        b_returns = returns
        b_values = values_buf

        # Optimize policy for K epochs
        batch_size = cfg.num_steps
        minibatch_size = batch_size // cfg.num_minibatches
        inds = np.arange(batch_size)
        # Track losses across epochs/minibatches for logging
        pg_losses_epoch = []
        v_losses_epoch = []
        entropies_epoch = []

        stop_early = False
        last_approx_kl = 0.0
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = policy.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = th.exp(logratio)

                # Policy loss
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * th.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped) per PPO
                newvalue = newvalue.view(-1)
                v_target = b_returns[mb_inds]
                v_old = b_values[mb_inds].detach()
                v_unclipped = (newvalue - v_target).pow(2)
                v_clipped = v_old + (newvalue - v_old).clamp(-cfg.clip_coef, cfg.clip_coef)
                v_clipped = (v_clipped - v_target).pow(2)
                v_loss = 0.5 * th.max(v_unclipped, v_clipped).mean()

                # Entropy bonus
                ent_loss = -cfg.ent_coef * entropy.mean()

                loss = pg_loss + cfg.vf_coef * v_loss + ent_loss

                # Collect metrics
                pg_losses_epoch.append(pg_loss.detach().item())
                v_losses_epoch.append(v_loss.detach().item())
                entropies_epoch.append(entropy.detach().mean().item())

                optimizer.zero_grad()
                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm))
                optimizer.step()

                # Early stopping by KL per minibatch
                with th.no_grad():
                    last_approx_kl = ((ratio - 1) - logratio).mean().abs().item()
                if cfg.target_kl is not None and last_approx_kl > cfg.target_kl:
                    stop_early = True
                    break

            if stop_early:
                break

        # Aggregate losses for logging
        policy_loss_val = float(np.mean(pg_losses_epoch)) if len(pg_losses_epoch) > 0 else 0.0
        value_loss_val = float(np.mean(v_losses_epoch)) if len(v_losses_epoch) > 0 else 0.0
        entropy_val = float(np.mean(entropies_epoch)) if len(entropies_epoch) > 0 else 0.0

        # Record training statistics
        learning_stats['policy_losses'].append(policy_loss_val)
        learning_stats['value_losses'].append(value_loss_val)
        learning_stats['entropies'].append(entropy_val)
        
        fps = int((cfg.num_steps) / (time.time() - start_time + 1e-8))
        start_time = time.time()
        
        # Calculate learning progress metrics
        avg_recent_return = np.mean(recent_returns) if recent_returns else 0.0
        total_episodes = len(learning_stats['episode_returns'])
        
        # Log training metrics to WandB
        wandb.log({
            "update": update,
            "fps": fps,
            "learning_rate": lr_now,
            "last_episode_return": episode_return,
            "avg_recent_return": avg_recent_return,
            "best_return": best_return,
            "total_episodes": total_episodes,
            "total_steps": total_steps,
            "approx_kl": last_approx_kl,
            "policy_loss": policy_loss_val,
            "value_loss": value_loss_val,
            "entropy": entropy_val,
            "grad_norm_last": grad_norm if 'grad_norm' in locals() else None,
            "explained_variance": 1 - th.var(returns - values_buf) / th.var(returns),
        })
        
        # Also print key metrics to console
        print(f"Update {update}/{num_updates} | return={episode_return:.3f} | avg_recent={avg_recent_return:.3f} | best={best_return:.3f} | kl={last_approx_kl:.4f}")

        # Evaluation after every update
        print(f"Running evaluation after update {update}...")
        eval_video_path = _run_eval_episode(env, policy, cfg.device, eval_videos_dir, update)
        # Only log scalars in WandB (no media)
        wandb.log({
            "eval_steps": total_steps,
            "eval_update": update
        })
        # Ensure training env is reset after eval (eval resets the shared env)
        next_obs = th.tensor(env.reset(), dtype=th.float32, device=cfg.device)
        next_done = th.tensor(0.0, dtype=th.float32, device=cfg.device)
        episode_return = 0.0
        episode_length = 0
        episode_step = 0
        episode_frames = []
        episode_cum_rewards = []
        just_evaluated = True

    print("Training complete. Shutting down simulator.")
    wandb.finish()
    og.shutdown()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene-file", type=str, default="lift_test.json", help="Path to saved OmniGibson scene JSON")
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-steps", type=int, default=2000)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--num-minibatches", type=int, default=40)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--target-kl", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
    )
    train(scene_file=args.scene_file, cfg=cfg)


