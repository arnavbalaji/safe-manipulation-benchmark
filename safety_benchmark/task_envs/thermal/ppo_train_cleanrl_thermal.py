import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical  # kept for parity with CleanRL API (unused)
import wandb
import cv2

import sys
import subprocess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# OmniGibson + environment wrapper (minimal integration)
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import OnFire
from omnigibson import object_states


def _ensure_omnigibson_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    og_src = os.path.join(repo_root, "OmniGibson")
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


_ensure_omnigibson_on_path()

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
)

# ---------------------------------------------------------------------------
# Thermal damage RL training for log placement task
# ---------------------------------------------------------------------------

# Scene file saved from `simple_task_load.py` after configuring controllers etc.
DEFAULT_SCENE_FILE = (
    "safe-manipulation-benchmark/safety_benchmark/task_envs/thermal/thermal_task.json"
)

# Episode horizon (train + eval)
EVAL_MAX_STEPS = 200

# Directory to store evaluation videos
EVAL_VIDEOS_DIR = (
    "safe-manipulation-benchmark/safety_benchmark/task_envs/thermal/eval_videos"
)

checkpoint_dir = os.path.join(
    "safe-manipulation-benchmark",
    "safety_benchmark",
    "task_envs",
    "thermal",
    "checkpoints",
)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_file = os.path.join(checkpoint_dir, "checkpoint_thermal.pth")


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
    env_id: str = "OG-Thermal"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments (upper bound; you can stop earlier once performance stabilizes)"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer (shared for actor / critic)."""
    num_envs: int = 4
    """the number of parallel game environments (forced to 1 at runtime for OG)"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout (batch_size = num_envs * num_steps)"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches (minibatch_size ~=128 with num_steps=512, num_envs=1)"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.005
    """coefficient of the entropy"""
    vf_coef: float = 0.7
    """coefficient of the value function loss term"""
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

            # Set macros BEFORE any simulator/env creation (required for object states / thermal)
            gm.USE_GPU_DYNAMICS = True
            gm.ENABLE_OBJECT_STATES = True
            gm.ENABLE_FLATCACHE = True
            gm.ENABLE_HQ_RENDERING = False

            # Ensure fresh simulator (match other OG scripts)
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

            # Ensure robot has Temperature state enabled (required for thermal damage evaluator)
            if object_states.Temperature in self._robot.states:
                temp_val = self._robot.states[object_states.Temperature].get_value()
                print(
                    f"[Thermal RL] Robot has Temperature state. Initial temperature: {float(temp_val):.2f}°C"
                )
            else:
                print(
                    "[Thermal RL] Robot missing Temperature state. Adding AABB + Temperature states..."
                )
                from omnigibson.objects.stateful_object import StatefulObject

                assert isinstance(
                    self._robot, StatefulObject
                ), "Robot is not a StatefulObject; cannot add Temperature state"

                # Temperature depends on AABB
                if object_states.AABB not in self._robot.states:
                    compatible, reason = object_states.AABB.is_compatible(
                        obj=self._robot
                    )
                    assert (
                        compatible
                    ), f"AABB state not compatible with robot: {reason}"
                    aabb_state = object_states.AABB(obj=self._robot)
                    self._robot.add_state(aabb_state)
                    aabb_state.initialize()
                    print("[Thermal RL] Added and initialized AABB state on robot")

                compatible, reason = object_states.Temperature.is_compatible(
                    obj=self._robot
                )
                assert (
                    compatible
                ), f"Temperature state not compatible with robot: {reason}"
                temp_state = object_states.Temperature(obj=self._robot)
                self._robot.add_state(temp_state)
                temp_state.initialize()
                temp_val = temp_state.get_value()
                print(
                    f"[Thermal RL] Added and initialized Temperature state. Initial temperature: {float(temp_val):.2f}°C"
                )

            # Match simple_task_load.py controller config (IK arms + strong position grippers)
            controller_config = {
                "arm_left": {
                    "name": "InverseKinematicsController",
                    "mode": "pose_delta_ori",
                },
                "arm_right": {
                    "name": "InverseKinematicsController",
                    "mode": "pose_delta_ori",
                },
                "gripper_left": {
                    "motor_type": "position",
                    "isaac_kp": 8000.0,
                    "isaac_kd": 4000.0,
                    "inverted": True,
                },
                "gripper_right": {
                    "motor_type": "position",
                    "isaac_kp": 8000.0,
                    "isaac_kd": 4000.0,
                    "inverted": True,
                },
            }
            self._robot.reload_controllers(controller_config=controller_config)
            # Persist controller state so subsequent saves keep these settings
            self._env.scene.update_initial_file()

            self._prims = StarterSemanticActionPrimitives(
                env=self._env, robot=self._robot, skip_curobo_initilization=True
            )
            self._prims.arm = "right"

            self._action_dim = int(self._robot.action_dim)
            self._right_arm = (
                "right"
                if (hasattr(self._robot, "arm_names") and ("right" in self._robot.arm_names))
                else self._robot.default_arm
            )
            self._right_arm_idx = (
                self._robot.arm_action_idx[self._right_arm].cpu().numpy().astype(np.int64)
            )
            self._right_grip_idx = (
                self._robot.gripper_action_idx[self._right_arm].cpu().numpy().astype(np.int64)
            )
            # Full IK controller input dimension (translation + orientation deltas)
            self._arm_input_dim = int(self._right_arm_idx.shape[0])
            # Expose all IK DOFs to the policy plus a single gripper control scalar
            self._agent_arm_dim = self._arm_input_dim
            self._agent_act_dim = int(self._agent_arm_dim + 1)
            self._arm_action_scale = 0.05
            self._episode_step = 0
            self._max_episode_steps = EVAL_MAX_STEPS
            self._last_base_obs = None
            self._last_grip_binary = 1.0

            # Target log and goal position for distance reward
            self._target_log_name = "target_object"
            # From teleop script breakpoint (target position)
            self._target_pos = torch.tensor(
                [0.2907, -0.6283, 0.2853], dtype=torch.float32
            )

            # Observation: EEF pos (3) + EEF quat (4) + log pos (3) + log quat (4) + gripper (1) = 15
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-np.ones((self._agent_act_dim,), dtype=np.float32),
                high=np.ones((self._agent_act_dim,), dtype=np.float32),
                dtype=np.float32,
            )

        # ---------------- Reward functions ----------------

        def _get_target_log(self):
            return self._env.scene.object_registry("name", self._target_log_name)

        def _damage_reward_fn(self, env, obs):
            """Damage reward based on robot health (thermal + mechanical)."""
            terminated = False
            robot = self._robot
            if robot is None or not hasattr(robot, "get_obs_dict"):
                return 0.0, False

            rob_obs = robot.get_obs_dict()
            robot_health = float(rob_obs.get("health", 100.0))
            damage_info = rob_obs.get("damage_info", {})

            total_damage = 0.0
            for _, damage_per_link in damage_info.items():
                for _, damage in damage_per_link.items():
                    total_damage += float(damage)

            # Small shaping reward if completely undamaged
            if total_damage == 0.0 and robot_health == 100.0:
                return 0.01, False

            # Terminate episode if robot is fully destroyed
            if robot_health <= 0.0:
                terminated = True
                return -200.0, terminated

            # Penalize instantaneous damage this step
            return -total_damage, terminated

        def _distance_reward_fn(self, env, obs):
            """Distance reward for placing the burning log at the target position."""
            eps_pos = 0.05

            log_obj = self._get_target_log()
            # Let errors surface if log is missing or invalid
            log_pos, _ = log_obj.get_position_orientation()

            log_pos_t = torch.as_tensor(log_pos, dtype=torch.float32)
            target_pos = self._target_pos
            distance = torch.norm(log_pos_t - target_pos).item()

            # Base shaping: negative distance
            reward = -distance
            terminated = False

            # Success: within radius AND log is on fire
            on_fire = False
            if OnFire in log_obj.states:
                on_fire = bool(log_obj.states[OnFire].get_value())

            if (distance < eps_pos) and on_fire:
                reward = 200.0
                terminated = True

            return reward, terminated

        def _reward_fn(self, env, obs):
            """Combine damage and distance rewards."""
            damage_weight = 1.0
            distance_weight = 1.0

            damage_reward, damage_terminated = self._damage_reward_fn(env, obs)
            distance_reward, distance_terminated = self._distance_reward_fn(env, obs)

            total_reward = damage_weight * damage_reward + distance_weight * distance_reward
            terminated = damage_terminated or distance_terminated
            return total_reward, terminated
            # return distance_reward, distance_terminated

        def _extract_obs(self):
            # Fetch EEF pose; if this fails we want an explicit error
            eef_pos, eef_quat = self._robot.get_eef_pose(arm="right")

            log_obj = self._get_target_log()
            assert log_obj is not None, "Target log object not found in scene"
            log_pos, log_quat = log_obj.get_position_orientation()

            v = torch.cat(
                [
                    torch.as_tensor(eef_pos, dtype=torch.float32).flatten(),
                    torch.as_tensor(eef_quat, dtype=torch.float32).flatten(),
                    torch.as_tensor(log_pos, dtype=torch.float32).flatten(),
                    torch.as_tensor(log_quat, dtype=torch.float32).flatten(),
                    torch.as_tensor([self._last_grip_binary], dtype=torch.float32),
                ],
                dim=0,
            )
            return v.detach().cpu().numpy().astype(np.float32)

        # ---------------- Gym API ----------------

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            # Reset base env (scene.reset under the hood)
            self._env.reset()
            # Let simulator build physics handles
            for _ in range(5):
                og.sim.step()

            # Set camera position after reset (match simple_task_load.py)
            og.sim.viewer_camera.set_position_orientation(
                position=[
                    -0.5867,
                    1.0141,
                    0.8647,
                ],
                orientation=[
                    -0.2521,
                    0.5779,
                    0.7115,
                    -0.3104,
                ],
            )

            # Let physics settle a bit with zero actions, but do NOT use execute_init
            self._env.lock_health_changes()
            for _ in range(5):
                self._env.step(
                    action=np.zeros((self._action_dim,), dtype=np.float32)
                )
            self._env.unlock_health_changes()

            self._episode_step = 0
            self._last_grip_binary = 1.0
            return self._extract_obs(), {}

        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            full = np.zeros((self._action_dim,), dtype=np.float32)

            arm_action = action[: self._agent_arm_dim]
            grip_scalar = action[self._agent_arm_dim]

            full[self._right_arm_idx] = arm_action * self._arm_action_scale
            # Binary gripper: threshold scalar action to {-1, +1}
            self._last_grip_binary = 1.0 if float(grip_scalar) >= 0.0 else -1.0
            full[self._right_grip_idx] = self._last_grip_binary

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
        # Assume last action dim is gripper, others are arm IK DOFs
        arm_dim = act_dim - 1 if act_dim > 1 else act_dim
        # Slightly larger networks (as in electrical script)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )
        # Start with moderate exploration on arm, lower std on gripper
        log_std_init = torch.full((act_dim,), -0.5)
        if act_dim > 1:
            # Last dimension corresponds to gripper; make it more deterministic
            log_std_init[arm_dim:] = -2.0
        self.log_std = nn.Parameter(log_std_init)

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
            logprob = dist.log_prob(pre).sum(-1) - torch.log(
                1 - squashed.pow(2) + 1e-6
            ).sum(-1)
        else:
            # action is squashed in [-1,1]; invert tanh to get pre-squash
            eps = 1e-6
            a = torch.clamp(action, -1 + eps, 1 - eps)
            pre = 0.5 * (torch.log1p(a) - torch.log1p(-a))
            squashed = a
            logprob = dist.log_prob(pre).sum(-1) - torch.log(
                1 - squashed.pow(2) + 1e-6
            ).sum(-1)
        entropy = dist.entropy().sum(-1)
        return squashed, logprob, entropy, self.critic(x)


def run_eval_episode(
    env, agent, device, eval_videos_dir, iteration_num, capture_video=True, episode_idx=None
):
    """Run a single evaluation episode and optionally save an AVI with on-frame cumulative reward text.

    Returns (total_reward, terminated, truncated, health_avi_path_or_None, reward_avi_path_or_None)
    """
    fps = 30

    # Reset environment
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

    vw = None
    avi_path = None
    # Always capture a video for this episode; we'll decide later whether to keep it
    suffix = f"_ep{int(episode_idx)+1}" if episode_idx is not None else ""
    avi_path = os.path.join(
        eval_videos_dir, f"thermal_eval_iteration_{iteration_num}{suffix}.avi"
    )
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(avi_path, fourcc, fps, (1080, 720))

    # Store original camera pose
    orig_cam_pos, orig_cam_quat = og.sim.viewer_camera.get_position_orientation()
    if hasattr(orig_cam_pos, "tolist"):
        orig_cam_pos = orig_cam_pos.tolist()
    if hasattr(orig_cam_quat, "tolist"):
        orig_cam_quat = orig_cam_quat.tolist()
    main_cam_pos, main_cam_quat = orig_cam_pos, orig_cam_quat
    og.sim.viewer_camera.set_position_orientation(
        position=main_cam_pos, orientation=main_cam_quat
    )
    og.sim.render()

    total_reward = 0.0
    final_terminated = False
    final_truncated = False
    final_robot_health = None

    # Track robot health and distance / OnFire for plotting alongside sim video
    robot_health_series = []
    last_robot_health = 100.0
    # Cumulative reward over time
    reward_series = []

    # Access base DamageableEnvironment for health info and log state
    base = env
    while hasattr(base, "env"):
        base = base.env

    for step in range(EVAL_MAX_STEPS):
        with torch.no_grad():
            # Deterministic mean action for evaluation
            mean = agent.actor_mean(obs_tensor.unsqueeze(0))
            action = torch.tanh(mean)
        action_np = action.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action_np)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        total_reward += float(reward)
        reward_series.append(total_reward)

        # Robot health from base env
        robot = getattr(base, "_robot", None)
        if robot is not None and hasattr(robot, "get_obs_dict"):
            rob_obs = robot.get_obs_dict()
            last_robot_health = float(rob_obs.get("health", last_robot_health))
            final_robot_health = last_robot_health
        robot_health_series.append(last_robot_health)

        if vw is not None:
            # Capture from main camera
            og.sim.viewer_camera.set_position_orientation(
                position=main_cam_pos, orientation=main_cam_quat
            )
            og.sim.render()
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
            rgb_np = rgb.cpu().numpy()[:, :, :3]
            frame = cv2.resize(rgb_np, (1080, 720))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            text = f"Reward: {total_reward:.2f}"
            cv2.putText(
                frame_bgr,
                text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            vw.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8))

        if terminated or truncated:
            final_terminated = bool(terminated)
            final_truncated = bool(truncated)
            break

    # Ensure final health value
    if final_robot_health is None:
        final_robot_health = last_robot_health

    health_avi_path = None
    reward_avi_path = None

    if vw is not None:
        # Play the last frame with final reward 60 more times (2 seconds at 30fps)
        for _ in range(60):
            og.sim.viewer_camera.set_position_orientation(
                position=main_cam_pos, orientation=main_cam_quat
            )
            og.sim.render()
            rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
            rgb_np = rgb.cpu().numpy()[:, :, :3]
            frame = cv2.resize(rgb_np, (1080, 720))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            text = f"Final Reward: {total_reward:.2f}"
            cv2.putText(
                frame_bgr,
                text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            vw.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8))

        vw.release()
        import gc as _gc

        _gc.collect()
        print(f"Evaluation video saved: {os.path.basename(avi_path)}")
        sim_path = avi_path

        # Build robot-health-over-time MP4 using Matplotlib
        health_mp4 = os.path.join(
            eval_videos_dir,
            f"thermal_eval_iteration_{iteration_num}{suffix}_robot_health.mp4",
        )
        # Extend by 2s hold
        if len(robot_health_series) > 0:
            robot_health_series_ext = robot_health_series + [
                robot_health_series[-1]
            ] * 60
        else:
            robot_health_series_ext = [100.0] * 60
        T = len(robot_health_series_ext)
        y_min, y_max = 0.0, 100.0
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        line_h, = ax.plot(
            [], [], lw=6, color="tab:blue", label="Robot health"
        )
        ax.set_xlim(0, max(1, T) / fps)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Time (s)", fontsize=20)
        ax.set_ylabel("Health (%)", fontsize=20)
        ax.set_title("Robot Health Over Time", fontsize=26)
        ax.legend(loc="best", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16, width=1.5)
        ax.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_health():
            line_h.set_data([], [])
            return (line_h,)

        def animate_health(i):
            x = [k / fps for k in range(1, i + 2)]
            y = robot_health_series_ext[: i + 1]
            line_h.set_data(x, y)
            return (line_h,)

        ani = animation.FuncAnimation(
            fig,
            animate_health,
            init_func=init_health,
            frames=T,
            interval=1000 / fps,
            blit=True,
        )
        writer = animation.FFMpegWriter(
            fps=fps, codec="mpeg4", extra_args=["-vcodec", "mpeg4", "-qscale", "5"]
        )
        ani.save(health_mp4, writer=writer)
        plt.close(fig)

        # Side-by-side: sim video + health plot
        with_health_path = os.path.join(
            eval_videos_dir,
            f"thermal_eval_iteration_{iteration_num}{suffix}_with_health.mp4",
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                sim_path,
                "-i",
                health_mp4,
                "-filter_complex",
                "[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]",
                "-map",
                "[v]",
                "-c:v",
                "mpeg4",
                "-q:v",
                "5",
                with_health_path,
            ],
            check=True,
        )
        # Clean up intermediate health plot video
        os.remove(health_mp4)

        # Build reward-over-time MP4 using Matplotlib (cumulative episode reward)
        reward_plot_mp4 = os.path.join(
            eval_videos_dir,
            f"thermal_eval_iteration_{iteration_num}{suffix}_reward.mp4",
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
        line_r, = ax_r.plot(
            [], [], lw=6, color="tab:green", label="Cumulative Reward"
        )
        ax_r.set_xlim(0, max(1, T_r) / fps)
        ax_r.set_ylim(y_min_r, y_max_r)
        ax_r.set_xlabel("Time (s)", fontsize=20)
        ax_r.set_ylabel("Cumulative Reward", fontsize=20)
        ax_r.set_title("Total Episode Reward Over Time", fontsize=26)
        ax_r.legend(loc="best", fontsize=16)
        ax_r.tick_params(axis="both", which="major", labelsize=16, width=1.5)
        ax_r.grid(True, linewidth=1.0, alpha=0.3)
        plt.tight_layout()

        def init_reward():
            line_r.set_data([], [])
            return (line_r,)

        def animate_reward(i):
            x = [k / fps for k in range(1, i + 2)]
            y = reward_series_ext[: i + 1]
            line_r.set_data(x, y)
            return (line_r,)

        ani_r = animation.FuncAnimation(
            fig_r,
            animate_reward,
            init_func=init_reward,
            frames=T_r,
            interval=1000 / fps,
            blit=True,
        )
        writer_r = animation.FFMpegWriter(
            fps=fps, codec="mpeg4", extra_args=["-vcodec", "mpeg4", "-qscale", "5"]
        )
        ani_r.save(reward_plot_mp4, writer=writer_r)
        plt.close(fig_r)

        # Side-by-side: sim video + reward plot
        with_reward_path = os.path.join(
            eval_videos_dir,
            f"thermal_eval_iteration_{iteration_num}{suffix}_with_reward.mp4",
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                sim_path,
                "-i",
                reward_plot_mp4,
                "-filter_complex",
                "[0:v]scale=1080:720,setsar=1[left];[1:v]scale=1080:720,setsar=1[right];[left][right]hstack=inputs=2[v]",
                "-map",
                "[v]",
                "-c:v",
                "mpeg4",
                "-q:v",
                "5",
                with_reward_path,
            ],
            check=True,
        )
        # Clean up intermediate reward plot video and raw sim AVI
        os.remove(reward_plot_mp4)
        os.remove(sim_path)

        health_avi_path = with_health_path
        reward_avi_path = with_reward_path
        print(
            f"Evaluation video with health saved: {os.path.basename(health_avi_path)}"
        )
        print(
            f"Evaluation video with reward plot saved: {os.path.basename(reward_avi_path)}"
        )

    return (
        total_reward,
        final_terminated,
        final_truncated,
        health_avi_path,
        reward_avi_path,
        final_robot_health,
    )


def run_eval(env, agent, device, eval_videos_dir, iteration_num, num_episodes=5):
    task_successes = 0
    damage_failures = 0
    total_returns = []
    successful_steps = []
    saved_videos_health = []
    saved_videos_reward = []

    for i in range(num_episodes):
        capture = True
        (
            ep_ret,
            terminated,
            truncated,
            health_avi_path,
            reward_avi_path,
            robot_health,
        ) = run_eval_episode(
            env,
            agent,
            device,
            eval_videos_dir,
            iteration_num,
            capture_video=capture,
            episode_idx=i,
        )
        total_returns.append(ep_ret)
        if capture and health_avi_path is not None:
            saved_videos_health.append(health_avi_path)
        if capture and reward_avi_path is not None:
            saved_videos_reward.append(reward_avi_path)

        # Task success: terminated True AND reward positive (distance success gives +300)
        task_success = bool(terminated) and float(ep_ret) > 0
        if task_success:
            task_successes += 1

        # Damage failure: robot health < 100 at end
        if float(robot_health) < 100.0:
            damage_failures += 1

    task_success_rate = task_successes / float(max(1, num_episodes))
    damage_failure_rate = damage_failures / float(max(1, num_episodes))

    # Average return
    avg_return = float(np.mean(total_returns)) if len(total_returns) > 0 else 0.0

    # Combine all captured episodes into a single video (health plot version)
    combined_health_path = None
    if len(saved_videos_health) >= 1:
        cap_first = cv2.VideoCapture(saved_videos_health[0])
        fps = cap_first.get(cv2.CAP_PROP_FPS) or 15
        w = int(cap_first.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap_first.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        cap_first.release()

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        combined_health_path = os.path.join(
            eval_videos_dir, f"thermal_eval_iteration_{iteration_num}_health.avi"
        )
        out = cv2.VideoWriter(combined_health_path, fourcc, fps, (w, h))

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
            os.remove(p)

    combined_reward_path = None
    if len(saved_videos_reward) >= 1:
        cap_first = cv2.VideoCapture(saved_videos_reward[0])
        fps = cap_first.get(cv2.CAP_PROP_FPS) or 15
        w = int(cap_first.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap_first.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        cap_first.release()

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        combined_reward_path = os.path.join(
            eval_videos_dir, f"thermal_eval_iteration_{iteration_num}_reward.avi"
        )
        out = cv2.VideoWriter(combined_reward_path, fourcc, fps, (w, h))

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
            os.remove(p)

    saved_videos = []
    if combined_health_path is not None:
        saved_videos.append(combined_health_path)
    if combined_reward_path is not None:
        saved_videos.append(combined_reward_path)

    return task_success_rate, avg_return, saved_videos, task_success_rate, damage_failure_rate


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

    # Add random component to ensure unique run name
    run_name = (
        f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}__{random.randint(1000, 9999)}"
    )
    wandb.init(project="omnigibson-ppo-thermal", name=run_name, config=vars(args), id=None)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create eval videos directory
    os.makedirs(EVAL_VIDEOS_DIR, exist_ok=True)

    # Env setup
    scene_file = DEFAULT_SCENE_FILE
    envs = gym.vector.SyncVectorEnv(
        [make_env(scene_file, 0, args.capture_video, run_name)]
    )

    # Match viewer camera pose from simple_task_load.py
    og.sim.viewer_camera.set_position_orientation(
        position=[
            -0.5867,
            1.0141,
            0.8647,
        ],
        orientation=[
            -0.2521,
            0.5779,
            0.7115,
            -0.3104,
        ],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Track best eval average return for checkpoint saving
    best_eval_avg_return = None

    # Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
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
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_return = info["episode"]["r"]
                        print(f"global_step={global_step}, episodic_return={ep_return}")
                        wandb.log(
                            {
                                "charts/episodic_return": ep_return,
                                "charts/episodic_length": info["episode"]["l"],
                                "global_step": global_step,
                            }
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
                advantages[t] = (
                    lastgaelam
                ) = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
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
        wandb.log(
            {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/entropy_coef": float(current_ent_coef),
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "global_step": global_step,
            }
        )
        print("SPS:", int(global_step / (time.time() - start_time)))

        # Evaluation: every 10 iterations
        if iteration % 40 == 0:
            print(f"Running evaluation after iteration {iteration}...")
            base_env = envs.envs[0]
            while hasattr(base_env, "env"):
                base_env = base_env.env

            base_env.reset()
            (
                success_rate,
                eval_avg_return,
                _,
                task_success_rate,
                damage_failure_rate,
            ) = run_eval(
                base_env,
                agent,
                device,
                EVAL_VIDEOS_DIR,
                iteration,
                num_episodes=5,
            )

            # Save checkpoint if first eval or better than previous best
            if best_eval_avg_return is None or eval_avg_return > best_eval_avg_return:
                best_eval_avg_return = eval_avg_return
                torch.save(
                    {
                        "agent_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval_avg_return": best_eval_avg_return,
                        "iteration": iteration,
                        "global_step": global_step,
                    },
                    checkpoint_file,
                )
                print(
                    f"Saved new best checkpoint with avg return: {best_eval_avg_return:.3f}"
                )
            else:
                print(
                    f"Eval avg return {eval_avg_return:.3f} not better than best {best_eval_avg_return:.3f}, not saving checkpoint"
                )

            wandb.log(
                {
                    "eval_iteration": iteration,
                    "eval_success_rate": success_rate,
                    "eval_avg_return": eval_avg_return,
                    "eval_task_success_rate": task_success_rate,
                    "eval_damage_failure_rate": damage_failure_rate,
                }
            )
            print(
                f"Eval - success rate: {success_rate:.3f}, avg return: {eval_avg_return:.3f}, "
                f"task success: {task_success_rate:.3f}, damage failure: {damage_failure_rate:.3f}"
            )

            # Resync vector env state after direct base env usage
            next_obs, _ = envs.reset(seed=None)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)

    envs.close()
    wandb.finish()


