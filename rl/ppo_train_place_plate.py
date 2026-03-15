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
import pickle
import yaml

# OmniGibson + environment wrapper (minimal integration)
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson import object_states
from omnigibson.controllers.controller_base import IsGraspingState

def _ensure_omnigibson_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    og_src = os.path.join(repo_root, "OmniGibson")
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


_ensure_omnigibson_on_path()

from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.utils.misc_utils import (
    save_rgb_camera_video,
    save_rgb_health_video,
    save_rgb_force_video,
)
from omnigibson.macros import gm
from omnigibson.robots import manipulation_robot

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_OBJECT_STATES = True
manipulation_robot.m.MAX_ASSIST_FORCE = 500

DEFAULT_STATE_FILE = "safe-manipulation-benchmark/resources/saved_states/place_plate_init_state_well_over_mat.pkl"

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
    env_id: str = "OG-PlacePlate"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
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
    anneal_ent_coef: bool = True
    """Toggle entropy coefficient annealing (linear decay to 0 over training, like learning rate)"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.01

    init_log_std: float = -0.5
    """initial policy log_std (per-dim); std=exp(init_log_std) — higher = more exploration"""
    init_log_std_gripper: float = -1.5
    """initial log_std for gripper dimension only (slightly lower than arm for some stability)"""
    state_path: str = DEFAULT_STATE_FILE
    """path to the saved state pkl file"""

    # Logging
    wandb: bool = True
    """if toggled, Weights & Biases logging will be enabled"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(state_path, idx, capture_video, run_name):
    """Create an OmniGibson-wrapped env compatible with CleanRL's expectations."""

    class OGPlacePlateEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, state_file_path):
            # Persist state file path for reliable reloads
            self._state_file = state_file_path
            
            # Ensure fresh simulator
            # NOTE: We avoid bootstrapping a dummy DamageableEnvironment with no robots because
            # DamageableEnvironment.reset() expects at least one robot (robots[0]).
            if og.sim is not None:
                try:
                    og.sim.stop()
                except Exception:
                    pass
                og.clear()

            # Load config from teleop_pick_place.py
            config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
            cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

            # Overwrite configs for place plate task
            cfg["scene"]["scene_model"] = "house_single_floor"
            cfg["scene"]["not_load_object_categories"] = ["ottoman"]
            cfg["scene"]["load_room_instances"] = ["kitchen_0"]
            
            # FrankaMounted robot setup
            cfg["robots"][0] = {
                "type": "FrankaMounted",
                "name": "franka0",
                "position": [5.7, -1.4, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "grasping_mode": "assisted",
                "obs_modalities": ["rgb", "depth"],
                "action_normalize": False,
                "self_collisions": True,
                "controller_config": {
                    "arm_0": {
                        "name": "InverseKinematicsController",
                        "mode": "pose_delta_ori",  # 6D delta position + delta orientation
                        "command_input_limits": None,
                    },
                    "gripper_0": {
                        "name": "MultiFingerGripperController",
                        "command_input_limits": (0.0, 1.0),
                        "mode": "smooth",
                        "isaac_kp": 4000.0,
                        "isaac_kd": 2000.0,
                    },
                },
            }

            cfg["env"]["external_sensors"] = []
            # Include plate and place_mat (BEHAVIOR nxzfmz) to match teleop_pick_place_load.py
            PLACE_MAT_POS = [5.185426712036133, -1.8776537656784058, 0.9251976013183594]
            PLACE_MAT_SCALE = [0.3, 0.3, 0.3]
            cfg["objects"] = [
                {
                    "type": "DatasetObject",
                    "name": "plate",
                    "category": "plate",
                    "model": "ntedfx",
                    "position": [5.4, -1.7, 0.95],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                {
                    "type": "DatasetObject",
                    "name": "place_mat",
                    "category": "place_mat",
                    "model": "nxzfmz",
                    "position": PLACE_MAT_POS,
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "scale": PLACE_MAT_SCALE,
                },
            ]

            # Create environment
            self._env = DamageableEnvironment(configs=cfg, reward_fn=self._reward_fn)
            self._env.reset()
            
            # Step simulation to ensure robot is fully initialized
            for _ in range(10):
                og.sim.step()
            
            # Load state from pkl file
            if not os.path.exists(self._state_file):
                raise FileNotFoundError(f"State file not found at {self._state_file}")
            
            with open(self._state_file, "rb") as f:
                # Cache init state so reset() can reuse it without re-reading from disk
                self._init_state_flat_array = pickle.load(f)
            og.sim.load_state(self._init_state_flat_array, serialized=True)
            # Reset all damage / health state after loading the initial snapshot
            try:
                if hasattr(self._env, "get_damageable_objects"):
                    for obj in self._env.get_damageable_objects():
                        if hasattr(obj, "_initialize_health"):
                            obj._initialize_health()
                        if hasattr(obj, "reset_damage_evaluators"):
                            obj.reset_damage_evaluators()
            except Exception:
                pass
            
            # Sync robot and run sim steps
            assert len(self._env.robots) > 0
            self._robot = self._env.robots[0]
            self._robot.keep_still()
            for _ in range(10):
                self._robot.keep_still()
                og.sim.step()

            # Cache place_mat object handle for OnTop(success) checks
            self._place_mat_obj = None
            try:
                self._place_mat_obj = self._env.scene.object_registry("name", "place_mat")
            except Exception:
                self._place_mat_obj = None
            # Set place_mat fixed_base after settling (like fireplace in firewood.py / teleop_firewood.py)
            if self._place_mat_obj is not None:
                self._place_mat_obj.fixed_base = True
                self._place_mat_obj.keep_still()

            # Set camera position
            try:
                og.sim.viewer_camera.set_position_orientation(
                    position=[4.2998127937316895, -0.5513805747032166, 1.6389135122299194],
                    orientation=[-0.21554666757583618, 0.5899057388305664, 0.7309074997901917, -0.2670675814151764],
                )
            except Exception:
                pass

            # Action space: arm position control + binary gripper.
            self._action_dim = int(self._robot.action_dim)
            self._default_arm = self._robot.default_arm
            self._arm_idx = self._robot.arm_action_idx[self._default_arm].cpu().numpy().astype(np.int64)
            self._gripper_idx = self._robot.gripper_action_idx[self._default_arm].cpu().numpy().astype(np.int64)
            # IK controller command is 6D (dx,dy,dz, d_angular[3]); we expose first 3 dims + gripper to the agent.
            self._agent_arm_dim = 3  # position-only control
            self._agent_act_dim = self._agent_arm_dim + 1  # arm (3) + gripper (1)
            
            # Action scaling for position deltas
            self._pos_action_scale = 0.01  # Scale for delta position
            
            self._episode_step = 0
            self._max_episode_steps = 200
            self._last_base_obs = None
            self._episode_distance_reward = 0.0
            self._episode_damage_reward = 0.0
            # Goal for place-plate task (world coordinates)
            self._goal_pos = torch.tensor(
                [5.156736850738525, -1.7949283123016357, 0.9105868339538574],
                dtype=torch.float32,
            )
            self._success_eps = 0.2
            # Success hold: require N consecutive steps satisfying success condition
            self._success_hold_steps = 5
            self._success_hold_counter = 0
            # Previous step's plate health (for per-step damage = change in health)
            self._plate_health_prev = 100.0

            # Observation space: 6D (EEF pos + plate pos)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            # Action space: 4D (dx, dy, dz, gripper)
            self.action_space = gym.spaces.Box(
                low=-np.ones((self._agent_act_dim,), dtype=np.float32),
                high=np.ones((self._agent_act_dim,), dtype=np.float32),
                dtype=np.float32
            )

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

        def _damage_reward_fn(self, env, obs):
            """
            Reward is negative (change in plate health) this timestep only.
            step_damage = max(0, prev_health - current_health). Not from 100.
            Terminate episode when plate health reaches 0 (broke the plate).
            """
            plate_obj = self._get_plate_obj()
            if plate_obj is None or not hasattr(plate_obj, "health"):
                return 0.0, False
            try:
                current_health = float(plate_obj.health)
            except Exception:
                return 0.0, False
            step_damage = max(0.0, self._plate_health_prev - current_health)
            self._plate_health_prev = current_health
            terminate_due_to_damage = current_health <= 0.0
            return -float(step_damage), terminate_due_to_damage

        def _distance_reward_fn(self, env, obs):
            """Reward = -distance to goal. Success: plate OnTop of place_mat, held for 5 steps."""
            plate_obj = self._get_plate_obj()
            if plate_obj is None:
                return 0.0, False
            try:
                plate_pos, _ = plate_obj.get_position_orientation()
                dist = float(torch.norm(torch.as_tensor(plate_pos, dtype=torch.float32) - self._goal_pos).item())
            except Exception:
                return 0.0, False
            reward = -dist
            try:
                on_top_mat = (
                    self._place_mat_obj is not None
                    and object_states.OnTop in plate_obj.states
                    and bool(plate_obj.states[object_states.OnTop].get_value(self._place_mat_obj))
                )
            except Exception:
                on_top_mat = False
            meets = on_top_mat
            if self._success_hold_counter > 0:
                if meets:
                    self._success_hold_counter -= 1
                    success = self._success_hold_counter == 0
                else:
                    self._success_hold_counter = 0
                    success = False
            else:
                success = False
                if meets:
                    self._success_hold_counter = int(self._success_hold_steps)
            if success:
                reward = 100.0
            return reward, success

        def _reward_fn(self, env, obs):
            # Reward: damage (2.0) + distance. Terminate when: success (place on mat) OR plate health <= 0.
            dist_r, success = self._distance_reward_fn(env, obs)
            dmg_r, terminate_due_to_damage = self._damage_reward_fn(env, obs)
            reward = float(2.0 * dmg_r + dist_r)
            terminated = bool(success or terminate_due_to_damage)
            # Store components on base env so wrapper can accumulate for logging (no duplicate computation)
            env._last_step_dist_r = float(dist_r)
            env._last_step_dmg_r = float(dmg_r)
            return max(-300.0, reward), terminated

        def _extract_obs(self):
            # Safely fetch EEF position
            try:
                eef_pos = self._robot.get_eef_position(self._default_arm)
            except Exception:
                try:
                    og.sim.play()
                except Exception:
                    pass
                try:
                    for _ in range(2):
                        og.sim.step()
                    eef_pos = self._robot.get_eef_position(self._default_arm)
                except Exception:
                    eef_pos = torch.zeros(3)
            
            plate = self._get_plate_obj()
            if plate is not None:
                plate_pos, _ = plate.get_position_orientation()
            else:
                plate_pos = torch.zeros(3)
            
            v = torch.cat([
                torch.as_tensor(eef_pos, dtype=torch.float32).flatten(),
                torch.as_tensor(plate_pos, dtype=torch.float32).flatten()
            ], dim=0)
            return v.detach().cpu().numpy().astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._success_hold_counter = 0

            # Data-collection style reset: keep the env alive, reset, then restore the saved pkl state.
            self._env.reset()

            # Let simulator rebuild any handles
            try:
                for _ in range(2):
                    og.sim.step()
            except Exception:
                pass

            # Restore exact initial state (no pose randomization requested)
            og.sim.load_state(self._init_state_flat_array, serialized=True)

            # Reset all damage / health state after loading the initial snapshot
            try:
                if hasattr(self._env, "get_damageable_objects"):
                    for obj in self._env.get_damageable_objects():
                        if hasattr(obj, "_initialize_health"):
                            obj._initialize_health()
                        if hasattr(obj, "reset_damage_evaluators"):
                            obj.reset_damage_evaluators()
            except Exception:
                pass

            # Sync robot and step for stability
            self._robot = self._env.robots[0]
            self._robot.keep_still()
            for _ in range(10):
                self._robot.keep_still()
                og.sim.step()

            # Reset health and damage evaluators so plate starts at 100
            try:
                if hasattr(self._env, "get_damageable_objects"):
                    for obj in self._env.get_damageable_objects():
                        if hasattr(obj, "_initialize_health"):
                            obj._initialize_health()
                        if hasattr(obj, "reset_damage_evaluators"):
                            obj.reset_damage_evaluators()
            except Exception:
                pass

            # Ensure plate health is 100 before starting the next episode (retry reset if not)
            self._plate_health_prev = 100.0
            plate_obj = self._get_plate_obj()
            if plate_obj is not None and hasattr(plate_obj, "health"):
                for _ in range(5):
                    try:
                        h = float(plate_obj.health)
                        if h == 100.0:
                            self._plate_health_prev = h
                            break
                        # Re-initialize health and evaluators for damageable objects and try again
                        if hasattr(self._env, "get_damageable_objects"):
                            for obj in self._env.get_damageable_objects():
                                if hasattr(obj, "_initialize_health"):
                                    obj._initialize_health()
                                if hasattr(obj, "reset_damage_evaluators"):
                                    obj.reset_damage_evaluators()
                        for _ in range(5):
                            og.sim.step()
                    except Exception:
                        break
                else:
                    try:
                        h = float(plate_obj.health)
                        if h != 100.0:
                            raise RuntimeError(
                                f"Plate health is {h:.2f} after reset; must be 100 before starting episode. "
                                "Check that _initialize_health() and reset_damage_evaluators() run correctly."
                            )
                    except (TypeError, AttributeError):
                        pass
                self._plate_health_prev = float(plate_obj.health) if hasattr(plate_obj, "health") else 100.0

            # Re-cache place_mat (handles can change across resets / state loads)
            self._place_mat_obj = None
            try:
                self._place_mat_obj = self._env.scene.object_registry("name", "place_mat")
            except Exception:
                self._place_mat_obj = None

            # Force gripper closed each reset (plate is grasped in init state)
            try:
                close = np.zeros((self._action_dim,), dtype=np.float32)
                close[self._gripper_idx] = -1.0
                # IMPORTANT: do not call env.step() inside reset() (it pollutes episode stats / step counters).
                # Apply the command directly to the robot and step the simulator.
                close_t = torch.as_tensor(close, dtype=torch.float32)
                for _ in range(5):
                    self._robot.apply_action(close_t)
                    try:
                        og.sim.step()
                    except Exception:
                        pass
            except Exception:
                pass

            # Set place_mat fixed_base after env settles (like fireplace in firewood.py / teleop_firewood.py)
            if self._place_mat_obj is not None:
                self._place_mat_obj.fixed_base = True
                self._place_mat_obj.keep_still()

            try:
                og.sim.viewer_camera.set_position_orientation(
                    position=[4.2998127937316895, -0.5513805747032166, 1.6389135122299194],
                    orientation=[-0.21554666757583618, 0.5899057388305664, 0.7309074997901917, -0.2670675814151764],
                )
            except Exception:
                pass
            
            self._episode_step = 0
            self._episode_distance_reward = 0.0
            self._episode_damage_reward = 0.0
            return self._extract_obs(), {}

        def step(self, action):
            # Re-apply place_mat fixed_base every step (fixed_base from reset wasn't persisting)
            if self._place_mat_obj is not None:
                try:
                    self._place_mat_obj.fixed_base = True
                    self._place_mat_obj.keep_still()
                except Exception:
                    pass

            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            
            # Build full action vector
            full = np.zeros((self._action_dim,), dtype=np.float32)
            
            # Extract arm action (3D) and gripper (1D)
            arm_action = action[:self._agent_arm_dim]
            gripper_raw = float(action[self._agent_arm_dim]) if len(action) > self._agent_arm_dim else -1.0
            # Simple binary gripper: >= 0 opens, < 0 closes (matches working ppo_train_cleanrl.py)
            # Policy can explore both; damage reward will teach it not to open early
            gripper_cmd = 1.0 if gripper_raw >= 0.0 else -1.0
            
            # Scale position delta; keep orientation command at zero so policy cannot control it
            arm_action_scaled = np.zeros((self._arm_idx.shape[0],), dtype=np.float32)
            arm_action_scaled[:3] = arm_action[:3] * self._pos_action_scale  # Delta position
            
            full[self._arm_idx] = arm_action_scaled
            full[self._gripper_idx] = gripper_cmd
            
            obs, reward, terminated, truncated, info = self._env.step(action=full)
            # Cache the last base observation dict for downstream eval access
            self._last_base_obs = obs if isinstance(obs, dict) else None
            # Accumulate distance and damage from the same values used in _reward_fn (set there on env)
            self._episode_distance_reward += getattr(self._env, "_last_step_dist_r", 0.0)
            self._episode_damage_reward += getattr(self._env, "_last_step_dmg_r", 0.0)
            self._episode_step += 1
            if self._episode_step >= self._max_episode_steps:
                truncated = True
            # On episode end, add episode-level metrics to info for WandB
            if terminated or truncated:
                final_health = 100.0
                plate_obj = self._get_plate_obj()
                if plate_obj is not None and hasattr(plate_obj, "health"):
                    try:
                        final_health = float(plate_obj.health)
                    except Exception:
                        pass
                info["distance_reward"] = self._episode_distance_reward
                info["damage_reward"] = self._episode_damage_reward
                info["final_plate_health"] = final_health
                self._episode_distance_reward = 0.0
                self._episode_damage_reward = 0.0
            # Distinguish success (placed on mat) from failure (plate broke): only count as success if terminated with plate health > 0
            if terminated:
                plate_obj = self._get_plate_obj()
                if plate_obj is not None and hasattr(plate_obj, "health"):
                    try:
                        info["episode_success"] = float(plate_obj.health) > 0.0
                    except Exception:
                        info["episode_success"] = True
                else:
                    info["episode_success"] = True
            return self._extract_obs(), float(reward), bool(terminated), bool(truncated), info

    def thunk():
        env = OGPlacePlateEnv(state_file_path=state_path)
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
    def __init__(self, envs, init_log_std: float = -1.0, init_log_std_gripper: float = -2.5):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.array(envs.single_action_space.shape).prod())
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)), nn.Tanh(),
            layer_init(nn.Linear(512, 512)), nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)), nn.Tanh(),
            layer_init(nn.Linear(512, 512)), nn.Tanh(),
            layer_init(nn.Linear(512, act_dim), std=0.01),
        )
        # Lower std for gripper dimension when present (act_dim > 3)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        with torch.no_grad():
            if act_dim > 3:
                self.log_std[act_dim - 1] = float(init_log_std_gripper)
                # NO bias: let the policy explore open/close ~50/50, then damage reward teaches it not to open early

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
    # Initialize WandB (optional; disabled by default)
    if args.wandb:
        wandb.init(project="omnigibson-ppo-place-plate", name=run_name, config=vars(args))
        _wandb_log = wandb.log
        _wandb_finish = wandb.finish
    else:
        _wandb_log = lambda *_, **__: None  # noqa: E731
        _wandb_finish = lambda *_, **__: None  # noqa: E731

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create videos directory for early training episode recordings
    videos_dir = "safe-manipulation-benchmark/rl/eval_videos"
    os.makedirs(videos_dir, exist_ok=True)
    early_train_videos_dir = os.path.join(videos_dir, "training_early")
    os.makedirs(early_train_videos_dir, exist_ok=True)
    checkpoint_dir = "safe-manipulation-benchmark/rl/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    NUM_EARLY_TRAIN_VIDEOS = 3  # record force/health videos for first N training episodes

    # env setup
    # Build OmniGibson vector envs using our adapter
    state_file = args.state_path
    # Use a single env wrapper directly to avoid vectorization overhead with global simulator
    envs = gym.vector.SyncVectorEnv([make_env(state_file, 0, args.capture_video, run_name)])
    # Set viewer camera pose
    try:
        og.sim.viewer_camera.set_position_orientation(
            position=[4.2998127937316895, -0.5513805747032166, 1.6389135122299194],
            orientation=[-0.21554666757583618, 0.5899057388305664, 0.7309074997901917, -0.2670675814151764],
        )
    except Exception:
        pass
    # Continuous action space expected
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(envs, init_log_std=args.init_log_std, init_log_std_gripper=args.init_log_std_gripper).to(device)
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

    # Buffers for early training episode videos (same style as eval: camera + health + forces)
    early_episode_video_count = 0
    train_frame_list = []
    train_health_list = []
    train_force_impact = []
    train_force_qs = []
    plate_force_key_train = None
    _train_fps = 15

    for iteration in range(1, args.num_iterations + 1):
        # Annealing: frac goes from 1 to 0 over training (same schedule for lr and ent_coef)
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        if args.anneal_lr:
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if args.anneal_ent_coef:
            ent_coef_now = frac * args.ent_coef
        else:
            ent_coef_now = args.ent_coef

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

            # Grab latest plate health via DamageableMixin.health on plate object
            final_plate_health = None
            try:
                base = envs.envs[0]
                while hasattr(base, "env"):
                    base = base.env
                dmg_env = getattr(base, "_env", None)
                if dmg_env is not None:
                    plate_obj = None
                    for nm in ["plate", "glass_plate", "target_object"]:
                        try:
                            plate_obj = dmg_env.scene.object_registry("name", nm)
                        except Exception:
                            plate_obj = None
                        if plate_obj is not None:
                            break
                    if plate_obj is not None and hasattr(plate_obj, "health"):
                        final_plate_health = float(plate_obj.health)
            except Exception:
                final_plate_health = None

            # Collect frames/health/forces for first few training episodes (same style as eval videos)
            episode_just_ended = bool(next_done.any().item()) if hasattr(next_done, "any") else bool(next_done[0].item())
            if early_episode_video_count < NUM_EARLY_TRAIN_VIDEOS:
                try:
                    rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
                    rgb_np = np.asarray(rgb.cpu().numpy()[:, :, :3], dtype=np.uint8)
                    frame = cv2.resize(rgb_np, (640, 360))
                    train_frame_list.append(frame)
                    train_health_list.append(final_plate_health if final_plate_health is not None else 100.0)
                    dmg = infos.get("damage_info") or {}
                    plate_dmg = dmg.get("plate") or {}
                    link_name = next(iter(plate_dmg.keys()), "base_link")
                    plate_force_key_train = plate_force_key_train or f"plate@{link_name}"
                    mech = plate_dmg.get(link_name, {}).get("mechanical") or {}
                    i_val = mech.get("impact_forces", 0.0)
                    q_val = mech.get("filtered_qs_forces", 0.0)
                    if isinstance(i_val, (list, np.ndarray)):
                        i_val = float(i_val[-1]) if len(i_val) > 0 else 0.0
                    else:
                        i_val = float(i_val)
                    if isinstance(q_val, (list, np.ndarray)):
                        q_val = float(q_val[-1]) if len(q_val) > 0 else 0.0
                    else:
                        q_val = float(q_val)
                    train_force_impact.append(i_val)
                    train_force_qs.append(q_val)
                except Exception:
                    pass  # skip this step's force; padding when saving will align lengths
            if episode_just_ended and early_episode_video_count < NUM_EARLY_TRAIN_VIDEOS and len(train_frame_list) > 0:
                try:
                    n_hold = 30
                    last_f = train_frame_list[-1]
                    frames_ext = train_frame_list + [last_f] * n_hold
                    T = len(frames_ext)
                    health_ext = (train_health_list + [train_health_list[-1]] * n_hold) if train_health_list else [100.0] * T
                    fi_ext = train_force_impact + [train_force_impact[-1] if train_force_impact else 0.0] * n_hold
                    fq_ext = train_force_qs + [train_force_qs[-1] if train_force_qs else 0.0] * n_hold
                    fi_ext = (fi_ext + [fi_ext[-1]] * T)[:T]
                    fq_ext = (fq_ext + [fq_ext[-1]] * T)[:T]
                    health_ext = (health_ext + [health_ext[-1]] * T)[:T] if len(health_ext) < T else health_ext[:T]
                    imgs = np.array(frames_ext)
                    suffix = f"train_ep{early_episode_video_count}"
                    save_rgb_camera_video(os.path.join(early_train_videos_dir, suffix), imgs, fps=_train_fps)
                    save_rgb_health_video(os.path.join(early_train_videos_dir, f"{suffix}_health.mp4"), imgs, ["plate"], {"plate": list(health_ext)}, fps=_train_fps)
                    pk = plate_force_key_train or "plate@base_link"
                    save_rgb_force_video(os.path.join(early_train_videos_dir, f"{suffix}_forces.mp4"), imgs, [pk], {pk: {"impact_forces": fi_ext, "filtered_qs_forces": fq_ext}}, forces_to_plot=("impact_forces", "filtered_qs_forces"), fps=_train_fps, force_ylim=(0, 100))
                    print(f"Early training video saved: {suffix} (camera + health + forces)")
                except Exception as e:
                    print(f"Early training video save failed: {e}")
                train_frame_list = []
                train_health_list = []
                train_force_impact = []
                train_force_qs = []
                early_episode_video_count += 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_return = float(info["episode"]["r"])
                        dmg_r = info.get("damage_reward")
                        dmg_r = float(dmg_r) if dmg_r is not None else 0.0
                        dist_r = info.get("distance_reward")
                        dist_r = float(dist_r) if dist_r is not None else 0.0
                        fph = info.get("final_plate_health", final_plate_health)
                        fph = 100.0 if fph is None else float(fph)
                        damage_done = 100.0 - fph
                        if final_plate_health is not None:
                            print(f"global_step={global_step}, episodic_return={ep_return}, final_plate_health={final_plate_health:.2f}")
                        else:
                            print(f"global_step={global_step}, episodic_return={ep_return}")
                        _wandb_log({
                            "charts/episodic_return": ep_return,
                            "charts/episodic_length": info["episode"]["l"],
                            "charts/final_plate_health": fph,
                            "charts/distance_reward": dist_r,
                            "charts/damage_reward": dmg_r,
                            "charts/damage_done": damage_done,
                        }, step=int(global_step))

            # Gymnasium sometimes reports episode stats directly under infos["episode"]
            # (depending on wrapper / vector env behavior). Handle that too.
            if "episode" in infos:
                try:
                    ep = infos["episode"]
                    # ep is usually a dict with keys {"r","l","t"}; values may be scalars or 1D arrays
                    r = ep.get("r", None) if isinstance(ep, dict) else None
                    l = ep.get("l", None) if isinstance(ep, dict) else None
                    if r is not None and l is not None:
                        # If vectorized, take per-env entries; we force num_envs=1, so index 0 is fine
                        r0 = float(r[0]) if hasattr(r, "__len__") and not np.isscalar(r) else float(r)
                        l0 = int(l[0]) if hasattr(l, "__len__") and not np.isscalar(l) else int(l)
                        dmg_r = infos.get("damage_reward")
                        if dmg_r is not None and hasattr(dmg_r, "__len__") and not np.isscalar(dmg_r):
                            dmg_r = float(dmg_r[0])
                        else:
                            dmg_r = float(dmg_r) if dmg_r is not None else 0.0
                        dist_r = infos.get("distance_reward")
                        if dist_r is not None and hasattr(dist_r, "__len__") and not np.isscalar(dist_r):
                            dist_r = float(dist_r[0])
                        else:
                            dist_r = float(dist_r) if dist_r is not None else 0.0
                        fph = infos.get("final_plate_health", final_plate_health)
                        if fph is not None and hasattr(fph, "__len__") and not np.isscalar(fph):
                            fph = float(fph[0])
                        elif fph is not None:
                            fph = float(fph)
                        else:
                            fph = 100.0
                        damage_done = 100.0 - fph
                        if final_plate_health is not None:
                            print(f"global_step={global_step}, episodic_return={r0}, final_plate_health={final_plate_health:.2f}")
                        else:
                            print(f"global_step={global_step}, episodic_return={r0}")
                        _wandb_log(
                            {
                                "charts/episodic_return": r0,
                                "charts/episodic_length": l0,
                                "charts/final_plate_health": fph,
                                "charts/distance_reward": dist_r,
                                "charts/damage_reward": dmg_r,
                                "charts/damage_done": damage_done,
                            },
                            step=int(global_step),
                        )
                except Exception:
                    pass

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
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

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
        _wandb_log({
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/ent_coef": ent_coef_now,
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time)),
        }, step=int(global_step))
        print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()
    _wandb_finish()
