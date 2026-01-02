# TODO(junhong): we haven't made the global class ID mapping in the observation processor yet.

import sys
sys.path.insert(0, "/home/juxu/Research/safe-manipulation/rl-flow-matching")

from ast import Pass
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import json
import h5py
import pickle
import torch as th
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass

import omnigibson as og
from omnigibson import object_states
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm

from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.controllers.controller_base import IsGraspingState

from omnigibson.utils.transform_utils import quat2axisangle


from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper

from models.cfm_policy import CFMPolicy, PolicyConfig

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_TRANSITION_RULES = False


# ======================== Observation Processing ========================

@dataclass
class ObsProcessorConfig:
    """Configuration for observation processing."""
    # Image settings
    seg_img_size: tuple = (128, 128)  # Target size for segmentation images
    frame_stack: int = 2  # Number of frames to stack
    
    # Observation keys - mapping from env obs keys to policy keys
    seg_obs_keys: List[str] = None  # Will be set in __post_init__
    proprio_key: str = "franka0::proprio"
    
    # Robot settings
    robot_name: str = "franka0"
    
    def __post_init__(self):
        if self.seg_obs_keys is None:
            self.seg_obs_keys = [
                f"{self.robot_name}::{self.robot_name}:eef_link:Camera:0::seg_instance",
                "external::external_sensor0::seg_instance",
                "external::external_sensor1::seg_instance",
            ]


class ObservationProcessor:
    """
    Processes raw environment observations into the format expected by the policy.
    
    The policy expects:
        - seg_images: Dict[str, Tensor] with shape [B, frame_stack, H, W] (int32 segmentation IDs)
        - state: Tensor [B, state_dim] (proprio)
    """
    
    def __init__(self, config: ObsProcessorConfig = None, device: str = "cuda"):
        self.config = config or ObsProcessorConfig()
        self.device = device
        
        # Frame buffer for temporal stacking
        self.seg_buffers: Dict[str, deque] = {
            key: deque(maxlen=self.config.frame_stack) 
            for key in self.config.seg_obs_keys
        }
        self.initialized = False
    
    def reset(self):
        """Reset frame buffers on episode reset."""
        for key in self.seg_buffers:
            self.seg_buffers[key].clear()
        self.initialized = False
    
    def _resize_segmentation(self, seg_img: th.Tensor) -> th.Tensor:
        """
        Resize segmentation image using nearest-neighbor interpolation.
        Preserves integer class IDs without interpolation artifacts.
        
        Args:
            seg_img: Segmentation image [H, W] or [C, H, W]
        
        Returns:
            Resized image [target_H, target_W]
        """
        if seg_img.dim() == 2:
            seg_img = seg_img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif seg_img.dim() == 3:
            seg_img = seg_img.unsqueeze(0)  # [1, C, H, W]
        
        # Use nearest neighbor to preserve class IDs
        resized = F.interpolate(
            seg_img.float(),
            size=self.config.seg_img_size,
            mode='nearest'
        )
        
        return resized.squeeze(0).squeeze(0).long()  # [target_H, target_W]
    
    def _extract_proprio(self, obs: dict, robot) -> th.Tensor:
        """
        Extract proprioceptive state from observation.
        
        The proprio state typically includes:
            - Joint positions (7 for Franka arm)
            - Joint velocities (7)
            - EEF position (3) 
            - EEF orientation (4 quaternion)
            - Gripper state (1 or 2)
        
        Args:
            obs: Raw observation dict from environment
            robot: Robot object for additional state extraction
        
        Returns:
            Proprio tensor [state_dim]
        """
        # Try to get proprio from obs dict first (if saved by data collection)
        proprio_key = self.config.proprio_key
        if proprio_key in obs:
            proprio = obs[proprio_key]
            if isinstance(proprio, np.ndarray):
                proprio = th.from_numpy(proprio).float()
            return proprio.to(self.device)
        
        # Otherwise, construct proprio from robot state
        # Get joint positions and velocities
        joint_pos = robot.get_joint_positions()
        joint_vel = robot.get_joint_velocities()
        
        # Get EEF pose
        default_arm = robot.default_arm if hasattr(robot, "default_arm") else "0"
        eef_pos = robot.get_eef_position(default_arm)
        eef_ori = robot.get_eef_orientation(default_arm)
        eef_ori_aa = quat2axisangle(eef_ori)  # Convert to axis-angle (3,)

        
        # Get gripper state
        is_grasping = robot.is_grasping()
        gripper_state = th.tensor([1.0 if is_grasping.value == IsGraspingState.TRUE else 0.0])
        
        # Concatenate all proprio components
        proprio = th.cat([
            joint_pos.flatten(),
            joint_vel.flatten(),
            eef_pos.flatten(),
            eef_ori_aa.flatten(),
        ]).float()
        
        return proprio.to(self.device)
    
    def _extract_segmentation(self, obs: dict) -> th.Tensor:
        """
        Extract and resize a segmentation image from observation.
        
        Args:
            obs: Raw observation dict
            key: Key for the segmentation observation
        
        Returns:
            Resized segmentation [H, W] as long tensor
        """
        frank_seg = obs['franka0']['franka0:eef_link:Camera:0']['seg_instance']
        external_seg_0 = obs['external']['external_sensor0']['seg_instance']
        external_seg_1 = obs['external']['external_sensor1']['seg_instance']
        
        # seg_img = obs[key]
        
        # Convert to tensor if needed
        # if isinstance(seg_img, np.ndarray):
        #     seg_img = th.from_numpy(seg_img)
        
        # Handle different input shapes
        # if seg_img.dim() == 3:
        #     # [H, W, C] or [C, H, W] - take first channel or squeeze
        #     if seg_img.shape[-1] in [1, 3, 4]:  # [H, W, C]
        #         seg_img = seg_img[..., 0]
        #     else:  # [C, H, W]
        #         seg_img = seg_img[0]
        
        # Resize to target size
        # seg_img = self._resize_segmentation(seg_img.to(self.device))
        frank_seg = self._resize_segmentation(frank_seg.to(self.device))
        external_seg_0 = self._resize_segmentation(external_seg_0.to(self.device))
        external_seg_1 = self._resize_segmentation(external_seg_1.to(self.device))
        
        return frank_seg, external_seg_0, external_seg_1
    
    def process(self, obs: dict, robot) -> Dict[str, th.Tensor]:
        """
        Process raw environment observation into policy input format.
        
        Args:
            obs: Raw observation dict from env.step() or env.reset()
            robot: Robot object for proprio extraction
        
        Returns:
            Dict with keys:
                - 'extero': Dict[str, Tensor] with shape [1, frame_stack, H, W]
                - 'proprio': Tensor [1, state_dim]
        """
        # Extract and buffer segmentation images
        # for key in self.config.seg_obs_keys:

        frank_seg, external_seg_0, external_seg_1 = self._extract_segmentation(obs)
        self.seg_buffers['franka0::franka0:eef_link:Camera:0::seg_instance'].append(frank_seg)
        self.seg_buffers['external::external_sensor0::seg_instance'].append(external_seg_0)
        self.seg_buffers['external::external_sensor1::seg_instance'].append(external_seg_1)
        
        # Initialize buffers by repeating first frame if needed
        if not self.initialized:
            for key in self.config.seg_obs_keys:
                while len(self.seg_buffers[key]) < self.config.frame_stack:
                    # Repeat the first frame
                    first_frame = self.seg_buffers[key][0]
                    self.seg_buffers[key].appendleft(first_frame.clone())
            self.initialized = True
        
        # Stack frames for each segmentation view
        seg_images = {}
        for key in self.config.seg_obs_keys:
            # Stack: list of [H, W] -> [frame_stack, H, W]
            stacked = th.stack(list(self.seg_buffers[key]), dim=0)
            # Add batch dim: [1, frame_stack, H, W]
            seg_images[key] = stacked.unsqueeze(0)
        
        # Extract proprio state
        proprio = self._extract_proprio(obs, robot)
        # Add batch dim: [1, state_dim]
        proprio = proprio.unsqueeze(0)
        return {
            'extero': seg_images,
            'proprio': proprio,
        }


class ActionChunker:
    """
    Handles action chunking for policies that output multiple future actions.
    
    The policy outputs [action_chunk_size] actions, but we may want to:
    1. Execute only the first N actions before re-planning
    2. Blend between old and new action chunks for smoother motion
    """
    
    def __init__(
        self, 
        action_chunk_size: int = 8,
        execute_horizon: int = 4,  # How many actions to execute before re-planning
    ):
        self.action_chunk_size = action_chunk_size
        self.execute_horizon = execute_horizon
        
        self.current_chunk: Optional[th.Tensor] = None
        self.chunk_idx: int = 0
    
    def reset(self):
        """Reset on episode start."""
        self.current_chunk = None
        self.chunk_idx = 0
    
    def update_chunk(self, new_chunk: th.Tensor):
        """
        Update with a new action chunk from the policy.
        
        Args:
            new_chunk: Action chunk [action_chunk_size, action_dim]
        """
        self.current_chunk = new_chunk
        self.chunk_idx = 0
    
    def get_action(self) -> Optional[th.Tensor]:
        """
        Get the next action to execute.
        
        Returns:
            Action tensor [action_dim] or None if chunk exhausted
        """
        if self.current_chunk is None:
            return None
        
        if self.chunk_idx >= len(self.current_chunk):
            return None
        
        action = self.current_chunk[self.chunk_idx]
        self.chunk_idx += 1
        return action
    
    def needs_replan(self) -> bool:
        """Check if we need to query the policy for a new action chunk."""
        if self.current_chunk is None:
            return True
        return self.chunk_idx >= self.execute_horizon


# ======================== Environment Configuration ========================

FLOUR_INIT_POS = [6.00, 0.35, 1.3]
FLOUR_INIT_ORI = [0.0, 0.0, 0.0, 1.0]

BOTTLE_OF_WINE_INIT_POS = [6.00, 0.2, 1.3]
BOTTLE_OF_WINE_INIT_ORI = [0.0, 0.0, 0.0, 1.0]

WINEGLASS_INIT_POS = [6.00, 0.12, 1.3]
WINEGLASS_INIT_ORI = [0.0, 0.0, 0.0, 1.0]

BOTTLE_OF_WHISKEY_INIT_POS = [6.00, 0.0, 1.3]
BOTTLE_OF_WHISKEY_INIT_ORI = [0.0, 0.0, 0.0, 1.0]

BOTTLE_OF_BEER_INIT_POS = [6.00, 0.0, 1.3]
BOTTLE_OF_BEER_INIT_ORI = [0.0, 0.0, 0.0, 1.0]

SHELF_INIT_POS = [6.00, 0.2, 1.3]
SHELF_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
SHELF_SCALE = [0.4, 0.8, 0.5]

# Task objects are located in BEHAVIOR-1k/datasets/objects/*
TASK_OBJECTS = {
    "box_of_crackers": {
        "type": "DatasetObject",
        "name": "box_of_crackers",
        "category": "box_of_crackers",
        "model": "cmdigf",
        "position": [6.0, 0.2, 2.0],
        "orientation": [0.0, 0.0, 0.70710678, 0.70710678],
    }, 
    "bag_of_flour": {
        "type": "DatasetObject",
        "name": "book",
        "category": "bag_of_flour",
        "model": "rlejxx",
        "position": FLOUR_INIT_POS,
        "orientation": FLOUR_INIT_ORI,
        "scale": [1.0, 1.0, 1.0],
    },
    "bottle_of_wine": {
        "type": "DatasetObject",
        "name": "bottle_of_wine",
        "category": "bottle_of_wine",
        "model": "hnkiog",
        "position": BOTTLE_OF_WINE_INIT_POS,
        "orientation": BOTTLE_OF_WINE_INIT_ORI,
        "scale": [1.0, 1.0, 1.0],
    },
    "wineglass": {
        "type": "DatasetObject",
        "name": "wineglass",
        "category": "wineglass",
        "model": "adiwil",
        "position": WINEGLASS_INIT_POS,
        "orientation": WINEGLASS_INIT_ORI,
        "scale": [1.0, 1.0, 1.0],
    },
    "bottle_of_beer": {
        "type": "DatasetObject",
        "name": "bottle_of_beer",
        "category": "bottle_of_beer",
        "model": "dqfsgv",
        "position": BOTTLE_OF_BEER_INIT_POS,
        "orientation": BOTTLE_OF_BEER_INIT_ORI,
        "scale": [1.0, 1.0, 1.0],
    },
    "stand": {
        "type": "DatasetObject",
        "name": "stand",
        "category": "stand",
        "model": "vyrick",
        "position": SHELF_INIT_POS,
        "orientation": SHELF_INIT_ORI,
        "scale": SHELF_SCALE,
        "fixed_base": True,
    },
}


def reset_env(env):
    obs, info = env.reset()
    # load state
    with open("resources/saved_states/shelf_init_state.pkl", "rb") as f: state_flat_array = pickle.load(f)
    og.sim.load_state(state_flat_array, serialized=True)

    # TODO: Add object pose and scale randomization

    for _ in range(10): og.sim.step()
    return obs, info

def load_policy(checkpoint_path: str, device: str = "cuda") -> CFMPolicy:
    """
    Load a trained CFMPolicy from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on
    
    Returns:
        Loaded policy in eval mode
    """
    # Load checkpoint
    checkpoint = th.load(checkpoint_path, map_location=device)
    
    # Create policy with config (use default or load from checkpoint)
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = PolicyConfig()
    
    policy = CFMPolicy(config)
    
    # Load weights
    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
    elif "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint is just the state dict
        policy.load_state_dict(checkpoint)
    
    policy.to(device)
    policy.eval()
    
    return policy


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default="../rl-flow-matching/checkpoints/final.pth",
                        help='Path to policy checkpoint')
    parser.add_argument('--load_state', action='store_true', help='Load a saved state')
    parser.add_argument('--n_episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--device', type=str, default='cuda', help='Device for policy')
    parser.add_argument('--execute_horizon', type=int, default=1, 
                        help='Actions to execute before re-planning')
    parser.add_argument('--save_data', action='store_true', help='Save trajectory data')
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(0)
    th.manual_seed(0)
    
    # Load policy
    device = args.device if th.cuda.is_available() else "cpu"
    print(f"Loading policy from {args.checkpoint}...")
    policy = load_policy(args.checkpoint, device=device)
    policy.print_parameter_summary()
    
    # Create observation processor and action chunker
    obs_processor = ObservationProcessor(device=device)
    action_chunker = ActionChunker(
        action_chunk_size=policy.config.action_chunk_size,
        execute_horizon=args.execute_horizon,
    )
    
    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Overwrite any configs here
    cfg["scene"]["scene_model"] = "house_single_floor"
    cfg["scene"]["not_load_object_categories"] = ["ottoman"]
    cfg["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
    
    ############### Franka robot ###############
    # TODO(junhong): if we have a better way (a franka-specific config file), we should use that
    # Completely replace robot config to avoid Tiago-specific settings carrying over
    cfg["robots"][0] = {
        "type": "FrankaPanda",
        "name": "franka0",
        "position": [6.8, 0.2, 1.0],  # Match Tiago base position
        "orientation": [0.0, 0.0, 1.0, 0.0],
        "grasping_mode": "assisted",
        "obs_modalities": ["rgb", "depth"],
        "action_normalize": False,
        "self_collisions": True,
        # Franka has single arm (arm_0, gripper_0) instead of left/right
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                # "name": "JointController",
                "command_input_limits": None,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            },
        },
    }

    # Generate external sensors config automatically
    # Get robot name and type from config to construct correct prim path
    robot_name = cfg["robots"][0].get("name", "franka0")
    robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()

    # Add objects here
    cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]
    EXTERNAL_CAMERA_CONFIGS = {
        # Side camera (fixed to base_link frame)
        "external_sensor_0": {
            "position": [7.3920, -0.6436, 1.7519],
            "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
            "horizontal_aperture": 15.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
        },
        # Left Shoulder (fixed to base_link frame)
        "external_sensor_1": {
            # wrt base frame
            "position": [7.1264, 1.1205, 2.0117],
            "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
            "horizontal_aperture": 15.0,
            "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1",
        },
    }
    external_sensors_config = []
    for name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
        i = name.split("_")[-1]
        position = camera_cfg["position"]
        orientation = camera_cfg["orientation"]
        external_sensors_config.append({
            "sensor_type": "VisionSensor",
            "name": f"external_sensor{i}",
            "relative_prim_path": camera_cfg["relative_prim_path"],
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {
                "image_height": 256,
                "image_width": 256,
                "horizontal_aperture": camera_cfg["horizontal_aperture"],
            },
            "position": th.tensor(position, dtype=th.float32),
            "orientation": th.tensor(orientation, dtype=th.float32),
            "pose_frame": "world",
        })

    # In case we want to modify the robot sensors that were used during data collection
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {
                "image_height": 256,
                "image_width": 256,
            },
        },
    }
    cfg["env"]["external_sensors"] = external_sensors_config
    for robot_cfg in cfg["robots"]:
        robot_cfg["sensor_config"] = robot_sensor_config


    env = DamageableEnvironment(configs=cfg)        
    # env = DamageableDataCollectionWrapper(
    #     env=env,
    #     output_path=args.collect_hdf5_path,
    #     only_successes=False,
    #     enable_dump_filters=False,
    # )

    robot = env.robots[0]
    # set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([ 7.0659, -0.7141,  1.9185]),
        orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
    )
    for _ in range(10): og.sim.step()

    robot = env.robots[0]
    
    for _ in range(10):
        og.sim.step()

    # Set initial robot pose
    robot.set_joint_positions(th.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]))

    for _ in range(10):
        og.sim.step()

    # ======================== Evaluation Loop ========================
    print(f"\n{'='*60}")
    print("Starting Closed-Loop Evaluation")
    print(f"Episodes: {args.n_episodes}, Max steps: {args.max_steps}")
    print(f"Execute horizon: {args.execute_horizon}")
    print(f"{'='*60}\n")
    
    for episode in range(args.n_episodes):
        print(f"\n--- Episode {episode + 1}/{args.n_episodes} ---")
        
        # Reset environment and processors
        obs, info = reset_env(env)
        # obs, info = env.reset()
        obs_processor.reset()
        action_chunker.reset()
        
        if args.load_state:
            for _ in range(50):
                og.sim.step()
        
        episode_reward = 0.0
        episode_damage = 0.0
        
        for step in range(args.max_steps):
            # Query policy for new action chunk if needed
            if action_chunker.needs_replan():
                # Process observation
                policy_input = obs_processor.process(obs, robot)
                
                # Generate action chunk
                with th.no_grad():
                    action_chunk = policy.generate_action(
                        seg_images=policy_input['extero'],
                        state=policy_input['proprio'],
                    )
                
                # Update chunker with new actions
                action_chunker.update_chunk(action_chunk[0])  # Remove batch dim
            
            # Get next action from chunk
            action = action_chunker.get_action()
            
            if action is None:
                print(f"Warning: No action available at step {step}")
                break
            
            # Convert to numpy for environment
            action = action.cpu().numpy()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            if "damage_info" in info:
                # Sum up damage across objects
                for obj_name, damage_data in info["damage_info"].items():
                    if isinstance(damage_data, dict) and "total_damage" in damage_data:
                        episode_damage += damage_data["total_damage"]
            
            # Check termination
            if terminated or truncated:
                print(f"Episode ended at step {step + 1}: terminated={terminated}, truncated={truncated}")
                break
            
            # Progress logging
            if (step + 1) % 50 == 0:
                print(f"  Step {step + 1}/{args.max_steps}")
        
        print(f"Episode {episode + 1} complete:")
        print(f"  Total steps: {step + 1}")
        print(f"  Total reward: {episode_reward:.4f}")
        print(f"  Total damage: {episode_damage:.4f}")
    
    # Save collected data
    if args.save_data:
        print("\nSaving data...")
        env.save_data()
    
    print("\nEvaluation complete!")
    og.shutdown()


if __name__ == "__main__":
    __main__()
