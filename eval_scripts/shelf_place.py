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
import cv2
import torch as th
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass
from collections import defaultdict
import omnigibson as og
from omnigibson import object_states
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm
from scipy.spatial.transform import Rotation as R
import omnigibson.utils.transform_utils as T

from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.controllers.controller_base import IsGraspingState

from omnigibson.utils.transform_utils import quat2axisangle


from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper
from safety_benchmark.utils.misc_utils import save_rgb_camera_video, save_rgb_force_video, save_rgb_health_video

from models.cfm_policy import CFMPolicy, PolicyConfig
from dataset.b1k_dataset import B1KDataset 

gm.USE_GPU_DYNAMICS=False
gm.ENABLE_TRANSITION_RULES = False

def get_visualization_config(task_name, robot_name):
    if task_name == "shelve_item":
        return {
            "target_objects_health_with_links": [f"{robot_name}@eef_link", f"{robot_name}@panda_hand", f"{robot_name}@panda_leftfinger", f"{robot_name}@panda_rightfinger", "box_of_crackers@base_link", "stand@base_link", "book@base_link", "bottle_of_wine@base_link", "wineglass@base_link", "bottle_of_beer@base_link"],
            "target_objects_health": [robot_name, "box_of_crackers", "stand", "book", "bottle_of_wine", "wineglass", "bottle_of_beer"],
            "target_objects_forces": ["box_of_crackers@base_link", "book@base_link", "bottle_of_wine@base_link", "wineglass@base_link", "bottle_of_beer@base_link"],
            "force_keys": ["impact_forces"],
            "target_contact_bodies": ["stand"]
        }

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
    
    # Global class vocabulary - path to HDF5 file used for training
    # This is used to build consistent class_to_id mapping at inference
    vocab_hdf5_path: Optional[str] = None
    
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
        - seg_images: Dict[str, Tensor] with shape [B, frame_stack, H, W] (global class IDs)
        - state: Tensor [B, state_dim] (proprio)
    
    Global class ID mapping ensures consistent segmentation class indices between
    training (from HDF5 dataset) and inference (from live environment).
    """
    
    def __init__(self, config: ObsProcessorConfig = None, device: str = "cuda", objects_of_interest: List[str] = None):
        self.config = config or ObsProcessorConfig()
        self.device = device
        self.objects_of_interest = ["box_of_crackers", "book", "bottle_of_wine", "bottle_of_beer", "stand"] 
        # Frame buffer for temporal stacking
        self.seg_buffers: Dict[str, deque] = {
            key: deque(maxlen=self.config.frame_stack) 
            for key in self.config.seg_obs_keys
        }
        self.initialized = False
        
        # Global class vocabulary (built from training HDF5)
        self.class_to_id: Dict[str, int] = {"unknown": 0}
        self.id_to_class: Dict[int, str] = {0: "unknown"}
        self.num_seg_classes: int = 1
        
        # Load vocabulary from HDF5 if provided
        if self.config.vocab_hdf5_path:
            self._build_class_vocabulary_from_hdf5(self.config.vocab_hdf5_path)
    
    def _build_class_vocabulary_from_hdf5(self, hdf5_path: str):
        """
        Build global class vocabulary from training HDF5 file.
        Scans ALL timesteps in ALL demos to ensure all possible classes are captured.
        
        This should use the SAME HDF5 file used for training to ensure
        consistent class_to_id mapping at inference time.
        
        Args:
            hdf5_path: Path to training HDF5 file
        """
        all_classes = set()
        
        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]
            demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
            
            for demo_key in demo_keys:
                demo_grp = data_grp[demo_key]
                
                if "info" in demo_grp and "obs_info" in demo_grp["info"]:
                    obs_info_data = demo_grp["info"]["obs_info"]
                    num_timesteps = len(obs_info_data)
                    
                    # Iterate through ALL timesteps to gather all classes
                    for timestep in range(num_timesteps):
                        obs_info = json.loads(obs_info_data[timestep].decode("utf-8"))
                        
                        # Parse structure: obs_info[camera_type][camera_name]["seg_instance"]
                        for camera_type in obs_info:
                            if isinstance(obs_info[camera_type], dict):
                                for camera_name in obs_info[camera_type]:
                                    if isinstance(obs_info[camera_type][camera_name], dict):
                                        if "seg_instance" in obs_info[camera_type][camera_name]:
                                            seg_mapping = obs_info[camera_type][camera_name]["seg_instance"]
                                            all_classes.update(seg_mapping.values())
        
        # Create consistent global mapping (sorted for reproducibility)
        # Reserve index 0 for "unknown" class
        sorted_classes = sorted(all_classes)
        self.class_to_id = {"unknown": 0}
        self.id_to_class = {0: "unknown"}
        idx = 1
        for cls_name in sorted_classes:
            if self.objects_of_interest is None:
                self.class_to_id[cls_name] = idx
                self.id_to_class[idx] = cls_name
                idx += 1
            else:
                for obj in self.objects_of_interest:
                    # print(f"Checking if {cls_name} in {obj}, result: {cls_name in obj}")
                    if obj in cls_name: 
                        self.class_to_id[cls_name] = idx
                        self.id_to_class[idx] = cls_name
                        idx += 1
                        break
        self.num_seg_classes = len(self.class_to_id)
        print(f"Built class vocabulary with {self.num_seg_classes} classes (including 'unknown'):")
        for cls_name, idx in self.class_to_id.items():
            print(f"  {idx}: {cls_name}")
    
    def set_class_vocabulary(self, class_to_id: Dict[str, int], id_to_class: Dict[int, str]):
        """
        Manually set the class vocabulary (e.g., from a saved checkpoint or dataset).
        
        Args:
            class_to_id: Dict mapping class name -> global ID
            id_to_class: Dict mapping global ID -> class name
        """
        self.class_to_id = class_to_id
        self.id_to_class = id_to_class
        self.num_seg_classes = len(class_to_id)
        print(f"Set class vocabulary with {self.num_seg_classes} classes")
    
    def remap_seg_to_global_ids(self, seg_image: th.Tensor, obs_info: dict, camera_type: str, camera_name: str) -> th.Tensor:
        """
        Remap segmentation image from environment seg IDs to global class IDs.
        
        During inference, the environment provides seg IDs that may differ from training.
        This function maps them to consistent global class IDs using obs_info metadata.
        
        Args:
            seg_image: Segmentation image tensor [H, W] with environment seg IDs
            obs_info: Observation info dict from environment containing seg_instance mapping
            camera_type: Camera type (e.g., "franka0", "external")
            camera_name: Camera name (e.g., "franka0:eef_link:Camera:0", "external_sensor0")
        
        Returns:
            Remapped segmentation image with global class IDs
        """
        remapped = th.zeros_like(seg_image)
        
        # Get seg_id -> class_name mapping from obs_info
        if camera_type in obs_info and camera_name in obs_info[camera_type]:
            if "seg_instance" in obs_info[camera_type][camera_name]:
                seg_mapping = obs_info[camera_type][camera_name]["seg_instance"]
                
                for seg_id_str, class_name in seg_mapping.items():
                    seg_id = int(seg_id_str)
                    global_class_id = self.class_to_id.get(class_name, 0)  # 0 = unknown
                    remapped[seg_image == seg_id] = global_class_id
        
        return remapped
    
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
            mode='nearest-exact'
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
    
    def _extract_segmentation(self, obs: dict, obs_info: Optional[dict] = None) -> th.Tensor:
        """
        Extract, resize, and remap segmentation images to global class IDs.
        
        Args:
            obs: Raw observation dict
            obs_info: Optional observation info dict with seg_instance mappings
                      If provided, remaps to global class IDs
        
        Returns:
            Tuple of (frank_seg, external_seg_0, external_seg_1) as long tensors [H, W]
        """
        frank_seg = obs['franka0']['franka0:eef_link:Camera:0']['seg_instance']
        external_seg_0 = obs['external']['external_sensor0']['seg_instance']
        external_seg_1 = obs['external']['external_sensor1']['seg_instance']
        
        # Resize to target size
        frank_seg = self._resize_segmentation(frank_seg.to(self.device))
        external_seg_0 = self._resize_segmentation(external_seg_0.to(self.device))
        external_seg_1 = self._resize_segmentation(external_seg_1.to(self.device))
        
        # Remap to global class IDs if obs_info is provided and vocabulary is loaded
        if obs_info is not None and self.num_seg_classes > 1:
            frank_seg = self.remap_seg_to_global_ids(
                frank_seg, obs_info, "franka0", "franka0:eef_link:Camera:0"
            )
            external_seg_0 = self.remap_seg_to_global_ids(
                external_seg_0, obs_info, "external", "external_sensor0"
            )
            external_seg_1 = self.remap_seg_to_global_ids(
                external_seg_1, obs_info, "external", "external_sensor1"
            )
        
        return frank_seg, external_seg_0, external_seg_1
    
    def process(self, obs: dict, robot, obs_info: Optional[dict] = None) -> Dict[str, th.Tensor]:
        """
        Process raw environment observation into policy input format.
        
        Args:
            obs: Raw observation dict from env.step() or env.reset()
            robot: Robot object for proprio extraction
            obs_info: Optional observation info dict with seg_instance mappings.
                      If provided, remaps segmentation to global class IDs.
        
        Returns:
            Dict with keys:
                - 'extero': Dict[str, Tensor] with shape [1, frame_stack, H, W] (global class IDs)
                - 'proprio': Tensor [1, state_dim]
        """
        # Extract, resize, and optionally remap segmentation images
        frank_seg, external_seg_0, external_seg_1 = self._extract_segmentation(obs, obs_info)
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


class EpisodeVideoRecorder:
    """
    Minimal video recorder that streams RGB frames from a chosen camera in the
    observation dict directly to disk (one video per episode).
    """

    def __init__(
        self,
        enabled: bool = False,
        output_dir: str = "resources/videos/shelf_place",
        camera_type: str = "external",
        camera_name: str = "external_sensor0",
        fps: int = 30,
    ):
        self.enabled = bool(enabled)
        self.output_dir = output_dir
        self.camera_type = camera_type
        self.camera_name = camera_name
        self.fps = fps

        self.writer = None
        self.current_episode = None
        self.output_path = None
        self.warned_missing = False

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)

    def start_episode(self, episode_idx: int):
        if not self.enabled:
            return
        # Close any previous writer
        self.close_episode()
        self.current_episode = episode_idx
        self.output_path = os.path.join(
            self.output_dir, f"episode_{episode_idx + 1:03d}_{self.camera_name}.mp4"
        )
        self.warned_missing = False
        self.writer = None

    def record_frame(self, obs: dict):
        if not self.enabled:
            return

        try:
            camera_obs = obs[self.camera_type][self.camera_name]
        except Exception:
            if not self.warned_missing:
                print(
                    f"[VideoRecorder] Missing camera '{self.camera_type}/{self.camera_name}' in observation; "
                    "skipping video recording."
                )
                self.warned_missing = True
            return

        if "rgb" not in camera_obs:
            if not self.warned_missing:
                print(
                    f"[VideoRecorder] Camera '{self.camera_type}/{self.camera_name}' has no 'rgb' data; "
                    "skipping video recording."
                )
                self.warned_missing = True
            return

        frame = np.asarray(camera_obs["rgb"])
        if frame.shape[-1] >= 3:
            frame = frame[:, :, :3]
        frame = frame.astype(np.uint8)

        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (width, height)
            )
            if not self.writer.isOpened():
                print(
                    f"[VideoRecorder] Failed to open video writer at {self.output_path}. Video recording disabled."
                )
                self.enabled = False
                return

        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close_episode(self):
        if self.writer is not None:
            self.writer.release()
            print(f"[VideoRecorder] Saved video to {self.output_path}")
        self.writer = None
        self.current_episode = None


def get_frame_from_obs(obs, camera_type, camera_name):
    camera_obs = obs[camera_type][camera_name]
    frame = np.asarray(camera_obs["rgb"])
    if frame.shape[-1] >= 3:
        frame = frame[:, :, :3]
    frame = frame.astype(np.uint8)
    return frame


def update_health(obs, health_list_link_names, target_objects_health_with_links, target_objects_health, health):
    # initialize arrays
    all_obj_healths = obs["health"].numpy()
    try:
        for obj_name in target_objects_health_with_links:
            health[obj_name].append(float(all_obj_healths[np.where(health_list_link_names == obj_name)[0][0]]))
    except Exception as e:
        print("Error: ", e)
        breakpoint()

    # Obtain health information for the entire target objects 
    for obj_name in target_objects_health:
        arrays = [
            v[-1] for k, v in health.items()
            if k.startswith(f"{obj_name}@") and len(v) > 0
        ]
        # Compute element-wise min
        if arrays:
            health[obj_name].append(float(min(arrays)))
        else:
            print(f"No health data for {obj_name}")

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

def check_object_upright(obj):
    q = obj.get_position_orientation()[1]
    r = R.from_quat(q)

    # Rotate the up vector
    up_rotated = r.apply([0, 0, 1])
    z_alignment = up_rotated[2]  # should be close to 1 if not toppled

    threshold = 0.995  # cos(small angle) ~1
    upright = z_alignment > threshold
    
    return upright

def reset_env(env):
    obs, info = env.reset()
    # load state
    with open("resources/saved_states/shelf_init_state.pkl", "rb") as f: state_flat_array = pickle.load(f)
    og.sim.load_state(state_flat_array, serialized=True)

    # TODO: Add object pose and scale randomization
    flour = env.scene.object_registry("name", "book")
    wineglass = env.scene.object_registry("name", "wineglass")
    winebottle = env.scene.object_registry("name", "bottle_of_wine")
    beerbottle = env.scene.object_registry("name", "bottle_of_beer")
    objects = [flour, wineglass, winebottle, beerbottle]
    trial_number = 0
    while True:
        print("trial number: ", trial_number)
        for obj in objects:
            pos, orn = obj.get_position_orientation()
            pos_magnitude = [-0.05, 0.05] 
            rot_magnitude = np.pi / 12 # 15 degrees
            pos_diff_xy = np.random.uniform(pos_magnitude[0], pos_magnitude[1], size=2)
            pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
            new_pos = pos + pos_diff
            orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
            new_orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
            obj.set_position_orientation(new_pos, new_orn)

        # randomize scale
        temp_state = og.sim.dump_state(serialized=False)
        og.sim.stop()
        for obj in objects:
            x_scale_magnitude = np.random.uniform(0.9, 1.1)
            y_scale_magnitude = np.random.uniform(0.9, 1.1)
            z_scale_magnitude = np.random.uniform(0.9, 1.1)
            new_scale = [obj.scale[0] * x_scale_magnitude, obj.scale[1] * y_scale_magnitude, obj.scale[2] * z_scale_magnitude]
            obj.scale = th.tensor(new_scale)
        og.sim.play()
        og.sim.load_state(temp_state)

        # Make sure all objects are upright
        all_upright = True
        for obj in objects:
            upright = check_object_upright(obj)
            print("object, upright: ", obj.name, upright)
            if not upright:
                print(f"Object {obj.name} is not upright, randomizing again")
                all_upright = False
                break
        if all_upright:
            print("All objects are upright, breaking")
            break
        trial_number += 1

    for _ in range(10): og.sim.step()

    return obs, info

def load_policy(checkpoint_path: str, device: str = "cuda", action_min: th.Tensor = None, action_max: th.Tensor = None) -> CFMPolicy:
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
        config = PolicyConfig(action_min=action_min, action_max=action_max)
    
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
                        # default="../rl-flow-matching/checkpoints/new-data/step_12500.pth",
                        # default="../rl-flow-matching/checkpoints/step_13500.pth",
                        default="../rl-flow-matching/checkpoints/no-gripper-state/step_27000.pth",
                        # default="../rl-flow-matching/checkpoints/no-gripper-state-no-idle/step_13000.pth",
                        # default="../rl-flow-matching/checkpoints/no-gripper-state/final.pth",
                        help='Path to policy checkpoint')
    parser.add_argument('--save_raw_hdf5_path', type=str, 
                        default="resources/evals/shelve_item_raw.hdf5",
                        help='Path to save raw HDF5 file')
    parser.add_argument('--load_state', action='store_true', help='Load a saved state')
    parser.add_argument('--n_episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=400, help='Max steps per episode')
    parser.add_argument('--device', type=str, default='cuda', help='Device for policy')
    parser.add_argument('--execute_horizon', type=int, default=1, 
                        help='Actions to execute before re-planning')
    parser.add_argument('--save_data', action='store_true', help='Save trajectory data')
    parser.add_argument('--vocab_hdf5', type=str, 
                        default="resources/playback_data/20260108-shelf-place-playback.hdf5",
                        help='Path to HDF5 file for building class vocabulary (should match training data)')
    parser.add_argument('--normalize_action', action='store_true', help='Normalize action', default=False)
    parser.add_argument('--save_videos', action='store_true', help='Save an RGB video for each episode', default=False)
    parser.add_argument('--video_dir', type=str, default='resources/videos/sheve_item', help='Directory to save episode videos')
    parser.add_argument('--video_camera_type', type=str, default='external', help='Observation camera_type to record (e.g., external or franka0)')
    parser.add_argument('--video_camera_name', type=str, default='external_sensor0', help='Observation camera_name to record (e.g., external_sensor0)')
    parser.add_argument('--video_fps', type=int, default=30, help='FPS for saved videos')
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(0)
    th.manual_seed(0)

    #### Load dataset for the normalization statistics ####
    dataset = B1KDataset(
        data_path="resources/playback_data/20260108-shelf-place-playback.hdf5",
        frame_stack=2,
        action_chunk_size=8,
        seg_img_size=(128, 128),
        normalize_action=args.normalize_action,
    )
    if args.normalize_action:
        action_min = dataset.action_min
        action_max = dataset.action_max
    else:
        action_min = None
        action_max = None
    
    # Load policy
    device = args.device if th.cuda.is_available() else "cpu"
    print(f"Loading policy from {args.checkpoint}...")
    policy = load_policy(args.checkpoint, device=device, action_min=action_min, action_max=action_max)
    policy.print_parameter_summary()
    
    # Create observation processor with class vocabulary from training HDF5
    obs_config = ObsProcessorConfig(vocab_hdf5_path=args.vocab_hdf5)
    obs_processor = ObservationProcessor(config=obs_config, device=device)
    action_chunker = ActionChunker(
        action_chunk_size=policy.config.action_chunk_size,
        execute_horizon=args.execute_horizon,
    )

    save_video_at_run_time = False
    os.makedirs(args.video_dir, exist_ok=True)
    if save_video_at_run_time:
        video_recorder = EpisodeVideoRecorder(
            enabled=args.save_videos,
            output_dir=args.video_dir,
            camera_type=args.video_camera_type,
            camera_name=args.video_camera_name,
            fps=args.video_fps,
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
        robot_cfg["obs_modalities"] = ["proprio", "rgb", "seg_instance"]


    env = DamageableEnvironment(configs=cfg)        
    # env = DamageableDataCollectionWrapper(
    #     env=env,
    #     output_path=args.save_raw_hdf5_path,
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

    visualization_config = get_visualization_config("shelve_item", robot_name)
    target_objects_health_with_links = visualization_config["target_objects_health_with_links"]
    target_objects_health = visualization_config["target_objects_health"]
    target_objects_forces = visualization_config["target_objects_forces"]
    target_contact_bodies = visualization_config["target_contact_bodies"]
    force_keys = visualization_config["force_keys"]
    health_list_link_names = np.array(env.health_list_link_names)
    all_eps_gripper_opened = list()
    all_eps_task_success = list()
    all_eps_health_dict = defaultdict(list)
    all_eps_info_list = list()

    for episode in range(args.n_episodes):
        print(f"\n--- Episode {episode + 1}/{args.n_episodes} ---")

        imgs = []
        info_list = list()
        
        # Reset environment and processors
        obs, info = reset_env(env)
        # obs, info = env.reset()
        obs_processor.reset()
        action_chunker.reset()
        if save_video_at_run_time:
            video_recorder.start_episode(episode)
            video_recorder.record_frame(obs)

        health = defaultdict(list)
        env_health = list()
        update_health(obs, health_list_link_names, target_objects_health_with_links, target_objects_health, health)
        
        if args.load_state:
            for _ in range(50):
                og.sim.step()
        
        episode_reward = 0.0
        episode_damage = 0.0
        
        # Get initial obs_info for global class ID remapping
        current_obs_info = info.get("obs_info", None)
        
        for step in range(args.max_steps):
            # Query policy for new action chunk if needed
            # if action_chunker.needs_replan():
                # Process observation with global class ID remapping
            policy_input = obs_processor.process(obs, robot, obs_info=current_obs_info)
                
            # Generate action chunk
            with th.no_grad():
                proprio = obs['franka0']['proprio'][None].to(device)
                action_chunk = policy.generate_action(
                    seg_images=policy_input['extero'],
                    state=proprio[..., :-1],
                    # n_actions=4
                )
                # import ipdb; ipdb.set_trace()
                # action_chunk = action_chunk.mean(dim=1)

                
                # Update chunker with new actions
                # action_chunker.update_chunk(action_chunk[0])  # Remove batch dim
            
            # Get next action from chunk
            # action = action_chunker.get_action()
            action = action_chunk[0, 0]
            # if action[-1] > 0:
            #     action[-1] = 1.0
            # TODO(junhong): force the gripper to be closed, just for testing!
            # action[-1] = -1.0
            
            if action is None:
                print(f"Warning: No action available at step {step}")
                break
            
            # Convert to numpy for environment
            action = action.cpu().numpy()
            # print("Step", step, "Action", action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            update_health(obs, health_list_link_names, target_objects_health_with_links, target_objects_health, health)
            info_list.append(info)

            # If want to save video during episode run itself
            if save_video_at_run_time:
                video_recorder.record_frame(obs)
            # If want to save the video at the end of the episode
            else:
                frame = get_frame_from_obs(obs, args.video_camera_type, args.video_camera_name)
                imgs.append(frame)
                # import matplotlib.pyplot as plt
                # plt.imshow(frame)
                # plt.show()
            
            # Update obs_info for next iteration's global class ID remapping
            current_obs_info = info.get("obs_info", current_obs_info)
            
            episode_reward += reward
            # if "damage_info" in info:
            #     # Sum up damage across objects
            #     for obj_name, damage_data in info["damage_info"].items():
            #         if isinstance(damage_data, dict) and "total_damage" in damage_data:
            #             episode_damage += damage_data["total_damage"]
            # Compute env health
            current_env_health = 0.0
            for obj_name in target_objects_health:
                current_env_health += health[obj_name][-1]
            env_health.append(current_env_health / len(target_objects_health))
            # print(f"Current environment health: ", env_health[-1])
            
            # Check termination
            if terminated or truncated:
                print(f"Episode ended at step {step + 1}: terminated={terminated}, truncated={truncated}")
                break
            
            # Progress logging
            if (step + 1) % 50 == 0:
                print(f"  Step {step + 1}/{args.max_steps}")
        
        print(f"Episode {episode + 1} complete:")
        print(f"  Total steps: {step + 1}")
        # print(f"  Total reward: {episode_reward:.4f}")
        # print(f"  Total damage: {episode_damage:.4f}")
        if save_video_at_run_time:
            video_recorder.close_episode()
        else:
            save_rgb_camera_video(os.path.join(args.video_dir, f"episode_{episode:03d}_{args.video_camera_name}"), imgs, args.video_fps)

        for k in health.keys(): health[k] = health[k][1:]
        
        # Saving health and force graphs
        data = dict()
        for obj_name in target_objects_forces:
            data[obj_name] = dict()
            for force_key in force_keys:
                data[obj_name][force_key] = []
        for i in range(len(info_list)):
            damage_info = info_list[i]["damage_info"]
            for obj_name in target_objects_forces:
                for force_key in force_keys:
                    data[obj_name][force_key].append(damage_info[obj_name.split("@")[0]][obj_name.split("@")[1]]["mechanical"][force_key])

        # Save videos for forces plot
        forces_video_path = os.path.join(args.video_dir, f"{episode:03d}_forces_video.mp4")
        save_rgb_force_video(output_video_path=forces_video_path, imgs=imgs, target_objects=target_objects_forces, data=data, forces_to_plot=force_keys)

        # Save video for health plot
        health_video_path = os.path.join(args.video_dir, f"{episode:03d}_health_video.mp4")
        save_rgb_health_video(output_video_path=health_video_path, imgs=imgs, target_objects=target_objects_health, health=health)


        # Check if gripper was opened at all
        gripper_opened = False
        if robot.is_grasping().value != IsGraspingState.TRUE:
            gripper_opened = True
            print("Gripper was opened at least once")
        all_eps_gripper_opened.append(gripper_opened)

        # Check task success
        stand = env.scene.object_registry("name", "stand")
        box_of_crackers = env.scene.object_registry("name", "box_of_crackers")
        box_inside_stand = box_of_crackers.states[object_states.Inside].get_value(other=stand)
        task_success = False
        if box_inside_stand and robot.is_grasping(candidate_obj=box_of_crackers).value == IsGraspingState.FALSE:
            task_success = True
            print("Task success")
        all_eps_task_success.append(task_success)

        # Print environment health
        print("All object healths:")
        for obj_name in target_objects_health:
            print(f"{obj_name} health: {health[obj_name][-1]}")
            all_eps_health_dict[obj_name].append(health[obj_name][-1])
        print(f"Environment health: {env_health[-1]}")
        all_eps_health_dict["env_health"].append(env_health[-1])
        # breakpoint()

        # Save info list
        all_eps_info_list.append(info_list)

    
    # Save to json file
    json_dict = dict()
    json_dict["all_eps_gripper_opened"] = all_eps_gripper_opened
    json_dict["all_eps_task_success"] = all_eps_task_success
    json_dict["all_eps_health_dict"] = dict(all_eps_health_dict)
    json_dict["average_gripper_opened"] = float(np.mean(all_eps_gripper_opened))
    json_dict["average_task_success"] = float(np.mean(all_eps_task_success))
    json_dict["average_obj_healths"] = dict()
    for obj_name in target_objects_health:
        json_dict["average_obj_healths"][obj_name] = float(np.mean(all_eps_health_dict[obj_name]))
    json_dict["average_env_health"] = float(np.mean(all_eps_health_dict["env_health"]))
    os.makedirs("resources/eval_results", exist_ok=True)
    breakpoint()
    with open("resources/eval_results/shelve_item.json", "w") as f:
        json.dump(json_dict, f)
    # Save collected data
    if args.save_data:
        print("\nSaving data...")
        env.save_data()
    
    print("\nEvaluation complete!")
    og.shutdown()


if __name__ == "__main__":
    __main__()
