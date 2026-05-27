import torch as th
import numpy as np
import argparse
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import yaml
import json
import h5py
from omnigibson.macros import gm

import omnigibson as og
from omnigibson.utils.ui_utils import draw_line, clear_debug_drawing
from omnigibson.utils.transform_utils import pose_transform, pose2mat
from safety_benchmark.damageable_env import DamageableDataPlaybackWrapper

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False

# Default configuration
collect_hdf5_path = "resources/teleop_data/shelve_item/trial_5.hdf5"
playback_hdf5_path = "resources/playback_data/temp.hdf5"

robot_name = "franka0"
robot_type = "FrankaPanda"

# Data files to visualize
data = [
    {
        "collect_hdf5_path": "resources/teleop_data/shelve_item/trial_1.hdf5",
        "data_path": "resources/playback_data/shelve_item/trial_1_playback.hdf5",
        "color": (1.0, 0.0, 0.0, 1.0),  # Red
        "line_size": 4.0
    },
    {
        "collect_hdf5_path": "resources/teleop_data/shelve_item/trial_2.hdf5",
        "data_path": "resources/playback_data/shelve_item/trial_2_playback.hdf5",
        "color": (0.0, 1.0, 0.0, 1.0),  # Green
        "line_size": 4.0
    },
    #     {
    #     "collect_hdf5_path": "resources/teleop_data/shelve_item/trial_5.hdf5",
    #     "data_path": "resources/playback_data/shelve_item/trial_5_playback.hdf5",
    #     "color": (1.0, 0.0, 0.0, 1.0),  # Red
    #     "line_size": 4.0
    # },
    # {
    #     "collect_hdf5_path": "resources/teleop_data/shelve_item/trial_6.hdf5",
    #     "data_path": "resources/playback_data/shelve_item/trial_6_playback.hdf5",
    #     "color": (0.0, 1.0, 0.0, 1.0),  # Green
    #     "line_size": 4.0
    # },
]


def extract_eef_poses_from_hdf5(hdf5_file, episode_id, robot):
    """
    Extract EEF poses from hdf5 file for a given episode.
    
    Args:
        hdf5_file: Open hdf5 file
        episode_id: Episode ID to extract
        robot_name: Name of the robot
        
    Returns:
        List of EEF positions (each as numpy array of shape (3,))
    """
    data_grp = hdf5_file["data"]
    traj_grp = data_grp[f"demo_{episode_id}"]
    
    eef_positions = []
    
    # Try to get EEF poses from obs if available
    obs_grp = traj_grp.get("obs", None)
    if obs_grp is not None:
        # Method 1: Try direct eef_pos key (if stored separately)
        if "eef_pos" in obs_grp:
            eef_pos_data = obs_grp["eef_pos"][:]
            if len(eef_pos_data.shape) == 2 and eef_pos_data.shape[1] >= 3:
                eef_positions = [eef_pos_data[i, :3] for i in range(len(eef_pos_data))]
                # return eef_positions

    # convert eef_pos from robot to world frame
    robot_base_pose = robot.get_position_orientation()
    T_robot_wrt_world = pose2mat(robot_base_pose)
    # breakpoint()
    eef_positions = th.tensor(eef_positions)
    ones = th.ones(eef_positions.shape[0], 1, device=eef_positions.device, dtype=eef_positions.dtype)
    eef_positions_homo = th.cat([eef_positions, ones], dim=1)   # (330, 4)
    eef_positions_world = eef_positions_homo @ T_robot_wrt_world.T
    eef_positions_world = eef_positions_world[:, :3]

    # eef_positions, _ = pose_transform(robot_base_pose[0], robot_base_pose[1], eef_positions[:, :3], eef_pos_data[:, 3:])
    
    return eef_positions_world


def visualize_trajectories(data_list, robot_name="franka0", max_episodes_per_file=None):
    """
    Visualize EEF trajectories from multiple hdf5 files.
    
    Args:
        data_list: List of dicts with keys: collect_hdf5_path, color, line_size
        robot_name: Name of the robot
        max_episodes_per_file: Maximum number of episodes to visualize per file (None for all)
    """
    # Initialize environment with first hdf5 file
    first_hdf5_path = data_list[0]["collect_hdf5_path"]
    env = DamageableDataPlaybackWrapper.create_from_hdf5(
        input_path=first_hdf5_path,
        output_path=playback_hdf5_path,
        robot_obs_modalities=["proprio"],
        robot_sensor_config=None,
        external_sensors_config=None,
        n_render_iterations=1,
        only_successes=False,
    )
    robot = env.robots[0]
    # Load scene
    input_hdf5 = h5py.File(first_hdf5_path, "r")
    scene_file = json.loads(input_hdf5["data"].attrs["scene_file"])
    env.scene.restore(scene_file, update_initial_file=True)
    input_hdf5.close()
    
    # Render a few times to initialize
    for _ in range(10):
        og.sim.render()
    
    # Clear any existing debug drawings
    clear_debug_drawing()
    
    # Set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([7.0659, -0.7141, 1.9185]),
        orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
    )
    for _ in range(10):
        og.sim.step()
    
    # Visualize trajectories from each file
    for data_idx, data_config in enumerate(data_list):
        hdf5_path = data_config["data_path"]
        color = data_config["color"]
        line_size = data_config.get("line_size", 2.0)
        
        print(f"Processing {hdf5_path}...")
        
        input_hdf5 = h5py.File(hdf5_path, "r")
        num_demos = input_hdf5["data"].attrs["n_episodes"]
        print(f"  Number of demos: {num_demos}")

        
        # breakpoint()
        # last_state = th.tensor(input_hdf5["data/demo_0/state"][10])
        # last_state_size = input_hdf5["data/demo_0/state_size"][10]
        # # load the last state into the simulation
        # og.sim.load_state(last_state[:int(last_state_size)], serialized=True)
        
        # Limit number of episodes if specified
        episodes_to_process = num_demos
        if max_episodes_per_file is not None:
            episodes_to_process = min(num_demos, max_episodes_per_file)
            print(f"  Processing {episodes_to_process} episodes (limited from {num_demos})")
        
        # Extract and visualize trajectories for each episode
        for episode_id in range(episodes_to_process):
            print(f"  Extracting EEF poses for episode {episode_id}...")
            eef_positions = extract_eef_poses_from_hdf5(input_hdf5, episode_id, robot)
            # breakpoint()
            
            # remove later
            eef_positions = eef_positions[:-50]

            if len(eef_positions) < 2:
                print(f"    Warning: Episode {episode_id} has less than 2 EEF positions, skipping")
                continue
            
            # Draw lines connecting consecutive EEF positions
            print(f"    Drawing {len(eef_positions) - 1} line segments...")
            for i in range(len(eef_positions) - 1):
                start = tuple(eef_positions[i].tolist())
                end = tuple(eef_positions[i + 1].tolist())
                draw_line(start, end, color=color, size=line_size)
            
            # Render to update display
            for _ in range(5):
                og.sim.step()
        
        input_hdf5.close()
        print(f"  Completed visualization for {hdf5_path}")
    
    print("\nVisualization complete! Press any key to exit...")
    breakpoint()
    # Keep rendering to maintain visualization
    while True:
        og.sim.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize EEF trajectories from hdf5 files")
    parser.add_argument(
        "--data_config",
        type=str,
        default=None,
        help="Path to YAML file with data configuration (optional)"
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default="franka0",
        help="Name of the robot"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to visualize per file (default: all)"
    )
    
    args = parser.parse_args()
    
    # Load data configuration if provided
    if args.data_config and os.path.exists(args.data_config):
        with open(args.data_config, "r") as f:
            data = yaml.safe_load(f)
    
    # # debugging
    # f = h5py.File(data[0]["collect_hdf5_path"], "r")
    # f["data/demo_0"].keys()
    
    visualize_trajectories(data, robot_name=args.robot_name, max_episodes_per_file=args.max_episodes)

