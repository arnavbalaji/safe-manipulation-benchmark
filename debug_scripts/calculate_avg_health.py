#!/usr/bin/env python3
"""
Calculate average environment health from HDF5 files and compare with visualization.
"""

import h5py
import numpy as np
import json
import sys
from pathlib import Path

def get_visualization_config(robot_name):
    """Get visualization config matching firewood.py"""
    target_objects_health_with_links = [
        "franka0@panda_link0",
        "franka0@panda_link1",
        "franka0@panda_link2",
        "franka0@panda_link3",
        "franka0@panda_link4",
        "franka0@panda_link5",
        "franka0@panda_link6",
        "franka0@panda_link7",
        "franka0@panda_hand",
        "franka0@panda_leftfinger",
        "franka0@panda_rightfinger",
        "franka0@eef_link",
        "log_center@base_link",
        "log_left@base_link",
        "target_object@base_link",
        "fireplace@base_link",
    ]
    
    target_objects_health = ["franka0", "log_center", "log_left", "target_object", "fireplace"]
    
    return {
        "target_objects_health_with_links": target_objects_health_with_links,
        "target_objects_health": target_objects_health,
    }

def calculate_avg_health_from_hdf5(hdf5_path, file_type="unknown"):
    """Calculate average environment health from HDF5 file using same logic as visualization."""
    
    if not Path(hdf5_path).exists():
        print(f"ERROR: File not found: {hdf5_path}")
        return None
    
    print("\n" + "=" * 80)
    print(f"Calculating average environment health from {file_type.upper()} HDF5")
    print(f"File: {hdf5_path}")
    print("=" * 80)
    
    vis_cfg = get_visualization_config("franka0")
    target_objects_health_with_links = vis_cfg["target_objects_health_with_links"]
    target_objects_health = vis_cfg["target_objects_health"]
    
    final_obj_healths = {}
    final_env_healths = []
    
    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
        
        print(f"\nFound {len(demo_keys)} episodes")
        
        for demo_key in demo_keys:
            demo_grp = data_grp[demo_key]
            demo_idx = int(demo_key.split("_")[-1])
            
            # Get health data
            if "obs" not in demo_grp or "health" not in demo_grp["obs"]:
                print(f"  {demo_key}: No health data, skipping")
                continue
            
            all_obj_healths = np.array(demo_grp["obs"]["health"])
            health_list_link_names = demo_grp.attrs.get("health_list_link_names", [])
            
            # Convert health_list_link_names to list if needed
            if isinstance(health_list_link_names, bytes):
                health_list_link_names = health_list_link_names.decode('utf-8')
            if isinstance(health_list_link_names, np.ndarray):
                health_list_link_names = health_list_link_names.tolist()
            elif isinstance(health_list_link_names, str):
                # Try to parse as JSON or split
                try:
                    health_list_link_names = json.loads(health_list_link_names)
                except:
                    health_list_link_names = health_list_link_names.split()
            
            # Map health to link names (same logic as visualization)
            health = {}
            for obj_name in target_objects_health_with_links:
                matching_indices = np.where(np.array(health_list_link_names) == obj_name)[0]
                if len(matching_indices) > 0:
                    health[obj_name] = all_obj_healths[:, matching_indices[0]]
                    health[obj_name] = health[obj_name][1:]  # Skip first step (same as visualization)
                else:
                    health[obj_name] = None
            
            # Aggregate health per object (min across links) - same as visualization
            for obj_name in target_objects_health:
                arrays = [v for k, v in health.items() if k.startswith(f"{obj_name}@") and v is not None]
                if arrays:
                    health[obj_name] = np.minimum.reduce(arrays)
                else:
                    health[obj_name] = None
            
            # Calculate environment health for this episode
            current_env_health = 0.0
            valid_objects = 0
            
            print(f"\n  Episode {demo_idx}:")
            for obj_name in target_objects_health:
                if health[obj_name] is not None and len(health[obj_name]) > 0:
                    final_health = health[obj_name][-1]
                    if obj_name not in final_obj_healths:
                        final_obj_healths[obj_name] = []
                    final_obj_healths[obj_name].append(final_health)
                    current_env_health += final_health
                    valid_objects += 1
                    print(f"    {obj_name}: final_health = {final_health:.2f}")
                else:
                    print(f"    {obj_name}: No health data")
            
            if valid_objects > 0:
                avg_episode_health = current_env_health / valid_objects
                final_env_healths.append(avg_episode_health)
                print(f"    Episode avg health: {avg_episode_health:.2f}")
            else:
                print(f"    Episode {demo_idx}: No valid objects, skipping")
    
    # Calculate overall average
    if final_env_healths:
        overall_avg = np.mean(final_env_healths)
        print(f"\n" + "=" * 80)
        print(f"RESULTS for {file_type.upper()}:")
        print("=" * 80)
        print(f"  Number of episodes with valid health: {len(final_env_healths)}")
        print(f"  Average environment health: {overall_avg:.2f}")
        print(f"\n  Per-object averages:")
        for obj_name in target_objects_health:
            if obj_name in final_obj_healths and len(final_obj_healths[obj_name]) > 0:
                obj_avg = np.mean(final_obj_healths[obj_name])
                print(f"    {obj_name}: {obj_avg:.2f} (from {len(final_obj_healths[obj_name])} episodes)")
        
        return overall_avg, final_obj_healths, final_env_healths
    else:
        print(f"\nERROR: No valid episodes found!")
        return None, None, None

if __name__ == "__main__":
    # Default paths
    default_teleop = "/home/arnav/projects/BEHAVIOR-1K/resources/teleop_data/firewood_trial_2.hdf5"
    default_playback = "/home/arnav/projects/BEHAVIOR-1K/resources/playback_data/firewood_trial_2_playback.hdf5"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        teleop_path = sys.argv[1]
    else:
        teleop_path = default_teleop
    
    if len(sys.argv) > 2:
        playback_path = sys.argv[2]
    else:
        playback_path = default_playback
    
    # Calculate for teleop
    teleop_result = calculate_avg_health_from_hdf5(teleop_path, "TELEOP")
    
    # Calculate for playback
    playback_result = calculate_avg_health_from_hdf5(playback_path, "PLAYBACK")
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    if teleop_result[0] is not None and playback_result[0] is not None:
        print(f"Teleop average:   {teleop_result[0]:.2f}")
        print(f"Playback average: {playback_result[0]:.2f}")
        print(f"Visualization:    19.84")
        print(f"\nDifference:")
        print(f"  Teleop vs Visualization:   {teleop_result[0] - 19.84:.2f}")
        print(f"  Playback vs Visualization: {playback_result[0] - 19.84:.2f}")
