#!/usr/bin/env python3
"""
Deep inspection of HDF5 files to check for mechanical damage tracking.
Analyzes both teleop (collection) and playback HDF5 files.
"""

import h5py
import numpy as np
import json
import sys
import os
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def inspect_hdf5_health(hdf5_path, file_type="unknown"):
    """Deep inspection of HDF5 file for health and damage tracking."""
    
    if not os.path.exists(hdf5_path):
        print(f"ERROR: File not found: {hdf5_path}")
        return
    
    print_section(f"Inspecting {file_type.upper()} HDF5: {hdf5_path}")
    
    with h5py.File(hdf5_path, "r") as f:
        # Check top-level structure
        print("\n--- Top-level keys ---")
        print(f"Keys: {list(f.keys())}")
        
        if "data" not in f:
            print("ERROR: No 'data' group found!")
            return
        
        data_grp = f["data"]
        
        # Check attributes
        print("\n--- Data group attributes ---")
        if data_grp.attrs:
            for key in data_grp.attrs.keys():
                try:
                    val = data_grp.attrs[key]
                    if isinstance(val, bytes):
                        try:
                            val = json.loads(val.decode('utf-8'))
                        except:
                            val = val.decode('utf-8')
                    print(f"  {key}: {val}")
                except Exception as e:
                    print(f"  {key}: <error reading: {e}>")
        
        # Get all demo keys
        demo_keys = [k for k in data_grp.keys() if k.startswith("demo_")]
        print(f"\n--- Found {len(demo_keys)} episodes ---")
        print(f"Episodes: {sorted(demo_keys)}")
        
        if len(demo_keys) == 0:
            print("WARNING: No episodes found!")
            return
        
        # Analyze each episode
        for demo_key in sorted(demo_keys):
            print_section(f"Episode: {demo_key}")
            demo_grp = data_grp[demo_key]
            
            # Check episode attributes
            print("\n--- Episode attributes ---")
            if demo_grp.attrs:
                for key in demo_grp.attrs.keys():
                    try:
                        val = demo_grp.attrs[key]
                        if key == "health_list_link_names":
                            if isinstance(val, bytes):
                                val = val.decode('utf-8')
                            print(f"  {key}: {val}")
                        else:
                            print(f"  {key}: {val}")
                    except Exception as e:
                        print(f"  {key}: <error: {e}>")
            
            # Check health_list_link_names
            health_list_link_names = None
            if "health_list_link_names" in demo_grp.attrs:
                health_list_link_names = demo_grp.attrs["health_list_link_names"]
                if isinstance(health_list_link_names, bytes):
                    health_list_link_names = health_list_link_names.decode('utf-8')
                elif isinstance(health_list_link_names, np.ndarray):
                    health_list_link_names = health_list_link_names.tolist()
                print(f"\n--- Health list link names ({len(health_list_link_names)} links) ---")
                for i, link_name in enumerate(health_list_link_names):
                    print(f"  [{i}] {link_name}")
            
            # Check obs/health
            if "obs" in demo_grp and "health" in demo_grp["obs"]:
                health_data = np.array(demo_grp["obs"]["health"])
                print(f"\n--- Health data shape: {health_data.shape} ---")
                print(f"  Data type: {health_data.dtype}")
                print(f"  Min: {np.min(health_data):.2f}, Max: {np.max(health_data):.2f}, Mean: {np.mean(health_data):.2f}")
                
                # Map health to link names
                if health_list_link_names is not None:
                    print(f"\n--- Health per link (first 10 steps, then last 10 steps) ---")
                    n_steps = health_data.shape[0]
                    n_links = health_data.shape[1] if len(health_data.shape) > 1 else 1
                    
                    # First 10 steps
                    print("\n  First 10 steps:")
                    for step_idx in range(min(10, n_steps)):
                        print(f"    Step {step_idx}: ", end="")
                        for link_idx in range(min(5, n_links)):  # Show first 5 links
                            if link_idx < len(health_list_link_names):
                                link_name = health_list_link_names[link_idx]
                                health_val = health_data[step_idx, link_idx] if len(health_data.shape) > 1 else health_data[step_idx]
                                print(f"{link_name}={health_val:.2f} ", end="")
                        print("...")
                    
                    # Last 10 steps
                    if n_steps > 10:
                        print("\n  Last 10 steps:")
                        for step_idx in range(max(0, n_steps - 10), n_steps):
                            print(f"    Step {step_idx}: ", end="")
                            for link_idx in range(min(5, n_links)):  # Show first 5 links
                                if link_idx < len(health_list_link_names):
                                    link_name = health_list_link_names[link_idx]
                                    health_val = health_data[step_idx, link_idx] if len(health_data.shape) > 1 else health_data[step_idx]
                                    print(f"{link_name}={health_val:.2f} ", end="")
                            print("...")
                    
                    # Check for robot links specifically
                    print(f"\n--- Robot health analysis ---")
                    robot_link_indices = []
                    for idx, link_name in enumerate(health_list_link_names):
                        if link_name.startswith("franka0@") or link_name.startswith("tiago0@") or link_name.startswith("r1pro0@"):
                            robot_link_indices.append((idx, link_name))
                    
                    if robot_link_indices:
                        print(f"  Found {len(robot_link_indices)} robot links:")
                        for idx, link_name in robot_link_indices:
                            robot_health = health_data[:, idx] if len(health_data.shape) > 1 else health_data
                            min_health = np.min(robot_health)
                            max_health = np.max(robot_health)
                            mean_health = np.mean(robot_health)
                            final_health = robot_health[-1] if len(robot_health) > 0 else None
                            print(f"    {link_name}:")
                            print(f"      Min: {min_health:.2f}, Max: {max_health:.2f}, Mean: {mean_health:.2f}, Final: {final_health:.2f}")
                            
                            # Check if health decreased (indicating damage)
                            if min_health < 100.0:
                                print(f"      ⚠️  DAMAGE DETECTED: Health dropped from 100.0 to {min_health:.2f}")
                                # Find when damage occurred
                                damage_steps = np.where(robot_health < 100.0)[0]
                                if len(damage_steps) > 0:
                                    print(f"      First damage at step {damage_steps[0]}, health={robot_health[damage_steps[0]]:.2f}")
                    else:
                        print("  No robot links found in health_list_link_names")
            else:
                print("\n--- WARNING: No obs/health data found ---")
            
            # Check info/damage_info
            if "info" in demo_grp and "damage_info" in demo_grp["info"]:
                damage_info_data = demo_grp["info"]["damage_info"]
                print(f"\n--- Damage info data shape: {damage_info_data.shape} ---")
                print(f"  Data type: {damage_info_data.dtype}")
                
                # Parse damage_info
                print(f"\n--- Damage info analysis (first 5 and last 5 steps) ---")
                n_steps = len(damage_info_data)
                
                # Sample steps
                sample_steps = list(range(min(5, n_steps))) + list(range(max(0, n_steps - 5), n_steps))
                sample_steps = sorted(set(sample_steps))
                
                for step_idx in sample_steps:
                    try:
                        damage_info_str = damage_info_data[step_idx]
                        if isinstance(damage_info_str, bytes):
                            damage_info_str = damage_info_str.decode('utf-8')
                        damage_info = json.loads(damage_info_str)
                        
                        print(f"\n  Step {step_idx} damage_info:")
                        for obj_name, obj_damage in damage_info.items():
                            print(f"    {obj_name}:")
                            for link_name, link_damage in obj_damage.items():
                                print(f"      {link_name}:")
                                for damage_type, damage_data in link_damage.items():
                                    if damage_type == "mechanical":
                                        print(f"        MECHANICAL DAMAGE:")
                                        if isinstance(damage_data, dict):
                                            if "damage" in damage_data:
                                                print(f"          damage: {damage_data['damage']}")
                                            if "impact_forces" in damage_data:
                                                impact_forces = damage_data["impact_forces"]
                                                if isinstance(impact_forces, list) and len(impact_forces) > 0:
                                                    print(f"          impact_forces (last): {impact_forces[-1] if isinstance(impact_forces[-1], (int, float)) else 'complex'}")
                                            if "contacts" in damage_data:
                                                contacts = damage_data["contacts"]
                                                if isinstance(contacts, list) and len(contacts) > 0:
                                                    print(f"          contacts (count): {len(contacts[-1]) if isinstance(contacts[-1], list) else 'N/A'}")
                                    elif damage_type == "thermal":
                                        print(f"        thermal: {damage_data}")
                    except Exception as e:
                        print(f"    Step {step_idx}: <error parsing: {e}>")
            else:
                print("\n--- WARNING: No info/damage_info data found ---")
            
            # Check if there are other relevant groups
            print(f"\n--- Other groups in episode ---")
            for key in demo_grp.keys():
                if key not in ["obs", "info", "action", "state", "reward", "terminated", "truncated"]:
                    print(f"  {key}: {demo_grp[key]}")


def compare_hdf5_files(teleop_path, playback_path):
    """Compare teleop and playback HDF5 files."""
    print_section("COMPARING TELEOP vs PLAYBACK HDF5 FILES")
    
    if not os.path.exists(teleop_path):
        print(f"ERROR: Teleop file not found: {teleop_path}")
        return
    
    if not os.path.exists(playback_path):
        print(f"ERROR: Playback file not found: {playback_path}")
        return
    
    with h5py.File(teleop_path, "r") as f_teleop, h5py.File(playback_path, "r") as f_playback:
        # Get demo keys
        teleop_demos = sorted([k for k in f_teleop["data"].keys() if k.startswith("demo_")])
        playback_demos = sorted([k for k in f_playback["data"].keys() if k.startswith("demo_")])
        
        print(f"\nTeleop episodes: {len(teleop_demos)}")
        print(f"Playback episodes: {len(playback_demos)}")
        
        # Compare each episode
        common_demos = set(teleop_demos) & set(playback_demos)
        print(f"\nCommon episodes: {len(common_demos)}")
        
        for demo_key in sorted(common_demos):
            print_section(f"Comparing {demo_key}")
            
            teleop_grp = f_teleop["data"][demo_key]
            playback_grp = f_playback["data"][demo_key]
            
            # Compare health
            if "obs" in teleop_grp and "health" in teleop_grp["obs"]:
                teleop_health = np.array(teleop_grp["obs"]["health"])
                playback_health = np.array(playback_grp["obs"]["health"]) if "obs" in playback_grp and "health" in playback_grp["obs"] else None
                
                print(f"\n--- Health comparison ---")
                print(f"Teleop health shape: {teleop_health.shape}")
                if playback_health is not None:
                    print(f"Playback health shape: {playback_health.shape}")
                    
                    # Compare robot health
                    teleop_health_list = teleop_grp.attrs.get("health_list_link_names", [])
                    playback_health_list = playback_grp.attrs.get("health_list_link_names", [])
                    
                    if isinstance(teleop_health_list, bytes):
                        teleop_health_list = teleop_health_list.decode('utf-8')
                    if isinstance(playback_health_list, bytes):
                        playback_health_list = playback_health_list.decode('utf-8')
                    
                    # Find robot links
                    robot_indices_teleop = []
                    robot_indices_playback = []
                    
                    for idx, link_name in enumerate(teleop_health_list if isinstance(teleop_health_list, list) else []):
                        if link_name.startswith("franka0@"):
                            robot_indices_teleop.append((idx, link_name))
                    
                    for idx, link_name in enumerate(playback_health_list if isinstance(playback_health_list, list) else []):
                        if link_name.startswith("franka0@"):
                            robot_indices_playback.append((idx, link_name))
                    
                    if robot_indices_teleop and robot_indices_playback:
                        print(f"\n  Robot health comparison:")
                        for (t_idx, t_link), (p_idx, p_link) in zip(robot_indices_teleop[:3], robot_indices_playback[:3]):  # Compare first 3 links
                            t_health = teleop_health[:, t_idx]
                            p_health = playback_health[:, p_idx]
                            
                            print(f"\n    {t_link}:")
                            print(f"      Teleop:   min={np.min(t_health):.2f}, max={np.max(t_health):.2f}, final={t_health[-1]:.2f}")
                            print(f"      Playback: min={np.min(p_health):.2f}, max={np.max(p_health):.2f}, final={p_health[-1]:.2f}")
                            print(f"      Difference: final_diff={t_health[-1] - p_health[-1]:.2f}")


if __name__ == "__main__":
    # Default paths
    default_teleop = "resources/teleop_data/firewood.hdf5"
    default_playback = "resources/playback_data/firewood_playback.hdf5"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        teleop_path = sys.argv[1]
    else:
        teleop_path = default_teleop
    
    if len(sys.argv) > 2:
        playback_path = sys.argv[2]
    else:
        playback_path = default_playback
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent.parent
    teleop_path = str(script_dir / teleop_path)
    playback_path = str(script_dir / playback_path)
    
    # Inspect teleop file
    if os.path.exists(teleop_path):
        inspect_hdf5_health(teleop_path, "TELEOP")
    else:
        print(f"Teleop file not found: {teleop_path}")
        print("Looking for alternative files...")
        # Try to find any firewood HDF5 files
        for root, dirs, files in os.walk(script_dir):
            for file in files:
                if "firewood" in file.lower() and file.endswith(".hdf5"):
                    full_path = os.path.join(root, file)
                    print(f"Found: {full_path}")
    
    # Inspect playback file
    if os.path.exists(playback_path):
        inspect_hdf5_health(playback_path, "PLAYBACK")
    else:
        print(f"Playback file not found: {playback_path}")
    
    # Compare if both exist
    if os.path.exists(teleop_path) and os.path.exists(playback_path):
        compare_hdf5_files(teleop_path, playback_path)
