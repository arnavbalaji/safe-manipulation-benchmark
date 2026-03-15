#!/usr/bin/env python3
"""
Proof that mechanical damage is NOT being tracked during teleop.
Shows damage breakdown if any damage exists.
"""

import h5py
import numpy as np
import json
import sys
from pathlib import Path

def analyze_damage_in_hdf5(hdf5_path, file_type="unknown"):
    """Analyze damage_info in HDF5 to show damage breakdown."""
    
    if not Path(hdf5_path).exists():
        print(f"ERROR: File not found: {hdf5_path}")
        return
    
    print("\n" + "=" * 80)
    print(f"ANALYZING {file_type.upper()} HDF5 FOR DAMAGE BREAKDOWN")
    print(f"File: {hdf5_path}")
    print("=" * 80)
    
    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
        
        print(f"\nFound {len(demo_keys)} episodes")
        
        episodes_with_damage_info = 0
        episodes_with_mech_damage = []
        episodes_with_thermal_damage = []
        total_damage_breakdown = {"mechanical": 0.0, "thermal": 0.0, "other": 0.0}
        
        for demo_key in demo_keys:
            demo_grp = data_grp[demo_key]
            
            # Check if damage_info exists
            if "info" not in demo_grp or "damage_info" not in demo_grp["info"]:
                continue
            
            episodes_with_damage_info += 1
            damage_info_data = demo_grp["info"]["damage_info"]
            
            # Analyze all steps for this episode
            episode_mech_damage = []
            episode_thermal_damage = []
            
            for step_idx in range(len(damage_info_data)):
                try:
                    damage_info_str = damage_info_data[step_idx]
                    if isinstance(damage_info_str, bytes):
                        damage_info_str = damage_info_str.decode('utf-8')
                    damage_info = json.loads(damage_info_str)
                    
                    # Check robot damage
                    if "franka0" in damage_info:
                        robot_damage = damage_info["franka0"]
                        
                        for link_name, link_damage in robot_damage.items():
                            if isinstance(link_damage, dict):
                                # Check mechanical
                                if "mechanical" in link_damage:
                                    mech_data = link_damage["mechanical"]
                                    if isinstance(mech_data, dict) and "damage" in mech_data:
                                        mech_damage_val = mech_data["damage"]
                                        if mech_damage_val > 0:
                                            episode_mech_damage.append((step_idx, link_name, mech_damage_val))
                                            total_damage_breakdown["mechanical"] += mech_damage_val
                                
                                # Check thermal
                                if "thermal" in link_damage:
                                    thermal_data = link_damage["thermal"]
                                    if isinstance(thermal_data, dict) and "damage" in thermal_data:
                                        thermal_damage_val = thermal_data["damage"]
                                        if thermal_damage_val > 0:
                                            episode_thermal_damage.append((step_idx, link_name, thermal_damage_val))
                                            total_damage_breakdown["thermal"] += thermal_damage_val
                except Exception as e:
                    pass
            
            if len(episode_mech_damage) > 0:
                episodes_with_mech_damage.append((demo_key, episode_mech_damage))
            if len(episode_thermal_damage) > 0:
                episodes_with_thermal_damage.append((demo_key, episode_thermal_damage))
        
        # Print results
        print(f"\n" + "=" * 80)
        print("RESULTS:")
        print("=" * 80)
        print(f"Episodes with damage_info: {episodes_with_damage_info}")
        print(f"Episodes with mechanical damage: {len(episodes_with_mech_damage)}")
        print(f"Episodes with thermal damage: {len(episodes_with_thermal_damage)}")
        
        if len(episodes_with_mech_damage) > 0:
            print(f"\n⚠️  MECHANICAL DAMAGE FOUND:")
            for demo_key, mech_damage_list in episodes_with_mech_damage[:5]:  # Show first 5
                print(f"\n  {demo_key}:")
                for step_idx, link_name, damage_val in mech_damage_list[:3]:  # Show first 3 instances
                    print(f"    Step {step_idx}, {link_name}: {damage_val:.4f}")
        else:
            print(f"\n✓ NO MECHANICAL DAMAGE FOUND")
        
        if len(episodes_with_thermal_damage) > 0:
            print(f"\n⚠️  THERMAL DAMAGE FOUND:")
            for demo_key, thermal_damage_list in episodes_with_thermal_damage[:5]:  # Show first 5
                print(f"\n  {demo_key}:")
                for step_idx, link_name, damage_val in thermal_damage_list[:3]:  # Show first 3 instances
                    print(f"    Step {step_idx}, {link_name}: {damage_val:.4f}")
        else:
            print(f"\n✓ NO THERMAL DAMAGE FOUND")
        
        print(f"\nTotal damage breakdown:")
        print(f"  Mechanical: {total_damage_breakdown['mechanical']:.4f}")
        print(f"  Thermal: {total_damage_breakdown['thermal']:.4f}")
        print(f"  Other: {total_damage_breakdown['other']:.4f}")
        
        # Also check health values
        print(f"\n" + "=" * 80)
        print("HEALTH VALUES CHECK:")
        print("=" * 80)
        episodes_with_health_drop = []
        for demo_key in demo_keys[:10]:  # Check first 10
            demo_grp = data_grp[demo_key]
            if "obs" in demo_grp and "health" in demo_grp["obs"]:
                health_data = np.array(demo_grp["obs"]["health"])
                min_health = np.min(health_data)
                if min_health < 100.0:
                    episodes_with_health_drop.append((demo_key, min_health))
        
        if episodes_with_health_drop:
            print(f"Episodes with health < 100:")
            for demo_key, min_health in episodes_with_health_drop:
                print(f"  {demo_key}: min_health = {min_health:.2f}")
        else:
            print(f"✓ All checked episodes have health = 100.0 (no damage)")


if __name__ == "__main__":
    default_teleop = "/home/arnav/projects/BEHAVIOR-1K/resources/teleop_data/firewood_trial_2.hdf5"
    default_playback = "/home/arnav/projects/BEHAVIOR-1K/resources/playback_data/firewood_trial_2_playback.hdf5"
    
    if len(sys.argv) > 1:
        teleop_path = sys.argv[1]
    else:
        teleop_path = default_teleop
    
    if len(sys.argv) > 2:
        playback_path = sys.argv[2]
    else:
        playback_path = default_playback
    
    # Analyze teleop
    analyze_damage_in_hdf5(teleop_path, "TELEOP")
    
    # Analyze playback
    analyze_damage_in_hdf5(playback_path, "PLAYBACK")
