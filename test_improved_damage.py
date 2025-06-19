"""
Test script for the improved mechanical damage system.
Demonstrates realistic physics-based damage calculation based on impact energy and velocity changes.
"""

import os
import sys
import torch as th
import numpy as np
import time

import omnigibson as og
from omnigibson.macros import gm
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS

# Configure OmniGibson settings
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True

def create_test_environment():
    """Create a test environment with a wine glass and table."""
    
    # Scene configuration
    scene_cfg = {"type": "Scene"}
    
    # Robot configuration
    robot_cfg = {
        "type": "Tiago",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        "position": [0., 0.75, 0.0],
        "orientation": [0, 0, -1, 1],
        "grasping_mode": "assisted",
        "damage_params": PARAMS["tiago_robot"]
    }
    
    # Object configurations
    objects = [
        {
            "type": "DatasetObject",
            "name": "coffee_table",
            "category": "coffee_table",
            "model": "aoojzy",
            "position": [0.0, 0.0, 0.0],
            "orientation": [0, 0, 0, 1],
            "scale": [1.0, 1.0, 1.0],
            "damage_params": PARAMS["coffee_table"]
        },
        {
            "type": "DatasetObject",
            "name": "wine_glass",
            "category": "wineglass",
            "model": "cmdagy",
            "position": [0.0, 0.0, 0.5],  # Start above the table
            "orientation": [0, 0, 0, 1],
            "scale": [1.0, 1.0, 1.0],
            "damage_params": PARAMS["bowl"],  # Using bowl params for wine glass
            "mass": 0.2,  # Light mass for realistic glass
            "friction": 0.8,
            "restitution": 0.1,
        }
    ]
    
    # Compile configuration
    cfg = {
        "scene": scene_cfg,
        "robots": [robot_cfg],
        "objects": objects
    }
    
    return DamageableEnvironment(configs=cfg)

def test_static_contact():
    """Test that static contact doesn't cause damage."""
    print("🧪 Testing static contact damage...")
    
    env = create_test_environment()
    env.reset()
    
    # Get objects
    wine_glass = env.scene.object_registry("name", "wine_glass")
    table = env.scene.object_registry("name", "coffee_table")
    
    # Let physics settle
    for _ in range(50):
        env.step(env.action_space.sample())
    
    # Check health after settling
    initial_health = wine_glass.health
    print(f"   Wine glass health after settling: {initial_health:.2f}")
    
    # Let it sit for more steps
    for _ in range(100):
        env.step(env.action_space.sample())
    
    final_health = wine_glass.health
    health_change = initial_health - final_health
    
    print(f"   Wine glass health after sitting: {final_health:.2f}")
    print(f"   Health change: {health_change:.2f}")
    
    if health_change < 1.0:
        print("   ✅ PASS: Static contact doesn't cause significant damage")
    else:
        print("   ❌ FAIL: Static contact is causing damage")
    
    return health_change < 1.0

def test_drop_impact():
    """Test that dropping the wine glass causes realistic damage."""
    print("\n🧪 Testing drop impact damage...")
    
    env = create_test_environment()
    env.reset()
    
    # Get objects
    wine_glass = env.scene.object_registry("name", "wine_glass")
    table = env.scene.object_registry("name", "coffee_table")
    
    # Let physics settle
    for _ in range(50):
        env.step(env.action_space.sample())
    
    initial_health = wine_glass.health
    print(f"   Initial wine glass health: {initial_health:.2f}")
    
    # Drop the wine glass from a height
    wine_glass.set_position_orientation(
        position=[0.0, 0.0, 1.0],  # Higher drop
        orientation=[0, 0, 0, 1]
    )
    
    # Let it fall and impact
    for _ in range(200):
        env.step(env.action_space.sample())
        
        # Check if it hit the table
        if wine_glass.get_position()[2] < 0.1:
            break
    
    # Let it settle after impact
    for _ in range(50):
        env.step(env.action_space.sample())
    
    final_health = wine_glass.health
    health_change = initial_health - final_health
    
    print(f"   Final wine glass health: {final_health:.2f}")
    print(f"   Health change: {health_change:.2f}")
    
    # Get impact history
    impact_history = wine_glass.get_impact_history()
    print(f"   Number of impacts detected: {len(impact_history.get('wine_glass', []))}")
    
    if health_change > 5.0:
        print("   ✅ PASS: Drop impact causes significant damage")
    else:
        print("   ❌ FAIL: Drop impact doesn't cause enough damage")
    
    return health_change > 5.0

def test_velocity_based_damage():
    """Test that damage scales with impact velocity."""
    print("\n🧪 Testing velocity-based damage scaling...")
    
    env = create_test_environment()
    env.reset()
    
    # Get objects
    wine_glass = env.scene.object_registry("name", "wine_glass")
    
    # Test different drop heights
    heights = [0.5, 1.0, 1.5]
    damages = []
    
    for height in heights:
        # Reset environment
        env.reset()
        
        # Let physics settle
        for _ in range(50):
            env.step(env.action_space.sample())
        
        initial_health = wine_glass.health
        
        # Drop from different height
        wine_glass.set_position_orientation(
            position=[0.0, 0.0, height],
            orientation=[0, 0, 0, 1]
        )
        
        # Let it fall and impact
        for _ in range(200):
            env.step(env.action_space.sample())
            if wine_glass.get_position()[2] < 0.1:
                break
        
        # Let it settle
        for _ in range(50):
            env.step(env.action_space.sample())
        
        final_health = wine_glass.health
        damage = initial_health - final_health
        damages.append(damage)
        
        print(f"   Drop height {height}m: {damage:.2f} damage")
    
    # Check if damage increases with height (velocity)
    if damages[1] > damages[0] and damages[2] > damages[1]:
        print("   ✅ PASS: Damage increases with drop height/velocity")
    else:
        print("   ❌ FAIL: Damage doesn't scale properly with velocity")
    
    return damages[1] > damages[0] and damages[2] > damages[1]

def test_material_properties():
    """Test that different materials have different damage responses."""
    print("\n🧪 Testing material property effects...")
    
    # Create environment with different objects
    scene_cfg = {"type": "Scene"}
    robot_cfg = {
        "type": "Tiago",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        "position": [0., 0.75, 0.0],
        "orientation": [0, 0, -1, 1],
        "grasping_mode": "assisted",
        "damage_params": PARAMS["tiago_robot"]
    }
    
    objects = [
        {
            "type": "DatasetObject",
            "name": "coffee_table",
            "category": "coffee_table",
            "model": "aoojzy",
            "position": [0.0, 0.0, 0.0],
            "orientation": [0, 0, 0, 1],
            "scale": [1.0, 1.0, 1.0],
            "damage_params": PARAMS["coffee_table"]
        },
        {
            "type": "DatasetObject",
            "name": "wine_glass",
            "category": "wineglass",
            "model": "cmdagy",
            "position": [0.0, 0.0, 0.5],
            "orientation": [0, 0, 0, 1],
            "scale": [1.0, 1.0, 1.0],
            "damage_params": PARAMS["bowl"],
            "mass": 0.2,
        },
        {
            "type": "DatasetObject",
            "name": "baseball",
            "category": "baseball",
            "model": "zanmar",
            "position": [0.5, 0.0, 0.5],
            "orientation": [0, 0, 0, 1],
            "scale": [1.0, 1.0, 1.0],
            "damage_params": PARAMS["baseball"],
            "mass": 0.15,
        }
    ]
    
    cfg = {"scene": scene_cfg, "robots": [robot_cfg], "objects": objects}
    env = DamageableEnvironment(configs=cfg)
    env.reset()
    
    # Get objects
    wine_glass = env.scene.object_registry("name", "wine_glass")
    baseball = env.scene.object_registry("name", "baseball")
    table = env.scene.object_registry("name", "coffee_table")
    
    # Let physics settle
    for _ in range(50):
        env.step(env.action_space.sample())
    
    # Drop both objects from same height
    wine_glass.set_position_orientation(position=[0.0, 0.0, 1.0], orientation=[0, 0, 0, 1])
    baseball.set_position_orientation(position=[0.5, 0.0, 1.0], orientation=[0, 0, 0, 1])
    
    # Let them fall and impact
    for _ in range(200):
        env.step(env.action_space.sample())
        if wine_glass.get_position()[2] < 0.1 and baseball.get_position()[2] < 0.1:
            break
    
    # Let them settle
    for _ in range(50):
        env.step(env.action_space.sample())
    
    wine_damage = 100.0 - wine_glass.health
    baseball_damage = 100.0 - baseball.health
    
    print(f"   Wine glass damage: {wine_damage:.2f}")
    print(f"   Baseball damage: {baseball_damage:.2f}")
    
    # Wine glass should take more damage than baseball (more fragile)
    if wine_damage > baseball_damage:
        print("   ✅ PASS: Fragile materials take more damage")
    else:
        print("   ❌ FAIL: Material properties not working correctly")
    
    return wine_damage > baseball_damage

def main():
    """Run all damage system tests."""
    print("🚀 Testing Improved Mechanical Damage System")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Static Contact", test_static_contact),
        ("Drop Impact", test_drop_impact),
        ("Velocity Scaling", test_velocity_based_damage),
        ("Material Properties", test_material_properties),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improved damage system is working correctly.")
    else:
        print("⚠️  Some tests failed. The damage system may need adjustments.")

if __name__ == "__main__":
    main() 