"""
Demonstration script for the improved mechanical damage system.
Shows realistic physics-based damage calculation with visual feedback.
"""

import os
import sys
import torch as th
import numpy as np
import time

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS

# Configure OmniGibson settings
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True

def create_demo_environment():
    """Create a demonstration environment with various objects."""
    
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
            "position": [0.0, 0.0, 0.5],
            "orientation": [0, 0, 0, 1],
            "scale": [1.0, 1.0, 1.0],
            "damage_params": PARAMS["bowl"],
            "mass": 0.2,
            "friction": 0.8,
            "restitution": 0.1,
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
    
    # Compile configuration
    cfg = {
        "scene": scene_cfg,
        "robots": [robot_cfg],
        "objects": objects
    }
    
    return DamageableEnvironment(configs=cfg)

def print_health_status(env):
    """Print current health status of all objects."""
    print("\n" + "="*60)
    print("🏥 HEALTH STATUS")
    print("="*60)
    
    for obj in env.scene.objects:
        if hasattr(obj, 'health'):
            obj_name = obj.name if hasattr(obj, 'name') else obj.__class__.__name__
            health = obj.health
            status = obj.damage_status
            
            # Color coding based on health
            if health > 80:
                health_emoji = "🟢"
            elif health > 60:
                health_emoji = "🟡"
            elif health > 30:
                health_emoji = "🟠"
            else:
                health_emoji = "🔴"
            
            print(f"{health_emoji} {obj_name}: {health:.1f}/100 ({status})")
            
            # Show link-specific health if available
            if hasattr(obj, 'link_healths') and len(obj.link_healths) > 1:
                for link_name, link_health in obj.link_healths.items():
                    print(f"   └─ {link_name}: {link_health:.1f}/100")

def print_impact_history(env):
    """Print recent impact history for debugging."""
    print("\n" + "="*60)
    print("💥 RECENT IMPACTS")
    print("="*60)
    
    for obj in env.scene.objects:
        if hasattr(obj, 'get_impact_history'):
            obj_name = obj.name if hasattr(obj, 'name') else obj.__class__.__name__
            history = obj.get_impact_history()
            
            if history:
                print(f"📊 {obj_name} impact history:")
                for link_name, impacts in history.items():
                    if impacts:
                        print(f"   └─ {link_name}: {len(impacts)} impacts")
                        # Show details of most recent impact
                        latest = impacts[-1]
                        print(f"      └─ Latest: {latest['impact_energy']:.3f}J energy, "
                              f"{latest['velocity_change']:.2f}m/s velocity change")
            else:
                print(f"📊 {obj_name}: No impacts detected")

def demo_instructions():
    """Print demo instructions."""
    print("\n" + "="*60)
    print("🎮 DEMO CONTROLS")
    print("="*60)
    print("Keyboard Controls:")
    print("  H - Print health status")
    print("  I - Print impact history")
    print("  R - Reset environment")
    print("  D - Drop wine glass from height")
    print("  B - Drop baseball from height")
    print("  T - Drop both objects from height")
    print("  Q - Quit demo")
    print("\nRobot Controls:")
    print("  WASD - Move robot base")
    print("  Arrow keys - Control robot arms")
    print("  Space - Open/close grippers")
    print("  Mouse - Control camera")
    print("="*60)

def drop_object(env, obj_name, height=1.0):
    """Drop an object from a specified height."""
    obj = env.scene.object_registry("name", obj_name)
    if obj is None:
        print(f"❌ Object '{obj_name}' not found!")
        return
    
    print(f"📦 Dropping {obj_name} from {height}m height...")
    
    # Get current position and set new height
    current_pos = obj.get_position()
    obj.set_position_orientation(
        position=[current_pos[0], current_pos[1], height],
        orientation=[0, 0, 0, 1]
    )
    
    # Let it fall
    for _ in range(100):
        env.step(env.action_space.sample())
        if obj.get_position()[2] < 0.1:
            break
    
    # Let it settle
    for _ in range(50):
        env.step(env.action_space.sample())
    
    # Check damage
    damage = 100.0 - obj.health
    print(f"   💥 {obj_name} took {damage:.2f} damage")

def main():
    """Main demonstration function."""
    print("🚀 Improved Mechanical Damage System Demo")
    print("="*60)
    print("This demo shows realistic physics-based damage calculation")
    print("based on impact energy and velocity changes.")
    print("="*60)
    
    # Create environment
    env = create_demo_environment()
    env.reset()
    
    # Get objects
    wine_glass = env.scene.object_registry("name", "wine_glass")
    baseball = env.scene.object_registry("name", "baseball")
    table = env.scene.object_registry("name", "coffee_table")
    robot = env.robots[0]
    
    # Let physics settle
    print("⏳ Settling physics...")
    for _ in range(50):
        env.step(env.action_space.sample())
    
    # Create robot controller
    action_generator = KeyboardRobotController(robot=robot)
    
    # Register custom key bindings
    def print_health():
        print_health_status(env)
    
    def print_impacts():
        print_impact_history(env)
    
    def reset_env():
        print("🔄 Resetting environment...")
        env.reset()
        for _ in range(50):
            env.step(env.action_space.sample())
        print("✅ Environment reset complete")
    
    def drop_wine_glass():
        drop_object(env, "wine_glass", height=1.0)
    
    def drop_baseball():
        drop_object(env, "baseball", height=1.0)
    
    def drop_both():
        print("📦 Dropping both objects from 1.0m height...")
        drop_object(env, "wine_glass", height=1.0)
        drop_object(env, "baseball", height=1.0)
    
    # Register key bindings
    import omnigibson.lazy as lazy
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.H,
        description="Print health status",
        callback_fn=print_health,
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.I,
        description="Print impact history",
        callback_fn=print_impacts,
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset environment",
        callback_fn=reset_env,
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.D,
        description="Drop wine glass",
        callback_fn=drop_wine_glass,
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.B,
        description="Drop baseball",
        callback_fn=drop_baseball,
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.T,
        description="Drop both objects",
        callback_fn=drop_both,
    )
    
    # Print initial status
    print_health_status(env)
    demo_instructions()
    
    # Main loop
    print("\n🎮 Starting demo... Press keys to interact!")
    
    try:
        while True:
            # Get action from controller
            action = action_generator.get_random_action()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check for quit
            if action_generator.check_quit():
                break
            
            # Small delay for visualization
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    main() 