import os
import sys


def _ensure_omnigibson_on_path():
    # Allow running without installing the package by appending the local OmniGibson repo
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    og_src = os.path.join(repo_root, "OmniGibson")
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


def main():
    _ensure_omnigibson_on_path()

    import omnigibson as og
    from omnigibson.utils.ui_utils import KeyboardRobotController

    # Stop sim if already running in this process
    if og.sim is not None:
        og.sim.stop()

    # Minimal empty scene with a Tiago robot
    scene_cfg = {"type": "Scene"}

    # Controllers suitable for keyboard teleoperation (base + IK arms + grippers)
    controller_choices = {
        "base": "HolonomicBaseJointController",
        "arm_left": "InverseKinematicsController",
        "arm_right": "InverseKinematicsController",
        "gripper_left": "MultiFingerGripperController",
        "gripper_right": "MultiFingerGripperController",
        "camera": "JointController",
        "trunk": "JointController",
    }
    controller_config = {comp: {"name": name} for comp, name in controller_choices.items()}
    # Tiago's left gripper often needs inverted finger direction
    controller_config["gripper_left"]["inverted"] = True

    robot_cfg = {
        "type": "Tiago",
        "action_normalize": False,
        "grasping_mode": "assisted",
        "controller_config": controller_config,
        # Place the robot at origin facing +X
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        # Keep RGB camera for optional ego view
        "obs_modalities": ["rgb"],
    }

    cfg = {
        "env": {"action_timestep": 1.0 / 60.0, "physics_timestep": 1.0 / 120.0},
        "scene": scene_cfg,
        "robots": [robot_cfg],
    }

    # Create and reset environment
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    env.reset()
    robot.reset()

    # Keyboard teleoperation
    teleop = KeyboardRobotController(robot=robot)
    
    # Register TAB key to save simulation state
    import omnigibson.lazy as lazy
    teleop.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Save simulation state to disk",
        callback_fn=lambda: save_simulation_state(),
    )
    
    def save_simulation_state():
        """Save current simulation state to disk"""
        try:
            # Save to a timestamped file
            save_path = f"sim_state_test.json"
            og.sim.save([save_path])
            print(f"Simulation state saved to: {save_path}")
        except Exception as e:
            print(f"Failed to save simulation state: {e}")
        finally:
            breakpoint()
    
    teleop.print_keyboard_teleop_info()

    steps = 1000
    for _ in range(steps):
        action = teleop.get_teleop_action()
        env.step(action=action)

    # Clean shutdown
    og.shutdown()


if __name__ == "__main__":
    main()


