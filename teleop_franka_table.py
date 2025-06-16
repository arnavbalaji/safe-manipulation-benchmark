"""
Example script for teleoperating a FrankaPanda robot in a damage environment.
"""

import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS

# Configure OmniGibson settings
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True

# Object configurations
OBJECT_CONFIGS = {
    "baseball": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "baseball",
        "model": "zanmar",
        "position": [0.2, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.5, 1.5, 1.5],
        "damage_params": PARAMS["baseball"]
    },
    "bowl": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "bowl",
        "model": "bexgtn",
        "position": [0.2, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [0.75, 0.75, 0.65],
        "damage_params": PARAMS["bowl"]
    }
}

def main():
    """
    Robot control demo with FrankaPanda robot
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create scene configuration
    scene_cfg = {
        "type": "Scene",
        "physics_dt": 1.0 / 60.0,
        "render_dt": 1.0 / 60.0,
        "gravity": [0.0, 0.0, -9.81]
    }

    # Add the robot we want to load
    robot0_cfg = {
        "type": "FrankaPanda",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        "grasping_mode": "assisted",
        "position": [0.0, 0.0, 0.0],  # Placed on ground
        "orientation": [0, 0, 0, 1],
        "scale": [0.6, 0.6, 0.6],
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "mode": "pose_absolute_ori",
                "smoothing_filter_size": 5,
                "use_impedances": True,
                "command_output_limits": None
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "mode": "independent",
                "command_output_limits": None
            }
        }
    }

    # Add objects
    target_object = "bowl"  # or "baseball"
    objects = [OBJECT_CONFIGS[target_object]] if target_object in OBJECT_CONFIGS else []

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg], objects=objects)

    # Create the environment using DamageableEnvironment
    env = DamageableEnvironment(configs=cfg)

    # Get robot and set controllers
    robot = env.robots[0]
    controller_config = {
        "arm_0": {"name": "InverseKinematicsController"},
        "gripper_0": {"name": "MultiFingerGripperController"}
    }
    robot.reload_controllers(controller_config=controller_config)

    # Update initial state
    env.scene.update_initial_state()

    # Let physics settle
    for _ in range(50):
        og.sim.step()

    # Update the simulator's viewer camera's pose so it points towards the robot
    # og.sim.viewer_camera.set_position_orientation(
    #     position=th.tensor([1.46949, -3.97358, 2.21529]),
    #     orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    # )

    # Reset environment and robot
    env.reset()
    robot.reset()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Print out keyboard teleop info
    action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    step = 0
    while True:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1

    # Always shut down the environment cleanly at the end
    og.clear()


if __name__ == "__main__":
    main()  