"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import torch as th
import json
import os
import cv2
from datetime import datetime
import numpy as np
import subprocess

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.object_states import OnTop
from omnigibson.utils.ui_utils import KeyboardRobotController
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

OBJECT_CONFIGS = {
    "baseball": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "baseball",
        "model": "zanmar",
        "position": [0.1, 0.0, 3.0],
        "orientation": [0, 0, 0, 1],
        "scale": [2.0, 2.0, 2.0],
        "damage_params": PARAMS["baseball"]
    },
    "bowl": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "wineglass",
        "model": "cmdagy",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],  # Identity quaternion for upright orientation
        # "scale": [0.5, 0.5, 0.4],  # Reduced scale to make it lighter and easier to grasp
        # "scale": [0.75, 0.75, 0.65],
        "scale": [1.5, 1.5, 1.5],
        "damage_params": PARAMS["bowl"],
        "mass": 0.005,  # Very light mass for easier manipulation
        "friction": 1.0,  # Higher friction for better grip
        "restitution": 0.1,  # Low bounciness
    },
    "coffee_table": {
        "type": "DatasetObject",    
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 1.0, 1.0],
        "damage_params": PARAMS["coffee_table"]
    }
}


def main():
    """
    Minimal teleop demo with Tiago in an empty scene.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Always use empty scene and Tiago robot
    scene_cfg = {"type": "Scene"}
    robot0_cfg = dict()
    robot0_cfg["type"] = "Tiago"
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True
    robot0_cfg["position"] = [0., 0.65, 0.0]  # Position next to the table
    robot0_cfg["orientation"] = [0, 0, -1, 1]  # Facing the table
    robot0_cfg["grasping_mode"] = "assisted"
    robot0_cfg["damage_params"] = PARAMS["tiago_robot"]

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    objects = [OBJECT_CONFIGS["coffee_table"], OBJECT_CONFIGS["bowl"]]
    cfg["objects"] = objects

    # Create the environment
    env = DamageableEnvironment(configs=cfg)

    # Load saved simulation state if it exists

    # Choose robot controller to use
    robot = env.robots[0]
    controller_choices = {
        "base": "JointController",
        "arm_left": "InverseKinematicsController",
        "arm_right": "InverseKinematicsController",
        "gripper_left": "MultiFingerGripperController",
        "gripper_right": "MultiFingerGripperController",
        "camera": "JointController",
    }

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    
    # Fix gripper controller configuration for Tiago
    controller_config["gripper_left"]["inverted"] = True
    controller_config["gripper_right"]["inverted"] = True
    
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_state()
    

    # Update the simulator's viewer camera's pose so it points towards the robot
    # og.sim.viewer_camera.set_position_orientation(
    #     position=th.tensor([1.46949, -3.97358, 2.21529]),
    #     orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    # )

    # Reset environment and robot
    env.reset()
    robot.reset()

    coffee_table = env.scene.object_registry("name", "coffee_table")
    obj = env.scene.object_registry("name", "target_object")
    
    # First set the OnTop state
    assert obj.states[OnTop].set_value(coffee_table, True), "Failed to set OnTop state"
    
    # Get table top position and place bowl there
    table_pos = coffee_table.get_position()

    # Let physics settle
    for _ in range(20):  # Increased settling steps
        og.sim.step()

    save_state_path = "safety_benchmark/grasp_save_state.json"
    if os.path.exists(save_state_path):
        print(f"🔄 Loading saved simulation state from: {save_state_path}")
        og.sim.restore(scene_files=[save_state_path])
        env.inialize_damageable_objects()
        print("✅ Successfully loaded saved simulation state")
    else:
        print(f"📝 No saved state found at: {save_state_path}")
        print("Starting with fresh environment...")


    for _ in range(20):  # Increased settling steps
        og.sim.step()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    
    # Enable camera teleoperation with custom key bindings (excluding 'T' key)
    from omnigibson.utils.ui_utils import CameraMover
    class CustomCameraMover(CameraMover):
        @property
        def input_to_command(self):
            """
            Returns:
                dict: Mapping from relevant keypresses to corresponding delta command to apply to the camera pose
            """
            return {
                lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
                lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
                # Removed T key to avoid conflict with gripper control
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }
    
    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()
    
    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )
    
    # Function to save simulation state with breakpoint
    def save_sim_state():
        filepath = f"safety_benchmark/grasp_save_state2.json"
        og.sim.save(json_paths=[filepath])
        print(f"✅ Simulation state saved to: {filepath}")
        breakpoint()
    
    # Register TAB key for breakpoint and state saving
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Breakpoint and save simulation state",
        callback_fn=save_sim_state,
    )

    # Print out relevant keyboard info
    action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = 500
    step = 0

    images = []
    healths = []
    link_healths = []
    damage_statuses = []

    while step != max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1

        rgb_img = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_img = rgb_img.cpu().numpy()[:, :, :3]
        rgb_img = cv2.resize(rgb_img, (512, 512))
        images.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        obj = env.scene.object_registry("name", "target_object")
        healths.append(obj.health)
        damage_statuses.append(obj.damage_status)
        link_healths.append(obj.link_healths.copy())


    # Clean up camera mover
    camera_mover.clear()

    # Save video
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    avi_path = f'videos_and_images/bowl_grasp_teleop.avi'
    mp4_path = f'videos_and_images/bowl_grasp_teleop.mp4'
    out = cv2.VideoWriter(avi_path, fourcc, 30, (width, height))

    for i, image in enumerate(images):
        # Add health captions
        frame_copy = image.copy()
        y_pos = 30
        cv2.putText(frame_copy, f"Object Health: {healths[i]:.2f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Handle link health wrapping
        y_pos += 30
        link_health_text = f"Link Health: {', '.join([f'{key}: {value:.2f}' for key, value in link_healths[i].items()])}"
        words = link_health_text.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) * 10 < width - 20:  # Approximate character width
                current_line = test_line
            else:
                cv2.putText(frame_copy, current_line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 25
                current_line = word
        if current_line:
            cv2.putText(frame_copy, current_line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 25

        # Add damage status
        cv2.putText(frame_copy, f"Damage Status: {damage_statuses[i]}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(np.ascontiguousarray(frame_copy, dtype=np.uint8))
    out.release()

    # Convert AVI to MP4 using ffmpeg
    subprocess.run([
        'ffmpeg', '-y', '-i', avi_path,
        '-c:v', 'mpeg4', mp4_path
    ], check=True)
    
    # Clean up AVI file
    os.remove(avi_path)

    # Always shut down the environment cleanly at the end
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main()