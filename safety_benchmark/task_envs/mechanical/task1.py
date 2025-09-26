import os
import sys

import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.object_states import OnTop, Inside
from omnigibson.macros import gm


def main():
    """
    Load a simple damageable environment from task1.yaml (Tiago in an empty scene)
    and allow keyboard teleoperation.
    """
    # Resolve config path next to this script
    cfg_path = os.path.join(os.path.dirname(__file__), "task1.yaml")

    # Create environment (Damageable wrapper supports loading configs from file path)
    env = DamageableEnvironment(configs=cfg_path)

    # Grab the single robot (Tiago) and set controller configuration suitable for teleop
    robot = env.robots[0]
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
    # Tiago grippers are inverted
    controller_config["gripper_left"]["inverted"] = True
    controller_config["gripper_right"]["inverted"] = True
    robot.reload_controllers(controller_config=controller_config)

    # Persist initial state after controller reload
    env.scene.update_initial_file()

    # Reset before teleoperation
    env.reset()
    robot.reset()

    # Try to place plate on the top rack; if it fails, remove the plate cleanly
    # try:
    #     rack = env.scene.object_registry("name", "dish_rack")
    #     plate = env.scene.object_registry("name", "plate")
    #     if rack is not None and plate is not None:
    #         # Flatten the plate (face up), align yaw with rack, and nudge above rack before OnTop
    #         _, rack_quat = rack.get_position_orientation()
    #         plate.set_position_orientation(orientation=rack_quat)
    #         # Ensure plate is thin enough in Z if needed
    #         # plate.set_scale(th.tensor([0.5, 0.5, 0.4]))
    #         og.sim.step()
    #         if not plate.states[OnTop].set_value(rack, True):
    #             raise RuntimeError("OnTop sampling failed for plate->rack")
    #         og.sim.step()
    #         env.scene.update_initial_file()
    # except Exception as e:
    #     try:
    #         plate = env.scene.object_registry("name", "plate")
    #         if plate is not None:
    #             env.scene.remove_object(plate)
    #             og.sim.step()
    #             env.scene.update_initial_file()
    #     except Exception:
    #         pass

    # Place dish rack on top of sink
    # try:
    #     sink = env.scene.object_registry("name", "commercial_kitchen_sink")
    #     rack = env.scene.object_registry("name", "dish_rack")
    #     if sink is not None and rack is not None:
    #         og.sim.step()
    #         assert rack.states[OnTop].set_value(sink, True), "Failed to set dish rack OnTop of sink"
    #         og.sim.step()
    #         env.scene.update_initial_file()
    # except Exception as e:
    #     print(f"OnTop placement failed: {e}")

    # Spawn robot in kitchen if possible
    # try:
    #     seg_map = getattr(env.scene, "_seg_map", None)
    #     spawn_pos = None
    #     if seg_map is not None:
    #         _, pos = seg_map.get_random_point_by_room_instance("kitchen_0")
    #         if pos is None:
    #             _, pos = seg_map.get_random_point_by_room_type("kitchen")
    #         if pos is not None:
    #             spawn_pos = [float(pos[0]), float(pos[1]), float(pos[2] + 0.02)]
    #     if spawn_pos is not None:
    #         robot.set_position_orientation(position=spawn_pos)
    #         env.scene.update_initial_file()
    # except Exception:
    #     pass

    # Keyboard teleoperation
    teleop = KeyboardRobotController(robot=robot)

    # Camera teleoperation (custom key bindings)
    from omnigibson.utils.ui_utils import CameraMover
    class CustomCameraMover(CameraMover):
        @property
        def input_to_command(self):
            return {
                lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
                lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }

    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()

    # Register reset key (R)
    teleop.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    teleop.print_keyboard_teleop_info()
    print("Press ESC in the viewer to quit.")

    try:
        while True:
            action = teleop.get_teleop_action()
            env.step(action=action)
    except KeyboardInterrupt:
        pass
    finally:
        camera_mover.clear()
        og.clear()


if __name__ == "__main__":
    main()
