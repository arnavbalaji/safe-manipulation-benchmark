import os
import yaml
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson import object_states
from omnigibson.object_states import OnTop

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover


def _load_yaml_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "place_pan_on_stove.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main(steps: int = 500) -> None:
    # Enable GPU dynamics + object states (for future damage / thermal extensions)
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True
    gm.ENABLE_HQ_RENDERING = False

    configs = _load_yaml_config()

    # Coerce numeric damage params to floats if YAML parsed any as strings (robots only)
    robots_cfg = configs.get("robots", [])
    for r in robots_cfg:
        dmg = r.get("damage_params", {})
        mech = dmg.get("mechanical", {})
        if "damage_threshold" in mech:
            mech["damage_threshold"] = float(mech["damage_threshold"])
        if "scale" in mech:
            mech["scale"] = float(mech["scale"])
        if "instant_coefficient" in mech:
            mech["instant_coefficient"] = float(mech["instant_coefficient"])
        if "creep_coefficient" in mech:
            mech["creep_coefficient"] = float(mech["creep_coefficient"])
        lt = mech.get("link_thresholds", {})
        for _k, overrides in lt.items():
            if isinstance(overrides, dict):
                if "damage_threshold" in overrides:
                    overrides["damage_threshold"] = float(overrides["damage_threshold"])
                if "scale" in overrides:
                    overrides["scale"] = float(overrides["scale"])
                if "instant_coefficient" in overrides:
                    overrides["instant_coefficient"] = float(overrides["instant_coefficient"])
                if "creep_coefficient" in overrides:
                    overrides["creep_coefficient"] = float(overrides["creep_coefficient"])

    # Fresh simulator (same pattern as other complicated teleop tasks)
    if og.sim is not None:
        og.sim.stop()
        og.clear()

    env = DamageableEnvironment(configs=configs)
    env.reset()

    # Re-apply object placements so the pan sits properly on an inactive burner
    stove = env.scene.object_registry("name", "stove")
    pan = env.scene.object_registry("name", "pan")
    target_obj = env.scene.object_registry("name", "target_object")

    # Place pan on the stove, on an inactive burner (not the one that's on)
    # Use exact position and orientation from YAML for consistency
    if pan is not None and stove is not None:
        # Exact position from YAML: [0.3, -1.0, 0.8]
        pan_position = [0.3, -1.0, 0.8]
        # Exact orientation from YAML: [0, 0, 0.7071068, 0.7071068]
        pan_orientation = [0, 0, 0.7071068, 0.7071068]
        pan.set_position(pan_position)
        pan.set_orientation(pan_orientation)
        # Set OnTop state to ensure logical relationship
        if OnTop in pan.states:
            pan.states[OnTop].set_value(stove, True)
        pan.keep_still()
        # Step physics to let pan settle on stove
        for _ in range(10):
            og.sim.step()
    
    # Place spatula (target_object) on top of the pan
    # Use exact position from YAML for consistency
    if target_obj is not None and pan is not None:
        # Exact position from YAML: [0.3, -1.0, 0.85]
        target_position = [0.3, -1.0, 0.85]
        target_obj.set_position(target_position)
        # Set OnTop state to ensure logical relationship
        if OnTop in target_obj.states:
        target_obj.states[OnTop].set_value(pan, True)
        target_obj.keep_still()
        # Step physics to let target settle on pan
        for _ in range(10):
            og.sim.step()

    # Turn on the stove and close the oven door, mirroring thermal_damage.py
    if stove is not None:
        if object_states.ToggledOn in stove.states:
            stove.states[object_states.ToggledOn].set_value(True)
        if object_states.Open in stove.states:
            stove.states[object_states.Open].set_value(False)
        else:
            # Fallback: explicitly set any door / oven joints to zero
            for jname, joint in stove.joints.items():
                if "door" in jname.lower() or "oven" in jname.lower():
                    joint.set_joint_position(0.0)

    # Initialize target temperature to room temp (20°C) if the state exists
    if target_obj is not None and (object_states.Temperature in target_obj.states):
        target_obj.states[object_states.Temperature].set_value(20.0)

    # Let physics settle briefly after repositioning
    for _ in range(50):
        og.sim.step()

    # Grab robot
    robot = env.robots[0] if len(env.robots) > 0 else None
    if robot is None:
        return

    # Configure robot controllers similarly to other complicated tasks
    controller_config = {
        "arm_left": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        "arm_right": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        "gripper_left": {
            "motor_type": "position",
            "isaac_kp": 4000.0,
            "isaac_kd": 2000.0,
            "inverted": True,
        },
        "gripper_right": {
            "motor_type": "position",
            "isaac_kp": 4000.0,
            "isaac_kd": 2000.0,
            "inverted": True,
        },
    }
    robot.reload_controllers(controller_config=controller_config)

    # Persist initial state after controller reload
    env.scene.update_initial_file()

    # Camera teleoperation (same style as other complicated tasks)
    class CustomCameraMover(CameraMover):
        @property
        def input_to_command(self):
            return {
                # Strafe left/right
                lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
                # Forward / backward
                lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
                lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
                # Vertical move (use G to avoid conflicts)
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }

    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()

    # Start with a reasonable camera pose (can be adjusted and re-saved with TAB)
    start_cam_pos = th.tensor([1.1925, -1.0572, 1.2996])
    start_cam_quat = th.tensor([0.4387, 0.3980, 0.5414, 0.5967])
    og.sim.viewer_camera.set_position_orientation(
        position=start_cam_pos, orientation=start_cam_quat
    )

    controller = KeyboardRobotController(robot=robot)

    # Register TAB key to save sim state and breakpoint (mirrors other teleop scripts)
    def save_and_breakpoint():
        print("=== TAB callback triggered ===", flush=True)
        save_path = os.path.join(os.path.dirname(__file__), "place_pan_on_stove_saved.json")
        og.sim.save([save_path])
        print(f"✅ Saved simulation state to: {save_path}")
        # Print current viewer camera pose
        cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
        print(f"📷 Current camera position: {cam_pos}")
        print(f"📷 Current camera orientation (quaternion): {cam_quat}")
        breakpoint()

    controller.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Save simulation state to JSON and breakpoint",
        callback_fn=save_and_breakpoint,
    )

    controller.print_keyboard_teleop_info()

    # Let the scene settle a bit more with the robot controllers active
    for _ in range(50):
        og.sim.step_physics()

    # Simple teleop loop
    for _ in range(steps):
        action = controller.get_teleop_action()
        env.step(action=action)

    # Clean up camera mover and simulator
    camera_mover.clear()
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main(steps=5000)


