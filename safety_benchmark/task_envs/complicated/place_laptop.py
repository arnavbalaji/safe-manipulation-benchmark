import os
import yaml
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.object_states import Filled

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover


def _load_yaml_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "place_laptop.yaml")
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

    # Fresh simulator (same pattern as fill_mug.py and electrical/simple_task.py)
    if og.sim is not None:
        og.sim.stop()
        og.clear()

    env = DamageableEnvironment(configs=configs)
    env.reset()

    for _ in range(50):
        og.sim.step()

    # Ensure the water glass starts filled and seeded with particles (matches electrical reset behavior)
    water_glass = None
    try:
        water_glass = env.scene.object_registry("name", "water_glass")
    except Exception:
        water_glass = None
    water_system = env.scene.get_system("water", force_init=True)
    if water_glass is not None and Filled in water_glass.states and water_system is not None:
        water_glass.states[Filled].set_value(water_system, True)
        glass_pos, _ = water_glass.get_position_orientation()
        if isinstance(glass_pos, th.Tensor):
            glass_pos = glass_pos.tolist()
        z_offset = 0.05
        drop_pos = [glass_pos[0], glass_pos[1], glass_pos[2] + z_offset]
        # for _ in range(100):
        #     water_system.generate_particles(positions=[drop_pos])
        #     og.sim.step()

    # Grab robot
    robot = env.robots[0] if len(env.robots) > 0 else None
    if robot is None:
        return

    # Configure robot controllers similarly to electrical/simple_task.py
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

    # Increase friction on gripper fingers to reduce slip during grasp
    gripper_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
        prim_path=f"{robot.prim_path}/Looks/gripper_mat",
        name="gripper_material",
        static_friction=2.0,
        dynamic_friction=2.0,
    )
    for link_name, link in robot.links.items():
        link_n = link_name.lower()
        if ("gripper" in link_n) or ("finger" in link_n):
            for mesh in link.collision_meshes.values():
                mesh.apply_physics_material(gripper_mat)

    # Persist initial state after controller reload
    env.scene.update_initial_file()

    # Camera teleoperation (same style as electrical/simple_task.py and fill_mug.py)
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
    # Use the latest saved viewer camera pose from teleoperation
    start_cam_pos = th.tensor(
        [0.5973, -0.7110, 1.3160]
    )
    start_cam_quat = th.tensor(
        [0.5281, -0.1252, -0.1938, 0.8173]
    )
    og.sim.viewer_camera.set_position_orientation(
        position=start_cam_pos, orientation=start_cam_quat
    )

    controller = KeyboardRobotController(robot=robot)

    # Register TAB key to save sim state and breakpoint (mirrors other teleop scripts)
    def save_and_breakpoint():
        print("=== TAB callback triggered ===", flush=True)
        save_path = os.path.join(os.path.dirname(__file__), "place_laptop_saved.json")
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

    # Let the scene settle for a bit
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


