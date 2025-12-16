import os
import yaml
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
import math

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
from omnigibson.object_states import ToggledOn, Filled, Open


def _load_yaml_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "simple_task.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main(steps: int = 10) -> None:
    # Enable GPU dynamics + object states (required for particles and Filled)
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_FLATCACHE = True
    gm.ENABLE_HQ_RENDERING = False

    configs = _load_yaml_config()

    # Coerce numeric damage params to floats if YAML parsed any as strings
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

    if og.sim is not None:
        og.sim.stop()
        og.clear()

    env = DamageableEnvironment(configs=configs)
    env.reset()

    # Force-enable GPU dynamics on the PhysX scene as well (in addition to macros)
    stage = lazy.omni.usd.get_context().get_stage()
    phys_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    if phys_prim and hasattr(lazy, "pxr") and hasattr(lazy.pxr, "PhysxSchema"):
        physx_scene = lazy.pxr.PhysxSchema.PhysxSceneAPI.Apply(phys_prim)
        physx_scene.CreateEnableGPUDynamicsAttr().Set(True)

    robot = env.robots[0] if len(env.robots) > 0 else None
    if robot is None:
        return

    # Configure robot controllers with higher gripper force (match mech_damage.py)
    controller_config = {
        # Set arms to IK so teleop supports EEF translation and rotation (arrows/P/;/N/B/O/U/V/C)
        "arm_left": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        "arm_right": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        # Stronger gripper position control
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
    # Create a PhysicsMaterial and apply it to gripper/finger collision meshes
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

    # Camera teleoperation (match mech_damage.py style)
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
    # Set starting viewer camera pose to match mech_damage.py
    start_cam_pos = th.tensor([-1.3225984573364258, 0.2645236849784851, 0.9981386065483093])
    start_cam_quat = th.tensor([0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452])
    og.sim.viewer_camera.set_position_orientation(position=start_cam_pos, orientation=start_cam_quat)

    # Configure sinks: near sink OFF, far sink ON for particle generation
    near_sink = env.scene.object_registry("name", "furniture_sink")
    far_sink = env.scene.object_registry("name", "furniture_sink_far")
    if near_sink is not None and ToggledOn in near_sink.states:
        near_sink.states[ToggledOn].set_value(False)
    if far_sink is not None and ToggledOn in far_sink.states:
        far_sink.states[ToggledOn].set_value(True)
    # Also enforce nested toggles for any child parts (e.g., faucet sub-objects)
    for obj in getattr(env.scene, "objects", []):
        try:
            if near_sink is not None and hasattr(obj, "prim_path") and obj.prim_path.startswith(near_sink.prim_path):
                if ToggledOn in obj.states:
                    obj.states[ToggledOn].set_value(False)
            if far_sink is not None and hasattr(obj, "prim_path") and obj.prim_path.startswith(far_sink.prim_path):
                if ToggledOn in obj.states:
                    obj.states[ToggledOn].set_value(True)
        except Exception:
            pass
    # Clear any residual water particles near view; far sink will regenerate as needed
    try:
        water_system = env.scene.get_system("water", force_init=True)
        if hasattr(water_system, "remove_all_particles"):
            water_system.remove_all_particles()
    except Exception:
        pass
    
    # Open laptop to fixed angle and fix base to prevent drift
    laptop = env.scene.object_registry("name", "laptop")

    controller = KeyboardRobotController(robot=robot)
    
    # Register TAB key to save sim state and breakpoint
    def save_and_breakpoint():
        print("=== TAB callback triggered ===", flush=True)
        save_path = os.path.join(os.path.dirname(__file__), "simple_task_saved.json")
        og.sim.save([save_path])
        print(f"✅ Saved simulation state to: {save_path}")
        breakpoint()
    
    controller.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="Save simulation state to JSON and breakpoint",
        callback_fn=save_and_breakpoint,
    )
    
    controller.print_keyboard_teleop_info()

    for _ in range(50):
        og.sim.step_physics()

    def set_laptop_pose(laptop):
        if laptop is not None:
            # Drive hinge joints to target angle, clamped to limits
            target_deg = 130.0
            target_rad = math.radians(target_deg)
            if hasattr(laptop, "joints"):
                for joint in laptop.joints.values():
                    lo = joint.lower_limit
                    hi = joint.upper_limit
                    target = max(lo, min(hi, target_rad))
                    joint.set_pos(target)
                    joint.keep_still()
            # Fix the articulation root to hold pose
            laptop.keep_still()

    for _ in range(steps):
        action = controller.get_teleop_action()
        set_laptop_pose(laptop)
        env.step(action=action)

    # Clean up camera mover and simulator
    camera_mover.clear()
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main(steps=5000)


