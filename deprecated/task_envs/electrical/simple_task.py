import os
import yaml
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
from omnigibson.object_states import OnTop, Filled


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

    # Defensive: coerce numeric damage params to floats if YAML parsed any as strings
    try:
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
    except Exception:
        pass

    if og.sim is not None:
        og.sim.stop()
        og.clear()

    env = DamageableEnvironment(configs=configs)
    env.reset()

    # Force-enable GPU dynamics on the PhysX scene as well (in addition to macros)
    try:
        stage = lazy.omni.usd.get_context().get_stage()
        phys_prim = stage.GetPrimAtPath("/World/PhysicsScene")
        if phys_prim and hasattr(lazy, "pxr") and hasattr(lazy.pxr, "PhysxSchema"):
            physx_scene = lazy.pxr.PhysxSchema.PhysxSceneAPI.Apply(phys_prim)
            physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
    except Exception:
        pass

    robot = env.robots[0] if len(env.robots) > 0 else None
    if robot is None:
        return

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

    # Place laptop and mug OnTop of the coffee table, and stabilize the mug to prevent slipping
    table = env.scene.object_registry("name", "coffee_table")
    for obj_name in ("laptop", "mug"):
        obj = env.scene.object_registry("name", obj_name)
        assert obj is not None and table is not None and OnTop in obj.states, f"Missing OnTop or object: {obj_name}"
        success = obj.states[OnTop].set_value(table, True)
        assert success, f"OnTop set failed for {obj_name}"
        if obj_name == "mug":
            # Enable CCD and increase friction on all mug links to prevent slip
            for _ln, _link in obj.links.items():
                _link.ccd_enabled = True
                _link.set_attribute("physxMaterial:staticFriction", 2.5)
                _link.set_attribute("physxMaterial:dynamicFriction", 2.5)
            obj.keep_still()
    # Let physics settle a bit to avoid immediate sliding
    for _ in range(20):
        og.sim.step()

    # Ensure a particle system exists, then set mug to Filled (force-init 'water')
    mug = env.scene.object_registry("name", "mug")
    assert mug is not None and Filled in mug.states, "Mug missing or not Fillable"
    env.scene.get_system("water")
    system_name = "water"
    mug.states[Filled].set_value(system_name, True)
    is_filled = mug.states[Filled].get_value(system_name)
    print(f"Mug Filled[{system_name}]: {bool(is_filled)}")
    breakpoint()

    controller = KeyboardRobotController(robot=robot)
    controller.print_keyboard_teleop_info()

    for _ in range(steps):
        action = controller.get_teleop_action()
        env.step(action=action)

    # Clean up camera mover and simulator
    camera_mover.clear()
    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main(steps=500)


