import os
import yaml
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
from omnigibson.object_states import OnTop


def _load_yaml_config():
  cfg_path = os.path.join(os.path.dirname(__file__), "fill_mug.yaml")
  with open(cfg_path, "r") as f:
      return yaml.safe_load(f)


def main(steps: int = 10) -> None:
  # Enable GPU dynamics + object states (required for particles and damage)
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

  # Place dish rack OnTop of the sink (supports automatic stable placement)
  sink = env.scene.object_registry("name", "furniture_sink")
  dish_rack = env.scene.object_registry("name", "dish_rack")
#   if (sink is not None) and (dish_rack is not None):
#       try:
#           dish_rack.states[OnTop].set_value(sink, True)
#       except Exception:
#           # If OnTop is not available or placement fails, continue without crashing
#           pass

  # Force-enable GPU dynamics on the PhysX scene as well (in addition to macros)
  stage = lazy.omni.usd.get_context().get_stage()
  phys_prim = stage.GetPrimAtPath("/World/PhysicsScene")
  if phys_prim and hasattr(lazy, "pxr") and hasattr(lazy.pxr, "PhysxSchema"):
      physx_scene = lazy.pxr.PhysxSchema.PhysxSceneAPI.Apply(phys_prim)
      physx_scene.CreateEnableGPUDynamicsAttr().Set(True)

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

  # Camera teleoperation (same style as electrical/simple_task.py)
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
  # Use the same starting viewer camera pose as electrical/simple_task.py
  start_cam_pos = th.tensor([-1.3225984573364258, 0.2645236849784851, 0.9981386065483093])
  start_cam_quat = th.tensor([0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452])
  og.sim.viewer_camera.set_position_orientation(position=start_cam_pos, orientation=start_cam_quat)

  controller = KeyboardRobotController(robot=robot)

  # Register TAB key to save sim state and breakpoint (mirrors electrical/simple_task.py)
  def save_and_breakpoint():
    print("=== TAB callback triggered ===", flush=True)
    save_path = os.path.join(os.path.dirname(__file__), "fill_mug_saved.json")
    og.sim.save([save_path])
    print(f"✅ Saved simulation state to: {save_path}")
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

  for _ in range(steps):
      action = controller.get_teleop_action()
      env.step(action=action)

  # Clean up camera mover and simulator
  camera_mover.clear()
  og.clear()
  og.shutdown()


if __name__ == "__main__":
  main(steps=5000)


