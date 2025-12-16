import os
import yaml
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm

from safety_benchmark.damageable_env import DamageableEnvironment
from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
from omnigibson.object_states import ToggledOn, OnFire
from omnigibson import object_states


def _load_yaml_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "simple_task.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main(steps: int = 10) -> None:
    # Enable GPU dynamics + object states (required for thermal simulation)
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
        # Also handle thermal damage params
        thermal = dmg.get("thermal", {})
        if "heating_threshold" in thermal:
            thermal["heating_threshold"] = float(thermal["heating_threshold"])
        if "cooling_threshold" in thermal:
            thermal["cooling_threshold"] = float(thermal["cooling_threshold"])
        if "scale" in thermal:
            thermal["scale"] = float(thermal["scale"])

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

    # Check if robot has Temperature state enabled (required for thermal damage)
    # If not, add it along with its dependencies
    if object_states.Temperature in robot.states:
        initial_temp = robot.states[object_states.Temperature].get_value()
        print(f"✅ Robot has Temperature state enabled. Initial temperature: {initial_temp:.2f}°C")
    else:
        print("⚠️  Robot does NOT have Temperature state enabled. Adding it now...")
        
        # Check if robot is a StatefulObject (it should be)
        from omnigibson.objects.stateful_object import StatefulObject
        if not isinstance(robot, StatefulObject):
            print("❌ ERROR: Robot is not a StatefulObject, cannot add Temperature state!")
        else:
            # Temperature requires AABB as a dependency - check and add if needed
            if object_states.AABB not in robot.states:
                print("   Adding AABB state (required dependency)...")
                compatible, reason = object_states.AABB.is_compatible(obj=robot)
                if compatible:
                    aabb_state = object_states.AABB(obj=robot)
                    robot.add_state(aabb_state)
                    aabb_state.initialize()
                    print("   ✅ AABB state added and initialized")
                else:
                    print(f"   ❌ Cannot add AABB state: {reason}")
            
            # Now add Temperature state
            compatible, reason = object_states.Temperature.is_compatible(obj=robot)
            if compatible:
                temp_state = object_states.Temperature(obj=robot)
                robot.add_state(temp_state)
                temp_state.initialize()
                initial_temp = temp_state.get_value()
                print(f"   ✅ Temperature state added and initialized. Initial temperature: {initial_temp:.2f}°C")
            else:
                print(f"   ❌ Cannot add Temperature state: {reason}")
                print(f"   Available robot states: {[s.__name__ for s in robot.states.keys()]}")

    # ------------------------------------------------------------------
    # Arrange a single burning log INSIDE the (scaled, shifted) fireplace
    # and align the flame with the log
    # ------------------------------------------------------------------
    fireplace = env.scene.object_registry("name", "fireplace")
    log_obj = env.scene.object_registry("name", "log_center")
    if fireplace is not None and log_obj is not None:
        fire_pos, _ = fireplace.get_position_orientation()

        # Ensure fireplace itself is not visually on fire; flame should come from the log
        if OnFire in fireplace.states:
            fireplace.states[OnFire].set_value(False)

        # Filter collisions between robot and fireplace's fillable meta link only
        # This allows the robot arm to reach inside the fireplace interior to place logs
        # while keeping collisions with the base and other structural parts
        if robot is not None:
            # Find the fillable meta link (interior of fireplace)
            target_link_name = "meta__base_link_fillable_0_0_link"
            if target_link_name in fireplace.links:
                target_link = fireplace.links[target_link_name]
                for robot_link_name, robot_link in robot.links.items():
                    robot_link.add_filtered_collision_pair(target_link)
                    print(f"✅ Filtered collisions between robot link '{robot_link_name}' and fireplace fillable link '{target_link_name}'")
            else:
                print(f"⚠️  Fireplace fillable link '{target_link_name}' not found. Available links: {list(fireplace.links.keys())}")

        # Place the log centered in the fireplace, slightly above the base
        dx, dy, dz = 0.0, 0.0, -0.18
        new_pos = [fire_pos[0] + dx, fire_pos[1] + dy, fire_pos[2] + dz]
        try:
            if hasattr(log_obj, "set_position"):
                log_obj.set_position(new_pos)
            if hasattr(log_obj, "scale"):
                log_obj.scale = [0.6, 0.6, 0.6]
        except Exception:
            pass

        # Flame should follow the log
        if OnFire in log_obj.states:
            log_obj.states[OnFire].set_value(True)

    # Step a few frames so particle / visual effects spawn
    for _ in range(10):
        og.sim.step()

    # Configure robot controllers with higher gripper force
    controller_config = {
        # Set arms to IK so teleop supports EEF translation and rotation
        "arm_left": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        "arm_right": {
            "name": "InverseKinematicsController",
            "mode": "pose_delta_ori",
        },
        # Stronger gripper position control (higher gains for easier grasping)
        "gripper_left": {
            "motor_type": "position",
            "isaac_kp": 8000.0,
            "isaac_kd": 4000.0,
            "inverted": True,
        },
        "gripper_right": {
            "motor_type": "position",
            "isaac_kp": 8000.0,
            "isaac_kd": 4000.0,
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

    # Set torso to lowest position (minimum height)
    if hasattr(robot, "trunk_control_idx") and hasattr(robot, "joint_lower_limits"):
        trunk_idx = robot.trunk_control_idx
        trunk_lower_limits = robot.joint_lower_limits[trunk_idx]
        # Set trunk joints to their minimum positions
        robot.set_joint_positions(positions=trunk_lower_limits, indices=trunk_idx, drive=False)
        # Step simulation a few times to let the torso settle
        for _ in range(10):
            og.sim.step()
        print("✅ Torso set to lowest position")

    # Camera teleoperation
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
    # Set starting viewer camera pose
    start_cam_pos = th.tensor([-1.3225984573364258, 0.2645236849784851, 0.9981386065483093])
    start_cam_quat = th.tensor([0.4495110511779785, -0.4464901387691498, -0.5452293753623962, 0.5489183068275452])
    og.sim.viewer_camera.set_position_orientation(position=start_cam_pos, orientation=start_cam_quat)

    # Fireplace is configured as heatSource in YAML, no additional setup needed

    controller = KeyboardRobotController(robot=robot)
    
    # Register TAB key to save sim state and breakpoint
    def save_and_breakpoint():
        print("=== TAB callback triggered ===", flush=True)
        save_path = os.path.join(os.path.dirname(__file__), "simple_task_saved.json")
        og.sim.save([save_path])
        print(f"✅ Saved simulation state to: {save_path}")
        # Print current camera position and orientation
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

    for _ in range(50):
        og.sim.step_physics()

    for _ in range(steps):
        action = controller.get_teleop_action()
        env.step(action=action)

    # Clean up camera mover and simulator
    camera_mover.clear()
    # Only shutdown - og.clear() can cause registry errors during cleanup
    og.shutdown()


if __name__ == "__main__":
    main(steps=5000)

