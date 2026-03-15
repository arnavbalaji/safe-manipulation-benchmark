import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import json
import h5py
import pickle
import torch as th
import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T

from omnigibson.utils.ui_utils import KeyboardRobotController, CameraMover
import omnigibson.lazy as lazy

from omnigibson.controllers.controller_base import IsGraspingState

from safety_benchmark.damageable_env import DamageableEnvironment, DamageableDataCollectionWrapper

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_OBJECT_STATES = True  # Required for OnTop state

# Increase assisted grasp force for better grasping
# Import manipulation_robot module to access its macros
from omnigibson.robots import manipulation_robot
# Increase MAX_ASSIST_FORCE from default 100 to 500 for stronger grasp
manipulation_robot.m.MAX_ASSIST_FORCE = 500
print(f"Assisted grasp force increased: MAX_ASSIST_FORCE = {manipulation_robot.m.MAX_ASSIST_FORCE}")


def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_state', action='store_true', help='Load a state')
    args = parser.parse_args()
    if args.load_state:
        load_state_path = f"resources/teleop_data/pick_place.hdf5"
        load_f = h5py.File(load_state_path, "r")

    # TODO: Set this
    collect_hdf5_path = f"resources/teleop_data/pick_place.hdf5"
    if not os.path.exists(collect_hdf5_path):
        os.makedirs(os.path.dirname(collect_hdf5_path), exist_ok=True)

    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Overwrite any configs here
    cfg["scene"]["scene_model"] = "house_single_floor"
    cfg["scene"]["not_load_object_categories"] = ["ottoman"]
    cfg["scene"]["load_room_instances"] = ["kitchen_0"]
    
    ############### FrankaMounted robot ###############
    # Mounted Franka robot positioned in front of kitchen sink, facing it
    # Position: in front of sink area (sink is typically on counter around y=-2.0 to y=-1.5)
    # Robot should face negative y direction (towards sink)
    cfg["robots"][0] = {
        "type": "FrankaMounted",
        "name": "franka0",
        "position": [5.7, -1.4, 0.0],  # In front of sink area, Z=0.0 for ground level
        "orientation": [0.0, 0.0, 0.0, 1.0],  # Default orientation (will face sink)
        "grasping_mode": "assisted",
        "obs_modalities": ["rgb", "depth"],
        "action_normalize": False,
        "self_collisions": True,
        # FrankaMounted has single arm (arm_0, gripper_0) like FrankaPanda
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "command_input_limits": None,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            },
        },
    }
    ############### FrankaMounted robot ###############

    # Generate external sensors config automatically
    # Get robot name and type from config to construct correct prim path
    robot_name = cfg["robots"][0].get("name", "franka0")
    robot_type = cfg["robots"][0].get("type", "FrankaMounted").lower()

    # Set external cameras for videos (optional, can be added later)
    # For now, we'll skip external cameras and just use viewer camera
    cfg["env"]["external_sensors"] = []

    # Add plate to kitchen counter to the right of the robot
    # Robot is at [6.7, -1.3, 0.0], plate positioned on counter (Z will be set correctly) to the right (positive X)
    # Plate X, Y will be preserved, Z will be set based on countertop height
    cfg["objects"] = [
        {
            "type": "DatasetObject",
            "name": "plate",
            "category": "plate",
            "model": "ntedfx",
            "position": [5.4, -1.7, 0.95],  # X, Y preserved, Z will be adjusted to countertop height
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
    ]

    env = DamageableEnvironment(configs=cfg)        
    env = DamageableDataCollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
        enable_dump_filters=False,
    )

    robot = env.robots[0]
    
    # Step simulation to ensure robot is fully initialized
    for _ in range(10): 
        og.sim.step()
    
    # Rotate robot orientation by 90 degrees to the right (around Z axis)
    # 90 degrees = π/2 radians, negative for clockwise (right)
    current_pos, current_orn = robot.get_position_orientation()
    rotation_90_right = T.euler2quat(th.tensor([0.0, 0.0, -np.pi/2]))  # Negative for clockwise
    new_robot_orn = T.quat_multiply(current_orn, rotation_90_right)
    robot.set_position_orientation(position=current_pos, orientation=new_robot_orn)
    
    # Step again after rotation
    for _ in range(10): 
        og.sim.step()
    
    # Set viewer camera to specified position and orientation
    camera_pos = th.tensor([4.691344738006592, -0.8605108261108398, 1.533362627029419])
    camera_orn = th.tensor([-0.215546652674675, 0.5899057388305664, 0.7309074997901917, -0.2670675814151764])
    og.sim.viewer_camera.set_position_orientation(
        position=camera_pos,
        orientation=camera_orn,
    )

    # FrankaMounted default joint positions (same as FrankaPanda: 7 arm joints + 2 gripper joints)
    robot.set_joint_positions(th.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]))

    # Increase gripper force by raising joint stiffness / damping on the gripper controller
    # Following pattern from mech_damage.py
    controller_config = {
        "arm_0": {
            "name": "InverseKinematicsController",
            "command_input_limits": None,
        },
        "gripper_0": {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": "smooth",
            "motor_type": "position",
            "isaac_kp": 4000.0,  # Increased from default for stronger grasp
            "isaac_kd": 2000.0,  # Increased from default for stronger grasp
        },
    }
    robot.reload_controllers(controller_config=controller_config)

    # Increase friction on gripper fingers to reduce slip during grasp
    try:
        for link_name, link in robot.links.items():
            n = link_name.lower()
            if ("gripper" in n) or ("finger" in n):
                try:
                    link.set_attribute("physxMaterial:staticFriction", 2.0)
                    link.set_attribute("physxMaterial:dynamicFriction", 2.0)
                except Exception:
                    pass
    except Exception:
        pass

    for _ in range(10):
        og.sim.step()

    # Set OnTop relationship: plate on counter/table
    # Find the countertop/table in the kitchen
    countertop = None
    for obj in env.scene.objects:
        if "countertop" in obj.name.lower() or "counter" in obj.name.lower():
            countertop = obj
            print(f"Found countertop: {obj.name}")
            break
    
    if countertop is None:
        print("Warning: Could not find countertop in scene, trying to find table or bar")
        for obj in env.scene.objects:
            if "table" in obj.name.lower() or "bar" in obj.name.lower():
                countertop = obj
                print(f"Found table/bar: {obj.name}")
                break
    
    # Get the plate object we just added
    plate = env.scene.object_registry("name", "plate")
    
    if countertop and plate:
        # Desired X, Y position from config (preserve these)
        desired_x = 5.4
        desired_y = -1.7
        
        # Get countertop's top surface Z coordinate
        countertop_aabb_min, countertop_aabb_max = countertop.aabb
        countertop_top_z = countertop_aabb_max[2].item()
        
        # Get plate's half height (from center to bottom)
        plate_aabb_min, plate_aabb_max = plate.aabb
        plate_half_height = (plate_aabb_max[2] - plate_aabb_min[2]).item() / 2.0
        
        # Calculate Z position: countertop top + plate half height + small offset
        desired_z = countertop_top_z + plate_half_height + 0.001
        
        # Set plate position manually (preserving X, Y, setting Z correctly)
        plate.set_position_orientation(
            position=th.tensor([desired_x, desired_y, desired_z]),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0])
        )
        
        print(f"Set plate position to [{desired_x}, {desired_y}, {desired_z:.3f}]")
        print(f"  Countertop top Z: {countertop_top_z:.3f}")
        print(f"  Plate half height: {plate_half_height:.3f}")
        
        # Step simulation to let physics settle
        for _ in range(20):
            og.sim.step()
        
        # Verify OnTop relationship (but don't resample position)
        if object_states.OnTop in plate.states:
            is_ontop = plate.states[object_states.OnTop].get_value(countertop)
            if is_ontop:
                print(f"Plate is OnTop of {countertop.name} ✓")
            else:
                print(f"Warning: Plate may not be OnTop of {countertop.name} (checking relationship)")
        else:
            print("Warning: plate does not have OnTop state")
    else:
        if not countertop:
            print("Warning: Could not find countertop/table in scene")
        if not plate:
            print("Warning: Could not find plate object")
    
    # Increase friction on plate to reduce slip during grasp (following mech_damage.py pattern)
    if plate is not None:
        try:
            for plate_link in plate.links.values():
                try:
                    plate_link.set_attribute("physxMaterial:staticFriction", 1.5)
                    plate_link.set_attribute("physxMaterial:dynamicFriction", 1.5)
                except Exception:
                    pass
        except Exception:
            pass

    # Camera teleoperation with WASD keys
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

    # Keyboard Teleop
    action_generator = KeyboardRobotController(robot=robot)
    
    # Function to save simulation state to pkl file
    def save_state_to_pkl():
        """Save current simulation state to pkl file"""
        os.makedirs("safe-manipulation-benchmark/resources/saved_states", exist_ok=True)
        save_path = "safe-manipulation-benchmark/resources/saved_states/pick_place_init_state.pkl"
        og.sim.update_handles()  # Update handles before dumping state
        state = og.sim.dump_state(serialized=True)
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        print(f"Simulation state saved to: {save_path}")
    
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.S,
        description="Save simulation state to pkl file",
        callback_fn=save_state_to_pkl,
    )
    action_generator.print_keyboard_teleop_info()

    # Print robot and sink info
    print(f"Robot position: {robot.get_position_orientation()[0]}")
    # Try to find sink in the scene
    sink = None
    for obj in env.scene.objects:
        if "sink" in obj.name.lower():
            sink = obj
            print(f"Found sink: {obj.name} at position {obj.get_position_orientation()[0]}")
            break
    if sink is None:
        print("No sink found in scene - manipulation will be done in the sink area")

    # ======================== Data collection ========================
    n_episodes = 1
    for i in range(n_episodes):
        print(f"Episode {i} starts")
        # if load a state
        if args.load_state:
            # state = th.tensor(load_f["data/demo_0/state"][700])
            # og.sim.load_state(state, serialized=True)
            breakpoint()
            for _ in range(50): 
                og.sim.step()

        prev_is_grasping = False
        while True:
            # Use KeyboardRobotController for default keybindings (as printed in terminal)
            # Handle different return types from get_teleop_action()
            ret = action_generator.get_teleop_action()
            if isinstance(ret, tuple) and len(ret) == 2:
                action, keypress_str = ret
            else:
                action = ret
                keypress_str = None
            
            # Skip robot action if camera movement keys are pressed (let camera mover handle them)
            # Camera mover handles these keys via its keyboard event subscription
            camera_keys = {"W", "A", "S", "D", "G"}
            if keypress_str and keypress_str in camera_keys:
                # Just step the environment without robot action to allow camera movement to be visible
                env.step(th.zeros(robot.action_dim))
                continue
            
            if keypress_str == "TAB":
                # Print viewer camera position and orientation, then breakpoint
                cam_pos, cam_orn = og.sim.viewer_camera.get_position_orientation()
                print("\n" + "=" * 60)
                print("VIEWER CAMERA POSE (TAB pressed)")
                print("=" * 60)
                print(f"Position (x, y, z): {cam_pos.tolist()}")
                print(f"Orientation (quaternion w, x, y, z): {cam_orn.tolist()}")
                print("=" * 60 + "\n")
                breakpoint()
            
            # Check if grasping state changed and update handles if so
            try:
                from omnigibson.utils.constants import IsGraspingState
                is_grasping = robot.is_grasping().value == IsGraspingState.TRUE
                if is_grasping != prev_is_grasping:
                    if og.sim.is_playing():
                        try:
                            og.sim.update_handles()
                        except Exception:
                            pass
                    prev_is_grasping = is_grasping
            except Exception:
                pass
            
            # Update handles before step to prevent articulation view errors during state dumping
            if og.sim.is_playing():
                try:
                    og.sim.update_handles()
                except Exception:
                    pass
            
            # Retry step with handle update if articulation view error occurs
            for attempt in range(3):
                try:
                    env.step(action)
                    break
                except AttributeError as e:
                    if "'NoneType' object has no attribute" in str(e) and attempt < 2:
                        if og.sim.is_playing():
                            try:
                                og.sim.update_handles()
                            except Exception:
                                pass
                    else:
                        raise
    print("Data saved")
    env.save_data()
    camera_mover.clear()
    og.shutdown()


if __name__ == "__main__":
    __main__()
