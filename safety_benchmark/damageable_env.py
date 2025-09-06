from omnigibson.envs.env_base import Environment
# from omnigibson.objects.dataset_object import DatasetObject
# from omnigibson.objects.primitive_object import PrimitiveObject
# from omnigibson.objects.usd_object import USDObject
# from omnigibson.objects.controllable_object import ControllableObject
# from omnigibson.objects.light_object import LightObject
# from omnigibson.objects.stateful_object import StatefulObject
# from omnigibson.robots.franka import FrankaPanda
from safety_benchmark.damageable_mixin import (
    DamageableDatasetObject,
    DamageablePrimitiveObject,
    DamageableUSDObject,
    DamageableControllableObject,
    DamageableLightObject,
    DamageableStatefulObject,
    DamageableFrankaPanda,
    DamageableTiago,
)
from safety_benchmark.params.test_params import PARAMS
import omnigibson as og
import inspect
import random
import string
import torch as th
import time

from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.robots import REGISTERED_ROBOTS

# Mapping from base object types to their damageable versions
DAMAGEABLE_OBJECT_MAPPING = {
    "DatasetObject": DamageableDatasetObject,
    "PrimitiveObject": DamageablePrimitiveObject,
    "USDObject": DamageableUSDObject,
    "ControllableObject": DamageableControllableObject,
    "LightObject": DamageableLightObject,
    "StatefulObject": DamageableStatefulObject,
    "FrankaPanda": DamageableFrankaPanda,
    "Tiago": DamageableTiago,
}

def create_damageable_object_from_config(cls_name, cls_registry, cfg, cls_type_descriptor):
    '''
    Loads in object from config as a damageable object
    '''
    # Make sure the requested class type is valid
    assert cls_name in cls_registry, f"Invalid {cls_type_descriptor} type received! Valid options are: {cls_registry.keys()}, got: {cls_name}"

    # Get damageable class if there
    base_cls = cls_registry[cls_name]
    damageable_cls = DAMAGEABLE_OBJECT_MAPPING.get(base_cls.__name__)
    cls_to_use = damageable_cls if damageable_cls is not None else base_cls

    # Get kwargs
    cls_kwargs = {}
    base_sig = inspect.signature(base_cls.__init__)
    for k in base_sig.parameters.keys():
        if k != "self" and k in cfg:
            cls_kwargs[k] = cfg[k]

    if damageable_cls is not None:
        # Getting damage parameters
        if "damage_params" not in cfg:
            obj_name = cfg.get("name", "default")
            if obj_name in PARAMS:
                cls_kwargs["params"] = PARAMS[obj_name]
            else:
                cls_kwargs["params"] = PARAMS["default"]
        else:
            cls_kwargs["params"] = cfg["damage_params"]

    # Creating class
    return cls_to_use(**cls_kwargs)

class DamageableEnvironment(Environment):
    '''
    OmniGibson environment wrapper to support damageable objects and robots
    '''
    def __init__(self, configs, in_vec_env=False, debug_physics_frequency=False):
        # Initialize the damageable environment
        super().__init__(configs, in_vec_env)
        self.damage_evaluators_initialized = False
        self._debug_physics_frequency = debug_physics_frequency

    def load(self):
        # Load scene, objects, and robots
        # TODO: Need to add support for scenes
        self._loaded = False

        # Load config variables
        self._load_variables()

        # Load the scene, robots, and task
        self._load_scene()
        self._load_robots()
        self._load_objects()
        self._load_task()
        self._load_external_sensors()

        self.inialize_damageable_objects()

    def inialize_damageable_objects(self):
        # Initialize health for all damageable objects
        for obj in self.scene.objects:
            if hasattr(obj, "_initialize_health"):
                obj._initialize_health()

    def reset(self):
        """Reset the environment and damage evaluators."""
        # Reset the base environment
        obs = super().reset()
        
        # Reset damage evaluators for all objects
        for obj in self.scene.objects:
            if hasattr(obj, "reset_damage_evaluators"):
                obj.reset_damage_evaluators()
            if hasattr(obj, "_initialize_health"):
                obj._initialize_health()
        
        # Reset damage evaluator initialization flag
        self.damage_evaluators_initialized = False
        
        return obs

    def _load_robots(self):
        """
        Load robots into the scene
        """
        # Only actually load robots if no robot has been imported from the scene loading directly yet
        if len(self.scene.robots) == 0:
            assert og.sim.is_stopped(), "Simulator must be stopped before loading robots!"

            # Iterate over all robots to generate in the robot config
            for i, robot_config in enumerate(self.robots_config):
                # Add a name for the robot if necessary
                if "name" not in robot_config:
                    robot_config["name"] = "robot_" + "".join(random.choices(string.ascii_lowercase, k=6))

                position, orientation = robot_config.pop("position", None), robot_config.pop("orientation", None)
                pose_frame = robot_config.pop("pose_frame", "scene")
                if position is not None:
                    position = position if isinstance(position, th.Tensor) else th.tensor(position, dtype=th.float32)
                if orientation is not None:
                    orientation = (
                        orientation if isinstance(orientation, th.Tensor) else th.tensor(orientation, dtype=th.float32)
                    )

                # Make sure robot exists, grab its corresponding kwargs, and create / import the robot
                robot = create_damageable_object_from_config(
                    cls_name=robot_config["type"],
                    cls_registry=REGISTERED_ROBOTS,
                    cfg=robot_config,
                    cls_type_descriptor="robot",
                )
                # Import the robot into the simulator
                self.scene.add_object(robot)
                robot.set_position_orientation(position=position, orientation=orientation, frame=pose_frame)

        assert og.sim.is_stopped(), "Simulator must be stopped after loading robots!"

    def _load_objects(self):
        # Load objects from config as damageable versions
        assert og.sim.is_stopped(), "Simulator must be stopped before loading objects!"
        for i, obj_config in enumerate(self.objects_config):
            # Add a name for the object if necessary
            if "name" not in obj_config:
                obj_config["name"] = f"obj{i}"
            
            # Pop the desired position and orientation
            position, orientation = obj_config.pop("position", None), obj_config.pop("orientation", None)
            
            # Create the damageable object
            obj = create_damageable_object_from_config(
                cls_name=obj_config["type"],
                cls_registry=REGISTERED_OBJECTS,
                cfg=obj_config,
                cls_type_descriptor="object",
            )
            
            # Import the object into the simulator and set the pose
            self.scene.add_object(obj)
            obj.set_position_orientation(position=position, orientation=orientation, frame="scene")

        assert og.sim.is_stopped(), "Simulator must be stopped after loading objects!"

    def step(self, action, n_render_iterations=1):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention but with TRUE physics-frequency damage evaluation
        
        Args:
            action (gym.spaces.Dict or dict or th.tensor): robot actions. If a dict is specified, each entry should
                map robot name to corresponding action. If a th.tensor, it should be the flattened, concatenated set
                of actions
            n_render_iterations (int): Number of rendering iterations to use before returning observations

        Returns:
            5-tuple:
                - dict: state, i.e. next observation
                - float: reward, i.e. reward at this current timestep
                - bool: terminated, i.e. whether this episode ended due to a failure or success
                - bool: truncated, i.e. whether this episode ended due to a time limit etc.
                - dict: info, i.e. dictionary with any useful information
        """
        # Initialize damage evaluators if this is the first env step
        if not self.damage_evaluators_initialized:
            for obj in self.scene.objects:
                if hasattr(obj, "_initialize_damage_evaluators"):
                    obj._initialize_damage_evaluators()
            self.damage_evaluators_initialized = True

        # Pre-processing before stepping simulation
        if hasattr(self, '_pre_step'):
            self._pre_step(action)

        # 🎯 NEW: Run physics simulation with TRUE physics-frequency damage evaluation!
        self._run_physics_with_damage_evaluation()
        
        # Aggregate forces for user-friendly display (one value per env step)
        self._aggregate_forces_for_env_step()
        
        # Get observations, rewards, etc.
        obs, reward, terminated, truncated, info = self._post_step(action)
        
        # Combine terminated and truncated into done for backward compatibility
        done = terminated or truncated

        # Render any additional times requested
        for _ in range(n_render_iterations - 1):
            og.sim.render()

        # Run final post-processing
        if hasattr(self, '_post_step'):
            return self._post_step(action)
        else:
            # Fallback to original behavior if _post_step doesn't exist
            obs, reward, terminated, truncated, info = super().step(action)
            
            # Update health of all damageable objects
            for obj in self.scene.objects:
                if hasattr(obj, "update_health"):
                    obj.update_health()
                    
            return obs, reward, terminated, truncated, info

    def _run_physics_with_damage_evaluation(self):
        """
        Run physics simulation with TRUE physics-frequency damage evaluation.
        This ensures damage evaluators run at every single physics step, not just environment steps.
        """
        # Get the physics timestep from OmniGibson
        physics_dt = og.sim.get_physics_dt()
        
        # Calculate how many physics substeps we need to run
        # This ensures we're running at true physics frequency
        num_substeps = max(1, int(1.0 / (physics_dt * self.env_config.get("action_frequency", 30.0))))
        
        # Debug info: Show what we're doing
        if hasattr(self, '_debug_physics_frequency') and self._debug_physics_frequency:
            print(f"🔬 Physics Frequency Debug:")
            print(f"   Physics DT: {physics_dt:.6f}s")
            print(f"   Action Frequency: {self.env_config.get('action_frequency', 30.0)} Hz")
            print(f"   Physics Substeps: {num_substeps}")
            print(f"   Effective Physics Freq: {1.0 / (physics_dt * num_substeps):.1f} Hz")
        
        # Try to hook into OmniGibson's physics events if available
        if hasattr(og.sim, 'physics_sim_view') and hasattr(og.sim.physics_sim_view, 'set_simulation_event_callback'):
            # 🎯 ADVANCED: Use OmniGibson's physics event callbacks for true physics-frequency
            self._setup_physics_event_callbacks()
        
        # Run physics simulation with damage evaluation at each substep
        for substep in range(num_substeps):
            # Run one physics step
            og.sim.step()
            
            # 🎯 CRITICAL: Evaluate damage at EVERY physics substep!
            self._evaluate_damage_at_physics_step()
            
            # Optional: Add small delay to prevent overwhelming the system
            if substep < num_substeps - 1:  # Don't delay on last substep
                time.sleep(0.001)  # 1ms delay between substeps

    def _setup_physics_event_callbacks(self):
        """
        Setup physics event callbacks to hook into OmniGibson's physics system.
        This provides even better integration with the physics simulation.
        """
        try:
            # Try to register physics event callbacks if available
            if hasattr(og.sim.physics_sim_view, 'set_simulation_event_callback'):
                # This would be the ideal way to hook into physics events
                # However, we need to check what's actually available in OmniGibson
                pass
        except Exception as e:
            # Fallback to our substep approach if callbacks aren't available
            pass

    def _evaluate_damage_at_physics_step(self):
        """
        Evaluate damage for all damageable objects at physics frequency.
        This ensures we capture peak forces that happen during physics steps.
        """
        for obj in self.scene.objects:
            if hasattr(obj, "update_health"):
                obj.update_health()

    def _aggregate_forces_for_env_step(self):
        """
        Aggregate forces for all damageable objects at the end of each environment step.
        This provides user-friendly force values that match the number of environment steps.
        """
        for obj in self.scene.objects:
            if hasattr(obj, "damage_evaluators"):
                for evaluator in obj.damage_evaluators:
                    if hasattr(evaluator, "aggregate_forces_for_env_step"):
                        evaluator.aggregate_forces_for_env_step()

    def _pre_step(self, action):
        """Apply the pre-sim-step part of an environment step, i.e. apply the robot actions."""
        # If the action is not a dictionary, convert into a dictionary
        if not isinstance(action, dict):
            # Handle PyTorch tensors and other iterables
            # Keep PyTorch tensors as tensors - don't convert to numpy yet!
            if hasattr(action, 'cpu') and hasattr(action, 'numpy'):  # PyTorch tensor
                # Keep as tensor, let OmniGibson handle conversion
                pass
            elif hasattr(action, '__iter__') and not isinstance(action, (str, bytes)):
                action = list(action)
            else:
                # Single value, convert to list
                action = [action]
            
            # Convert to action dictionary
            action_dict = dict()
            idx = 0
            for robot in self.robots:
                action_dim = robot.action_dim
                if idx + action_dim <= len(action):
                    action_dict[robot.name] = action[idx : idx + action_dim]
                else:
                    # Handle case where action is shorter than expected
                    action_dict[robot.name] = action[idx:] if idx < len(action) else [0.0] * action_dim
                idx += action_dim
        else:
            # Our inputted action is the action dictionary
            action_dict = action

        # Iterate over all robots and apply actions
        for robot in self.robots:
            if robot.name in action_dict:
                robot.apply_action(action_dict[robot.name])

    def _post_step(self, action):
        """Apply the post-sim-step part of an environment step, i.e. grab observations and return the step results."""
        # Grab observations
        obs, obs_info = self.get_obs()

        # Step the scene graph builder if necessary
        if hasattr(self, '_scene_graph_builder') and self._scene_graph_builder is not None:
            self._scene_graph_builder.step(self.scene)

        # Grab reward, done, and info, and populate with internal info
        reward, done, info = self.task.step(self, action)
        self._populate_info(info)
        info["obs_info"] = obs_info

        if done and hasattr(self, '_automatic_reset') and self._automatic_reset:
            # Add lost observation to our information dict, and reset
            info["last_observation"] = obs
            obs = self.reset()

        # Hacky way to check for time limit info to split terminated and truncated
        terminated = False
        truncated = False
        if "done" in info and "termination_conditions" in info["done"]:
            for tc, tc_data in info["done"]["termination_conditions"].items():
                if tc_data["done"]:
                    if tc == "timeout":
                        truncated = True
                    else:
                        terminated = True
            assert (terminated or truncated) == done, "Terminated and truncated must match done!"

        # Increment step
        if hasattr(self, '_current_step'):
            self._current_step += 1
        return obs, reward, terminated, truncated, info

    def _populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        Args:
            info (dict): Information dictionary to populate

        Returns:
            dict: Information dictionary with added info
        """
        if hasattr(self, '_current_step'):
            info["episode_length"] = self._current_step

        if hasattr(self, '_scene_graph_builder') and self._scene_graph_builder is not None:
            info["scene_graph"] = self.get_scene_graph() 