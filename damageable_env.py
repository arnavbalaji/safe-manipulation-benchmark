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
import torch

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
    def __init__(self, configs, in_vec_env=False):
        # Initialize the damageable environment
        super().__init__(configs, in_vec_env)
        self.damage_generators_initialized = False

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
        """Reset the environment and damage generators."""
        # Reset the base environment
        obs = super().reset()
        
        # Reset damage generators for all objects
        for obj in self.scene.objects:
            if hasattr(obj, "reset_damage_generators"):
                obj.reset_damage_generators()
            if hasattr(obj, "_initialize_health"):
                obj._initialize_health()
        
        # Reset damage generator initialization flag
        self.damage_generators_initialized = False
        
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
                    position = position if isinstance(position, torch.Tensor) else torch.tensor(position, dtype=torch.float32)
                if orientation is not None:
                    orientation = (
                        orientation if isinstance(orientation, torch.Tensor) else torch.tensor(orientation, dtype=torch.float32)
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

    def step(self, action):
        # Step function wrapper for damage generation
        if not self.damage_generators_initialized:
            # Initializing damage generators if this is the first env step
            for obj in self.scene.objects:
                if hasattr(obj, "_initialize_damage_generators"):
                    obj._initialize_damage_generators()
            self.damage_generators_initialized = True

        obs, reward, terminated, truncated, info = super().step(action) # Stepping the base env
        
        # Update health of all damageable objects
        for obj in self.scene.objects:
            if hasattr(obj, "update_health"):
                obj.update_health()
                
        return obs, reward, terminated, truncated, info 