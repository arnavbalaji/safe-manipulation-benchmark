from omnigibson.envs.env_base import Environment
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.objects.usd_object import USDObject
from omnigibson.objects.controllable_object import ControllableObject
from omnigibson.objects.light_object import LightObject
from omnigibson.objects.stateful_object import StatefulObject
from safety_benchmark.damageable_mixin import (
    DamageableDatasetObject,
    DamageablePrimitiveObject,
    DamageableUSDObject,
    DamageableControllableObject,
    DamageableLightObject,
    DamageableStatefulObject,
)
from safety_benchmark.params.test_params import PARAMS
import omnigibson as og
import inspect

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
}

def create_damageable_object_from_config(cls_name, cls_registry, cfg, cls_type_descriptor):
    """
    Helper function to create a damageable object with str type @cls_name, which should be a valid entry in @cls_registry,
    using kwargs in dictionary form @cfg to pass to the constructor, with @cls_type_name specified for debugging.
    This is similar to create_class_from_registry_and_config but ensures objects are created as damageable versions.

    Args:
        cls_name (str): Name of the class to create. This should correspond to the actual class type, in string form
        cls_registry (dict): Class registry. This should map string names of valid classes to create to the
            actual class type itself
        cfg (dict): Any keyword arguments to pass to the class constructor
        cls_type_descriptor (str): Description of the class type being created. This can be any string and is used
            solely for debugging purposes

    Returns:
        any: Created damageable object instance
    """
    # Make sure the requested class type is valid
    assert cls_name in cls_registry, f"Invalid {cls_type_descriptor} type received! Valid options are: {cls_registry.keys()}, got: {cls_name}"

    # Get the base class
    base_cls = cls_registry[cls_name]
    
    # Get the damageable version of the class if it exists
    damageable_cls = DAMAGEABLE_OBJECT_MAPPING.get(base_cls.__name__)
    
    # If no damageable version exists, use the base class
    cls_to_use = damageable_cls if damageable_cls is not None else base_cls
    
    # Extract the kwargs relevant for the specific class
    cls_kwargs = {}
    # Get signatures for both the damageable class and base class
    damageable_sig = inspect.signature(damageable_cls.__init__) if damageable_cls is not None else None
    base_sig = inspect.signature(base_cls.__init__)
    
    # Extract kwargs for both classes
    for k in base_sig.parameters.keys():
        if k != "self" and k in cfg:
            cls_kwargs[k] = cfg[k]
    
    # If this is a damageable class, ensure params are passed correctly
    if damageable_cls is not None:
        # Ensure damage_params exists
        if "damage_params" not in cfg:
            obj_name = cfg.get("name", "default")
            if obj_name in PARAMS:
                cls_kwargs["params"] = PARAMS[obj_name]
            else:
                cls_kwargs["params"] = PARAMS["default"]
        else:
            cls_kwargs["params"] = cfg["damage_params"]
    
    # Create the class
    return cls_to_use(**cls_kwargs)

class DamageableEnvironment(Environment):
    """
    A version of the OmniGibson environment that loads objects as damageable versions.
    """

    def __init__(self, configs, in_vec_env=False):
        """
        Initialize the damageable environment.

        Args:
            configs (str or dict or list of str or dict): config_file path(s) or raw config dictionaries.
                If multiple configs are specified, they will be merged sequentially in the order specified.
            in_vec_env (bool): Whether this environment is part of a vectorized environment
        """
        super().__init__(configs, in_vec_env)
        self.damage_generators_initialized = False

    def load(self):
        """
        Load the scene and robot specified in the config file.
        """
        # This environment is not loaded
        self._loaded = False

        # Load config variables
        self._load_variables()

        # Load the scene, robots, and task
        self._load_scene()
        self._load_robots()
        self._load_objects()
        self._load_task()
        self._load_external_sensors()

        # Initialize health for all damageable objects
        for obj in self.scene.objects:
            if hasattr(obj, "_initialize_health"):
                obj._initialize_health()

    def _load_objects(self):
        """
        Load any additional custom objects into the scene as damageable versions.
        """
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
        """
        Step the environment and update health of all damageable objects.
        
        Args:
            action: The action to take in the environment
            
        Returns:
            tuple: (obs, reward, terminated, truncated, info) from the environment step
        """
        # Step the environment
        if not self.damage_generators_initialized:
            for obj in self.scene.objects:
                breakpoint()
                if hasattr(obj, "_initialize_damage_generators"):
                    obj._initialize_damage_generators()
            self.damage_generators_initialized = True

        obs, reward, terminated, truncated, info = super().step(action)
        
        
        # Update health of all damageable objects
        for obj in self.scene.objects:
            if hasattr(obj, "update_health"):
                obj.update_health()
                
        return obs, reward, terminated, truncated, info 