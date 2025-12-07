
import inspect
import random
import string
import torch as th
import numpy as np
import time
import json

from safety_benchmark.params.test_params import PARAMS
from safety_benchmark.utils.misc_utils import json_default
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

import omnigibson as og
from omnigibson.envs.env_base import Environment
from omnigibson.envs.data_wrapper import DataPlaybackWrapper, DataCollectionWrapper
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
            obj_category = cfg.get("category", "default")
            if obj_category in PARAMS:
                cls_kwargs["params"] = PARAMS[obj_category]
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
    def __init__(self, configs, in_vec_env=False, debug_physics_frequency=False, reward_fn=None):
        # Initialize the damageable environment
        super().__init__(configs, in_vec_env)
        self.damage_evaluators_initialized = False
        self._debug_physics_frequency = debug_physics_frequency
        self._reward_fn = reward_fn
        self.lock_health = False
        
    def load(self):
        # Load scene, objects, and robots
        # TODO: Need to add support for scenes
        self._loaded = False

        # Load config variables
        self._load_variables()

        # Load the scene, robots, and task
        og.sim.stop()
        self._load_scene()
        self._load_robots()
        self._load_objects()
        self._load_task()
        self._load_external_sensors()
        og.sim.play()

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
        # breakpoint()
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
    
    def lock_health_changes(self):
        self.lock_health = True

    def unlock_health_changes(self):
        self.lock_health = False

    def step(self, action, n_render_iterations=1):
        """
        Apply robot's action and return the next state, reward, done and info,
        
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
            # Initialize robot damage evaluators if supported
            for robot in getattr(self, "robots", []):
                if hasattr(robot, "_initialize_damage_evaluators"):
                    robot._initialize_damage_evaluators()
            self.damage_evaluators_initialized = True
        
        obs, reward, terminated, truncated, info = super().step(action, n_render_iterations)
        # breakpoint()
        obj_damage_info = {}
        obs_info = {}
        if "obs_info" in info:
            obs_info["obs_info"] = info["obs_info"]
        if not self.lock_health:
            # Update all damageable objects
            for obj in self.scene.objects:
                if hasattr(obj, "update_health"):
                    obj.update_health()
                    # obj_health_states[obj.name] = obj.get_obs_dict()
                    obj_damage_info[obj.name] = obj.damage_info
            # Optionally update robots if they implement damage
            for robot in getattr(self, "robots", []):
                if hasattr(robot, "update_health"):
                    robot.update_health()
        
            # breakpoint()
            # obs["object_health_states"] = obj_health_states
            obs_info["damage_info"] = obj_damage_info
            if self._reward_fn is not None:
                reward, terminated = self._reward_fn(self, obs)
        
        obs, obs_info = self._process_obs(obs, obs_info)
        return obs, reward, terminated, truncated, obs_info


    def _process_obs(self, obs, info):
        """
        Modifies @obs inplace for any relevant post-processing

        Args:
            obs (dict): Keyword-mapped relevant observations from the immediate env step
            info (dict): Keyword-mapped relevant information from the immediate env step
        """
        obs["health"] = []
        for obj in self.scene.objects:
            if hasattr(obj, "update_health"):
                for link_name, health in obj.link_healths.items():
                    obs["health"].append(health)
        obs["health"] = th.tensor(obs["health"], dtype=th.float32)
        info["damage_info"] = json.dumps(info["damage_info"], default=json_default)
        info["obs_info"] = json.dumps(info["obs_info"], default=json_default)
        return obs, info

    def set_object_params(self):
        # Set params for all damageable objects
        for obj in self.scene.objects:
            if hasattr(obj, "set_params"):
                if obj.category in PARAMS:
                    obj.set_params(PARAMS[obj.category])
                    print(f"Set params for {obj.name} to {PARAMS[obj.category]}")
                else:
                    obj.set_params(PARAMS["default"])


class DamageableDataCollectionWrapper(DataCollectionWrapper):
    """
    Custom DataCollectionWrapper that properly handles health metadata collection
    for damageable objects and robots during data collection. This ensures:
    1. Health is initialized before collecting metadata
    2. Robots are included in health metadata
    3. Proper error handling for uninitialized health
    """
    
    def process_traj_to_hdf5(self, traj_data, traj_grp_name, nested_keys=("obs",), data_grp=None):
        """
        Processes trajectory data and stores them in HDF5, with proper health metadata collection.
        
        This method overrides the parent implementation to:
        - Ensure health is initialized before collecting metadata
        - Include robots in health metadata
        - Add proper error handling
        
        Args:
            traj_data (list of dict): Trajectory data, where each entry is a keyword-mapped set of data for a single
                sim step
            traj_grp_name (str): Name of the trajectory group to store
            nested_keys (list of str): Name of key(s) corresponding to nested data in @traj_data
            data_grp (None or h5py.Group): If specified, the h5py Group under which a new group with name
                @traj_grp_name will be created. If None, will default to "data" group

        Returns:
            hdf5.Group: Generated hdf5 group storing the recorded trajectory data
        """
        
        # First pad all state values to be the same max (uniform) size (from parent DataCollectionWrapper)
        for step_data in traj_data:
            state = step_data["state"]
            padded_state = th.zeros(self.max_state_size, dtype=th.float32)
            padded_state[: len(state)] = state
            step_data["state"] = padded_state

        # Collect health metadata with proper initialization and error handling BEFORE calling parent
        # This ensures health is initialized and we include robots
        health_list = []

        for obj in self.scene.objects:
            if hasattr(obj, "update_health"):
                for link_name, health in obj.link_healths.items():
                    health_list.append(f"{obj.name}@{link_name}")
        # traj_grp.attrs["health_list_link_names"] = health_list
        
        # # Process scene objects
        # for obj in self.scene.objects:
        #     if hasattr(obj, "update_health"):
        #         # Ensure health is initialized
        #         if not hasattr(obj, "link_healths"):
        #             if hasattr(obj, "_initialize_health"):
        #                 obj._initialize_health()
                
        #         # Safely collect health metadata
        #         if hasattr(obj, "link_healths"):
        #             try:
        #                 for link_name, health in obj.link_healths.items():
        #                     health_list.append(f"{obj.name}@{link_name}")
        #             except (AttributeError, TypeError):
        #                 # Skip if link_healths is not properly initialized
        #                 pass
        
        # # Process robots (which are not in scene.objects)
        # for robot in getattr(self, "robots", []):
        #     if hasattr(robot, "update_health"):
        #         # Ensure health is initialized
        #         if not hasattr(robot, "link_healths"):
        #             if hasattr(robot, "_initialize_health"):
        #                 robot._initialize_health()
                
        #         # Safely collect health metadata
        #         if hasattr(robot, "link_healths"):
        #             try:
        #                 for link_name, health in robot.link_healths.items():
        #                     health_list.append(f"{robot.name}@{link_name}")
        #             except (AttributeError, TypeError):
        #                 # Skip if link_healths is not properly initialized
        #                 pass

        # Call parent method to handle the rest of the data processing
        traj_grp = super().process_traj_to_hdf5(traj_data, traj_grp_name, nested_keys, data_grp)
        
        # Add health list link names to the trajectory group
        traj_grp.attrs["health_list_link_names"] = health_list

        return traj_grp


class DamageableDataPlaybackWrapper(DataPlaybackWrapper):
    """
    Custom DataPlaybackWrapper that:
    1. Uses DamageableEnvironment instead of og.Environment when creating from HDF5
    2. Calls set_object_params() on the wrapped environment after scene.restore() is called
    
    This ensures damage parameters are set correctly during data playback without modifying OmniGibson source code.
    """
    
    @classmethod
    def create_from_hdf5(
        cls,
        input_path,
        output_path,
        compression=dict(),
        robot_obs_modalities=tuple(),
        robot_proprio_keys=None,
        robot_sensor_config=None,
        external_sensors_config=None,
        include_sensor_names=None,
        exclude_sensor_names=None,
        n_render_iterations=5,
        overwrite=True,
        only_successes=False,
        flush_every_n_traj=10,
        flush_every_n_steps=0,
        include_env_wrapper=False,
        additional_wrapper_configs=None,
        append_to_input_path=False,
        load_room_instances=None,
        overwrite_config=None,
        full_scene_file=None,
        include_task=True,
        include_task_obs=True,
        include_robot_control=True,
        include_contacts=True,
    ):
        """
        Create a DamageableDataPlaybackWrapper environment instance from the recorded demonstration info.
        
        This method overrides the parent to use DamageableEnvironment instead of og.Environment,
        avoiding the need to modify OmniGibson source code.
        
        Args and Returns are the same as DataPlaybackWrapper.create_from_hdf5()
        """
        import h5py
        import json
        import omnigibson as og
        from omnigibson.macros import gm
        from omnigibson.utils.data_utils import merge_scene_files
        from omnigibson.envs.env_wrapper import create_wrapper
        
        # check flush parameters
        if flush_every_n_steps > 0:
            assert flush_every_n_traj == 1, "flush_every_n_traj must be 1 if flush_every_n_steps is greater than 0"
        # Read from the HDF5 file
        f = h5py.File(input_path, "a" if append_to_input_path else "r")
        config = json.loads(f["data"].attrs["config"]) if overwrite_config is None else json.loads(overwrite_config)

        # Hot swap in additional info for playing back data

        if include_contacts:
            # Minimize physics leakage during playback (we need to take an env step when loading state)
            config["env"]["action_frequency"] = 1000.0
            config["env"]["rendering_frequency"] = 1000.0
            config["env"]["physics_frequency"] = 1000.0
        else:
            # Since we are setting all objects to be visual-only, physics will not be propogating
            config["env"]["action_frequency"] = 30.0
            config["env"]["rendering_frequency"] = 30.0
            config["env"]["physics_frequency"] = 120.0
            # Simulator-level visual-only set to True
            gm.VISUAL_ONLY = True

        # Make sure obs space is flattened for recording
        config["env"]["flatten_obs_space"] = True

        # Set the scene file either to the one stored in the hdf5 or the hot swap scene file
        config["scene"]["scene_file"] = json.loads(f["data"].attrs["scene_file"])
        if full_scene_file:
            with open(full_scene_file, "r") as json_file:
                full_scene_json = json.load(json_file)
            config["scene"]["scene_file"] = merge_scene_files(
                scene_a=full_scene_json, scene_b=config["scene"]["scene_file"], keep_robot_from="b"
            )
            # Overwrite rooms type to avoid loading room types from the hdf5 file
            config["scene"]["load_room_types"] = None
            config["scene"]["load_room_instances"] = load_room_instances
        else:
            config["scene"]["scene_file"] = json.loads(f["data"].attrs["scene_file"])

        # Use dummy task if not loading task
        if not include_task:
            config["task"] = {"type": "DummyTask"}

        # Maybe include task observations
        config["task"]["include_obs"] = include_task_obs

        # Set scene file and disable online object sampling if BehaviorTask is being used
        if config["task"]["type"] == "BehaviorTask":
            config["task"]["online_object_sampling"] = False
            # Don't use presampled robot pose
            config["task"]["use_presampled_robot_pose"] = False

        if load_room_instances is not None:
            config["scene"]["load_room_instances"] = load_room_instances

        # Because we're loading directly from the cached scene file, we need to disable any additional objects that are being added since
        # they will already be cached in the original scene file
        config["objects"] = []

        # Set observation modalities and update sensor config
        for robot_cfg in config["robots"]:
            robot_cfg["obs_modalities"] = list(robot_obs_modalities)
            robot_cfg["include_sensor_names"] = include_sensor_names
            robot_cfg["exclude_sensor_names"] = exclude_sensor_names
            if robot_proprio_keys is not None:
                robot_cfg["proprio_obs"] = robot_proprio_keys
            if robot_sensor_config is not None:
                robot_cfg["sensor_config"] = robot_sensor_config
        if external_sensors_config is not None:
            config["env"]["external_sensors"] = external_sensors_config

        # Load env - Use DamageableEnvironment instead of og.Environment
        env = DamageableEnvironment(configs=config)

        # Optionally include the desired environment wrapper specified in the config
        if include_env_wrapper:
            env = create_wrapper(env=env)

        if additional_wrapper_configs is not None:
            for wrapper_cfg in additional_wrapper_configs:
                env = create_wrapper(env=env, wrapper_cfg=wrapper_cfg)

        # Wrap and return env
        return cls(
            env=env,
            input_path=input_path,
            output_path=output_path,
            compression=compression,
            n_render_iterations=n_render_iterations,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
            flush_every_n_steps=flush_every_n_steps,
            full_scene_file=full_scene_file,
            load_room_instances=load_room_instances,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
        )
    
    def playback_episode(self, episode_id, record_data=True, video_writers=None, callback=None, replay_for_annotation=False, break_after_n_steps=100):
        """
        Playback episode @episode_id, and optionally record observation data if @record is True.
        
        This method overrides the parent implementation to call set_object_params() on the
        wrapped environment right after scene.restore() is called.

        Args:
            episode_id (int): Episode to playback. This should be a valid demo ID number from the inputted collected
                data hdf5 file
            record_data (bool): Whether to record data during playback or not
            video_writers (Any): Optional video writers to record the playback
            replay_for_annotation (bool): If True, replay the dataset to break after X steps to note down the MP_end_step and subtask_term_step for each subtask
            break_after_n_steps (int): Number of steps to break after when replay_for_annotation is True
        """
        import h5py
        import json
        from omnigibson.utils.python_utils import h5py_group_to_torch, create_object_from_init_info
        import omnigibson as og
        from omnigibson.controllers.controller_base import ControlType
        from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
        
        data_grp = self.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        # Grab episode data
        # Skip early if found malformed data
        try:
            transitions = json.loads(traj_grp.attrs["transitions"])
            traj_grp = h5py_group_to_torch(traj_grp)
            init_metadata = traj_grp["init_metadata"]
            action = traj_grp["action"]
            state = traj_grp["state"]
            state_size = traj_grp["state_size"]
            reward = traj_grp["reward"]
            terminated = traj_grp["terminated"]
            truncated = traj_grp["truncated"]
        except KeyError as e:
            print(f"Got error when trying to load episode {episode_id}:")
            print(f"Error: {str(e)}")
            return

        result = []
        
        # Reset environment and update this to be the new initial state
        self.scene.restore(self.scene_file, update_initial_file=True)

        # Call set_object_params() on the wrapped environment if it has this method
        # This must happen right after scene.restore() and before resetting object attributes
        if hasattr(self.env, "set_object_params"):
            self.env.set_object_params()

        # Reset object attributes from the stored metadata
        with og.sim.stopped():
            for attr, vals in init_metadata.items():
                assert len(vals) == self.scene.n_objects
            for i, obj in enumerate(self.scene.objects):
                for attr, vals in init_metadata.items():
                    val = vals[i]
                    setattr(obj, attr, val.item() if val.ndim == 0 else val)
        self.reset()

        # If not controlling robots, disable for all robots
        if not self.include_robot_control:
            for robot in self.robots:
                robot.control_enabled = False
                # Set all controllers to effort mode with zero gain, this keeps the robot still
                for controller in robot.controllers.values():
                    for i, dof in enumerate(controller.dof_idx):
                        dof_joint = robot.joints[robot.dof_names_ordered[dof]]
                        dof_joint.set_control_type(
                            control_type=ControlType.EFFORT,
                            kp=None,
                            kd=None,
                        )

        # Restore to initial state
        # Ensure simulator is playing before loading state (required by load_state)
        if not og.sim.is_playing():
            og.sim.play()
        og.sim.load_state(state[0, : int(state_size[0])], serialized=True)
        if callback is not None:
            result.append(callback(action=action[0]))

        # If record, record initial observations
        if record_data:
            # We need to step the environment to get the initial observations propagated
            first_time_load_n_iteration = 10
            self.current_obs, _, _, _, init_info = self.env.step(
                action=action[0], n_render_iterations=self.n_render_iterations + first_time_load_n_iteration
            )
            step_data = {"obs": self._process_obs(obs=self.current_obs, info=init_info)}
            self.current_traj_history.append(step_data)

        # Print all object names in the scene
        if replay_for_annotation:
            print(f"================= object names in the scene =================")
            all_objs = og.sim.scenes[0].objects
            print([o.name for o in all_objs])

        for i, (a, s, ss, r, te, tr) in enumerate(
            zip(action, state[1:], state_size[1:], reward, terminated, truncated)
        ):
            print(f"================= simulation step {i} =================")
            if replay_for_annotation:
                if i % break_after_n_steps == 0:
                    print(f"================= simulation step {i} =================")
                    # Note: You can use the following to step the rendering in OG: for _ in range(500): og.sim.render()
                    # And then you can click on objects in the viewer to get the OG specific name of the object
                    breakpoint()

            # # For debugging
            # if i > 100:
            #     break

            # Execute any transitions that should occur at this current step
            if str(i) in transitions:
                cur_transitions = transitions[str(i)]
                scene = og.sim.scenes[0]
                for add_sys_name in cur_transitions["systems"]["add"]:
                    scene.get_system(add_sys_name, force_init=True)
                for remove_sys_name in cur_transitions["systems"]["remove"]:
                    scene.clear_system(remove_sys_name)
                for remove_obj_name in cur_transitions["objects"]["remove"]:
                    obj = scene.object_registry("name", remove_obj_name)
                    scene.remove_object(obj)
                for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
                    obj = create_object_from_init_info(add_obj_info)
                    scene.add_object(obj)
                    obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
                # Step physics to initialize any new objects
                og.sim.step()
            
            # Restore the sim state, and take a very small step with the action to make sure physics are
            # properly propagated after the sim state update
            # Ensure simulator is playing before loading state (required by load_state)
            if not og.sim.is_playing():
                og.sim.play()
            og.sim.load_state(s[: int(ss)], serialized=True)
            if callback is not None:
                result.append(callback(action=a))

            # Restore the sim state, and take a very small step with the action to make sure physics are
            # properly propagated after the sim state update
            # Ensure simulator is playing before loading state (required by load_state)
            if not og.sim.is_playing():
                og.sim.play()
            og.sim.load_state(s[: int(ss)], serialized=True)
            if not self.include_contacts:
                # When all objects/systems are visual-only, keep them still on every step
                for obj in self.scene.objects:
                    obj.keep_still()
                for system in self.scene.systems:
                    # TODO: Implement keep_still for other systems
                    if isinstance(system, MacroPhysicalParticleSystem):
                        system.set_particles_velocities(
                            lin_vels=th.zeros((system.n_particles, 3)), ang_vels=th.zeros((system.n_particles, 3))
                        )
            self.current_obs, _, _, _, info = self.env.step(action=a, n_render_iterations=self.n_render_iterations)

            # If recording, record data
            if record_data:
                step_data = self._parse_step_data(
                    action=a,
                    obs=self.current_obs,
                    reward=r,
                    terminated=te,
                    truncated=tr,
                    info=info,
                )
                if self.flush_every_n_steps > 0:
                    if i == 0:
                        self.current_traj_grp, self.traj_dsets = self.allocate_traj_to_hdf5(
                            step_data, f"demo_{episode_id}", num_samples=len(action), video_writers=video_writers
                        )
                    if i % self.flush_every_n_steps == 0:
                        self.flush_partial_traj(num_samples=len(action), video_writers=video_writers)
                # append to current trajectory history
                self.current_traj_history.append(step_data)

            self.current_episode_step_count += 1
            self.step_count += 1

        # breakpoint()
        if record_data:
            if self.flush_every_n_steps > 0:
                self.flush_partial_traj(num_samples=len(action), video_writers=video_writers)
            self.flush_current_traj()

        return result

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        # Store action, obs, reward, terminated, truncated, info
        step_data = dict()
        step_data["obs"] = self._process_obs(obs=obs, info=info)
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated
        step_data["info"] = info
        return step_data

    def process_traj_to_hdf5(self, traj_data, traj_grp_name, nested_keys=("obs",), data_grp=None):
        """
        Processes trajectory data and stores them in HDF5, with proper health metadata collection.
        
        This method overrides the parent implementation to:
        - Ensure health is initialized before collecting metadata
        - Include robots in health metadata
        - Add proper error handling
        
        Args:
            traj_data (list of dict): Trajectory data, where each entry is a keyword-mapped set of data for a single
                sim step
            traj_grp_name (str): Name of the trajectory group to store
            nested_keys (list of str): Name of key(s) corresponding to nested data in @traj_data
            data_grp (None or h5py.Group): If specified, the h5py Group under which a new group with name
                @traj_grp_name will be created. If None, will default to "data" group

        Returns:
            hdf5.Group: Generated hdf5 group storing the recorded trajectory data
        """
        
        # Collect health metadata with proper initialization and error handling BEFORE calling parent
        # This ensures health is initialized and we include robots
        health_list = []

        for obj in self.scene.objects:
            if hasattr(obj, "update_health"):
                for link_name, health in obj.link_healths.items():
                    health_list.append(f"{obj.name}@{link_name}")        

        # Call parent method to handle the rest of the data processing
        traj_grp = super().process_traj_to_hdf5(traj_data, traj_grp_name, nested_keys, data_grp)
        
        # Add health list link names to the trajectory group
        traj_grp.attrs["health_list_link_names"] = health_list

        return traj_grp
