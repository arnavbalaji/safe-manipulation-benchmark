from omnigibson.utils.python_utils import Registerable
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.objects.usd_object import USDObject
from omnigibson.objects.controllable_object import ControllableObject
from omnigibson.objects.light_object import LightObject
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.robots.franka import FrankaPanda
from omnigibson.robots.franka_mounted import FrankaMounted
from omnigibson.robots.tiago import Tiago
from omnigibson.robots.r1pro import R1Pro
from safety_benchmark.params.test_params import PARAMS, DAMAGE_EVALUATORS


class DamageableMixin:
    '''
    Mixin adding damage functionality to the OmniGibson object classes'
    '''
    def __init__(self, *args, **kwargs):
        # Filter out usd_path if it exists, since robots construct their own path
        if 'usd_path' in kwargs:
            del kwargs['usd_path']
        
        super().__init__(*args, **kwargs)
        # Store params dict, set empty damage_evaluators list
        self.track_damage = False
        self.params = kwargs.get('params', {})
        self.damage_evaluators = []
        self.damageable_links = []

    def _initialize_health(self):
        # Initialize link healths to the maximum
        self.link_healths = {link_name: 100.0 for link_name in self.links.keys()}
        self.damage_info = {}

    def _initialize_damage_evaluators(self):
        for evaluator_name in self.params.get("damage_evaluators", []):
            eval_cls = DAMAGE_EVALUATORS[evaluator_name] # Getting correct damage evaluator
            self.damage_evaluators.append(eval_cls(self, **self.params[evaluator_name]))

    def reset_damage_evaluators(self):
        # Reset tracking in all damage evaluators (for env.reset())
        for evaluator in self.damage_evaluators:
            if hasattr(evaluator, 'reset_tracking'):
                evaluator.reset_tracking()

    def set_track_damage(self, track_damage):
        self.track_damage = track_damage
        
    def set_params(self, params):
        self.params = params

    def set_damageable_links(self, links=None):
        if links is None:
            self.damageable_links = self.links.keys()
        else:
            self.damageable_links = links

    @property
    def health(self):
        # Returns minimum health value across links
        return min(self.link_healths.values())

        # # Returning average health value across links
        # return sum(self.link_healths.values()) / len(self.link_healths)

    def update_health(self):
        # Updates health based on the damage evaluators
        self.damage_info = {}
        for evaluator in self.damage_evaluators:
            # print(f"Updating health for {self.name} with {evaluator.name}")
            link_damages = evaluator.generate_damage()
            for link_name, damage in link_damages.items():
                
                # For logging information related to damages to each link
                if link_name not in self.damage_info:
                    self.damage_info[link_name] = {}
               
                # Update link healths
                new_health = max(0.0, self.link_healths[link_name] - damage)
                self.link_healths[link_name] = new_health

                # # For debugging
                # if self.name == "coffee_cup_1" and link_name == "base_link":
                #     print("new_health: ", new_health)
                #     # if new_health == 0.0:
                #     #     breakpoint()

                # Update the mechanical damage information
                if evaluator.name == "mechanical":
                    if "mechanical" not in self.damage_info[link_name]:
                        self.damage_info[link_name]["mechanical"] = {}
                    self.damage_info[link_name]["mechanical"]["impact_forces"] = evaluator.impact_forces[link_name][-1]
                    self.damage_info[link_name]["mechanical"]["unfiltered_raw_sim_forces"] = evaluator.unfiltered_raw_sim_forces[link_name][-1]
                    self.damage_info[link_name]["mechanical"]["filtered_raw_sim_forces"] = evaluator.filtered_raw_sim_forces[link_name][-1]
                    self.damage_info[link_name]["mechanical"]["unfiltered_qs_forces"] = evaluator.unfiltered_qs_forces[link_name][-1]
                    self.damage_info[link_name]["mechanical"]["filtered_qs_forces"] = evaluator.filtered_qs_forces[link_name][-1]
                    self.damage_info[link_name]["mechanical"]["contacts"] = evaluator.contacts_by_link[link_name][-1]
                    self.damage_info[link_name]["mechanical"]["damage"] = damage

                # Update the electrical damage information (water particle contacts)
                if evaluator.name == "electrical":
                    if "electrical" not in self.damage_info[link_name]:
                        self.damage_info[link_name]["electrical"] = {}
                    # Get contact summary for particle counts
                    contact_summary = evaluator.get_contact_summary()
                    link_details = contact_summary.get("link_details", {})
                    particle_count = link_details.get(link_name, {}).get("particle_count", 0)
                    self.damage_info[link_name]["electrical"]["particle_count"] = particle_count
                    self.damage_info[link_name]["electrical"]["damage"] = damage

                # Update the thermal damage information (temperature-based damage)
                if evaluator.name == "thermal":
                    if "thermal" not in self.damage_info[link_name]:
                        self.damage_info[link_name]["thermal"] = {}
                    # Get current temperature from the evaluator
                    temperature = evaluator.get_temperature() if hasattr(evaluator, "get_temperature") else None
                    if temperature is None and hasattr(evaluator, "current_temperature"):
                        temperature = evaluator.current_temperature
                    self.damage_info[link_name]["thermal"]["temperature"] = temperature
                    self.damage_info[link_name]["thermal"]["damage"] = damage
                    # Store thresholds for reference
                    if hasattr(evaluator, "heating_threshold"):
                        self.damage_info[link_name]["thermal"]["heating_threshold"] = evaluator.heating_threshold
                    if hasattr(evaluator, "cooling_threshold"):
                        self.damage_info[link_name]["thermal"]["cooling_threshold"] = evaluator.cooling_threshold



'''Damageable Object subclasses'''
class DamageableDatasetObject(DamageableMixin, DatasetObject):
    pass

class DamageablePrimitiveObject(DamageableMixin, PrimitiveObject):
    pass

class DamageableUSDObject(DamageableMixin, USDObject):
    pass

class DamageableControllableObject(DamageableMixin, ControllableObject):
    pass

class DamageableLightObject(DamageableMixin, LightObject):
    pass

class DamageableStatefulObject(DamageableMixin, StatefulObject):
    pass

class DamageableFrankaPanda(DamageableMixin, FrankaPanda):
    @property
    def usd_path(self):
        # Override to use the original FrankaPanda model path, not the damageable version
        import os
        from omnigibson.utils.asset_utils import get_dataset_path
        return os.path.join(get_dataset_path("omnigibson-robot-assets"), "models/franka/franka_panda/usd/franka_panda.usda")


class DamageableFrankaMounted(DamageableMixin, FrankaMounted):
    @property
    def usd_path(self):
        # Override to use the original FrankaMounted model path
        import os
        from omnigibson.utils.asset_utils import get_dataset_path
        return os.path.join(get_dataset_path("omnigibson-robot-assets"), "models/franka/franka_mounted/usd/franka_mounted.usda")


class DamageableTiago(DamageableMixin, Tiago):
    # def __init__(self, *args, **kwargs):
    #     # Store the original name before super().__init__ modifies it
    #     name = kwargs.get("name", "robot")
    #     breakpoint()
    #     # Explicitly set the prim path to match the original Tiago format
    #     kwargs["relative_prim_path"] = f"/controllable_tiago_{name}"
    #     super().__init__(*args, **kwargs)

    @property
    def usd_path(self):
        # Override to use the original Tiago model path, not the damageable version
        model = "tiago"  # Use the original model name, not the class name
        import os
        from omnigibson.macros import gm
        # For older OG
        # return os.path.join(gm.ASSET_PATH, f"models/{model}/usd/{model}.usda")
        return os.path.join(gm.DATA_PATH, f"omnigibson-robot-assets/models/{model}/usd/{model}.usda")

class DamageableR1Pro(DamageableMixin, R1Pro):
    @property
    def usd_path(self):
        # Override to use the original Tiago model path, not the damageable version
        model = "r1pro"  # Use the original model name, not the class name
        import os
        from omnigibson.macros import gm
        # For older OG
        # return os.path.join(gm.ASSET_PATH, f"models/{model}/usd/{model}.usda")
        return os.path.join(gm.DATA_PATH, f"omnigibson-robot-assets/models/{model}/usd/{model}.usda")

    @property
    def model_name(self):
        """
        Returns:
            str: name of this robot model. usually corresponds to the class name of a given robot model
        """
        return "R1Pro"