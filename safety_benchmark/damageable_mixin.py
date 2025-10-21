from omnigibson.utils.python_utils import Registerable
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.objects.usd_object import USDObject
from omnigibson.objects.controllable_object import ControllableObject
from omnigibson.objects.light_object import LightObject
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.robots.franka import FrankaPanda
from omnigibson.robots.tiago import Tiago
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
        self.params = kwargs.get('params', {})
        self.damage_evaluators = []

        # Set thresholds
        thresholds = self.params.get("health_thresholds", [90.0, 60.0, 30.0])
        self.minor_threshold, self.major_threshold, self.critical_threshold = thresholds

    def _initialize_health(self):
        # Initialize link healths to the maximum
        self.link_healths = {link_name: 100.0 for link_name in self.links.keys()}
        self.damage_statuses = {link_name: "none" for link_name in self.links.keys()}
        self.damage_info = {}
        self.previous_health = 100.0

    def _initialize_damage_evaluators(self):
        # Set damage evaluators once sim is playing
        for evaluator_name in self.params.get("damage_evaluators", []):
            eval_cls = DAMAGE_EVALUATORS[evaluator_name] # Getting correct damage evaluator
            self.damage_evaluators.append(eval_cls(self, **self.params[evaluator_name]))

    def reset_damage_evaluators(self):
        # Reset tracking in all damage evaluators (for env.reset())
        for evaluator in self.damage_evaluators:
            if hasattr(evaluator, 'reset_tracking'):
                evaluator.reset_tracking()

    @property
    def health(self):
        # TODO: Change back to average health value across links when things are working
        # Returns minimum health value across links
        return min(self.link_healths.values())

        # # Returning average health value across links
        # return sum(self.link_healths.values()) / len(self.link_healths)
        

    @property
    def damage_status(self):
        # Returning damage status of average health
        h = self.health
        if h < self.critical_threshold:
            return "critical"
        elif h < self.major_threshold:
            return "major"
        elif h < self.minor_threshold:
            return "minor"
        elif h < 100.0:
            return "negligible"
        else:
            return "none"

    def update_health(self):
        # Updates health based on the damage evaluators
        self.damage_info = {}
        for evaluator in self.damage_evaluators:
            link_damages = evaluator.generate_damage()
            self.damage_info[evaluator.name] = link_damages
            for link_name, damage in link_damages.items():
                # Update link healths
                new_health = max(0.0, self.link_healths[link_name] - damage)
                self.link_healths[link_name] = new_health

                # Calculate and update individual link status
                if new_health < self.critical_threshold:
                    status = "critical"
                elif new_health < self.major_threshold:
                    status = "major"
                elif new_health < self.minor_threshold:
                    status = "minor"
                elif new_health < 100.0:
                    status = "negligible"
                else:
                    status = "none"
                self.damage_statuses[link_name] = status

    def get_impact_history(self, link_name: str = None):
        # Get impact history from damage evaluators
        impact_history = {}
        for evaluator in self.damage_evaluators:
            if hasattr(evaluator, 'get_impact_history'):
                history = evaluator.get_impact_history(link_name)
                if link_name is None:
                    impact_history.update(history)
                else:
                    impact_history[link_name] = history
        return impact_history

    def get_obs_dict(self):
        obs_dict = {}
        obs_dict["health"] = self.health
        obs_dict["damage_status"] = self.damage_status
        obs_dict["damage_info"] = self.damage_info
        return obs_dict


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
        from omnigibson.macros import gm
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_panda/usd/franka_panda.usda")

class DamageableTiago(DamageableMixin, Tiago):
    @property
    def usd_path(self):
        # Override to use the original Tiago model path, not the damageable version
        model = "tiago"  # Use the original model name, not the class name
        import os
        from omnigibson.macros import gm
        return os.path.join(gm.ASSET_PATH, f"models/{model}/usd/{model}.usda")