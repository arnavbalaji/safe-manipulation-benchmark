from omnigibson.utils.python_utils import Registerable
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.objects.usd_object import USDObject
from omnigibson.objects.controllable_object import ControllableObject
from omnigibson.objects.light_object import LightObject
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.robots.franka import FrankaPanda
from omnigibson.robots.tiago import Tiago
from safety_benchmark.params.test_params import PARAMS, DAMAGE_GENERATORS


class DamageableMixin:
    '''
    Mixin adding damage functionality to the OmniGibson object classes'
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store params dict, set empty damage_generators list
        self.params = kwargs.get('params', {})
        self.damage_generators = []

        # Set thresholds
        thresholds = self.params.get("health_thresholds", [90.0, 60.0, 30.0])
        self.minor_threshold, self.major_threshold, self.critical_threshold = thresholds

    def _initialize_health(self):
        # Initialize link healths to the maximum
        self.link_healths = {link_name: 100.0 for link_name in self.links.keys()}
        self.damage_statuses = {link_name: "none" for link_name in self.links.keys()}

    def _initialize_damage_generators(self):
        # Set damage generators once sim is playing
        for generator_name in self.params.get("damage_generators", []):
            gen_cls = DAMAGE_GENERATORS[generator_name] # Getting correct damage generator
            self.damage_generators.append(gen_cls(self, **self.params[generator_name]))

    def reset_damage_generators(self):
        # Reset tracking in all damage generators (for env.reset())
        for generator in self.damage_generators:
            if hasattr(generator, 'reset_tracking'):
                generator.reset_tracking()

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
        # Updates health based on the damage generators
        for generator in self.damage_generators:
            link_damages = generator.generate_damage()
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
        # Get impact history from damage generators
        impact_history = {}
        for generator in self.damage_generators:
            if hasattr(generator, 'get_impact_history'):
                history = generator.get_impact_history(link_name)
                if link_name is None:
                    impact_history.update(history)
                else:
                    impact_history[link_name] = history
        return impact_history


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
    pass

class DamageableTiago(DamageableMixin, Tiago):
    pass