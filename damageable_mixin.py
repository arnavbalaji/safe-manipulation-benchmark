from omnigibson.utils.python_utils import Registerable
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.objects.usd_object import USDObject
from omnigibson.objects.controllable_object import ControllableObject
from omnigibson.objects.light_object import LightObject
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.robots.franka import FrankaPanda
from safety_benchmark.params.test_params import PARAMS, DAMAGE_GENERATORS


class DamageableMixin:
    """Mixin adding damageable functionality to any object class."""
    def __init__(self, *args, **kwargs):
        # Initialize base class
        super().__init__(*args, **kwargs)
        # Store parameters and prepare empty generators list
        self.params = kwargs.get('params', {})
        self.damage_generators = []

    def _initialize_health(self):
        """Initialize health-related attributes after base object is initialized."""
        # Start each link at full health
        self.link_healths = {link_name: 100.0 for link_name in self.links.keys()}
        # Unpack thresholds
        thresholds = self.params.get("health_thresholds", [90.0, 60.0, 30.0])
        self.minor_threshold, self.major_threshold, self.critical_threshold = thresholds
        # Initial status for each link
        self.damage_statuses = {link_name: "none" for link_name in self.links.keys()}

    def _initialize_damage_generators(self):
        """Initialize damage generators after base object is initialized."""
        for generator_name in self.params.get("damage_generators", []):
            gen_cls = DAMAGE_GENERATORS[generator_name]
            self.damage_generators.append(gen_cls(self, **self.params[generator_name]))

    @property
    def health(self):
        """Returns the average health across all links."""
        return sum(self.link_healths.values()) / len(self.link_healths)

    @property
    def damage_status(self):
        """Returns the overall damage status of the object based on thresholds."""
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
        """Updates the health of all links based on damage from all generators."""
        for generator in self.damage_generators:
            link_damages = generator.generate_damage()
            for link_name, damage in link_damages.items():
                # Decrease health but not below zero
                new_health = max(0.0, self.link_healths[link_name] - damage)
                self.link_healths[link_name] = new_health
                # Update individual link status
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


# Damageable subclasses using the mixin
class DamageableDatasetObject(DamageableMixin, DatasetObject):
    """A DatasetObject that can be damaged."""
    pass


class DamageablePrimitiveObject(DamageableMixin, PrimitiveObject):
    """A PrimitiveObject that can be damaged."""
    pass


class DamageableUSDObject(DamageableMixin, USDObject):
    """A USDObject that can be damaged."""
    pass


class DamageableControllableObject(DamageableMixin, ControllableObject):
    """A ControllableObject that can be damaged."""
    pass


class DamageableLightObject(DamageableMixin, LightObject):
    """A LightObject that can be damaged."""
    pass


class DamageableStatefulObject(DamageableMixin, StatefulObject):
    """A StatefulObject that can be damaged."""
    pass


class DamageableFrankaPanda(DamageableMixin, FrankaPanda):
    """A FrankaPanda robot that can be damaged."""
    pass