import torch as th

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

from safety_benchmark.damage_evaluators.damage_evaluator import DamageEvaluator
from omnigibson.objects.object_base import BaseObject


class ThermalDamageEvaluator(DamageEvaluator):
    """
    Thermal damage evaluator that evaluates thermal damage based on the temperature of the object.
    """
    def __init__(self, entity: BaseObject, damage_threshold: float, scale: float):
        super().__init__(entity, damage_threshold, scale)
        self.entity = entity
        self.damage_threshold = damage_threshold
        self.scale = scale
        
    def generate_damage(self):
        damage = 0.0
        current_temperature = self.entity.states[object_states.Temperature].get_value()

        if current_temperature > self.damage_threshold:
            damage = self.scale * (current_temperature - self.damage_threshold)
        
        return {link_name: damage for link_name in self.entity.links.keys()}