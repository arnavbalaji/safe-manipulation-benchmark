import torch as th
import math
import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

from safety_benchmark.damage_evaluators.damage_evaluator import DamageEvaluator
from omnigibson.objects.object_base import BaseObject


class ThermalDamageEvaluator(DamageEvaluator):
    """
    Thermal damage evaluator that evaluates thermal damage based on the temperature of the object.
    """
    def __init__(self, entity: BaseObject, heating_threshold: float, scale: float, cooling_threshold: float):
        super().__init__(entity, heating_threshold, scale)
        self.entity = entity
        self.scale = scale
        self.heating_threshold = heating_threshold
        self.cooling_threshold = cooling_threshold
        self.name = "thermal"
    def generate_damage(self):
        damage = 0.0
        self.current_temperature = self.entity.states[object_states.Temperature].get_value()

        if self.current_temperature > self.heating_threshold:
            damage = self.scale * (self.current_temperature - self.heating_threshold)
            
            # Set the object's Heated state to True when temperature exceeds threshold
            # This will cause steam to appear
            # if object_states.Heated in self.entity.states:
            #     self.entity.states[object_states.Heated].set_value(True)
        elif self.current_temperature < self.cooling_threshold:
            damage = abs(self.scale * (self.current_temperature - self.cooling_threshold))
    
        return {link_name: damage for link_name in self.entity.links.keys()}
    
    def get_temperature(self):
        return self.current_temperature