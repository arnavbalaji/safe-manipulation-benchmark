from omnigibson.objects.object_base import BaseObject
from abc import ABC, abstractmethod
from typing import Dict


class DamageEvaluator(ABC):
    '''
    Damage Evaluator abstract class
    '''
    def __init__(self, entity: BaseObject, damage_threshold: float, scale: float):
        self.damage_threshold = damage_threshold
        self.scale = scale
        self.entity = entity
    
    @abstractmethod
    def generate_damage(self) -> Dict[str, float]:
        raise NotImplementedError