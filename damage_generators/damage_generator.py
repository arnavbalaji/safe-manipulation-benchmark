from omnigibson.objects.object_base import BaseObject


class DamageGenerator:
    def __init__(self, entity: BaseObject, damage_threshold: float, scale: float):
        self.damage_threshold = damage_threshold
        self.scale = scale
        self.entity = entity

    def generate_damage(self):
        return {link_name: 0.0 for link_name in self.entity.links.keys()}