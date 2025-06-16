from safety_benchmark.damage_generators.damage_generator import DamageGenerator
from omnigibson.objects.object_base import BaseObject
import omnigibson as og
import torch
from omnigibson.utils.usd_utils import RigidContactAPI
from typing import Dict

class MechanicalDamageGenerator(DamageGenerator):
    '''
    Damage generator for mechanical forces
    '''
    def __init__(self, entity: BaseObject, damage_threshold: float, scale: float):
        super().__init__(entity, damage_threshold, scale)
        # Initialize the contact API
        RigidContactAPI.initialize_view()

    def generate_damage(self) -> Dict[str, float]:
        link_damages = {}
        # Tracking contact forces for each link
        for link_name, link in self.entity.links.items():
            total_force = 0.0

            # Old implementation using RigidContactAPI
            # scene_idx = RigidContactAPI.get_scene_idx(link.prim_path)
            # all_impulses = RigidContactAPI.get_all_impulses(scene_idx)
            # _, row_idx = RigidContactAPI.get_body_row_idx(link.prim_path)
            # link_impulses = all_impulses[row_idx]
            #
            # if link_impulses is not None and len(link_impulses) > 0:
            #     link_impulses_tensor = torch.tensor(link_impulses).clone().detach()
            #     total_force = torch.sum(torch.norm(link_impulses_tensor, dim=-1)).item()

            # New implementation using link.contact_list()
            contacts = link.contact_list()
            if len(contacts) > 0:
                # Extract impulse values from contact objects
                contact_forces = torch.tensor([c.impulse.tolist() for c in contacts])
                total_force = torch.sum(torch.norm(contact_forces, dim=-1)).item()
            else:
                total_force = 0.0

            if total_force >= self.damage_threshold:
                link_damages[link_name] = (total_force - self.damage_threshold) * self.scale
            else:
                link_damages[link_name] = 0.0
        
        return link_damages