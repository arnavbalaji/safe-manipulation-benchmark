from safety_benchmark.damage_evaluators.damage_evaluator import DamageEvaluator
from omnigibson.objects.object_base import BaseObject
import omnigibson as og
import torch as th
from typing import Dict
import numpy as np

class MechanicalDamageEvaluator(DamageEvaluator):
    """
    Evaluates damage based on mechanical forces from OmniGibson.
    """
    def __init__(self, entity: BaseObject, damage_threshold: float, scale: float, link_thresholds: Dict[str, dict] = None, enable_deceleration_detection: bool = None):
        super().__init__(entity, damage_threshold, scale)
        self.entity = entity
        
        # Store force values
        self.force_values = []
        
        # Track forces during current environment step
        self.current_env_step_forces = []
        
        # Per-link overrides for thresholds and scale
        self.link_thresholds = {k.lower(): v for k, v in (link_thresholds or {}).items()}
        
        # Initialize for relative force calculation
        self.initial_impulse_forces = {}
        self.initial_normal_forces = {}

    def generate_damage(self) -> Dict[str, float]:
        """
        Generate damage values using OmniGibson's contact forces.
        """
        link_damages = {}
        max_force_this_timestep = 0.0
        
        for link_name, link in self.entity.links.items():
            # Get contact forces from OmniGibson
            impulse_force = 0.0
            normal_force = 0.0
            
            contacts = link.contact_list()
            if len(contacts) > 0:
                # Get both impulse and normal forces from contacts
                contact_forces = th.tensor([c.impulse.tolist() for c in contacts])
                contact_normals = th.tensor([c.normal.tolist() for c in contacts])
                # Debug: break if any component is negative to test directionality
                # if (contact_forces < 0).any().item() or (contact_normals < 0).any().item():
                #     min_imp = contact_forces.min().item()
                #     min_norm = contact_normals.min().item()
                #     neg_impulse_vals = contact_forces[contact_forces < 0].tolist()
                #     neg_normal_vals = contact_normals[contact_normals < 0].tolist()
                #     print(f"[MechanicalDamageEvaluator] Negative component detected on link '{link_name}': min_impulse={min_imp:.6f}, min_normal={min_norm:.6f}")
                #     print(f"    neg_impulse_components={neg_impulse_vals}")
                #     print(f"    neg_normal_components={neg_normal_vals}")
                #     breakpoint()
                
                # Calculate current impulse and normal forces
                impulse_force = th.norm(th.sum(contact_forces, dim=0)).item()
                normal_force = th.norm(th.sum(contact_normals, dim=0)).item()
                
                # Set initial values if being called for the first time
                if link_name not in self.initial_impulse_forces:
                    self.initial_impulse_forces[link_name] = impulse_force
                if link_name not in self.initial_normal_forces:
                    self.initial_normal_forces[link_name] = normal_force
                
                # Calculate relative forces (floor at 0)
                relative_impulse = max(0.0, impulse_force - self.initial_impulse_forces[link_name])
                relative_normal = max(0.0, normal_force - self.initial_normal_forces[link_name])
                
                # Combine relative forces
                contact_force = relative_impulse + relative_normal
            else:
                contact_force = 0.0

            active_threshold = self.damage_threshold
            active_scale = self.scale
            
            # Check for per-link overrides
            if self.link_thresholds:
                lname = link_name.lower()
                matches = [k for k in self.link_thresholds.keys() if k in lname]
                if matches:
                    # Find the longest match, and if there are ties, prefer the first one listed in params
                    max_len = max(len(k) for k in matches)
                    longest_matches = [k for k in matches if len(k) == max_len]
                    # Use the first one that appears in the original link_thresholds dict (preserving order)
                    best = None
                    for k in self.link_thresholds.keys():
                        if k.lower() in longest_matches:
                            best = k.lower()
                            break
                    override = self.link_thresholds[best]
                    if 'damage_threshold' in override:
                        active_threshold = override['damage_threshold']
                    if 'scale' in override:
                        active_scale = override['scale']
            
            # Calculate damage
            damage = max(0.0, (contact_force - active_threshold) * active_scale)
            
            # Store damage for this link
            link_damages[link_name] = damage

            max_force_this_timestep = max(max_force_this_timestep, contact_force)

        # Track forces for current environment step
        if max_force_this_timestep > 0:
            self.current_env_step_forces.append(max_force_this_timestep)
        else:
            self.current_env_step_forces.append(0.0)
        
        return link_damages

    def aggregate_forces_for_env_step(self):
        """
        Aggregate forces for the current environment step.
        """
        if not self.current_env_step_forces:
            # No forces this env step, append 0
            self.force_values.append(0.0)
        else:
            # # Use MAXIMUM force from this env step instead of sum (for testing thresholds)
            # max_force = max(self.current_env_step_forces)
            # self.force_values.append(max_force)

            total_force = sum(self.current_env_step_forces)
            self.force_values.append(total_force)
        
        # Reset for next environment step
        self.current_env_step_forces = []

    def reset_tracking(self):
        """Reset force tracking state."""
        self.current_env_step_forces = []
        self.initial_impulse_forces = {}
        self.initial_normal_forces = {}
    
    def get_current_env_step_force(self):
        """
        Get the current environment step's aggregated force.
        """
        if not self.current_env_step_forces:
            return 0.0
        return sum(self.current_env_step_forces)
    
    @property
    def aggregated_force_values(self):
        """
        Property to access the aggregated force values.
        """
        return self.force_values