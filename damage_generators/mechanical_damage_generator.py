from safety_benchmark.damage_generators.damage_generator import DamageGenerator
from omnigibson.objects.object_base import BaseObject
import omnigibson as og
import torch as th
from omnigibson.utils.usd_utils import RigidContactAPI
from typing import Dict, Optional
import numpy as np

class MechanicalDamageGenerator(DamageGenerator):
    '''
    Realistic damage generator for mechanical forces based on impact energy and velocity changes.
    
    This implementation tracks:
    1. Impact energy (kinetic energy before impact)
    2. Velocity changes (sudden deceleration indicating impacts)
    3. Material properties (fragility, elasticity)
    4. Proper impact detection (vs static contact)
    '''
    def __init__(self, entity: BaseObject, damage_threshold: float, scale: float, 
                 material_properties: Optional[Dict] = None):
        super().__init__(entity, damage_threshold, scale)
        
        # Initialize material properties with defaults
        self.material_properties = material_properties or {
            "fragility": 1.0,  # How easily the object breaks (0.1 = very fragile, 10.0 = very tough)
            "elasticity": 0.3,  # How much energy is absorbed vs transferred (0.0 = no bounce, 1.0 = perfect bounce)
            "density": 1000.0,  # kg/m³, affects impact energy calculation
            "contact_threshold": 0.1,  # Minimum velocity change to consider as impact (m/s)
            "energy_threshold": 0.01,  # Minimum kinetic energy to consider as impact (J)
        }
        
        # Initialize the contact API
        RigidContactAPI.initialize_view()
        
        # Track previous velocities for impact detection
        self._prev_velocities = {}
        self._prev_contact_states = {}
        self._impact_history = {}
        
        # Initialize tracking for each link
        for link_name in self.entity.links.keys():
            self._prev_velocities[link_name] = th.zeros(3)
            self._prev_contact_states[link_name] = False
            self._impact_history[link_name] = []

    def _calculate_kinetic_energy(self, velocity: th.Tensor, mass: float) -> float:
        """Calculate kinetic energy: KE = 0.5 * mass * velocity^2"""
        return 0.5 * mass * th.sum(velocity ** 2).item()

    def _detect_impact(self, link_name: str, current_velocity: th.Tensor, 
                      current_contacts: bool) -> Optional[Dict]:
        """
        Detect if an impact occurred based on velocity change and contact state.
        
        Returns:
            Dict with impact information if impact detected, None otherwise
        """
        prev_velocity = self._prev_velocities[link_name]
        prev_contacts = self._prev_contact_states[link_name]
        
        # Calculate velocity change
        velocity_change = th.norm(current_velocity - prev_velocity).item()
        
        # Impact detection criteria:
        # 1. Sudden velocity change (deceleration)
        # 2. Contact state changed from no contact to contact
        # 3. Velocity change exceeds threshold
        # 4. Was moving before impact (prev_velocity magnitude > threshold)
        is_impact = (
            velocity_change > self.material_properties["contact_threshold"] and
            not prev_contacts and current_contacts and
            th.norm(prev_velocity) > self.material_properties["contact_threshold"]  # Was moving before impact
        )
        
        if is_impact:
            # Calculate impact energy using velocity before impact
            link = self.entity.links[link_name]
            mass = link.mass if hasattr(link, 'mass') else self.material_properties["density"] * 0.001
            impact_energy = self._calculate_kinetic_energy(prev_velocity, mass)
            
            return {
                "velocity_change": velocity_change,
                "impact_energy": impact_energy,
                "pre_impact_velocity": prev_velocity.clone(),
                "post_impact_velocity": current_velocity.clone(),
                "timestamp": og.sim.current_time
            }
        
        return None

    def _calculate_damage_from_impact(self, impact_info: Dict) -> float:
        """
        Calculate damage based on impact energy and material properties.
        
        Uses a realistic damage model:
        - Damage scales with impact energy
        - Material fragility affects damage multiplier
        - Elasticity reduces damage (energy absorbed)
        """
        impact_energy = impact_info["impact_energy"]
        velocity_change = impact_info["velocity_change"]
        
        # Base damage from impact energy - use a more aggressive scaling for fragile objects
        base_damage = impact_energy * self.scale
        
        # Apply material properties
        fragility_multiplier = self.material_properties["fragility"]
        elasticity_reduction = 1.0 - self.material_properties["elasticity"]
        
        # For very fragile objects (like glass), increase damage significantly
        if fragility_multiplier > 3.0:
            fragility_multiplier *= 2.0  # Double the damage for very fragile objects
        
        # Final damage calculation
        damage = base_damage * fragility_multiplier * elasticity_reduction
        
        # Apply threshold - only damage if impact is significant
        if impact_energy < self.material_properties["energy_threshold"]:
            damage = 0.0
        
        return max(0.0, damage)

    def _update_contact_forces_damage(self, link_name: str) -> float:
        """
        Calculate damage from sustained contact forces (for crushing, etc.)
        This should be ZERO for normal static contact like a wine glass sitting on a table.
        """
        # For now, return 0 to eliminate any damage from static contact
        # This can be expanded later for crushing scenarios if needed
        return 0.0

    def generate_damage(self) -> Dict[str, float]:
        link_damages = {}
        
        for link_name, link in self.entity.links.items():
            total_damage = 0.0
            
            # Get current state
            current_velocity = link.get_linear_velocity()
            current_contacts = len(link.contact_list()) > 0
            
            # Detect impacts
            impact_info = self._detect_impact(link_name, current_velocity, current_contacts)
            
            if impact_info is not None:
                # Calculate damage from impact
                impact_damage = self._calculate_damage_from_impact(impact_info)
                total_damage += impact_damage
                
                # Store impact history for debugging
                self._impact_history[link_name].append(impact_info)
                
                # Keep only recent impacts (last 10)
                if len(self._impact_history[link_name]) > 10:
                    self._impact_history[link_name] = self._impact_history[link_name][-10:]
            
            # Add damage from sustained contact forces (currently disabled)
            contact_damage = self._update_contact_forces_damage(link_name)
            total_damage += contact_damage
            
            # Update tracking state
            self._prev_velocities[link_name] = current_velocity.clone()
            self._prev_contact_states[link_name] = current_contacts
            
            link_damages[link_name] = total_damage
        
        return link_damages

    def get_impact_history(self, link_name: str = None) -> Dict:
        """Get impact history for debugging and analysis."""
        if link_name is None:
            return self._impact_history
        return self._impact_history.get(link_name, [])

    def reset_tracking(self):
        """Reset all tracking state (useful for environment resets)."""
        for link_name in self.entity.links.keys():
            self._prev_velocities[link_name] = th.zeros(3)
            self._prev_contact_states[link_name] = False
            self._impact_history[link_name] = []