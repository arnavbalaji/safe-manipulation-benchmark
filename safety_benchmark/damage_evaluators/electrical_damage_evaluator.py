import torch as th
import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

from safety_benchmark.damage_evaluators.damage_evaluator import DamageEvaluator
from omnigibson.objects.object_base import BaseObject


class ElectricalDamageEvaluator(DamageEvaluator):
    """
    Electrical damage evaluator that evaluates electrical damage based on water particle contact.
    
    Uses hybrid detection:
    1. ContactParticles state (primary method)
    2. Manual proximity detection (fallback method)
    
    Simple damage calculation: if particles > threshold, apply damage = scale
    """
    
    def __init__(self, entity: BaseObject, damage_threshold: float, scale: float, 
                 water_system_name: str = "sludge", proximity_threshold: float = 0.02):
        super().__init__(entity, damage_threshold, scale)
        self.entity = entity
        self.damage_threshold = damage_threshold
        self.scale = scale
        
        # Water system configuration
        self.water_system_name = water_system_name
        self.water_system = None
        
        # Proximity detection settings
        self.proximity_threshold = proximity_threshold  # Distance threshold for manual detection
        
        # Tracking variables
        self.initialized = False
    
    def _initialize_water_system(self, scene):
        """Initialize the water system reference if not already done."""
        if self.initialized or self.water_system is not None:
            return
            
        try:
            # Try to find the water system
            possible_water_systems = [self.water_system_name, "sludge", "water", "fluid"]
            for system_name in possible_water_systems:
                if scene.is_physical_particle_system(system_name):
                    self.water_system = scene.get_system(system_name)
                    print(f"✅ ElectricalDamageEvaluator: Found water system: {system_name}")
                    break
            
            if self.water_system is None:
                print(f"⚠️ ElectricalDamageEvaluator: No water system found for {self.water_system_name}")
            else:
                print(f"✅ ElectricalDamageEvaluator: Water contact tracking enabled")
                
        except Exception as e:
            print(f"❌ ElectricalDamageEvaluator: Error setting up water system: {e}")
            self.water_system = None
        
        self.initialized = True
    
    def _get_contact_particles_data(self):
        """Get water contact data using ContactParticles state (primary method)."""
        if self.water_system is None:
            return {}
        
        contact_data = {}
        
        try:
            # Get water particles in contact with the robot using ContactParticles state
            contacting_particles = self.entity.states[object_states.ContactParticles].get_value(system=self.water_system)
            total_contact_count = len(contacting_particles)
            
            # Get per-link contact information if available
            for link_name, link in self.entity.links.items():
                try:
                    # Try to get link-specific contact particles
                    link_contact_particles = self.entity.states[object_states.ContactParticles].get_value(
                        system=self.water_system, link=link
                    )
                    link_contact_count = len(link_contact_particles)
                except:
                    # Fallback: estimate link contact based on total contact
                    link_contact_count = total_contact_count // len(self.entity.links) if total_contact_count > 0 else 0
                
                contact_data[link_name] = {
                    'particle_count': link_contact_count,
                    'total_particles': total_contact_count
                }
                
        except Exception as e:
            print(f"⚠️ ElectricalDamageEvaluator: Error getting ContactParticles data: {e}")
            # Return empty contact data on error
            for link_name in self.entity.links.keys():
                contact_data[link_name] = {'particle_count': 0, 'total_particles': 0}
        
        return contact_data
    
    def _manual_proximity_detection(self):
        """Manual water contact detection using particle proximity (fallback method)."""
        if self.water_system is None:
            return {}
        
        contact_data = {}
        
        try:
            # Get all water particle positions
            particle_positions = self.water_system.get_particles_position_orientation()[0]
            
            for link_name, link in self.entity.links.items():
                # Get link position and bounding box
                link_pos = link.get_position_orientation()[0]
                link_aabb = link.visual_aabb
                
                # Check particles within proximity threshold
                nearby_particles = 0
                for particle_pos in particle_positions:
                    if self._is_particle_near_link(particle_pos, link_pos, link_aabb):
                        nearby_particles += 1
                
                contact_data[link_name] = {
                    'particle_count': nearby_particles,
                    'total_particles': len(particle_positions)
                }
                
        except Exception as e:
            print(f"⚠️ ElectricalDamageEvaluator: Error in manual proximity detection: {e}")
            # Return empty contact data on error
            for link_name in self.entity.links.keys():
                contact_data[link_name] = {'particle_count': 0, 'total_particles': 0}
        
        return contact_data
    
    def _is_particle_near_link(self, particle_pos, link_pos, link_aabb):
        """Check if a particle is near a link using proximity threshold."""
        try:
            # Convert to numpy for easier calculations
            particle_pos = particle_pos.cpu().numpy() if hasattr(particle_pos, 'cpu') else particle_pos
            link_pos = link_pos.cpu().numpy() if hasattr(link_pos, 'cpu') else link_pos
            
            # Simple distance check from particle to link center
            distance = np.linalg.norm(particle_pos - link_pos)
            return distance <= self.proximity_threshold
            
        except Exception:
            return False
    
    def _get_water_contact_data(self):
        """Get water contact data using hybrid detection approach."""
        # Initialize water system if needed
        if not self.initialized and hasattr(self.entity, 'scene'):
            self._initialize_water_system(self.entity.scene)
        
        # Try ContactParticles first (primary method)
        contact_data = self._get_contact_particles_data()
        
        # If no contact detected, try manual proximity (fallback method)
        if not any(data['particle_count'] > 0 for data in contact_data.values()):
            contact_data = self._manual_proximity_detection()
        
        return contact_data
    
    def generate_damage(self) -> dict:
        """
        Generate electrical damage values based on water particle contact.
        
        Simple damage calculation: if particles > threshold, apply damage = scale
        
        Returns:
            dict: Mapping from link names to damage amounts
        """
        # Get current water contact data
        contact_data = self._get_water_contact_data()
        
        # Calculate damage for each link
        link_damages = {}
        for link_name in self.entity.links.keys():
            if link_name in contact_data:
                particle_count = contact_data[link_name]['particle_count']
                
                # Simple damage calculation: if above threshold, apply scale
                if particle_count > self.damage_threshold:
                    damage = (particle_count - self.damage_threshold) * self.scale
                else:
                    damage = 0.0
            else:
                damage = 0.0
            
            link_damages[link_name] = damage
        
        return link_damages
    
    def reset_tracking(self):
        """Reset all tracking variables (called on env.reset())."""
        # No complex tracking to reset in simplified version
        pass
    
    def get_contact_summary(self) -> dict:
        """Get a summary of current water contact status."""
        if self.water_system is None:
            return {'status': 'no_water_system', 'total_contact': 0}
        
        contact_data = self._get_water_contact_data()
        total_contact = sum(data['particle_count'] for data in contact_data.values())
        
        return {
            'status': 'active' if total_contact > 0 else 'no_contact',
            'total_contact': total_contact,
            'link_details': contact_data
        } 