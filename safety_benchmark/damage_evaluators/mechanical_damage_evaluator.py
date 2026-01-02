from safety_benchmark.damage_evaluators.damage_evaluator import DamageEvaluator
from omnigibson.objects.object_base import BaseObject
import torch as th
from typing import Dict
import omnigibson as og
import numpy as np
from omnigibson.utils.usd_utils import RigidContactAPI


class MechanicalDamageEvaluator(DamageEvaluator):
    """
    Evaluates damage based on mechanical forces from OmniGibson.
    """

    def __init__(
        self,
        entity: BaseObject,
        damage_threshold: float = 0.0,
        damage_scale: float = 1.0,
        impact_damage_sensitivity: float = 1.0,
        qs_damage_sensitivity: float = 1.0,
        link_config_overrides: Dict[str, dict] | None = None,
        **kwargs,
    ):
        """
        Args:
            entity: The "object" to evaluate damage for.
            damage_threshold: The threshold for damage. If the damage potential is greater than this threshold, then damage is applied.
            damage_scale: The scale for damage. The damage is scaled by this value.
            impact_damage_sensitivity: The sensitivity for impact damage. The impact force is scaled by this value.
            qs_damage_sensitivity: The sensitivity for quasistatic damage. The quasistatic forces are summed and scaled by this value.
            link_config_overrides: The damage_threshold, impact_damage_sensitivity, qs_damage_sensitivity, and damage_scale overrides for each link. Typically used if specific links take values differnt from the default values for the object.
        """
        super().__init__(entity, damage_threshold, damage_scale)
        self.damage_threshold = damage_threshold
        self.damage_scale = damage_scale
        self.entity = entity
        self.impact_damage_sensitivity = impact_damage_sensitivity
        self.qs_damage_sensitivity = qs_damage_sensitivity
        self.link_config_overrides = {k.lower(): v for k, v in (link_config_overrides or {}).items()}
        self.name = "mechanical"
        # For EMA filtering of the raw forces from simulation
        self.alpha = 0.2
        # Window size for averaging the filtered forces from simulation
        self.window_size = int((1.0 / og.sim.get_sim_step_dt()) / 2.0)

        # Running state for impact / sustained computations
        self.prev_link_positions: dict[str, th.Tensor] = {}
        init_link_positions = {link_name: link.get_position_orientation()[0] for link_name, link in self.entity.links.items()}
        self.prev_link_positions.update(init_link_positions)
        self.prev_link_velocities: dict[str, th.Tensor] = {}
        init_link_velocities = {link_name: th.zeros(3) for link_name in self.entity.links.keys()}
        self.prev_link_velocities.update(init_link_velocities)
        self.previous_unique_contact_bodies: dict[str, set[str]] = {}

        # For tracking the unique contact bodies for each link
        for link_name, link in self.entity.links.items():
            init_contacts_list = link.contact_list()
            body0_list = {c.body0 for c in init_contacts_list}
            body1_list = {c.body1 for c in init_contacts_list}
            init_previous_unique_contact_bodies = body0_list | body1_list
            self.previous_unique_contact_bodies[link_name] = init_previous_unique_contact_bodies

        # For additional logging
        self.damage_potentials: dict[str, list] = {}
        self.unfiltered_raw_sim_forces: dict[str, list] = {}
        self.filtered_raw_sim_forces: dict[str, list] = {}
        self.unfiltered_qs_forces: dict[str, list] = {}
        self.filtered_qs_forces: dict[str, list] = {}
        self.impact_forces: dict[str, list] = {}
        self.contacts_by_link: dict[str, list] = {}
        self.num_unique_contacts: dict[str, list] = {}

    def check_new_contact_body(self, current_unique_contact_bodies, link_name) -> bool:
        """
        Check if a new category object was contacted for the first time in this step.
        """
        if len(current_unique_contact_bodies) > len(self.previous_unique_contact_bodies[link_name]):
            return True, current_unique_contact_bodies - self.previous_unique_contact_bodies[link_name]
        else:
            return False, set()
    
    def generate_damage(self) -> Dict[str, float]:
        link_damages: Dict[str, float] = {}
        
        # I think dt should not be the physics frequency. It should be based on the time taken by one env.step() call.
        # TODO: However, the dt field in contact_now has 0.008 (which is the physics frequency)
        # dt = og.sim.get_physics_dt()
        dt = og.sim.get_sim_step_dt()

        # Apply damage to each link
        for i, (link_name, link) in enumerate(self.entity.links.items()):
            
            # Skip non-damageable links
            if link_name not in self.entity.damageable_links:
                continue

            # Initialize the relevant logging lists for the link if they don't exist.
            if link_name not in self.impact_forces:
                self.impact_forces[link_name] = []
                self.unfiltered_raw_sim_forces[link_name] = []
                self.filtered_raw_sim_forces[link_name] = []
                self.unfiltered_qs_forces[link_name] = []
                self.filtered_qs_forces[link_name] = []
                self.contacts_by_link[link_name] = []
            
            # Get contacts for the link (from the physics engine)
            try:
                contacts_list = link.contact_list()
            except Exception:
                contacts_list = []
            self.contacts_by_link[link_name].append(contacts_list)

            try:
                # Compute impact force (force that leads to acceleration) via finite-differenced acceleration
                position_previous = self.prev_link_positions.get(link_name)
                position_current, _ = link.get_position_orientation()
                displacement = position_current - position_previous
                velocity_previous = self.prev_link_velocities.get(link_name, th.zeros(3))
                velocity_current = displacement / max(dt, 1e-8) # dx/dt
                delta_velocity = velocity_current - velocity_previous
                acceleration = delta_velocity / max(dt, 1e-8) # dv/dt
                acceleration_unit_vector = acceleration / th.linalg.vector_norm(acceleration).item()
            except Exception as e:
                print("1 Error: ", e)
                breakpoint()

            # Track for next step
            self.prev_link_positions[link_name] = position_current.clone()
            self.prev_link_velocities[link_name] = velocity_current.clone()

            # # NOTE: Not using this approach anymore. 
            # # Impact forces is only computed if a contact was added at this time step, so check that
            # current_unique_contact_bodies = {c.body1 for c in contacts_now}
            # if len(current_unique_contact_bodies) > len(self.previous_unique_contact_bodies[link_name]):
            #     impact_magnitude = th.linalg.vector_norm(acceleration).item() * float(getattr(link, "mass", 1.0))
            # else:
            #     impact_magnitude = 0.0
            # self.previous_unique_contact_bodies[link_name] = current_unique_contact_bodies
            
            # If using only the acceleration as a proxy for impact force.
            # impact_force = 1.0 * acceleration
            impact_force =  float(getattr(link, "mass", 1.0)) * acceleration
            impact_force_magnitude = th.linalg.vector_norm(impact_force).item()
            self.impact_forces[link_name].append(impact_force_magnitude)

            # # For debugging
            # if self.entity.name == "coffee_cup_1" and link_name == "base_link":
            #     print("impact_force_magnitude: ", impact_force_magnitude)
            
            adjust_sim_forces = True
            # TODO: check if this condition is needed.
            # if acceleration is quite small, then we don't want to adjust the forces obtained from OG.
            # i.e. we don't want to remove the component of forces obtained from OG that are in the direction of acceleration.
            delta_velocity_norm = th.linalg.vector_norm(delta_velocity).item()
            if delta_velocity_norm < 1e-8:
                adjust_sim_forces = False

            # Compute rest of the forces on the object (quasistatic forces or in other words, forces other than the one causing acceleration)
            impulses: list[th.Tensor] = []
            current_unfiltered_qs_force_magnitude = 0.0
            current_filtered_qs_force_magnitude = 0.0
            for c in contacts_list:
                impulses.append(th.tensor(c.impulse.tolist(), dtype=th.float32))
            try:
                if impulses:
                    current_unfiltered_raw_sim_force_magnitude = (float(th.sum(th.stack([th.linalg.vector_norm(v) for v in impulses]))) / max(dt, 1e-8))
                    
                    # For debugging
                    # if self.entity.category == "agent" and link_name in ["right_gripper_finger_link1", "right_gripper_finger_link2"]:
                    #     print("link_name, current_unfiltered_raw_sim_force_magnitude: ", link_name, current_unfiltered_raw_sim_force_magnitude)                    

                    # =======================================================
                    # 1) Obtain only the quasistatic forces
                    if not adjust_sim_forces:
                        # Note that we sum the magnitudes of the impulses (for each contact point) and divide
                        # by dt to get the force. So, the output is a scalar value for the force on the link.
                        current_unfiltered_qs_force_magnitude += current_unfiltered_raw_sim_force_magnitude
                    else:
                        adjusted_qs_force_magnitudes = []
                        for j, impulse_vec in enumerate(impulses):
                            proj_imp = th.dot(impulse_vec, acceleration_unit_vector).item()
                            effective_impulse = impulse_vec - proj_imp * acceleration_unit_vector if proj_imp > 0 else impulse_vec
                            adjusted_qs_force_magnitudes.append(th.linalg.vector_norm(effective_impulse))                        
                        current_unfiltered_qs_force_magnitude += (float(th.sum(th.stack(adjusted_qs_force_magnitudes))) / max(dt, 1e-8))

                    self.unfiltered_qs_forces[link_name].append(current_unfiltered_qs_force_magnitude)
                    
                    # Option 2) Average over a window
                    if len(self.unfiltered_qs_forces[link_name]) >= self.window_size:
                        current_filtered_qs_force_magnitude = sum(self.unfiltered_qs_forces[link_name][-self.window_size:]) / self.window_size
                    else:
                        current_filtered_qs_force_magnitude = current_unfiltered_qs_force_magnitude
                    self.filtered_qs_forces[link_name].append(current_filtered_qs_force_magnitude)
                    # =======================================================
                    
                    # 2) Obtain the raw sim forces
                    self.unfiltered_raw_sim_forces[link_name].append(current_unfiltered_raw_sim_force_magnitude)
                    # Option 1: EMA
                    # if len(self.raw_forces_from_sim[link_name]) == 0:
                    #     filtered_force = current_force
                    # else:
                    #     filtered_force = self.alpha * current_force + (1 - self.alpha) * self.raw_forces_from_sim[link_name][-1]

                    # # Option 2: Average over a window
                    # if len(self.unfiltered_raw_sim_forces[link_name]) >= self.window_size:
                    #     filtered_raw_sim_force_magnitude = sum(self.unfiltered_raw_sim_forces[link_name][-self.window_size:]) / self.window_size
                    # else:
                    #     filtered_raw_sim_force_magnitude = current_unfiltered_raw_sim_force_magnitude
                    
                    # Option 3: Filter first contact force
                    # check if a new category object was contacted for the first time in this step
                    body0_list = {c.body0 for c in contacts_list}
                    body1_list = {c.body1 for c in contacts_list}
                    current_unique_contact_bodies = body0_list | body1_list
                    new_contact_bool, new_contact_bodies = self.check_new_contact_body(current_unique_contact_bodies, link_name)
                    if new_contact_bool:
                        # print("link_name, new_contact_body: ", link_name, new_contact_bodies)
                        filtered_raw_sim_force_magnitude = sum(self.unfiltered_raw_sim_forces[link_name][-self.window_size:]) / self.window_size
                    else:
                        filtered_raw_sim_force_magnitude = current_unfiltered_raw_sim_force_magnitude
                    self.previous_unique_contact_bodies[link_name] = current_unique_contact_bodies

                    self.filtered_raw_sim_forces[link_name].append(filtered_raw_sim_force_magnitude)
                    
                    # For debugging
                    # if self.entity.category == "agent" and link_name in ["right_gripper_finger_link1", "right_gripper_finger_link2"]:
                    #     print("link_name, filtered_raw_sim_force_magnitude: ", link_name, filtered_raw_sim_force_magnitude)                    
                    # if self.entity.category == "agent"  and link_name in ["gripper_right_left_finger_link", "gripper_right_right_finger_link"]:
                    #     if filtered_raw_sim_force_magnitude > 50.0:
                    #         print("link_name, filtered_raw_sim_force_magnitude: ", link_name, filtered_raw_sim_force_magnitude)
                    #         breakpoint()
                    
                else:
                    self.unfiltered_raw_sim_forces[link_name].append(0.0)
                    self.filtered_raw_sim_forces[link_name].append(0.0)
                    self.unfiltered_qs_forces[link_name].append(0.0)
                    self.filtered_qs_forces[link_name].append(0.0)
            
            except Exception as e:
                print("2 Error: ", e)
                breakpoint()

            # # For debugging
            # if self.entity.name == "cup" and link_name == "base_link":
            #     # breakpoint()
            #     unique_contact_bodies = {c.body1 for c in contacts_list}
            #     print("unique_contact_bodies: ", unique_contact_bodies)
            #     # This API gives the same contacts as the contact_list() API.
            #     # rigid_contacts = RigidContactAPI.get_contact_pairs(0, column_prim_paths=[link.prim_path])
            #     # if len(rigid_contacts) > 0:
            #     #     breakpoint()
            #     # print("rigid_contacts: ", rigid_contacts)
            #     print("impact_magnitude, raw_impulse_forces: ", impact_magnitude, self.raw_impulse_forces[link_name][-1])
            #     # print("position_current: ", position_current)
            #     if impact_magnitude > 50.0:
            #         breakpoint()

            # Per-link values
            link_impact_damage_sensitivity = self.impact_damage_sensitivity
            link_qs_damage_sensitivity = self.qs_damage_sensitivity
            link_damage_threshold = self.damage_threshold
            link_damage_scale = self.damage_scale

            lname = link_name.lower()
            for key, overrides in self.link_config_overrides.items():
                if key in lname:
                    if "impact_damage_sensitivity" in overrides:
                        link_impact_damage_sensitivity = overrides["impact_damage_sensitivity"]
                    if "qs_damage_sensitivity" in overrides:
                        link_qs_damage_sensitivity = overrides["qs_damage_sensitivity"]
                    if "damage_threshold" in overrides:
                        link_damage_threshold = overrides["damage_threshold"]
                    if "damage_scale" in overrides:
                        link_damage_scale = overrides["damage_scale"]
                    break

            try:
                impact_damage_potential = link_impact_damage_sensitivity * impact_force_magnitude
                qs_damage_potential = link_qs_damage_sensitivity * current_filtered_qs_force_magnitude

                link_damage_potential = impact_damage_potential + qs_damage_potential
                self.damage_potentials[link_name] = link_damage_potential
                link_damage = max(0.0, (link_damage_potential - link_damage_threshold)) * link_damage_scale
                link_damages[link_name] = link_damage
            
            except Exception as e:
                print("3 Error: ", e)
                breakpoint()
                
            # For debugging
            # if self.entity.name == "coffee_cup_1" and link_name == "base_link":
            #     print("impact_damage_potential, qs_damage_potential, link_damage_potential, link_damage_threshold, link_damage: ", impact_damage_potential, qs_damage_potential, link_damage_potential, link_damage_threshold, link_damage)
                # breakpoint()
                # if damage > 0.0:
                #     breakpoint()

        return link_damages

    def reset_tracking(self):
        """Reset mechanical damage tracking state"""
        self.prev_link_positions: dict[str, th.Tensor] = {}
        init_link_positions = {link_name: link.get_position_orientation()[0] for link_name, link in self.entity.links.items()}
        self.prev_link_positions.update(init_link_positions)
        self.prev_link_velocities: dict[str, th.Tensor] = {}
        init_link_velocities = {link_name: th.zeros(3) for link_name in self.entity.links.keys()}
        self.prev_link_velocities.update(init_link_velocities)
        self.previous_unique_contact_bodies: dict[str, set[str]] = {}
        for link_name, link in self.entity.links.items():
            init_contacts_list = link.contact_list()
            init_previous_unique_contact_bodies = {c.body1 for c in init_contacts_list}
            self.previous_unique_contact_bodies[link_name] = init_previous_unique_contact_bodies
            self.impact_forces[link_name] = []
            self.unfiltered_raw_sim_forces[link_name] = []
            self.filtered_raw_sim_forces[link_name] = []
            self.unfiltered_qs_forces[link_name] = []
            self.filtered_qs_forces[link_name] = []
            self.contacts_by_link[link_name] = []

    def update_link_positions_and_velocities(self):
        self.prev_link_positions = {link_name: link.get_position_orientation()[0] for link_name, link in self.entity.links.items()}
        self.prev_link_velocities = {link_name: th.zeros(3) for link_name in self.entity.links.keys()}
    