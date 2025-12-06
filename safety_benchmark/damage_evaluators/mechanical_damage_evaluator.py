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
        strain_threshold: float = 0.0,
        damage_scale: float = 1.0,
        dynamic_forces_coefficient: float = 1.0,
        static_forces_coefficient: float = 1.0,
        link_thresholds: Dict[str, dict] | None = None,
        **kwargs,
    ):
        super().__init__(entity, strain_threshold, damage_scale)
        self.strain_threshold = strain_threshold
        self.damage_scale = damage_scale
        self.entity = entity
        self.dynamic_forces_coefficient = dynamic_forces_coefficient
        self.static_forces_coefficient = static_forces_coefficient
        self.link_thresholds = {k.lower(): v for k, v in (link_thresholds or {}).items()}
        self.name = "mechanical"

        # Running state for impact / sustained computations
        self.prev_link_positions: dict[str, th.Tensor] = {}
        init_link_positions = {link_name: link.get_position_orientation()[0] for link_name, link in self.entity.links.items()}
        self.prev_link_positions.update(init_link_positions)
        self.prev_link_velocities: dict[str, th.Tensor] = {}
        init_link_velocities = {link_name: th.zeros(3) for link_name in self.entity.links.keys()}
        self.prev_link_velocities.update(init_link_velocities)
        self.last_accel_dir_by_link: dict[str, th.Tensor | None] = {}
        self.previous_unique_contact_bodies: dict[str, set[str]] = {}
        for link_name, link in self.entity.links.items():
            init_contacts_list = link.contact_list()
            init_previous_unique_contact_bodies = {c.body1 for c in init_contacts_list}
            self.previous_unique_contact_bodies[link_name] = init_previous_unique_contact_bodies

        # Aggregation helpers
        self.current_env_step_strains: dict[str, float] = {}
        self.strain_values: list[float] = []
        # Per-link last strain values for the most recent compute
        self.last_strain_by_link: dict[str, float] = {}

        # For additional logging
        self.dynamic_forces: dict[str, list] = {}
        self.raw_forces_from_sim: dict[str, list] = {}
        self.static_forces: dict[str, list] = {}
        self.contacts_by_link: dict[str, list] = {}
        self.num_unique_contacts: dict[str, list] = {}

    def generate_damage(self) -> Dict[str, float]:
        link_damages: Dict[str, float] = {}
        epsilon = 1e-8
        
        # I think dt should not be the physics frequency. It should be based on the time taken by one env.step() call.
        # TODO: However, the dt field in contact_now has 0.008 (which is the physics frequency)
        # dt = og.sim.get_physics_dt()
        dt = og.sim.get_sim_step_dt()

        # Apply damage to each link
        for link_name, link in self.entity.links.items():

            # Get contacts for the link (from the physics engine)
            try:
                contacts_list = link.contact_list()
            except Exception:
                contacts_list = []

            # Compute dynamic forces (due to acceleration) via finite-differenced acceleration
            position_current, _ = link.get_position_orientation()
            has_previous = link_name in self.prev_link_positions
            position_previous = self.prev_link_positions.get(link_name)
            velocity_previous = self.prev_link_velocities.get(link_name, th.zeros(3))

            # TODO: Check if this condition is really necessary. I think we always have a previous position.
            if has_previous and position_previous is not None:
                displacement = position_current - position_previous
                velocity_current = displacement / max(dt, 1e-8) # dx/dt
                delta_velocity = velocity_current - velocity_previous
                acceleration = delta_velocity / max(dt, 1e-8) # dv/dt

                # # NOTE: Not using this approach anymore. 
                # # Impact forces is only computed if a contact was added at this time step, so check that
                # current_unique_contact_bodies = {c.body1 for c in contacts_now}
                # if len(current_unique_contact_bodies) > len(self.previous_unique_contact_bodies[link_name]):
                #     impact_magnitude = th.linalg.vector_norm(acceleration).item() * float(getattr(link, "mass", 1.0))
                # else:
                #     impact_magnitude = 0.0
                # self.previous_unique_contact_bodies[link_name] = current_unique_contact_bodies
                
                # TODO: Currently using only the acceleration as a proxy for impact force.
                dynamic_force = th.linalg.vector_norm(acceleration).item() * 1.0
                # impact_magnitude = th.linalg.vector_norm(acceleration).item() * float(getattr(link, "mass", 1.0))

                # TODO: Question: Why is dynamic_force not used here and instead
                # we are using delta_velocity_norm?
                delta_velocity_norm = th.linalg.vector_norm(delta_velocity).item()
                if delta_velocity_norm > epsilon:
                    accel_unit = delta_velocity / delta_velocity_norm
                    self.last_accel_dir_by_link[link_name] = accel_unit.clone()
                else:
                    self.last_accel_dir_by_link[link_name] = None
            else:
                velocity_current = th.zeros(3)
                dynamic_force = 0.0
                self.last_accel_dir_by_link[link_name] = None
            
            # Log relevant information
            if link_name not in self.dynamic_forces:
                self.dynamic_forces[link_name] = []
                self.raw_forces_from_sim[link_name] = []
                self.static_forces[link_name] = []
                self.contacts_by_link[link_name] = []
            self.dynamic_forces[link_name].append(dynamic_force)

            # Compute rest of the forces on the object (other that the ones due to acceleration)
            impulses: list[th.Tensor] = []
            try:
                self.contacts_by_link[link_name].append(contacts_list)
                # if len(contacts_now) > 0:
                #     breakpoint()
                # print("contacts_list: ", len(contacts_list))
                for c in contacts_list:
                    try:
                        impulses.append(th.tensor(c.impulse.tolist(), dtype=th.float32))
                    except Exception:
                        pass
            except Exception:
                pass
            
            accel_unit = self.last_accel_dir_by_link.get(link_name, None)
            static_force = 0.0
            if impulses:
                if accel_unit is None:
                    # Note that we sum the magnitudes of the impulses (for each contact point) and divide
                    # by dt to get the force. So, the output is a scalar value for the force on the link.
                    static_force += (float(th.sum(th.stack([th.linalg.vector_norm(v) for v in impulses]))) / max(dt, 1e-8))
                    # For debugging
                    # if self.entity.name == "swivel_chair" and link_name == "base_link":
                    #     for j, impulse in enumerate(impulses):
                    #         print("j, impulses: ", j, th.linalg.vector_norm(impulse))
                else:
                    adjusted = []
                    for j, impulse_vec in enumerate(impulses):
                        proj_imp = th.dot(impulse_vec, accel_unit).item()
                        effective_impulse = impulse_vec - proj_imp * accel_unit if proj_imp > 0 else impulse_vec
                        adjusted.append(th.linalg.vector_norm(effective_impulse))

                        # For debugging
                        # if self.entity.name == "swivel_chair" and link_name == "base_link":
                        #     print("j, impulses: ", j, th.linalg.vector_norm(effective_impulse))
                    
                    if adjusted:
                        static_force += (float(th.sum(th.stack(adjusted))) / max(dt, 1e-8))

                self.raw_forces_from_sim[link_name].append(float(th.sum(th.stack([th.linalg.vector_norm(v) for v in impulses]))) / max(dt, 1e-8))
                self.static_forces[link_name].append(static_force)

            else:
                self.raw_forces_from_sim[link_name].append(0.0)
                self.static_forces[link_name].append(0.0)
            
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

            # Per-link overrides for coefficients / thresholds / scale (do not change object_type)
            strain_threshold = self.strain_threshold
            dynamic_forces_coefficient = self.dynamic_forces_coefficient
            static_forces_coefficient = self.static_forces_coefficient

            try:
                lname = link_name.lower()
                for key, overrides in self.link_thresholds.items():
                    if key in lname:
                        if "damage_threshold" in overrides:
                            strain_threshold = overrides["damage_threshold"]
                        if "dynamic_forces_coefficient" in overrides:
                            dynamic_forces_coefficient = overrides["dynamic_forces_coefficient"]
                        if "static_forces_coefficient" in overrides:
                            static_forces_coefficient = overrides["static_forces_coefficient"]
                        break
            except Exception:
                pass

            strain_due_to_dynamic_force = dynamic_forces_coefficient * dynamic_force
            strain_due_to_static_force = static_forces_coefficient * static_force

            current_strain = strain_due_to_dynamic_force + strain_due_to_static_force
            self.last_strain_by_link[link_name] = current_strain
            damage = max(0.0, (current_strain - strain_threshold)) * self.damage_scale
            link_damages[link_name] = damage
            # For debugging
            # if self.entity.name == "swivel_chair" and link_name == "base_link":
            #     print("strain_due_to_dynamic_force, strain_due_to_static_force, current_strain, strain_threshold, damage: ", strain_due_to_dynamic_force, strain_due_to_static_force, current_strain, strain_threshold, damage)
            #     if damage > 0.0:
            #         breakpoint()

            # Track for next step
            self.prev_link_positions[link_name] = position_current.clone()
            self.prev_link_velocities[link_name] = velocity_current.clone()

            # Record per-step strain proxy
            self.current_env_step_strains[link_name] = current_strain

        return link_damages

    def aggregate_strains_for_env_step(self):
        """Aggregate strains for the current environment step."""
        if not self.current_env_step_strains:
            self.strain_values.append(0.0)
        else:
            self.strain_values.append(max(self.current_env_step_strains.values()))
        self.current_env_step_strains = {}

    def reset_tracking(self):
        """Reset strain tracking state."""
        self.current_env_step_strains = {}
        self.prev_link_positions = {}
        self.prev_link_velocities = {}
        self.last_accel_dir_by_link = {}
        self.last_strain_by_link = {}
        self.strain_values = []
    
    def get_current_env_step_strain(self):
        """Get the current environment step's aggregated strain proxy."""
        if self.last_strain_by_link:
            return float(max(self.last_strain_by_link.values()))
        return 0.0