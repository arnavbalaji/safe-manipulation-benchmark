from safety_benchmark.damage_evaluators.damage_evaluator import DamageEvaluator
from omnigibson.objects.object_base import BaseObject
import torch as th
from typing import Dict

class MechanicalDamageEvaluator(DamageEvaluator):
    """
    Evaluates damage based on mechanical forces from OmniGibson.
    """
    def __init__(self, entity: BaseObject, damage_threshold: float = 0.0, scale: float = 0.0, link_thresholds: Dict[str, dict] = None,
                 impact_threshold: float = None,
                 crushing_threshold: float = None, crushing_scale: float = None,  # legacy names (back-compat)
                 impact_scale: float = None,
                 crushing_memory: float = 0.9,  # legacy name (back-compat)
                 sustained_threshold: float = None, sustained_scale: float = None, sustained_memory: float = None,
                 min_sustained_threshold: float = None,
                 **kwargs):
        # Keep base init for compatibility; mechanical damage ignores the generic threshold / scale
        super().__init__(entity, damage_threshold, scale)
        self.entity = entity
        self.name = "mechanical"
        
        # Aggregates across env steps and per-step tracking
        self.force_values = []
        self.current_env_step_forces = []

        # Per-link overrides
        self.link_thresholds = {k.lower(): v for k, v in (link_thresholds or {}).items()}
        
        # Running state for impact / sustained computations
        self.prev_link_positions = {}
        self.prev_link_velocities = {}
        self.last_accel_dir_by_link = {}
        
        # Separate thresholds / scales for impact vs. sustained (support legacy names)
        self.impact_threshold = 0.0 if impact_threshold is None else float(impact_threshold)
        self.impact_scale = 1.0 if impact_scale is None else float(impact_scale)
        
        # Prefer new sustained_* if provided; else fall back to legacy crushing_*
        chosen_sustained_threshold = sustained_threshold if sustained_threshold is not None else crushing_threshold
        chosen_sustained_scale = sustained_scale if sustained_scale is not None else crushing_scale
        chosen_sustained_memory = sustained_memory if sustained_memory is not None else crushing_memory

        self.sustained_threshold = 0.0 if chosen_sustained_threshold is None else float(chosen_sustained_threshold)
        self.sustained_scale = 1.0 if chosen_sustained_scale is None else float(chosen_sustained_scale)
        self.sustained_memory = 0.99 if chosen_sustained_memory is None else float(chosen_sustained_memory)
        self.min_sustained_threshold = (0.1 * self.sustained_threshold) if (min_sustained_threshold is None) else float(min_sustained_threshold)
        
        # Sustained aggregation using exponential memory: C(t) = S(t) + lambda * C(t-1)
        self._sustained_running_total = {}
        self._sustained_reset_next = {}
        
        # Exposed per-link last-step values for logging / plotting
        self.impact_forces_by_link = {}
        self.sustained_forces_by_link = {}

    def generate_damage(self) -> Dict[str, float]:
        """
        Generate damage values using the impact + sustained damage model:
            d_mech = d_impact + d_sustained
            d_impact = (F_impact - beta_i)^+ * alpha_i,
                where F_impact = || v_t - v_{t-1} ||  (magnitude of change in velocity per physics step).
            d_sustained = (C(t) - beta_s)^+ * alpha_s,
                where S(t) = sum over contacts i of || Fi - max(0, Fi·â) â || with â the unit acceleration direction,
                and C(t) = S(t) + lambda * C(t-1).

        Velocity approximation uses finite difference: v_t = x_t - x_{t-1}.
        """
        link_damages = {}
        max_force_this_timestep = 0.0
        epsilon = 1e-8
        
        for link_name, link in self.entity.links.items():
            contacts_now = link.contact_list()
            
            # Impact term and acceleration direction
            impact_magnitude = 0.0
            position_current, _ = link.get_position_orientation()
            has_previous = link_name in self.prev_link_positions
            position_previous = self.prev_link_positions.get(link_name, None)
            velocity_previous = self.prev_link_velocities.get(link_name, th.zeros(3))

            if has_previous:
                displacement = position_current - position_previous
                velocity_current = displacement
                delta_velocity = velocity_current - velocity_previous
                
                # Get physics timestep to convert velocity change to acceleration
                import omnigibson as og
                dt = og.sim.get_physics_dt()
                acceleration = delta_velocity / dt
                impact_magnitude = th.linalg.vector_norm(acceleration).item()

                delta_velocity_norm = th.linalg.vector_norm(delta_velocity).item()
                if delta_velocity_norm > epsilon:
                    accel_unit = delta_velocity / delta_velocity_norm
                    self.last_accel_dir_by_link[link_name] = accel_unit.clone()
                else:
                    self.last_accel_dir_by_link[link_name] = None
            else:
                velocity_current = th.zeros(3)
                impact_magnitude = 0.0
                self.last_accel_dir_by_link[link_name] = None

            impact_magnitude *= link.mass
            
            # Per-link overrides for impact
            active_impact_threshold = self.impact_threshold
            active_impact_scale = self.impact_scale
            if self.link_thresholds:
                lname = link_name.lower()
                matches = [k for k in self.link_thresholds.keys() if k in lname]
                if matches:
                    max_len = max(len(k) for k in matches)
                    longest_matches = [k for k in matches if len(k) == max_len]
                    best = None
                    for k in self.link_thresholds.keys():
                        if k.lower() in longest_matches:
                            best = k.lower()
                            break
                    override = self.link_thresholds[best]
                    if 'impact_threshold' in override:
                        active_impact_threshold = float(override['impact_threshold'])
                    if 'impact_scale' in override:
                        active_impact_scale = float(override['impact_scale'])

            impact_damage = max(0.0, (impact_magnitude - active_impact_threshold) * active_impact_scale)
            
            # Sustained term
            impulses = []
            normals = []
            try:
                if len(contacts_now) > 0:
                    for c in contacts_now:
                        try:
                            impulse = th.tensor(c.impulse.tolist(), dtype=th.float32)
                            impulses.append(impulse)
                        except Exception:
                            pass
                        try:
                            normals.append(th.tensor(c.normal.tolist(), dtype=th.float32))
                        except Exception:
                            pass
            except Exception:
                pass
            
            # if link_name == "base_link":
            #     breakpoint()
            
            accel_unit = self.last_accel_dir_by_link.get(link_name, None)
            sustained_step_value = 0.0
            if impulses:
                if accel_unit is None:
                    # sustained_step_value = float(th.sum(th.stack([th.linalg.vector_norm(v) for v in impulses]))) + float(th.sum(th.stack([th.linalg.vector_norm(v) for v in normals])))
                    sustained_step_value = float(th.sum(th.stack([th.linalg.vector_norm(v) for v in impulses])))
                else:
                    adjusted = []
                    for impulse_vec in impulses:
                        proj_imp = th.dot(impulse_vec, accel_unit).item()
                        effective_impulse = impulse_vec - proj_imp * accel_unit if proj_imp > 0 else impulse_vec
                        adjusted.append(th.linalg.vector_norm(effective_impulse))
                    # for normal_vec in normals:
                    #     up_dir = th.tensor([0.0, 0.0, 1.0], dtype=normal_vec.dtype, device=normal_vec.device)
                    #     up_comp = th.dot(normal_vec, up_dir).item()
                    #     if up_comp > 0.0:
                    #         # Subtract mg from the upward component, but floor at 0
                    #         new_up = max(0.0, up_comp - (link.mass * 9.81))
                    #         delta_up = up_comp - new_up
                    #         normal_vec = normal_vec - delta_up * up_dir
                    #     proj_norm = th.dot(normal_vec, accel_unit).item()
                    #     effective_normal = normal_vec - proj_norm * accel_unit if proj_norm > 0 else normal_vec
                    #     adjusted.append(th.linalg.vector_norm(effective_normal))
                    sustained_step_value = float(th.sum(th.stack(adjusted))) if adjusted else 0.0
            
            # Running total C(t) = S_eff(t) + lambda * C(t-1), where S_eff applies min threshold
            running_total_prev = self._sustained_running_total.get(link_name, 0.0)
            if self._sustained_reset_next.get(link_name, False):
                running_total_prev = 0.0
                self._sustained_reset_next[link_name] = False
            sustained_effective = sustained_step_value if (sustained_step_value >= self.min_sustained_threshold) else 0.0
            running_total_curr = float(sustained_effective) + self.sustained_memory * float(running_total_prev)
            self._sustained_running_total[link_name] = running_total_curr
            
            # Record values for external logging
            impact_list = self.impact_forces_by_link.get(link_name, [])
            impact_list.append(impact_magnitude)
            self.impact_forces_by_link[link_name] = impact_list

            sustained_list = self.sustained_forces_by_link.get(link_name, [])
            sustained_list.append(sustained_step_value)
            self.sustained_forces_by_link[link_name] = sustained_list
            
            # Sustained damage using running total
            active_sustained_threshold = self.sustained_threshold
            active_sustained_scale = self.sustained_scale
            if self.link_thresholds:
                lname = link_name.lower()
                matches = [k for k in self.link_thresholds.keys() if k in lname]
                if matches:
                    max_len = max(len(k) for k in matches)
                    longest_matches = [k for k in matches if len(k) == max_len]
                    best = None
                    for k in self.link_thresholds.keys():
                        if k.lower() in longest_matches:
                            best = k.lower()
                            break
                    override = self.link_thresholds[best]
                    # Prefer new keys; keep support for legacy ones
                    if 'sustained_threshold' in override:
                        active_sustained_threshold = float(override['sustained_threshold'])
                    elif 'crushing_threshold' in override:
                        active_sustained_threshold = float(override['crushing_threshold'])
                    if 'sustained_scale' in override:
                        active_sustained_scale = float(override['sustained_scale'])
                    elif 'crushing_scale' in override:
                        active_sustained_scale = float(override['crushing_scale'])
            
            sustained_damage = max(0.0, (running_total_curr - active_sustained_threshold)) * active_sustained_scale
            
            # If running total exceeds threshold, schedule reset for next timestep
            if running_total_curr > active_sustained_threshold:
                self._sustained_reset_next[link_name] = True
            
            # Combine
            link_damages[link_name] = impact_damage + sustained_damage
            
            # Max force summary
            sum_impulses = float(th.sum(th.stack([th.linalg.vector_norm(v) for v in impulses])).item()) if impulses else 0.0
            sum_normals = float(th.sum(th.stack([th.linalg.vector_norm(n) for n in normals])).item()) if normals else 0.0
            link_force_this_timestep = sum_impulses + sum_normals
            max_force_this_timestep = max(max_force_this_timestep, impact_magnitude, running_total_curr, link_force_this_timestep)
            
            # Update stored state for next call
            self.prev_link_positions[link_name] = position_current.clone()
            self.prev_link_velocities[link_name] = velocity_current.clone()

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
            total_force = max(self.current_env_step_forces)
            self.force_values.append(total_force)
        
        # Reset for next environment step
        self.current_env_step_forces = []

    def reset_tracking(self):
        """Reset force tracking state."""
        self.current_env_step_forces = []
        self.prev_link_positions = {}
        self.prev_link_velocities = {}
        self.last_accel_dir_by_link = {}
        self._sustained_running_total = {}
        self._sustained_reset_next = {}
        self.impact_forces_by_link = {}
        self.sustained_forces_by_link = {}
    
    def get_current_env_step_force(self):
        """
        Get the current environment step's aggregated force.
        """
        if not self.current_env_step_forces:
            return 0.0
        return max(self.current_env_step_forces)
    
    @property
    def aggregated_force_values(self):
        """
        Property to access the aggregated force values.
        """
        return self.force_values