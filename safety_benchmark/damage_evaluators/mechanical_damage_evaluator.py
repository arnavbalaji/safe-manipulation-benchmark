from safety_benchmark.damage_evaluators.damage_evaluator import DamageEvaluator
from omnigibson.objects.object_base import BaseObject
import torch as th
from typing import Dict
import omnigibson as og
import numpy as np

class MechanicalDamageEvaluator(DamageEvaluator):
    """
    Evaluates damage based on mechanical forces from OmniGibson.
    """

    def __init__(
        self,
        entity: BaseObject,
        damage_threshold: float = 0.0,
        scale: float = 1.0,
        instant_coefficient: float = 1.0,
        creep_coefficient: float = 1.0,
        link_thresholds: Dict[str, dict] | None = None,
        object_type: str | None = None,
        **kwargs,
    ):
        super().__init__(entity, damage_threshold, scale)
        self.entity = entity
        self.instant_coefficient = instant_coefficient
        self.creep_coefficient = creep_coefficient
        self.link_thresholds = {k.lower(): v for k, v in (link_thresholds or {}).items()}
        self.name = "mechanical"

        # Running state for impact / sustained computations
        self.prev_link_positions: dict[str, th.Tensor] = {}
        self.prev_link_velocities: dict[str, th.Tensor] = {}
        self.last_accel_dir_by_link: dict[str, th.Tensor | None] = {}

        if object_type == "brittle":
            self.creep_coefficient = 0.0

        # Creep accumulation (simple exponentially-decaying tail via dot with precomputed terms)
        self.creep_values = np.array([], dtype=float)
        self.creep_terms = np.linspace(0.001, 0.5, 50)

        # Aggregation helpers
        self.current_env_step_strains: list[float] = []
        self.strain_values: list[float] = []
        # Per-link last strain values for the most recent compute
        self.last_strain_by_link: dict[str, float] = {}

    def generate_damage(self) -> Dict[str, float]:
        link_damages: Dict[str, float] = {}
        epsilon = 1e-8
        dt = og.sim.get_physics_dt()

        for link_name, link in self.entity.links.items():
            try:
                contacts_now = link.contact_list()
            except Exception:
                contacts_now = []

            # Impact via finite-differenced acceleration
            position_current, _ = link.get_position_orientation()
            has_previous = link_name in self.prev_link_positions
            position_previous = self.prev_link_positions.get(link_name)
            velocity_previous = self.prev_link_velocities.get(link_name, th.zeros(3))

            if has_previous and position_previous is not None:
                displacement = position_current - position_previous
                velocity_current = displacement / max(dt, 1e-8) # dx/dt
                delta_velocity = velocity_current - velocity_previous
                acceleration = delta_velocity / max(dt, 1e-8) # dv/dt
                impact_magnitude = th.linalg.vector_norm(acceleration).item() * float(getattr(link, "mass", 1.0))

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

            # Contact impulses
            impulses: list[th.Tensor] = []
            try:
                for c in contacts_now:
                    try:
                        impulses.append(th.tensor(c.impulse.tolist(), dtype=th.float32))
                    except Exception:
                        pass
            except Exception:
                pass

            accel_unit = self.last_accel_dir_by_link.get(link_name, None)
            # Only count impact_magnitude if the link is in contact with any object
            F_t = impact_magnitude if contacts_now else 0.0
            if impulses:
                if accel_unit is None:
                    F_t += (float(th.sum(th.stack([th.linalg.vector_norm(v) for v in impulses]))) / max(dt, 1e-8))
                else:
                    adjusted = []
                    for impulse_vec in impulses:
                        proj_imp = th.dot(impulse_vec, accel_unit).item()
                        effective_impulse = impulse_vec - proj_imp * accel_unit if proj_imp > 0 else impulse_vec
                        adjusted.append(th.linalg.vector_norm(effective_impulse))
                    if adjusted:
                        F_t += (float(th.sum(th.stack(adjusted))) / max(dt, 1e-8))
            
            if F_t <= 0.1:
                self.creep_values = np.array([], dtype=float)
            skip_creep = len(self.creep_values) == 0

            # Update creep tail
            while self.creep_values.shape[0] >= self.creep_terms.shape[0]:
                self.creep_values = np.delete(self.creep_values, 0)
            self.creep_values = np.append(self.creep_values, F_t)

            # Per-link overrides for coefficients / thresholds / scale (do not change object_type)
            local_damage_threshold = self.damage_threshold
            local_scale = self.scale
            local_inst_coef = self.instant_coefficient
            local_creep_coef = self.creep_coefficient

            try:
                lname = link_name.lower()
                for key, overrides in self.link_thresholds.items():
                    if key in lname:
                        if "damage_threshold" in overrides:
                            local_damage_threshold = overrides["damage_threshold"]
                        if "scale" in overrides:
                            local_scale = overrides["scale"]
                        if "instant_coefficient" in overrides:
                            local_inst_coef = overrides["instant_coefficient"]
                        if "creep_coefficient" in overrides:
                            local_creep_coef = overrides["creep_coefficient"]
                        break
            except Exception:
                pass

            inst_term = local_inst_coef * F_t
            if local_creep_coef > 0.0 and not skip_creep:
                creep_term = local_creep_coef * float(np.dot(self.creep_values, self.creep_terms[: self.creep_values.shape[0]]))
            else:
                creep_term = 0.0

            current_strain = inst_term + creep_term
            # Track per-link last strain
            self.last_strain_by_link[link_name] = current_strain
            damage = max(0.0, (current_strain - local_damage_threshold)) * local_scale
            link_damages[link_name] = damage

            # Track for next step
            self.prev_link_positions[link_name] = position_current.clone()
            self.prev_link_velocities[link_name] = velocity_current.clone()

            # Record per-step strain proxy
            self.current_env_step_strains.append(current_strain)

        return link_damages

    def aggregate_strains_for_env_step(self):
        """Aggregate strains for the current environment step."""
        if not self.current_env_step_strains:
            self.strain_values.append(0.0)
        else:
            self.strain_values.append(max(self.current_env_step_strains))
        self.current_env_step_strains = []

    def reset_tracking(self):
        """Reset strain tracking state."""
        self.current_env_step_strains = []
        self.prev_link_positions = {}
        self.prev_link_velocities = {}
        self.last_accel_dir_by_link = {}
        self.creep_values = np.array([], dtype=float)
        self.last_strain_by_link = {}
        self.strain_values = []
    
    def get_current_env_step_strain(self):
        """Get the current environment step's aggregated strain proxy."""
        if self.last_strain_by_link:
            return float(max(self.last_strain_by_link.values()))
        return 0.0