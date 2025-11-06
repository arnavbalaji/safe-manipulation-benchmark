from __future__ import annotations

from typing import Dict

import numpy as np
import torch as th

import omnigibson as og

from safety_benchmark.damage_evaluators.damage_evaluator import DamageEvaluator
from omnigibson.objects.object_base import BaseObject


class MechanicalDamageEvaluator(DamageEvaluator):
    """
    Mechanical damage evaluator that derives per-step force from raw contact impulses, grouped by external body.

    Method for F_t (per link, per env step):
    - Gather all contacts for the entity across links (object.contact_list()).
    - For each contact CsRawData, identify which body in (body0, body1) is one of this entity's links; the other is
      the external body.
    - Convert contact impulse (vector) to force by dividing by dt.
    - Accumulate force vectors per (link_name, external_body).
    - For each link, set F_t(link) = sum_over_external_bodies(||force_vector||_2). This captures concurrent pressing
      from multiple bodies (e.g., robot and table) within this step without canceling directions.

    The resulting per-link F_t is then combined with optional creep term and threshold/scale to produce damage.
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
        self.instant_coefficient = float(instant_coefficient)
        self.creep_coefficient = float(creep_coefficient)
        self.link_thresholds = {k.lower(): v for k, v in (link_thresholds or {}).items()}
        self.name = "mechanical"

        if object_type == "brittle":
            self.creep_coefficient = 0.0

        # Rolling creep state per link (FIFO tail of past F_t magnitudes)
        self._creep_values_by_link: dict[str, np.ndarray] = {}
        self._creep_terms = np.linspace(0.001, 0.999, 100)

        # Aggregation helpers for telemetry
        self.last_strain_by_link: dict[str, float] = {}
        self.current_env_step_strains: list[float] = []
        self.strain_values: list[float] = []

        # Cache link prim paths to link names for quick lookup
        self._link_path_to_name: dict[str, str] = {
            link.prim_path: name for name, link in self.entity.links.items()
        }

    def _accumulate_force_by_external_body(self) -> dict[str, dict[str, th.Tensor]]:
        """
        Build mapping: link_name -> { external_body_prim_path -> force_vector (torch, world) }
        Force is computed as sum(impulse) / dt over all contacts in this step for that (link, external body).
        """
        # Ensure contacts are available
        contacts = []
        try:
            contacts = self.entity.contact_list()
        except Exception:
            contacts = []

        forces_per_link: dict[str, dict[str, th.Tensor]] = {}
        if not contacts:
            return forces_per_link

        for c in contacts:
            try:
                body0 = str(c.body0)
                body1 = str(c.body1)
                dt = float(c.dt) if float(c.dt) > 0.0 else 1e-8
                imp = th.tensor(tuple(c.impulse), dtype=th.float32)
            except Exception:
                continue

            # Determine which body belongs to this entity and which is external
            if body0 in self._link_path_to_name and body1 not in self._link_path_to_name:
                link_name = self._link_path_to_name[body0]
                other_body = body1
                # impulse direction reported is pair-wise; use as-is for magnitude aggregation
                force_vec = imp / dt
            elif body1 in self._link_path_to_name and body0 not in self._link_path_to_name:
                link_name = self._link_path_to_name[body1]
                other_body = body0
                force_vec = imp / dt
            else:
                # Either self-self or neither belongs to this entity
                continue

            if link_name not in forces_per_link:
                forces_per_link[link_name] = {}
            if other_body not in forces_per_link[link_name]:
                forces_per_link[link_name][other_body] = th.zeros(3, dtype=th.float32)

            # Accumulate net force from the same external body within this step (sum of impulses/dt)
            forces_per_link[link_name][other_body] = forces_per_link[link_name][other_body] + force_vec

        return forces_per_link

    def _update_creep_tail(self, link_name: str, f_t: float) -> float:
        """
        Update and evaluate the creep term for a given link, based on the current step's F_t magnitude.
        Returns the creep contribution (float).
        """
        if self.creep_coefficient <= 0.0:
            return 0.0

        tail = self._creep_values_by_link.get(link_name)
        if tail is None:
            tail = np.array([], dtype=float)
        # Maintain tail length up to len(self._creep_terms)
        if tail.shape[0] >= self._creep_terms.shape[0]:
            tail = np.delete(tail, 0)
        tail = np.append(tail, f_t)
        self._creep_values_by_link[link_name] = tail

        # Dot-product with precomputed terms
        return self.creep_coefficient * float(
            np.dot(tail, self._creep_terms[: tail.shape[0]])
        )

    def generate_damage(self) -> Dict[str, float]:
        link_damages: Dict[str, float] = {}

        # Build per-link, per-external-body force vectors
        # Many calls to torch operations here do not need autograd
        with th.no_grad():
            forces_per_link = self._accumulate_force_by_external_body()

            self.last_strain_by_link = {}
            self.current_env_step_strains = []

            # For links with forces, compute F_t(link) as sum of magnitudes across external bodies
            for link_name, external_forces in forces_per_link.items():
                if not external_forces:
                    continue
                # Sum magnitudes over external bodies (not vector sum, to avoid cancellation)
                f_t = 0.0
                for force_vec in external_forces.values():
                    try:
                        f_t += float(th.linalg.vector_norm(force_vec).item())
                    except Exception:
                        f_t += 0.0

                inst_term = self.instant_coefficient * f_t
                creep_term = self._update_creep_tail(link_name, f_t)
                strain = inst_term + creep_term

                self.last_strain_by_link[link_name] = strain
                self.current_env_step_strains.append(strain)

                # Damage calculation uses global threshold/scale (per-link thresholds optional in future)
                damage = max(0.0, (strain - self.damage_threshold)) * self.scale
                link_damages[link_name] = damage

        return link_damages

    def aggregate_strains_for_env_step(self):
        if not self.current_env_step_strains:
            self.strain_values.append(0.0)
        else:
            self.strain_values.append(max(self.current_env_step_strains))
        self.current_env_step_strains = []

    def reset_tracking(self):
        self.current_env_step_strains = []
        self.last_strain_by_link = {}
        self._creep_values_by_link = {}
        self.strain_values = []

    def get_current_env_step_strain(self) -> float:
        if self.last_strain_by_link:
            return float(max(self.last_strain_by_link.values()))
        return 0.0


