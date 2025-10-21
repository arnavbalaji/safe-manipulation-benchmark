from typing import Dict

from omnigibson import object_states
from safety_benchmark.damage_evaluators.damage_evaluator import DamageEvaluator
from omnigibson.objects.object_base import BaseObject


class ElectricalDamageEvaluator(DamageEvaluator):
    """
    Compute per-link electrical damage from water particle contacts.

    Damage model (simple):
        damage(link) = max(0, particles(link) - threshold_link) * scale_link

    - Particles are counted via ContactParticles state per link.
    - Water system is auto-detected once (by name or common fallbacks).
    - Optional per-link overrides via `link_thresholds`: a mapping of substrings
      (case-insensitive) to dicts with optional keys: {"damage_threshold", "scale"}.
    """

    def __init__(
        self,
        entity: BaseObject,
        damage_threshold: float,
        scale: float,
        water_system_name: str = "water",
        link_thresholds: Dict[str, dict] | None = None,
    ) -> None:
        super().__init__(entity, damage_threshold, scale)
        self.entity = entity
        self.name = "electrical"
        self.damage_threshold = float(damage_threshold)
        self.scale = float(scale)
        self.water_system_name = water_system_name
        self.link_thresholds = {k.lower(): v for k, v in (link_thresholds or {}).items()}

        self._water_system = None
        self._initialized = False

    def _ensure_water_system(self) -> None:
        if self._initialized:
            return
        scene = getattr(self.entity, "scene", None)
        if scene is None:
            self._initialized = True
            return

        candidate_names = [self.water_system_name, "water", "sludge", "fluid"]
        for name in candidate_names:
            try:
                if scene.is_physical_particle_system(name):
                    self._water_system = scene.get_system(name)
                    break
            except Exception:
                # If OG raises on lookup, skip to next candidate
                continue
        self._initialized = True

    def _best_link_overrides(self, link_name: str) -> tuple[float, float]:
        """Return (threshold, scale) for this link, applying optional overrides.

        Matching strategy: choose the longest substring key contained in the
        lowercased link name; fall back to global defaults if none match.
        """
        threshold = self.damage_threshold
        scale = self.scale
        if not self.link_thresholds:
            return threshold, scale

        lname = link_name.lower()
        matches = [k for k in self.link_thresholds.keys() if k in lname]
        if not matches:
            return threshold, scale

        # Prefer the longest matching key for specificity
        longest_len = max(len(k) for k in matches)
        candidates = [k for k in matches if len(k) == longest_len]
        chosen_key = candidates[0]
        override = self.link_thresholds.get(chosen_key, {})
        if isinstance(override, dict):
            if "damage_threshold" in override:
                try:
                    threshold = float(override["damage_threshold"])
                except Exception:
                    pass
            if "scale" in override:
                try:
                    scale = float(override["scale"])
                except Exception:
                    pass
        return threshold, scale

    def _count_particles_per_link(self) -> Dict[str, int]:
        """Return a mapping link_name -> number of contacting water particles."""
        if self._water_system is None:
            return {name: 0 for name in self.entity.links.keys()}
        results: Dict[str, int] = {}
        for link_name, link in self.entity.links.items():
            count = 0
            try:
                particles = self.entity.states[object_states.ContactParticles].get_value(
                    system=self._water_system, link=link
                )
                count = len(particles)
            except Exception:
                count = 0
            results[link_name] = count
        return results

    def generate_damage(self) -> Dict[str, float]:
        """Compute per-link damage from water contacts.

        Returns a dict mapping link names to scalar damage values.
        """
        self._ensure_water_system()
        if self._water_system is None:
            return {name: 0.0 for name in self.entity.links.keys()}

        counts = self._count_particles_per_link()
        damages: Dict[str, float] = {}
        for link_name, particle_count in counts.items():
            thr, scl = self._best_link_overrides(link_name)
            damages[link_name] = max(0.0, float(particle_count) - thr) * scl
        return damages

    def reset_tracking(self) -> None:
        """No persistent state to reset for this evaluator."""
        # Nothing to reset; kept for interface compatibility
        return None

    def get_contact_summary(self) -> Dict[str, object]:
        """Provide a lightweight summary of current contact counts per link."""
        self._ensure_water_system()
        if self._water_system is None:
            return {"status": "no_water_system", "total_contact": 0, "link_details": {}}

        counts = self._count_particles_per_link()
        total = int(sum(counts.values()))
        link_details = {name: {"particle_count": int(c)} for name, c in counts.items()}
        return {"status": ("active" if total > 0 else "no_contact"), "total_contact": total, "link_details": link_details}