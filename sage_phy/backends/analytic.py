"""Closed-form backend: ideal Coulomb friction, no rolling, no tipping."""
from __future__ import annotations

from .. import closed_form as cf


class AnalyticBackend:
    name = "analytic"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.g = cfg["contact"]["gravity"]

    def run(self, ep: dict) -> dict:
        mu_e = cf.effective_mu(ep["friction"], ep["contact_friction"])
        if ep["primitive"] == "push":
            p = ep["params"]
            d = cf.push_displacement(p["s"], p["v"], mu_e, ep["linear_damping"],
                                     ep["max_linear_velocity"], self.g)
            return {"d": d, "tilt_deg": 0.0, "mu_e": mu_e}
        accel = self.cfg["lift"]["accel"]
        F = ep["params"]["F"]
        held = cf.lift_holds(F, ep["mass"], mu_e, accel, self.g)
        return {"grasped": True, "held": held, "slip_m": 0.0, "mu_e": mu_e,
                "force_margin": cf.force_margin(F, ep["mass"], mu_e, accel, self.g)}
