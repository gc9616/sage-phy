"""Closed-form Coulomb-friction predictions used by gate G4 and the analytic backend.

Push: the object moves with the pusher over the stroke s, is released at speed v,
then slides. Without damping  d = s + v^2 / (2 mu_e g).
With linear damping c the slide obeys  dv/dt = -mu_e g - c v,  whose stopping
distance is  v/c - (a/c^2) ln(1 + v c / a),  a = mu_e g,  which is always < v/c.

Lift: a parallel-jaw grip with force F per finger holds iff 2 mu_e F >= m (g + a).
"""
from __future__ import annotations

import math

G = 9.81


def effective_mu(mu_object: float, mu_other: float) -> float:
    """PhysX 'average' combine mode."""
    return 0.5 * (mu_object + mu_other)


def object_mu_for_effective(mu_e: float, mu_other: float) -> float:
    """Object friction that gives effective friction mu_e against a surface of mu_other."""
    mu = 2.0 * mu_e - mu_other
    if mu < 0:
        raise ValueError(
            f"effective mu {mu_e} is unreachable against a surface with mu {mu_other} "
            "under 'average' combine; lower the other material too"
        )
    return mu


def slide_distance(v: float, mu_e: float, damping: float = 0.0, g: float = G) -> float:
    """Distance slid after release at speed v."""
    if v <= 0:
        return 0.0
    a = mu_e * g
    if damping <= 0:
        return math.inf if a <= 0 else v * v / (2.0 * a)
    if a <= 0:
        return v / damping
    c = damping
    return v / c - (a / (c * c)) * math.log1p(v * c / a)


def push_displacement(s: float, v: float, mu_e: float, damping: float = 0.0,
                      max_velocity: float | None = None, g: float = G) -> float:
    """Final displacement d along the push axis."""
    if max_velocity is not None:
        v = min(v, max_velocity)
    return s + slide_distance(v, mu_e, damping, g)


def damped_slide_bound(v: float, damping: float) -> float:
    """Upper bound v/c on the slide under linear damping, whatever the friction."""
    return v / damping


def min_grip_force(mass: float, mu_e: float, accel: float, g: float = G) -> float:
    """Smallest per-finger force that holds the object: m (g + a) / (2 mu_e)."""
    return mass * (g + accel) / (2.0 * mu_e)


def lift_holds(force: float, mass: float, mu_e: float, accel: float, g: float = G) -> bool:
    return 2.0 * mu_e * force >= mass * (g + accel)


def force_margin(force: float, mass: float, mu_e: float, accel: float, g: float = G) -> float:
    """Tuned grip force divided by the minimum force the closed form requires."""
    return force / min_grip_force(mass, mu_e, accel, g)
