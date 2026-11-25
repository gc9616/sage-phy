"""Gate G4: the simulator must reproduce elementary Coulomb-friction predictions
before any experiment runs. All checks are in clean mode unless stated."""
from __future__ import annotations

from dataclasses import dataclass

from . import closed_form as cf


@dataclass
class GateResult:
    check: str
    passed: bool
    detail: str


def check_slide(trials: list[dict], tol: float = 0.15, g: float = cf.G) -> GateResult:
    """trials: {mu_e, v, slide} with no damping. Pass if every simulated slide is
    within `tol` (relative) of v^2 / (2 mu_e g)."""
    worst = 0.0
    for t in trials:
        pred = cf.slide_distance(t["v"], t["mu_e"], 0.0, g)
        worst = max(worst, abs(t["slide"] - pred) / pred)
    return GateResult("slide distance", worst <= tol, f"worst relative error {worst:.3f} (limit {tol})")


def check_lift_boundary(sim_boundary_F: float, mass: float, mu_e: float, accel: float,
                        grid_step: float, g: float = cf.G) -> GateResult:
    pred = cf.min_grip_force(mass, mu_e, accel, g)
    err = abs(sim_boundary_F - pred)
    return GateResult("lift boundary", err <= grid_step,
                      f"simulated {sim_boundary_F:.2f} N vs predicted {pred:.2f} N (step {grid_step:.2f})")


def check_mass_control(outcomes_by_episode: list[list[bool]], min_share: float = 0.9) -> GateResult:
    """Each inner list: push outcomes for one (object, friction, seed) across the
    mass levels. Pass if outcomes are identical in at least min_share of them."""
    same = sum(1 for o in outcomes_by_episode if len(set(o)) == 1)
    share = same / len(outcomes_by_episode)
    return GateResult("mass control on push", share >= min_share, f"identical in {share:.0%}")


def check_shipped_damping(trials: list[dict], damping: float, slack: float = 0.01) -> GateResult:
    """trials: {v, slide} under shipped damping. Pass if slide <= v/c + slack."""
    worst = max(t["slide"] - cf.damped_slide_bound(t["v"], damping) for t in trials)
    return GateResult("shipped damping", worst <= slack, f"worst excess over v/c {worst*100:.2f} cm")
