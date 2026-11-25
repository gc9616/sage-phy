"""Outcome scoring for the two primitives. Fixed rules, no AI judge."""
from __future__ import annotations

UNDERSHOOT, OVERSHOOT, TIP_OVER = "undershoot", "overshoot", "tip_over"
SLIP, DROP, GRASP_MISS = "slip", "drop", "grasp_miss"


def score_push(d: float, tilt_deg: float, cfg: dict) -> tuple[bool, str | None]:
    """Success iff d is inside the band and the object did not tip beyond the limit."""
    lo, hi = cfg["push"]["band"]
    if tilt_deg > cfg["push"]["max_tilt_deg"]:
        return False, TIP_OVER
    if d < lo:
        return False, UNDERSHOOT
    if d > hi:
        return False, OVERSHOOT
    return True, None


def score_lift(grasped: bool, held: bool, slip_m: float, cfg: dict) -> tuple[bool, str | None]:
    """Success iff the object is still held after the hold with less than max slip."""
    if not grasped:
        return False, GRASP_MISS
    if not held:
        return False, DROP
    if slip_m >= cfg["lift"]["max_slip"]:
        return False, SLIP
    return True, None
