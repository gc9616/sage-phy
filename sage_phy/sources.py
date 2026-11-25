"""The seven property sources of E2 and how each resolves an object's physics.

Object records (see configs/objects.example.json) carry:
  id, scene, prim_path, split ('train'|'test'), category, material_class,
  layout_mass, collision_volume, mass_range [p5, median, p95],
  friction_range [lo, median, hi]
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

LAYOUT, DEFAULT_DENSITY, ABO_MEDIAN, ABO_RANGE = "layout", "default_density", "abo_median", "abo_range"
SHIPPED, FIXED, MAT_MEDIAN, MAT_RANGE = "shipped", "fixed", "material_median", "material_range"


@dataclass(frozen=True)
class PropertySource:
    name: str
    mass: str
    friction: str
    shipped_damping: bool
    role: str
    tier_push: int
    tier_lift: int = 3


SOURCES: dict[str, PropertySource] = {s.name: s for s in [
    PropertySource("shipped", LAYOUT, SHIPPED, True, "release", 1),
    PropertySource("simple_defaults", DEFAULT_DENSITY, FIXED, False, "baseline", 1),
    PropertySource("grounded_median", ABO_MEDIAN, MAT_MEDIAN, False, "single estimate", 1),
    PropertySource("grounded_range", ABO_RANGE, MAT_RANGE, False, "proposed", 1),
    PropertySource("ablate_friction", LAYOUT, MAT_MEDIAN, True, "ablation", 2),
    PropertySource("ablate_damping", LAYOUT, SHIPPED, False, "ablation", 2),
    PropertySource("ablate_mass", ABO_MEDIAN, SHIPPED, True, "ablation", 2),
]}

COMPARISON = ["shipped", "simple_defaults", "grounded_median", "grounded_range"]
ABLATIONS = ["ablate_friction", "ablate_damping", "ablate_mass"]


@dataclass(frozen=True)
class Props:
    mass: float
    friction: float
    linear_damping: float
    angular_damping: float
    max_linear_velocity: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def shipped_friction(cfg: dict) -> float:
    sh = cfg["shipped"]
    return sh["door_friction"] if sh.get("f0_override") else sh["friction"]


def sample_mass(obj: dict, rng: random.Random) -> float:
    """Log-uniform over the plausible range [p5, p95] (mass spans orders of magnitude)."""
    lo, _, hi = obj["mass_range"]
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample_friction(obj: dict, rng: random.Random) -> float:
    lo, _, hi = obj["friction_range"]
    return rng.uniform(lo, hi)


def resolve(source: PropertySource | str, obj: dict, cfg: dict,
            rng: random.Random | None = None) -> Props:
    """Physics properties of `obj` under `source`. Range sources need `rng`."""
    if isinstance(source, str):
        source = SOURCES[source]
    if source.mass == LAYOUT:
        m = obj["layout_mass"]
    elif source.mass == DEFAULT_DENSITY:
        m = cfg["defaults"]["density"] * obj["collision_volume"]
    elif source.mass == ABO_MEDIAN:
        m = obj["mass_range"][1]
    elif source.mass == ABO_RANGE:
        m = sample_mass(obj, _need(rng))
    else:
        raise ValueError(source.mass)

    if source.friction == SHIPPED:
        mu = shipped_friction(cfg)
    elif source.friction == FIXED:
        mu = cfg["defaults"]["friction"]
    elif source.friction == MAT_MEDIAN:
        mu = obj["friction_range"][1]
    elif source.friction == MAT_RANGE:
        mu = sample_friction(obj, _need(rng))
    else:
        raise ValueError(source.friction)

    sh = cfg["shipped"]
    if source.shipped_damping:
        return Props(m, mu, sh["linear_damping"], sh["angular_damping"], sh["max_linear_velocity"])
    return Props(m, mu, 0.0, 0.0, None)


def _need(rng):
    if rng is None:
        raise ValueError("range sources need an rng")
    return rng
