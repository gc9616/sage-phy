"""Episode generation for F0, E1 and E2, and controller selection.

An episode is a flat dict that fully determines one simulation run. Its `id`
is a hash of its contents, so runs are resumable and reproducible.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from statistics import mean

from .sources import COMPARISON, SOURCES, Props, resolve, sample_friction, sample_mass, shipped_friction


def episode(experiment: str, primitive: str, obj: dict, condition: str, props: Props,
            contact_friction: float, params: dict, seed: int, loader: str = "grounded",
            **extra) -> dict:
    ep = {"experiment": experiment, "primitive": primitive, "object_id": obj["id"],
          "scene": obj.get("scene"), "layout": obj.get("layout"), "prim_path": obj.get("prim_path"),
          "support_prim": obj.get("support_prim"), "push_axis": obj.get("push_axis"),
          "condition": condition, "loader": loader, **props.to_dict(),
          "contact_friction": contact_friction, "params": params, "seed": seed, **extra}
    ep["id"] = hashlib.sha1(json.dumps(ep, sort_keys=True).encode()).hexdigest()[:16]
    return ep


def geomspace(lo: float, hi: float, n: int) -> list[float]:
    if n == 1:
        return [lo]
    r = (hi / lo) ** (1 / (n - 1))
    return [lo * r ** i for i in range(n)]


def _seeded(*key) -> random.Random:
    return random.Random(int(hashlib.sha1(repr(key).encode()).hexdigest()[:12], 16))


def push_grid(cfg: dict) -> list[dict]:
    return [{"s": s, "v": v} for s in cfg["e2"]["push_s"] for v in cfg["e2"]["push_v"]]


def lift_grid(cfg: dict) -> list[dict]:
    e2 = cfg["e2"]
    return [{"F": F} for F in geomspace(e2["lift_F_min"], e2["lift_F_max"], e2["lift_F_n"])]


def _clean(m: float, mu: float) -> Props:
    return Props(m, mu, 0.0, 0.0, None)


def _shipped_damped(cfg, m: float, mu: float) -> Props:
    sh = cfg["shipped"]
    return Props(m, mu, sh["linear_damping"], sh["angular_damping"], sh["max_linear_velocity"],
                 sh["max_angular_velocity"])


# ---------------------------------------------------------------- F0 (audit)
def f0_episodes(obj: dict, cfg: dict, params: dict | None = None, seeds: int = 1) -> list[dict]:
    """Four settings: (a) loader as shipped; (b) bound mu=0.5; (c) bound mu=1e20;
    (d) shipped friction with damping and velocity cap removed."""
    params = params or {"s": 0.10, "v": 0.5}
    # Contact material matches the object so the combine mode cannot hide the value.
    m, mu_sh = obj["layout_mass"], cfg["shipped"]["friction"]
    settings = [
        ("a_shipped", _shipped_damped(cfg, m, mu_sh), mu_sh, "sage"),
        ("b_bound_0.5", _shipped_damped(cfg, m, 0.5), 0.5, "grounded"),
        ("c_bound_1e20", _shipped_damped(cfg, m, 1e20), 1e20, "grounded"),
        ("d_no_damping", _clean(m, mu_sh), mu_sh, "sage_no_damping"),
    ]
    return [episode("f0", "push", obj, name, p, c, params, seed, loader=ld)
            for name, p, c, ld in settings for seed in range(seeds)]


def classify_f0(d: dict[str, float], tol: float = 0.01) -> str:
    """'applied' if (a) behaves like (c), 'ignored' if like (b), 'indistinguishable'
    if (b) and (c) differ by less than tol (then damping, not friction, dominates)."""
    a, b, c = d["a_shipped"], d["b_bound_0.5"], d["c_bound_1e20"]
    if abs(b - c) < tol:
        return "indistinguishable"
    return "applied" if abs(a - c) < abs(a - b) else "ignored"


# ---------------------------------------------------------------- E1 (tolerance)
def e1_push_episodes(obj: dict, cfg: dict, params: dict) -> list[dict]:
    """Effective-friction sweep x mass (negative control) x damping mode x seeds,
    plus the exact shipped condition.

    Under 'average' combine an effective mu below support_friction/2 is unreachable
    by changing the object alone, so the sweep sets the support material to the
    same value: effective mu then equals the level exactly.
    """
    e1 = cfg["e1"]
    m_med = obj["mass_range"][1]
    levels = [(f"mu={x:.3g}", x) for x in geomspace(e1["friction_min"], e1["friction_max"], e1["friction_n"])]
    levels.append(("mu=shipped", shipped_friction(cfg)))
    eps = []
    for (lname, mu), scale, mode, seed in itertools.product(levels, e1["mass_scales"], e1["modes"], range(e1["seeds"])):
        m = m_med * scale
        props = _clean(m, mu) if mode == "clean" else _shipped_damped(cfg, m, mu)
        eps.append(episode("e1", "push", obj, f"{lname}|m×{scale:g}|{mode}", props, mu, params, seed,
                           mass_scale=scale, mode=mode, mu_level=lname))
    for seed in range(e1["seeds"]):
        mu = shipped_friction(cfg)
        eps.append(episode("e1", "push", obj, "shipped_exact", resolve("shipped", obj, cfg), mu, params, seed,
                           mode="shipped_exact"))
    return eps


def e1_lift_episodes(obj: dict, cfg: dict, params: dict) -> list[dict]:
    """Mass x object-friction grid against fixed pads, plus the shipped parameters.
    3x3 uses (p5, median, p95) x (lo, median, hi); 5x5 extends each axis outward."""
    p5, med, p95 = obj["mass_range"]
    lo, fmed, hi = obj["friction_range"]
    if cfg["e1"]["lift_grid"] >= 5:
        masses, frictions = [med * 0.1, p5, med, p95, med * 10], [lo / 2, lo, fmed, hi, hi * 2]
    else:
        masses, frictions = [p5, med, p95], [lo, fmed, hi]
    pad = cfg["contact"]["pad_friction"]
    eps = []
    for m, mu, seed in itertools.product(masses, frictions, range(cfg["e1"]["seeds"])):
        eps.append(episode("e1", "lift", obj, f"m={m:.3g}|mu={mu:.3g}", _clean(m, mu), pad, params, seed))
    for seed in range(cfg["e1"]["seeds"]):
        eps.append(episode("e1", "lift", obj, "shipped_exact", resolve("shipped", obj, cfg), pad, params, seed,
                           mode="shipped_exact"))
    return eps


def calibration_episodes(obj: dict, cfg: dict, primitive: str) -> list[dict]:
    """Controller calibration at the plausible median, clean mode."""
    mu = obj["friction_range"][1]
    props = _clean(obj["mass_range"][1], mu)
    contact = cfg["contact"]["support_friction" if primitive == "push" else "pad_friction"]
    grid = push_grid(cfg) if primitive == "push" else lift_grid(cfg)
    return [episode("calib", primitive, obj, "median", props, contact, p, seed)
            for p in grid for seed in range(cfg["e1"]["seeds"])]


# ---------------------------------------------------------------- E2 (transfer)
def e2_train_episodes(source: str, objects: list[dict], cfg: dict, primitive: str) -> list[dict]:
    src = SOURCES[source]
    ranged = "range" in (src.mass + src.friction)
    samples = cfg["e2"]["test_samples"] if ranged else 1
    contact = cfg["contact"]["support_friction" if primitive == "push" else "pad_friction"]
    grid = push_grid(cfg) if primitive == "push" else lift_grid(cfg)
    eps = []
    for obj, p, k, seed in itertools.product(objects, grid, range(samples), range(cfg["e2"]["seeds"])):
        props = resolve(src, obj, cfg, _seeded("train", source, obj["id"], k))
        eps.append(episode("e2_train", primitive, obj, source, props, contact, p, seed, sample=k))
    return eps


def e2_eval_episodes(source: str, params: dict, objects: list[dict], cfg: dict, primitive: str) -> list[dict]:
    """The same held-out test set for every source: parameters sampled from the
    plausible ranges with seeds that do not depend on the source."""
    contact = cfg["contact"]["support_friction" if primitive == "push" else "pad_friction"]
    eps = []
    for obj, k, seed in itertools.product(objects, range(cfg["e2"]["test_samples"]), range(cfg["e2"]["seeds"])):
        rng = _seeded("test", obj["id"], k)
        props = _clean(sample_mass(obj, rng), sample_friction(obj, rng))
        eps.append(episode("e2_eval", primitive, obj, source, props, contact, params, seed,
                           tuned_on=source, sample=k))
    return eps


def _tiebreak(params: dict) -> tuple:
    # Shorter stroke, then slower; smaller force.
    return (params["s"], params["v"]) if "s" in params else (params["F"],)


def select_controller(results: list[dict]) -> dict:
    """Highest success rate over all training episodes; ties to the gentler controller."""
    by = defaultdict(list)
    for r in results:
        by[json.dumps(r["params"], sort_keys=True)].append(r["success"])
    scored = [(mean(v), json.loads(k)) for k, v in by.items()]
    best = max(s for s, _ in scored)
    return min((p for s, p in scored if math.isclose(s, best)), key=_tiebreak)


def select_calibration(results: list[dict], cfg: dict) -> dict:
    """Among controllers that succeed in every seed, the one whose mean outcome is
    closest to the band centre (push) or the smallest force (lift)."""
    by = defaultdict(list)
    for r in results:
        by[json.dumps(r["params"], sort_keys=True)].append(r)
    ok = {k: v for k, v in by.items() if all(r["success"] for r in v)}
    if not ok:
        raise RuntimeError(f"no controller succeeds at the median for {results[0]['object_id']}")
    if "d" in results[0]:
        centre = sum(cfg["push"]["band"]) / 2
        k = min(ok, key=lambda k: abs(mean(r["d"] for r in ok[k]) - centre))
    else:
        k = min(ok, key=lambda k: json.loads(k)["F"])
    return json.loads(k)


__all__ = ["COMPARISON"]
