"""Mass plausibility against ABO ranges, with Gen2Physics' ALDE and APE."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median


def plausible_mass(density_pcts: dict, bbox_volume: float) -> tuple[float, float, float]:
    """(p5, median, p95) mass for an object of the given bounding-box volume."""
    return (density_pcts["p5"] * bbox_volume, density_pcts["p50"] * bbox_volume,
            density_pcts["p95"] * bbox_volume)


def classify(m: float, lo: float, hi: float) -> str:
    return "below" if m < lo else "above" if m > hi else "within"


def alde(m: float, m_ref: float) -> float:
    return abs(math.log(m) - math.log(m_ref))


def ape(m: float, m_ref: float) -> float:
    return abs(m - m_ref) / m_ref


def summarize(rows: list[dict]) -> dict:
    """rows: {category, mass, range: (p5, median, p95)} for one mass source.
    Returns overall below/within/above shares and per-category median ALDE/APE."""
    counts = Counter()
    per_cat = defaultdict(lambda: {"alde": [], "ape": []})
    for r in rows:
        lo, med, hi = r["range"]
        counts[classify(r["mass"], lo, hi)] += 1
        per_cat[r["category"]]["alde"].append(alde(r["mass"], med))
        per_cat[r["category"]]["ape"].append(ape(r["mass"], med))
    n = sum(counts.values()) or 1
    return {
        "n": sum(counts.values()),
        "shares": {k: counts[k] / n for k in ("below", "within", "above")},
        "per_category": {c: {"n": len(v["alde"]), "median_alde": median(v["alde"]),
                             "median_ape": median(v["ape"])} for c, v in per_cat.items()},
    }
