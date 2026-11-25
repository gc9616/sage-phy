"""Amazon Berkeley Objects (ABO) listings: parse weights and dimensions, then
compute effective-density percentiles per product category.

Input: the ABO listings metadata files (listings/metadata/listings_*.json.gz,
one JSON object per line). Only the metadata is needed, not the 3D models.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

MASS_KG = {"pounds": 0.45359237, "pound": 0.45359237, "lb": 0.45359237, "lbs": 0.45359237,
           "kilograms": 1.0, "kilogram": 1.0, "kg": 1.0,
           "grams": 1e-3, "gram": 1e-3, "g": 1e-3,
           "ounces": 0.028349523125, "ounce": 0.028349523125, "oz": 0.028349523125}
LENGTH_M = {"inches": 0.0254, "inch": 0.0254, "in": 0.0254,
            "centimeters": 0.01, "centimeter": 0.01, "cm": 0.01,
            "millimeters": 1e-3, "millimeter": 1e-3, "mm": 1e-3,
            "meters": 1.0, "meter": 1.0, "m": 1.0,
            "feet": 0.3048, "foot": 0.3048, "ft": 0.3048}


def _measure(entry, table: dict) -> float | None:
    """Value in SI units from an ABO measurement dict, preferring normalized_value."""
    if isinstance(entry, list):
        entry = entry[0] if entry else None
    if not isinstance(entry, dict):
        return None
    src = entry.get("normalized_value") or entry
    unit, value = str(src.get("unit", "")).strip().lower(), src.get("value")
    if unit not in table or value is None:
        return None
    try:
        v = float(value) * table[unit]
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def parse_listing(rec: dict) -> dict | None:
    """Return {category, mass_kg, dims_m (w, l, h)} or None if anything is missing."""
    pt = rec.get("product_type")
    if isinstance(pt, list):
        pt = pt[0].get("value") if pt and isinstance(pt[0], dict) else None
    mass = _measure(rec.get("item_weight"), MASS_KG)
    dims = rec.get("item_dimensions") or {}
    w, l, h = (_measure(dims.get(k), LENGTH_M) for k in ("width", "length", "height"))
    if not pt or mass is None or None in (w, l, h):
        return None
    return {"category": str(pt), "mass_kg": mass, "dims_m": (w, l, h)}


def iter_listings(paths: Iterable[str | Path]) -> Iterator[dict]:
    for p in paths:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = parse_listing(json.loads(line))
                if parsed:
                    yield parsed


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not sorted_vals:
        raise ValueError("empty")
    k = (len(sorted_vals) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def category_densities(listings: Iterable[dict], min_listings: int = 30) -> dict[str, dict]:
    """Bounding-box effective density percentiles (kg/m^3) for categories with
    at least `min_listings` complete listings."""
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in listings:
        w, l, h = r["dims_m"]
        by_cat[r["category"]].append(r["mass_kg"] / (w * l * h))
    out = {}
    for cat, vals in by_cat.items():
        if len(vals) < min_listings:
            continue
        vals.sort()
        out[cat] = {"n": len(vals), "p5": percentile(vals, 5),
                    "p50": percentile(vals, 50), "p95": percentile(vals, 95)}
    return out
