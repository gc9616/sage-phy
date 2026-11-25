"""Analysis for E1 and E2. The unit is the object; seeds are aggregated first."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

from .stats import gap_closed, transfer_gap, wilson


def e1_summary(results: list[dict]) -> dict:
    """Share of objects whose shipped parameters fall outside their success region
    (i.e. the exact shipped condition fails in any seed), with a Wilson interval."""
    by_obj = defaultdict(list)
    for r in results:
        if r.get("mode") == "shipped_exact":
            by_obj[r["object_id"]].append(r["success"])
    outside = sum(1 for v in by_obj.values() if not all(v))
    est, lo, hi = wilson(outside, len(by_obj))
    region = defaultdict(set)
    for r in results:
        if r.get("mode") != "shipped_exact":
            region[(r["object_id"], r["condition"])].add(r["success"])
    region_size = Counter(o for (o, _), s in region.items() if s == {True})
    return {"objects": len(by_obj), "shipped_outside": outside,
            "share": est, "ci95": [lo, hi], "region_size": dict(region_size)}


def object_rates(results: list[dict]) -> dict[str, dict[str, float]]:
    """rates[tuned_on][object_id] = success rate on the held-out test set."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in results:
        acc[r["tuned_on"]][r["object_id"]].append(r["success"])
    return {s: {o: mean(v) for o, v in objs.items()} for s, objs in acc.items()}


def e2_summary(results: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    rates = object_rates(results)
    out = {"success": {s: mean(v.values()) for s, v in rates.items()}, "gap": {}, "gap_closed": {}}
    if "grounded_range" not in rates:
        return out
    for s in rates:
        if s != "grounded_range":
            out["gap"][s] = transfer_gap(rates, s, n=n_boot, seed=seed)
    if "shipped" in rates:
        for s in rates:
            if s.startswith("ablate_"):
                out["gap_closed"][s] = gap_closed(rates, s, n=n_boot, seed=seed)
    fails = defaultdict(Counter)
    margins = defaultdict(list)
    for r in results:
        if r["failure"]:
            fails[r["tuned_on"]][r["failure"]] += 1
        if "force_margin" in r:
            margins[r["tuned_on"]].append(r["force_margin"])
    out["failures"] = {s: dict(c) for s, c in fails.items()}
    out["median_force_margin"] = {s: median(v) for s, v in margins.items()}
    return out
