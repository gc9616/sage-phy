"""Object-level statistics: Wilson intervals and bootstrap over objects.
Seeds are never treated as independent samples; aggregate to objects first."""
from __future__ import annotations

import math
import random
from statistics import mean
from typing import Callable, Sequence

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """(estimate, lo, hi) Wilson score interval for k successes out of n."""
    if n == 0:
        return (math.nan, 0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def bootstrap(objects: Sequence, stat: Callable[[Sequence], float], n: int = 1000,
              seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(estimate, lo, hi): percentile bootstrap, resampling objects with replacement."""
    rng = random.Random(seed)
    objs = list(objects)
    est = stat(objs)
    reps = []
    for _ in range(n):
        v = stat([objs[rng.randrange(len(objs))] for _ in objs])
        if not math.isnan(v):
            reps.append(v)
    reps.sort()
    if not reps:
        return (est, math.nan, math.nan)
    lo = reps[int(math.floor(alpha / 2 * (len(reps) - 1)))]
    hi = reps[int(math.ceil((1 - alpha / 2) * (len(reps) - 1)))]
    return (est, lo, hi)


def transfer_gap(rates: dict[str, dict[str, float]], source: str, ref: str = "grounded_range",
                 **kw) -> tuple[float, float, float]:
    """Success of the ref-tuned controller minus that of the source-tuned one,
    averaged over held-out objects. rates[source][object_id] = success rate."""
    objs = sorted(rates[ref])

    def stat(sample):
        return mean(rates[ref][o] - rates[source][o] for o in sample)
    return bootstrap(objs, stat, **kw)


def gap_closed(rates: dict[str, dict[str, float]], ablation: str, shipped: str = "shipped",
               ref: str = "grounded_range", **kw) -> tuple[float, float, float]:
    """Share of the shipped transfer gap that a single correction closes."""
    objs = sorted(rates[ref])

    def stat(sample):
        g_ship = mean(rates[ref][o] - rates[shipped][o] for o in sample)
        g_abl = mean(rates[ref][o] - rates[ablation][o] for o in sample)
        return math.nan if g_ship == 0 else (g_ship - g_abl) / g_ship
    return bootstrap(objs, stat, **kw)
