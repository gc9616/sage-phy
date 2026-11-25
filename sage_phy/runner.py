"""Episode runner: runs episodes on a backend, scores them, and appends JSONL.
Episodes already in the output file are skipped, so a crashed run resumes."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable, Protocol

from .primitives import score_lift, score_push


class Backend(Protocol):
    name: str

    def run(self, ep: dict) -> dict:
        """Push -> {d, tilt_deg}; lift -> {grasped, held, slip_m}."""


def load_results(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(ep: dict, out: dict, cfg: dict) -> dict:
    if ep["primitive"] == "push":
        ok, fail = score_push(out["d"], out.get("tilt_deg", 0.0), cfg)
    else:
        ok, fail = score_lift(out["grasped"], out["held"], out["slip_m"], cfg)
    return {**ep, **out, "success": ok, "failure": fail}


def run(episodes: Iterable[dict], backend: Backend, cfg: dict, out_path: str | Path,
        progress: bool = True) -> list[dict]:
    """Run every episode not yet in out_path; return all results for these episodes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"]: r for r in load_results(out_path)}
    episodes = list(episodes)
    todo = [e for e in episodes if e["id"] not in done]
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, ep in enumerate(todo, 1):
            res = score(ep, backend.run(ep), cfg)
            res["backend"] = backend.name
            f.write(json.dumps(res) + "\n")
            f.flush()
            done[ep["id"]] = res
            if progress and (i % 50 == 0 or i == len(todo)):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(todo)} episodes ({rate:.1f}/s)", file=sys.stderr)
    return [done[e["id"]] for e in episodes]
