"""Command line: python -m sage_phy <command> [--backend analytic|isaac] ...

The analytic backend runs anywhere and exercises the whole pipeline; only
isaac-backend results are reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from . import abo, experiments as ex, gates
from .analysis import e1_summary, e2_summary
from .backends import get_backend
from .config import load_config
from .runner import load_results, run
from .sources import ABLATIONS, COMPARISON, SOURCES


def _objects(path: str, split: str | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        objs = json.load(f)["objects"]
    return [o for o in objs if split is None or o["split"] == split]


def _out(args, name: str) -> Path:
    return Path(args.results) / args.backend / name


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_abo(args, cfg):
    paths = sorted(glob.glob(args.listings))
    if not paths:
        sys.exit(f"no files match {args.listings}")
    dens = abo.category_densities(abo.iter_listings(paths), cfg["abo"]["min_listings"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dens, indent=2))
    print(f"{len(dens)} categories with >= {cfg['abo']['min_listings']} listings -> {args.out}")


def cmd_readback(args, cfg):
    be = get_backend("isaac", cfg)
    be.load_scene(args.scene, "sage")
    rows = be.read_back_all()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"{len(rows)} rigid bodies -> {args.out}")
    be.close()


def cmd_f0(args, cfg):
    obj = next(o for o in _objects(args.objects) if o["id"] == args.object)
    res = run(ex.f0_episodes(obj, cfg), get_backend(args.backend, cfg), cfg, _out(args, "f0.jsonl"))
    d = {r["condition"]: r["d"] for r in res}
    _dump({"displacement": d, "verdict": ex.classify_f0(d)})


def cmd_g4(args, cfg):
    be = get_backend(args.backend, cfg)
    objs = _objects(args.objects)[:3]
    g = cfg["contact"]["gravity"]
    s, v = 0.05, 0.3
    eps, results = [], []
    for o in objs:
        for mu in (0.2, 0.5, 1.0):
            for scale in cfg["e1"]["mass_scales"]:
                p = ex._clean(o["mass_range"][1] * scale, mu)
                eps.append(ex.episode("g4", "push", o, f"clean|mu={mu}|m×{scale}", p, mu, {"s": s, "v": v}, 0,
                                      mu_e=mu, mass_scale=scale))
            p = ex._shipped_damped(cfg, o["mass_range"][1], mu)
            eps.append(ex.episode("g4", "push", o, f"damped|mu={mu}", p, mu, {"s": s, "v": v}, 0, mu_e=mu))
    results = run(eps, be, cfg, _out(args, "g4.jsonl"))
    clean = [r for r in results if r["condition"].startswith("clean")]
    damped = [r for r in results if r["condition"].startswith("damped")]
    groups = {}
    for r in clean:
        groups.setdefault((r["object_id"], r["mu_e"]), []).append(r["success"])
    checks = [
        gates.check_slide([{"mu_e": r["mu_e"], "v": v, "slide": r["d"] - s} for r in clean], g=g),
        gates.check_mass_control(list(groups.values())),
        gates.check_shipped_damping([{"v": min(v, cfg["shipped"]["max_linear_velocity"]), "slide": r["d"] - s}
                                     for r in damped], cfg["shipped"]["linear_damping"]),
    ]
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.check}: {c.detail}")
    sys.exit(0 if all(c.passed for c in checks) else 1)


def cmd_calibrate(args, cfg):
    be = get_backend(args.backend, cfg)
    out = {}
    for o in _objects(args.objects):
        res = run(ex.calibration_episodes(o, cfg, args.primitive), be, cfg, _out(args, f"calib_{args.primitive}.jsonl"))
        out[o["id"]] = ex.select_calibration(res, cfg)
    path = _out(args, f"calib_{args.primitive}.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"calibrated {len(out)} objects -> {path}")


def cmd_e1(args, cfg):
    be = get_backend(args.backend, cfg)
    calib = json.loads(_out(args, f"calib_{args.primitive}.json").read_text())
    gen = ex.e1_push_episodes if args.primitive == "push" else ex.e1_lift_episodes
    eps = [e for o in _objects(args.objects) for e in gen(o, cfg, calib[o["id"]])]
    res = run(eps, be, cfg, _out(args, f"e1_{args.primitive}.jsonl"))
    _dump(e1_summary(res))


def cmd_e2(args, cfg):
    be = get_backend(args.backend, cfg)
    names = {"comparison": COMPARISON, "ablations": ABLATIONS, "all": COMPARISON + ABLATIONS}[args.sources]
    train, test = _objects(args.objects, "train"), _objects(args.objects, "test")
    tuned = {}
    for s in names:
        res = run(ex.e2_train_episodes(s, train, cfg, args.primitive), be, cfg,
                  _out(args, f"e2_train_{args.primitive}.jsonl"))
        tuned[s] = ex.select_controller(res)
        print(f"{s}: tuned {tuned[s]}", file=sys.stderr)
    eval_path = _out(args, f"e2_eval_{args.primitive}.jsonl")
    for s in names:
        run(ex.e2_eval_episodes(s, tuned[s], test, cfg, args.primitive), be, cfg, eval_path)
    _out(args, f"e2_tuned_{args.primitive}.json").write_text(json.dumps(tuned, indent=2))
    _dump(e2_summary(load_results(eval_path), cfg["stats"]["bootstrap"], cfg["stats"]["seed"]))


def cmd_plan(args, cfg):
    """Episode counts per experiment (for the compute budget)."""
    objs = _objects(args.objects)
    train, test = [o for o in objs if o["split"] == "train"], [o for o in objs if o["split"] == "test"]
    dummy = {"s": 0.1, "v": 0.3, "F": 10.0}
    n = {
        "calib_push": sum(len(ex.calibration_episodes(o, cfg, "push")) for o in objs),
        "e1_push": sum(len(ex.e1_push_episodes(o, cfg, dummy)) for o in objs),
        "e1_lift": sum(len(ex.e1_lift_episodes(o, cfg, dummy)) for o in objs),
    }
    for s in COMPARISON + ABLATIONS:
        for prim in ("push", "lift"):
            n[f"e2_{prim}_{s}"] = (len(ex.e2_train_episodes(s, train, cfg, prim))
                                   + len(ex.e2_eval_episodes(s, dummy, test, cfg, prim)))
    tiers = {1: 0, 2: 0, 3: 0}
    for k, v in n.items():
        if k.startswith("e2_"):
            prim, s = k.split("_")[1], k.split("_", 2)[2]
            t = SOURCES[s].tier_push if prim == "push" else SOURCES[s].tier_lift
        else:
            t = 2 if "lift" in k else 1
        tiers[t] += v
    _dump({"episodes": n, "by_tier": tiers, "objects": len(objs)})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sage_phy")
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", default="analytic", choices=["analytic", "isaac"])
    ap.add_argument("--objects", default="configs/objects.example.json")
    ap.add_argument("--results", default="results")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("abo-densities", help="ABO density percentiles per category")
    p.add_argument("--listings", default="data/abo/listings/metadata/listings_*.json.gz")
    p.add_argument("--out", default="data/abo_densities.json")
    p.set_defaults(fn=cmd_abo)

    p = sub.add_parser("readback", help="Stage 1: read back applied physics (Isaac only)")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", default="results/isaac/readback.json")
    p.set_defaults(fn=cmd_readback)

    p = sub.add_parser("f0", help="Stage 1: F0 friction test")
    p.add_argument("--object", required=True)
    p.set_defaults(fn=cmd_f0)

    sub.add_parser("g4", help="Gate G4 closed-form checks").set_defaults(fn=cmd_g4)

    for name, fn in (("calibrate", cmd_calibrate), ("e1", cmd_e1)):
        p = sub.add_parser(name)
        p.add_argument("--primitive", default="push", choices=["push", "lift"])
        p.set_defaults(fn=fn)

    p = sub.add_parser("e2", help="Controller transfer")
    p.add_argument("--primitive", default="push", choices=["push", "lift"])
    p.add_argument("--sources", default="comparison", choices=["comparison", "ablations", "all"])
    p.set_defaults(fn=cmd_e2)

    sub.add_parser("plan", help="Episode counts by tier").set_defaults(fn=cmd_plan)

    args = ap.parse_args(argv)
    args.fn(args, load_config(args.config))


if __name__ == "__main__":
    main()
