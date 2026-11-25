import gzip
import json
import math
import random

import pytest

from sage_phy import abo, closed_form as cf, experiments as ex, gates, plausibility, stats
from sage_phy.backends.analytic import AnalyticBackend
from sage_phy.config import load_config
from sage_phy.primitives import score_lift, score_push
from sage_phy.runner import load_results, run
from sage_phy.sources import SOURCES, resolve

CFG = load_config()
OBJ = {"id": "o1", "scene": "s", "prim_path": "/W/o1", "split": "train", "layout_mass": 1.0,
       "collision_volume": 0.0004, "mass_range": [0.2, 0.3, 0.5], "friction_range": [0.4, 0.5, 0.6]}


# ---------------- closed form
def test_slide_undamped():
    assert cf.slide_distance(0.5, 0.5, 0.0, 9.81) == pytest.approx(0.25 / 9.81)


def test_damped_slide_below_bound_and_tends_to_v_over_c():
    for mu in (0.05, 0.5, 2.0):
        d = cf.slide_distance(0.5, mu, 10.0)
        assert 0 < d < cf.damped_slide_bound(0.5, 10.0)
    assert cf.slide_distance(0.5, 1e-9, 10.0) == pytest.approx(0.05, rel=1e-6)
    assert cf.slide_distance(0.5, 1e20, 10.0) == pytest.approx(0.0, abs=1e-12)


def test_damped_matches_numeric_integration():
    v, mu, c, g, dt, x = 0.4, 0.3, 10.0, 9.81, 1e-6, 0.0
    while v > 0:
        x += v * dt
        v -= (mu * g + c * v) * dt
    assert cf.slide_distance(0.4, 0.3, 10.0) == pytest.approx(x, rel=1e-3)


def test_lift_condition_and_margin():
    F = cf.min_grip_force(0.3, 0.75, 1.0)
    assert cf.lift_holds(F * 1.0001, 0.3, 0.75, 1.0)
    assert not cf.lift_holds(F * 0.999, 0.3, 0.75, 1.0)
    assert cf.force_margin(2 * F, 0.3, 0.75, 1.0) == pytest.approx(2.0)


def test_unreachable_effective_mu():
    with pytest.raises(ValueError):
        cf.object_mu_for_effective(0.05, 0.5)


# ---------------- scoring
def test_push_scoring():
    assert score_push(0.15, 0, CFG) == (True, None)
    assert score_push(0.10, 0, CFG) == (False, "undershoot")
    assert score_push(0.20, 0, CFG) == (False, "overshoot")
    assert score_push(0.15, 45, CFG) == (False, "tip_over")


def test_lift_scoring():
    assert score_lift(True, True, 0.0, CFG) == (True, None)
    assert score_lift(True, True, 0.02, CFG) == (False, "slip")
    assert score_lift(True, False, 0.0, CFG) == (False, "drop")
    assert score_lift(False, False, 0.0, CFG) == (False, "grasp_miss")


# ---------------- sources
def test_sources_table():
    assert len(SOURCES) == 7
    sh = resolve("shipped", OBJ, CFG)
    assert (sh.mass, sh.friction, sh.linear_damping, sh.max_linear_velocity) == (1.0, 1e20, 10.0, 0.5)
    sd = resolve("simple_defaults", OBJ, CFG)
    assert sd.mass == pytest.approx(0.4) and sd.friction == 0.5 and sd.linear_damping == 0
    am = resolve("ablate_mass", OBJ, CFG)
    assert am.mass == 0.3 and am.friction == 1e20 and am.linear_damping == 10.0
    rng = random.Random(0)
    for _ in range(100):
        r = resolve("grounded_range", OBJ, CFG, rng)
        assert 0.2 <= r.mass <= 0.5 and 0.4 <= r.friction <= 0.6


# ---------------- ABO
def test_abo_parse_and_densities(tmp_path):
    rec = {"product_type": [{"value": "CUP"}],
           "item_weight": [{"normalized_value": {"unit": "pounds", "value": 1.0}}],
           "item_dimensions": {k: {"normalized_value": {"unit": "inches", "value": 4.0}}
                               for k in ("width", "length", "height")}}
    p = tmp_path / "listings_0.json.gz"
    with gzip.open(p, "wt") as f:
        for _ in range(30):
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({"product_type": [{"value": "CUP"}]}) + "\n")  # incomplete: skipped
    dens = abo.category_densities(abo.iter_listings([p]), 30)
    expect = 0.45359237 / (4 * 0.0254) ** 3
    assert dens["CUP"]["n"] == 30 and dens["CUP"]["p50"] == pytest.approx(expect)


def test_percentile():
    assert abo.percentile([1, 2, 3, 4, 5], 50) == 3
    assert abo.percentile([0, 10], 5) == pytest.approx(0.5)


def test_plausibility():
    assert plausibility.classify(0.1, 0.2, 0.5) == "below"
    assert plausibility.alde(math.e, 1.0) == pytest.approx(1.0)
    assert plausibility.ape(1.5, 1.0) == pytest.approx(0.5)


# ---------------- stats
def test_wilson_known_value():
    est, lo, hi = stats.wilson(5, 10)
    assert est == 0.5 and lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)


def test_gap_and_gap_closed():
    objs = [f"o{i}" for i in range(10)]
    rates = {"grounded_range": {o: 0.9 for o in objs}, "shipped": {o: 0.3 for o in objs},
             "ablate_friction": {o: 0.75 for o in objs}}
    assert stats.transfer_gap(rates, "shipped", n=200)[0] == pytest.approx(0.6)
    assert stats.gap_closed(rates, "ablate_friction", n=200)[0] == pytest.approx(0.75)


# ---------------- experiments + gates through the analytic backend
def test_e1_push_counts_and_shipped_exact():
    eps = ex.e1_push_episodes(OBJ, CFG, {"s": 0.1, "v": 0.3})
    assert len(eps) == 8 * 3 * 2 * 2 + 2
    assert len({e["id"] for e in eps}) == len(eps)


def test_f0_classification():
    assert ex.classify_f0({"a_shipped": 0.10, "b_bound_0.5": 0.14, "c_bound_1e20": 0.10}) == "applied"
    assert ex.classify_f0({"a_shipped": 0.14, "b_bound_0.5": 0.14, "c_bound_1e20": 0.10}) == "ignored"
    assert ex.classify_f0({"a_shipped": 0.10, "b_bound_0.5": 0.105, "c_bound_1e20": 0.10}) == "indistinguishable"


def test_g4_passes_on_analytic():
    be = AnalyticBackend(CFG)
    trials = []
    for mu in (0.2, 0.5, 1.0):
        ep = ex.episode("g4", "push", OBJ, "c", ex._clean(0.3, mu), mu, {"s": 0.05, "v": 0.3}, 0)
        trials.append({"mu_e": mu, "v": 0.3, "slide": be.run(ep)["d"] - 0.05})
    assert gates.check_slide(trials).passed


def test_pipeline_end_to_end(tmp_path):
    be = AnalyticBackend(CFG)
    res = run(ex.calibration_episodes(OBJ, CFG, "push"), be, CFG, tmp_path / "c.jsonl", progress=False)
    params = ex.select_calibration(res, CFG)
    assert CFG["push"]["band"][0] <= be.run({**res[0], "params": params})["d"] <= CFG["push"]["band"][1]
    # mass is a negative control for push: outcomes must not depend on it
    e1 = run(ex.e1_push_episodes(OBJ, CFG, params), be, CFG, tmp_path / "e1.jsonl", progress=False)
    by = {}
    for r in e1:
        if "mass_scale" in r:
            by.setdefault((r["mu_level"], r["mode"], r["seed"]), set()).add(r["success"])
    assert all(len(v) == 1 for v in by.values())
    # resume: rerun appends nothing
    n = len(load_results(tmp_path / "e1.jsonl"))
    run(ex.e1_push_episodes(OBJ, CFG, params), be, CFG, tmp_path / "e1.jsonl", progress=False)
    assert len(load_results(tmp_path / "e1.jsonl")) == n


def test_select_controller_tiebreak():
    rs = [{"params": {"s": 0.1, "v": 0.3}, "success": True}, {"params": {"s": 0.08, "v": 0.4}, "success": True},
          {"params": {"s": 0.06, "v": 0.2}, "success": False}]
    assert ex.select_controller(rs) == {"s": 0.08, "v": 0.4}


# ---------------- assigner and paths
def test_assigner_on_sage_layout(tmp_path):
    from sage_phy.assigner import assign_object, normalize_type, write_layout_copy
    layout = {"id": "L", "rooms": [{"room_type": "kitchen", "objects": [
        {"id": "r_mug_1", "type": "mug_2", "description": "A ceramic mug", "mass": 0.25,
         "dimensions": {"width": 0.1, "length": 0.1, "height": 0.1}}]}]}
    src = tmp_path / "scene" / "layout_L.json"
    src.parent.mkdir()
    src.write_text(json.dumps(layout))
    assert normalize_type("Chair_12") == "chair"
    dens = {"DRINKING_CUP": {"p5": 100.0, "p50": 300.0, "p95": 500.0}}
    mats = {"ceramic": {"friction_lo": 0.4, "friction_median": 0.5, "friction_hi": 0.6}}

    def fn(o):
        d = o["dimensions"]
        return assign_object(o["type"], o["description"], (d["width"], d["length"], d["height"]),
                             1e-3, {"mug": "DRINKING_CUP"}, dens, mats, {"ceramic": "ceramic"})
    out = write_layout_copy(src, tmp_path / "grounded", fn)
    obj = json.loads(out.read_text())["rooms"][0]["objects"][0]["sage_phy"]
    assert obj["category"] == "DRINKING_CUP" and obj["mass_range"] == pytest.approx([0.1, 0.3, 0.5])
    assert obj["friction"] == [0.4, 0.5, 0.6] and obj["physics_source"] == "abo"
    assert json.loads(src.read_text()) == layout  # original untouched
    with pytest.raises(ValueError):
        write_layout_copy(src, src.parent, fn)


def test_repo_relative_paths():
    from sage_phy.config import REPO_ROOT, repo_path
    assert repo_path("configs/default.toml") == REPO_ROOT / "configs" / "default.toml"
    assert repo_path("configs/default.toml").exists()
