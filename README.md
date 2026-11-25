# sage-phy: Grounding SAGE's Physics

This is the CS 6672 final project. We **audit** the physics shipped with SAGE-10k, **ground** it in real-world object data (ABO masses and material friction), and measure the **consequence** for controllers tuned in simulation.

| Stage | What | Code |
|---|---|---|
| 1. Audit | Read back the applied physics. Run F0, the four-setting friction test. Check mass plausibility against ABO (ALDE/APE). | `physics_usd.read_back_all`, `experiments.f0_episodes`, `abo`, `plausibility` |
| 2. Ground | Property assigner, which writes copies of the layouts. Physics loader, which binds materials, sets mass and combine mode, and verifies by read-back. | `assigner`, `physics_usd`, `sources` |
| 3. Consequence | E1: tolerance sweeps. E2: controller transfer across 4 sources plus 3 ablations. | `experiments`, `runner`, `analysis`, `stats` |
| Validity | Gate G4: closed-form Coulomb checks before any experiment. | `closed_form`, `gates` |

## Layout

```
configs/default.toml          all constants; PILOT values are fixed Nov 29 and frozen Nov 30
configs/objects.example.json  synthetic objects for pipeline tests (real: configs/objects.json)
sage_phy/
  closed_form.py   d = s + v²/(2 μe g), damped slide ≤ v/c, 2 μe F ≥ m(g+a)
  sources.py       the 7 property sources and how each resolves an object's physics
  physics_usd.py   physics loader: bind material, set mass/damping, read back, verify
  abo.py           ABO listings → density P5/P50/P95 per category
  plausibility.py  below/within/above, ALDE, APE
  assigner.py      layout copy with material_class, mass_range, friction, restitution, physics_source
  primitives.py    push / lift scoring and failure categories
  experiments.py   F0, calibration, E1, E2 episode generation and controller selection
  runner.py        resumable JSONL episode runner
  analysis.py      E1 share outside the region (Wilson), E2 gaps (bootstrap over objects)
  backends/        analytic (closed form, no GPU) and isaac (Isaac Sim 5.0)
scripts/check_isaac.py         gate G1
scripts/fetch_scenes.py        SAGE kits + scenes by room type (reads only the layout JSON to search)
data/sage/scene_index.json     scanned scenes: room types and objects (git-ignored; rebuilt by fetch_scenes)
TESTING.md                     desktop checklist, G1 to G2
tests/                          unit and end-to-end tests (analytic backend)
```

## Setup

**Any machine** (analysis and the analytic backend):
```bash
pip install -e .[dev]
pytest
python -m sage_phy plan        # episode counts by tier
python -m sage_phy g4          # gate G4 against the analytic reference
```

**Desktop** (Isaac Sim 5.0, headless over SSH and tmux): see [TESTING.md](TESTING.md).

Every path in the code, configs and object files is relative to the repo root, whatever the working directory. `data/sage/kits` (SAGE's loader) and `data/sage/scenes` are fetched by `scripts/fetch_scenes.py` and git-ignored. `SAGE_KITS` overrides the location of the kits.

## Run order

The Isaac backend follows the same order. The analytic backend runs all of these today.

```bash
python -m sage_phy abo-densities                       # Stage 1 plausibility input
python -m sage_phy f0 --object <id>                     # F0  -> applied / ignored / indistinguishable
python -m sage_phy g4                                   # G4  (must pass before E1)
python -m sage_phy calibrate --primitive push
python -m sage_phy e1 --primitive push                  # Tier 1
python -m sage_phy e2 --primitive push --sources comparison   # Tier 1
python -m sage_phy e2 --primitive push --sources ablations    # Tier 2
python -m sage_phy calibrate --primitive lift && python -m sage_phy e1 --primitive lift   # Tier 2
python -m sage_phy e2 --primitive lift --sources all          # Tier 3
```

Results go to `results/<backend>/*.jsonl`, one line per episode. A run that is re-launched skips episodes it has already completed. Only results from the `isaac` backend are reported.

## Status and schedule

| Date | Gate | Work |
|---|---|---|
| Nov 23 | **G1** | `scripts/check_isaac.py` passes on the desktop |
| Nov 24 | | `readback` on one scene with SAGE's loader; check the `get_layout_scene_loaded` signature; `type_map.json`; ABO densities |
| Nov 25 | **G2** | F0; fill `materials.csv` from SceneSmith; pick and hand-label the 20 objects into `configs/objects.json` |
| Nov 26 | | Physics loader on Isaac (`apply_props`, contact materials, `verify`); assigner writes copies of the layouts |
| Nov 27–28 | **G3** | `IsaacBackend._push`, `_reset_object`. At G3, decide whether lift (Tier 2) goes ahead |
| Nov 29 | **G4**, pilot | G4 on Isaac; fix the PILOT values in `default.toml` |
| Nov 30 | freeze | No further code or config changes to the experiment path |
| Dec 1–3 | | Tier 1, then Tier 2, then Tier 3 runs |
| Dec 4 | | Final analysis and report |

Code marked `VERIFY` in `backends/isaac.py` and `scripts/check_isaac.py` uses Isaac Sim 5.0 module paths that haven't been run yet.

## Notes on the design

- **Average combine mode.** The effective friction is μe = (μ_object + μ_other)/2. Against a support surface with μ = 0.5, no object friction can give μe < 0.25. For that reason, the E1 friction sweep sets the support material to the same level as the object, so that μe equals the level exactly.
- **Range sampling.** Mass is sampled log-uniformly over [P5, P95]. Friction is sampled uniformly over the range for the material class.
- **What SAGE's loader does** (`kits/isaacsim_utils.py`, `convert_mesh_to_usd`). It applies friction 1e20 with `MaterialAPI` directly on each mesh prim and binds no material, sets linear and angular damping to 10, and caps linear velocity at 0.5 m/s and angular velocity at 0.5 **deg/s**. The read-back reports both the direct and the bound friction. The grounded loader binds a material and also overwrites the direct value.
- **Test set.** Every E2 source is evaluated on the same held-out test set. Its parameters are sampled with seeds that do not depend on the source.
