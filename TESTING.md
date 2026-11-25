# Desktop test checklist (gates G1 to G2)

Run these steps in order on the desktop, over SSH inside `tmux`. Commands are run from the repo root. All paths are relative to it, so the clone can live anywhere. `$ISAAC` means your Isaac Sim 5.0 install folder (the one containing `python.sh`, or `python.bat` on Windows).

After each step, paste the last ~20 lines of output into our chat, pass or fail.

## 0. Clone and check the pure-Python side (no GPU)

```bash
git clone https://github.com/gc9616/sage-phy.git && cd sage-phy
python3 -m pip install -e .[dev]     # any Python >= 3.10
python3 -m pytest -q                  # expect: all passed, 0-1 skipped
python3 -m sage_phy g4                # expect: three PASS lines (analytic reference)
```

## 1. G1: Isaac Sim runs headless

```bash
$ISAAC/python.sh scripts/check_isaac.py
```
**Pass:** the output ends with `G1 PASS` and shows the cube resting at z ≈ 0.05 m. Note the steps/s figure.
**If an import fails** (`isaacsim.core.api`, `isaacsim.core.prims` …): send the error. These are the lines marked `VERIFY`.

## 2. Install into Isaac's Python and get SAGE's loader and scenes

```bash
$ISAAC/python.sh -m pip install -r requirements-isaac.txt
$ISAAC/python.sh -m pip install -e .
$ISAAC/python.sh -m pytest -q tests/test_physics_usd.py   # USD loader tests: expect 3 passed
python3 scripts/fetch_scenes.py --kits-only               # SAGE loader -> data/sage/kits
python3 scripts/fetch_scenes.py --rooms kitchen dining --want 3   # ~50 MB per scene
```

## 3. SAGE's loader, unmodified, then read back the applied physics (Stage 1 audit)

```bash
ls data/sage/scenes/*/layout_*.json
$ISAAC/python.sh data/sage/kits/load_isaacsim_demo.py data/sage/scenes/<scene>/layout_<id>.json
$ISAAC/python.sh -m sage_phy --backend isaac readback --layout data/sage/scenes/<scene>/layout_<id>.json
```
**Pass:** SAGE's demo prints a stability ratio. `results/isaac/readback.json` lists every object, with `direct_static` = 1e20, `linear_damping` = 10, `max_linear_velocity` = 0.5 and `max_angular_velocity` = 0.5. Commit `readback.json`, or send it.

## 4. Where we expect breakage (send the errors)

- `SimulationContext.clear_instance` and `omni.physics.tensors` calls in `sage_phy/backends/isaac.py`.
- The push primitive is new and unrun. A first trial needs a real object entry in `configs/objects.json`. We'll fill that in together after step 3.

## Notes

- Tailscale and NoMachine are only for watching. Run everything headless over SSH in `tmux`, so a dropped connection doesn't kill a run.
- Results append to `results/isaac/*.jsonl`. If you re-run a command, it skips episodes that are already done.
- Don't edit `configs/default.toml` after Nov 30 (the freeze).
