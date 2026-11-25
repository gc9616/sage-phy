"""Isaac Sim 5.0 backend. Run with Isaac Sim's Python (./python.sh), headless.

Status: the structure and physics plumbing are in place; every line marked
VERIFY must be checked on the desktop once G1 passes (Isaac Sim 5.0 renamed
many omni.isaac.* modules to isaacsim.*).

Build order (from the plan):
  Nov 24  load_scene with SAGE's unmodified loader, read_back_all  -> audit
  Nov 25  F0 (four settings) through run()
  Nov 26  physics loader: apply_props + contact materials + verify
  Nov 27-28  push primitive, then lift (Tier 2), tensor property updates (Tier 2)
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

from .. import physics_usd


class IsaacBackend:
    name = "isaac"

    def __init__(self, cfg: dict, sage_root: str | None = None, headless: bool = True,
                 physics_dt: float = 1 / 240):
        from isaacsim import SimulationApp  # VERIFY: 5.0 entry point
        self.app = SimulationApp({"headless": headless})
        self.cfg = cfg
        self.dt = physics_dt
        self.sage_root = Path(sage_root or os.environ.get("SAGE_ROOT", "")).expanduser()
        self._scene = None      # (scene id, loader mode) currently on the stage
        self._world = None

    # ------------------------------------------------------------ scene
    def load_scene(self, scene: str, loader: str) -> None:
        """loader: 'sage' (release as shipped), 'sage_no_damping', or 'grounded'.
        All three start from SAGE's own loader; 'grounded' then overwrites physics
        per episode via apply_props."""
        if self._scene == (scene, loader):
            return
        sys.path.insert(0, str(self.sage_root / "kits"))
        from isaacsim_utils import get_layout_scene_loaded  # SAGE-10k kits loader
        # VERIFY: argument names of get_layout_scene_loaded against kits/isaacsim_utils.py
        get_layout_scene_loaded(str(self.sage_root / "scenes" / scene))
        import omni.usd
        self.stage = omni.usd.get_context().get_stage()
        from isaacsim.core.api import World  # VERIFY
        self._world = World(physics_dt=self.dt, rendering_dt=self.dt, stage_units_in_meters=1.0)
        self._world.reset()
        self._scene = (scene, loader)

    def read_back_all(self) -> list[dict]:
        return physics_usd.read_back_all(self.stage)

    # ------------------------------------------------------------ episodes
    def run(self, ep: dict) -> dict:
        self.load_scene(ep["scene"], "sage" if ep["loader"].startswith("sage") else "grounded")
        if ep["loader"] == "sage":
            pass  # release physics untouched (F0 setting a)
        elif ep["loader"] == "sage_no_damping":
            physics_usd.set_damping(self.stage.GetPrimAtPath(ep["prim_path"]), 0.0, 0.0, None)
        else:
            physics_usd.apply_props(self.stage, ep["prim_path"], ep)
            physics_usd.verify(self.stage, ep["prim_path"], ep)
        # TODO(Nov 26): bind support material (ep['contact_friction']) to the surface under the object.
        self._reset_object(ep)
        if ep["primitive"] == "push":
            return self._push(ep)
        return self._lift(ep)

    def _reset_object(self, ep: dict) -> None:
        """Restore the object's layout pose with seeded jitter (+-5 mm, +-3 deg)."""
        rng = random.Random(hash((ep["object_id"], ep["seed"])))
        e1 = self.cfg["e1"]
        self._jitter = (rng.uniform(-e1["jitter_pos"], e1["jitter_pos"]),
                        rng.uniform(-e1["jitter_pos"], e1["jitter_pos"]),
                        math.radians(rng.uniform(-e1["jitter_yaw_deg"], e1["jitter_yaw_deg"])))
        # TODO(Nov 27): write pose via the tensor view (set_transforms), zero velocities.
        raise NotImplementedError("_reset_object: implement on the desktop (Nov 27)")

    def _push(self, ep: dict) -> dict:
        """Kinematic pusher at COM height: accelerate uniformly to v over stroke s,
        stop, let the object slide; return displacement along the push axis and tilt.

        Plan:
          1. create a kinematic box (RigidBodyAPI kinematicEnabled + CollisionAPI)
             behind the object, bind the support material to it;
          2. each step set its kinematic target x(t) = 0.5 a t^2, a = v^2 / (2 s);
          3. after the stroke, retract it; step until object speed < 1 mm/s or 5 s;
          4. d = displacement along the axis; tilt from the final orientation.
        """
        raise NotImplementedError("_push: implement Nov 27 (see docstring)")

    def _lift(self, ep: dict) -> dict:
        """Floating parallel-jaw gripper (Tier 2): approach from above along the shorter
        horizontal axis, close with force F per finger, lift 0.10 m, hold 2 s."""
        raise NotImplementedError("_lift: Tier 2, implement Nov 28 after G3")

    # ------------------------------------------------------------ Tier 2
    def set_props_tensor(self, prim_path: str, mass: float, friction: float) -> None:
        """Change mass and friction between episodes without a reload.
        VERIFY: omni.physics.tensors RigidBodyView.set_masses / set_material_properties."""
        raise NotImplementedError("tensor interface: Tier 2; full reload is the fallback")

    def close(self) -> None:
        self.app.close()
