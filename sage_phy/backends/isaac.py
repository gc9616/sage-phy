"""Isaac Sim 5.0 backend. Run with Isaac Sim's Python, headless:

    ~/isaacsim/python.sh -m sage_phy --backend isaac --objects configs/objects.json f0 --object <id>

Scenes are loaded with SAGE's own, unmodified loader (kits/isaacsim_utils.py,
get_layout_scene_loaded), which puts every object at /World/<object id>.

Per episode (Tier 1 path): stop the timeline -> author physics in USD -> play ->
jitter the pose -> run the primitive. Stopping restores the authored poses.
Changing properties through the tensor interface without a stop is Tier 2.

Lines marked VERIFY use Isaac Sim 5.0 APIs that have not been run yet.
"""
from __future__ import annotations

import math
import os
import random
import sys

import numpy as np

from ..config import repo_path

physics_usd = None  # imported after SimulationApp starts: pxr must not load first

PUSHER = "/World/sagephy_pusher"
PUSHER_SIZE = (0.02, 0.10, 0.04)   # thickness along the push axis, width, height (m)
SETTLE_SPEED = 1e-3                # m/s: object counted as stopped
MAX_SLIDE_S = 5.0


def _quat_up(q_xyzw: np.ndarray) -> np.ndarray:
    """World direction of the body's +z axis for a quaternion (x, y, z, w)."""
    x, y, z, w = q_xyzw
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


class IsaacBackend:
    name = "isaac"

    def __init__(self, cfg: dict, sage_kits: str | None = None, headless: bool = True,
                 physics_dt: float = 1 / 240):
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": headless})
        global physics_usd
        from .. import physics_usd as _pu
        physics_usd = _pu
        self.cfg = cfg
        self.dt = physics_dt
        kits = repo_path(sage_kits or os.environ.get("SAGE_KITS", "data/sage/kits")).resolve()
        if not (kits / "isaacsim_utils.py").exists():
            raise FileNotFoundError(f"SAGE kits not found at {kits}; set SAGE_KITS")
        sys.path.insert(0, str(kits))
        self._layout = None
        self._dirty = False     # USD physics changed since the SAGE load
        self.sim = None

    # ------------------------------------------------------------ scene
    def load_scene(self, layout: str, fresh: bool = False) -> None:
        """Load a layout with SAGE's loader. `fresh` forces a reload (used when a
        'sage' episode follows episodes that rewrote physics)."""
        if self._layout == layout and not fresh:
            return
        from isaacsim_utils import get_layout_scene_loaded  # SAGE-10k kits
        import omni.usd
        if self.sim is not None:
            self.sim.stop()
        self.track_ids = get_layout_scene_loaded(str(repo_path(layout)))
        self.stage = omni.usd.get_context().get_stage()
        from isaacsim.core.api import SimulationContext  # VERIFY: 5.0 path
        SimulationContext.clear_instance()                # VERIFY
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt,
                                     stage_units_in_meters=1.0)
        self._make_pusher()
        self._layout, self._dirty = layout, False

    def read_back_all(self) -> list[dict]:
        return physics_usd.read_back_all(self.stage)

    def _make_pusher(self) -> None:
        from pxr import Gf, UsdGeom, UsdPhysics
        cube = UsdGeom.Cube.Define(self.stage, PUSHER)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.XformCommonAPI(cube)
        xf.SetScale(Gf.Vec3f(*PUSHER_SIZE))
        xf.SetTranslate(Gf.Vec3d(0, 0, -10))   # parked
        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr(True)

    # ------------------------------------------------------------ episodes
    def run(self, ep: dict) -> dict:
        sage_mode = ep["loader"].startswith("sage")
        self.load_scene(ep["layout"], fresh=sage_mode and self._dirty)
        self.sim.stop()
        if ep["loader"] == "sage_no_damping":
            physics_usd.set_damping(self.stage.GetPrimAtPath(ep["prim_path"]), 0.0, 0.0, None, None)
            self._dirty = True
        elif not sage_mode:
            physics_usd.apply_props(self.stage, ep["prim_path"], ep)
            physics_usd.verify(self.stage, ep["prim_path"], ep)
            mu_c = ep["contact_friction"]
            if ep.get("support_prim"):
                physics_usd.set_contact_material(self.stage, ep["support_prim"], mu_c, "support")
            physics_usd.set_contact_material(self.stage, PUSHER, mu_c, "pusher")
            self._dirty = True
        self.sim.play()
        self._views(ep["prim_path"])
        self._reset_object(ep)
        if ep["primitive"] == "push":
            return self._push(ep)
        return self._lift(ep)

    def _views(self, prim_path: str) -> None:
        import omni.physics.tensors as tensors  # VERIFY
        self.sim.step(render=False)              # physics must be initialised first
        self.tv = tensors.create_simulation_view("numpy")
        self.tv.set_subspace_roots("/")
        self.obj_view = self.tv.create_rigid_body_view(prim_path)
        self.pusher_view = self.tv.create_rigid_body_view(PUSHER)
        self.idx = np.array([0], dtype=np.uint32)

    def _pose(self) -> tuple[np.ndarray, np.ndarray]:
        tf = self.obj_view.get_transforms()[0]   # px py pz qx qy qz qw
        return tf[:3].copy(), tf[3:].copy()

    def _reset_object(self, ep: dict) -> None:
        """Seeded jitter of +-5 mm and +-3 deg about z on the authored pose."""
        rng = random.Random(f"{ep['object_id']}|{ep['seed']}")
        e1 = self.cfg["e1"]
        dx, dy = (rng.uniform(-e1["jitter_pos"], e1["jitter_pos"]) for _ in range(2))
        yaw = math.radians(rng.uniform(-e1["jitter_yaw_deg"], e1["jitter_yaw_deg"]))
        p, q = self._pose()
        qz = np.array([0, 0, math.sin(yaw / 2), math.cos(yaw / 2)])
        x1, y1, z1, w1 = qz
        x2, y2, z2, w2 = q
        qn = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                       w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
        self.obj_view.set_transforms(np.concatenate([p + [dx, dy, 0], qn])[None].astype(np.float32), self.idx)
        self.obj_view.set_velocities(np.zeros((1, 6), np.float32), self.idx)
        for _ in range(int(0.25 / self.dt)):     # settle
            self.sim.step(render=False)

    def _extent_along(self, prim_path: str, axis: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Half-extent of the object's world bbox along `axis`, COM height, centre."""
        from pxr import Usd, UsdGeom
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        r = cache.ComputeWorldBound(self.stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
        half = 0.5 * float(np.abs((hi - lo)[:2] @ np.abs(axis[:2])))
        return half, float(0.5 * (lo[2] + hi[2])), 0.5 * (lo + hi)

    def _push(self, ep: dict) -> dict:
        """Kinematic pusher at COM height accelerates uniformly to v over stroke s,
        stops, and the object slides. Returns displacement along the axis and tilt."""
        s, v = ep["params"]["s"], ep["params"]["v"]
        axis = np.array([*ep.get("push_axis", (1.0, 0.0)), 0.0])
        axis /= np.linalg.norm(axis)
        p0, q0 = self._pose()
        half, zc, _ = self._extent_along(ep["prim_path"], axis)
        start = p0 - axis * (half + PUSHER_SIZE[0] / 2 + 0.001)
        start[2] = zc
        yaw = math.atan2(axis[1], axis[0])
        qp = np.array([0, 0, math.sin(yaw / 2), math.cos(yaw / 2)], np.float32)

        def place(pos):
            self.pusher_view.set_kinematic_targets(np.concatenate([pos, qp])[None].astype(np.float32), self.idx)

        self.pusher_view.set_transforms(np.concatenate([start, qp])[None].astype(np.float32), self.idx)
        a = v * v / (2 * s)
        T = 2 * s / v
        t = 0.0
        while t < T:
            t += self.dt
            place(start + axis * 0.5 * a * min(t, T) ** 2)
            self.sim.step(render=False)
        place(start + axis * (s - 0.02))   # stop and back off 2 cm so it cannot re-contact
        t = 0.0
        while t < MAX_SLIDE_S:
            self.sim.step(render=False)
            t += self.dt
            if np.linalg.norm(self.obj_view.get_velocities()[0, :3]) < SETTLE_SPEED:
                break
        p1, q1 = self._pose()
        place(np.array([0, 0, -10.0]))
        cos_tilt = float(np.clip(_quat_up(q1) @ _quat_up(q0), -1, 1))
        return {"d": float((p1 - p0) @ axis), "tilt_deg": math.degrees(math.acos(cos_tilt)),
                "lateral": float(np.linalg.norm((p1 - p0) - ((p1 - p0) @ axis) * axis))}

    def _lift(self, ep: dict) -> dict:
        """Floating parallel-jaw gripper (Tier 2): approach from above along the shorter
        horizontal axis, close with force F per finger, lift 0.10 m, hold 2 s."""
        raise NotImplementedError("_lift: Tier 2, build Nov 28 after G3")

    # ------------------------------------------------------------ Tier 2
    def set_props_tensor(self, mass: float, friction: float) -> None:
        """Change mass and friction without a stop/play. VERIFY: RigidBodyView
        set_masses / set_material_properties ([static, dynamic, restitution] per shape)."""
        self.obj_view.set_masses(np.array([[mass]], np.float32), self.idx)
        mp = self.obj_view.get_material_properties()
        mp[..., 0] = friction
        mp[..., 1] = friction
        self.obj_view.set_material_properties(mp, self.idx)

    def close(self) -> None:
        self.app.close()
