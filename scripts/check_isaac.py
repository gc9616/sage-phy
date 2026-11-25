"""Gate G1: Isaac Sim runs headless on the desktop.

Run with Isaac Sim's Python, over SSH inside tmux:
    ~/isaacsim/python.sh scripts/check_isaac.py

Pass = a 1 kg cube dropped from 1 m lands on the ground plane in a headless run,
and the script prints the GPU, the Isaac Sim version and the step rate.
"""
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402
from isaacsim.core.api import World  # noqa: E402  VERIFY: 5.0 module path

world = World(physics_dt=1 / 240, rendering_dt=1 / 60, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

cube = UsdGeom.Cube.Define(stage, "/World/cube")
cube.GetSizeAttr().Set(0.1)
UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(0, 0, 1.0))
UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr().Set(1.0)

world.reset()
n, t0 = 480, time.time()
for _ in range(n):
    world.step(render=False)
rate = n / (time.time() - t0)

from isaacsim.core.prims import RigidPrim  # noqa: E402  VERIFY
z = float(RigidPrim("/World/cube").get_world_poses()[0][0][2])

try:
    from isaacsim.core.version import get_version  # VERIFY
    version = get_version()
except Exception:
    version = "unknown"

print(f"Isaac Sim {version}; cube z after 2 s = {z:.3f} m; {rate:.0f} physics steps/s")
ok = 0.0 < z < 0.1
print("G1 PASS" if ok else "G1 FAIL: cube did not come to rest on the ground")
app.close()
raise SystemExit(0 if ok else 1)
