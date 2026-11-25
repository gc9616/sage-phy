"""Physics loader on a stage built the way SAGE's convert_mesh_to_usd builds one:
a Mesh prim with RigidBodyAPI + CollisionAPI + MassAPI and MaterialAPI applied
directly with friction 1e20. Needs usd-core (Python <= 3.12): .venv/Scripts/python -m pytest
"""
import pytest

try:
    from pxr import Usd, UsdGeom, UsdPhysics
    from sage_phy import physics_usd as pu
except ImportError as e:  # no usd-core, or its DLLs are blocked (Windows Smart App Control)
    pytest.skip(f"pxr unavailable: {e}", allow_module_level=True)

from sage_phy.config import load_config  # noqa: E402
from sage_phy.sources import resolve  # noqa: E402

CFG = load_config()
OBJ = {"id": "mug", "layout_mass": 0.25, "collision_volume": 4e-4,
       "mass_range": [0.2, 0.33, 0.5], "friction_range": [0.4, 0.55, 0.7]}


def sage_like_stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/mug")
    prim = mesh.GetPrim()
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.25)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    m = UsdPhysics.MaterialAPI.Apply(prim)
    m.CreateDynamicFrictionAttr(1e20)
    m.CreateStaticFrictionAttr(1e20)
    table = UsdGeom.Mesh.Define(stage, "/World/table").GetPrim()
    UsdPhysics.CollisionAPI.Apply(table)
    UsdPhysics.MaterialAPI.Apply(table).CreateStaticFrictionAttr(1e20)
    return stage


def test_readback_reports_sage_direct_friction():
    rows = {r["prim_path"]: r for r in pu.read_back_all(sage_like_stage())}
    assert set(rows) == {"/World/mug", "/World/table"}
    mug = rows["/World/mug"]
    assert mug["mass"] == pytest.approx(0.25) and mug["rigid"]
    c = mug["colliders"][0]
    assert c["direct_static"] == 1e20 and c["material"] is None  # nothing bound
    assert not rows["/World/table"]["rigid"]


def test_apply_and_verify_grounded():
    stage = sage_like_stage()
    props = resolve("grounded_median", OBJ, CFG).to_dict()
    with pytest.warns(UserWarning):
        pu.apply_props(stage, "/World/mug", props)
    pu.verify(stage, "/World/mug", props)
    c = pu.read_back(stage, "/World/mug")["colliders"][0]
    assert c["static"] == c["dynamic"] == c["direct_static"] == pytest.approx(0.55)
    assert c["material"].startswith(pu.MATERIAL_ROOT)


def test_verify_catches_mismatch():
    stage = sage_like_stage()
    props = resolve("grounded_median", OBJ, CFG).to_dict()
    with pytest.warns(UserWarning):
        pu.apply_props(stage, "/World/mug", props)
    UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/mug")).GetMassAttr().Set(9.0)
    with pytest.raises(AssertionError, match="mass"):
        pu.verify(stage, "/World/mug", props)
