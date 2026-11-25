"""Physics loader components: bind materials, set mass and damping, read back.

This is the replacement for the physics part of SAGE's `convert_mesh_to_usd`
(server/isaacsim/.../usd_utils.py). It works on any USD stage, so it runs both
inside Isaac Sim and with plain `usd-core` for unit checks.

Every load must end with `verify(...)`; any mismatch is an error.
"""
from __future__ import annotations

import math

from pxr import Sdf, Usd, UsdPhysics, UsdShade

try:  # PhysxSchema ships with Isaac Sim, not with usd-core
    from pxr import PhysxSchema
except ImportError:  # pragma: no cover
    PhysxSchema = None

MATERIAL_ROOT = "/World/PhysicsMaterials"


def _material_path(prim_path: str) -> str:
    return f"{MATERIAL_ROOT}/{Sdf.Path(prim_path).name}_mat"


def define_material(stage: Usd.Stage, path: str, static_mu: float, dynamic_mu: float,
                    restitution: float = 0.0, combine: str = "average") -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(static_mu))
    api.CreateDynamicFrictionAttr().Set(float(dynamic_mu))
    api.CreateRestitutionAttr().Set(float(restitution))
    if PhysxSchema is not None:
        px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        px.CreateFrictionCombineModeAttr().Set(combine)
        px.CreateRestitutionCombineModeAttr().Set(combine)
    return mat


def bind_physics_material(stage: Usd.Stage, prim: Usd.Prim, mat: UsdShade.Material) -> None:
    """Bind with the 'physics' purpose on the body and every collider below it, with
    strongerThanDescendants so a material SAGE authored on a child cannot win."""
    for p in Usd.PrimRange(prim):
        if p == prim or p.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(
                mat, UsdShade.Tokens.strongerThanDescendants, "physics")


def set_mass(prim: Usd.Prim, mass: float) -> None:
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr().Set(float(mass))
    # Density is ignored when mass is authored; clear it to avoid ambiguity.
    if api.GetDensityAttr().HasAuthoredValue():
        api.GetDensityAttr().Clear()


def set_damping(prim: Usd.Prim, linear: float, angular: float, max_linear_velocity: float | None) -> None:
    if PhysxSchema is None:
        raise RuntimeError("PhysxSchema unavailable: run inside Isaac Sim")
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateLinearDampingAttr().Set(float(linear))
    rb.CreateAngularDampingAttr().Set(float(angular))
    # "No cap" = a value far above anything the primitives produce.
    rb.CreateMaxLinearVelocityAttr().Set(float(max_linear_velocity) if max_linear_velocity else 1.0e4)


def apply_props(stage: Usd.Stage, prim_path: str, props: dict, restitution: float = 0.0) -> None:
    """props: Props.to_dict() — mass, friction, linear/angular damping, max velocity."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise KeyError(prim_path)
    mu = props["friction"]
    mat = define_material(stage, _material_path(prim_path), mu, mu, restitution)
    bind_physics_material(stage, prim, mat)
    set_mass(prim, props["mass"])
    set_damping(prim, props["linear_damping"], props["angular_damping"], props["max_linear_velocity"])


def set_contact_material(stage: Usd.Stage, prim_path: str, mu: float, name: str) -> None:
    """Fixed, stated material for a support surface or gripper pad."""
    mat = define_material(stage, f"{MATERIAL_ROOT}/{name}", mu, mu)
    bind_physics_material(stage, stage.GetPrimAtPath(prim_path), mat)


def read_back(stage: Usd.Stage, prim_path: str) -> dict:
    """Authored physics on a body: mass, bound friction per collider, damping, cap."""
    prim = stage.GetPrimAtPath(prim_path)
    out = {"prim_path": prim_path, "mass": None, "colliders": [],
           "linear_damping": None, "angular_damping": None, "max_linear_velocity": None}
    if prim.HasAPI(UsdPhysics.MassAPI):
        out["mass"] = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
    if PhysxSchema is not None and prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        rb = PhysxSchema.PhysxRigidBodyAPI(prim)
        out["linear_damping"] = rb.GetLinearDampingAttr().Get()
        out["angular_damping"] = rb.GetAngularDampingAttr().Get()
        out["max_linear_velocity"] = rb.GetMaxLinearVelocityAttr().Get()
    for p in Usd.PrimRange(prim):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
        entry = {"collider": str(p.GetPath()), "material": None, "static": None, "dynamic": None}
        if mat:
            api = UsdPhysics.MaterialAPI(mat.GetPrim())
            entry.update(material=str(mat.GetPath()), static=api.GetStaticFrictionAttr().Get(),
                         dynamic=api.GetDynamicFrictionAttr().Get())
        out["colliders"].append(entry)
    return out


def read_back_all(stage: Usd.Stage) -> list[dict]:
    """Read back every rigid body on the stage (the Stage 1 applied-parameters audit)."""
    return [read_back(stage, str(p.GetPath())) for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def verify(stage: Usd.Stage, prim_path: str, props: dict, rtol: float = 1e-6) -> None:
    """Raise if anything authored differs from what was intended."""
    rb = read_back(stage, prim_path)
    errs = []

    def close(a, b):
        return a is not None and math.isclose(a, b, rel_tol=rtol)

    if not close(rb["mass"], props["mass"]):
        errs.append(f"mass {rb['mass']} != {props['mass']}")
    if not rb["colliders"]:
        errs.append("no colliders")
    for c in rb["colliders"]:
        if not (close(c["static"], props["friction"]) and close(c["dynamic"], props["friction"])):
            errs.append(f"{c['collider']}: friction {c['static']}/{c['dynamic']} != {props['friction']}")
    if PhysxSchema is not None:
        if not close(rb["linear_damping"], props["linear_damping"]):
            errs.append(f"linear damping {rb['linear_damping']} != {props['linear_damping']}")
    if errs:
        raise AssertionError(f"read-back mismatch on {prim_path}: " + "; ".join(errs))
