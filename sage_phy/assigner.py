"""Property assigner: give each SAGE object a category, material class, mass range
and friction range, written to a COPY of the layout. Originals are never modified.

Inputs:
  data/type_map.json       SAGE object type -> ABO category (hand-built, ~15 categories)
  data/material_keywords.json  keyword -> material class (SceneSmith's 19 classes)
  data/materials.csv       material class -> friction/restitution ranges and density
  ABO densities            from abo.category_densities
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .plausibility import plausible_mass

FIELDS = ("material_class", "mass_range", "friction", "restitution", "physics_source")


def load_materials(path: str | Path) -> dict[str, dict]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["material_class"]] = {k: float(v) for k, v in row.items()
                                          if k != "material_class" and v not in ("", None)}
    return out


def material_from_keywords(text: str, keywords: dict[str, str], default: str | None = None) -> str | None:
    t = text.lower()
    for kw, cls in keywords.items():
        if kw.lower() in t:
            return cls
    return default


def assign_object(obj_type: str, description: str, bbox_dims: tuple[float, float, float],
                  mesh_volume: float, type_map: dict, densities: dict, materials: dict,
                  keywords: dict, hand_label: str | None = None) -> dict:
    """Physics fields for one object. `hand_label` overrides the keyword map
    (used for the 20 experiment objects)."""
    mat = hand_label or material_from_keywords(f"{obj_type} {description}", keywords)
    mprops = materials.get(mat, {}) if mat else {}
    w, l, h = bbox_dims
    cat = type_map.get(normalize_type(obj_type))

    if cat in densities:
        mass_range = list(plausible_mass(densities[cat], w * l * h))
        source = "abo"
    elif "density" in mprops:
        m = mprops["density"] * mesh_volume
        mass_range = [m, m, m]
        source = "material_density_low_confidence"
    else:
        mass_range, source = None, "unassigned"

    def rng(prefix):
        keys = [f"{prefix}_lo", f"{prefix}_median", f"{prefix}_hi"]
        return [mprops[k] for k in keys] if all(k in mprops for k in keys) else None

    return {"category": cat, "material_class": mat, "mass_range": mass_range,
            "friction": rng("friction"), "restitution": rng("restitution"),
            "physics_source": source}


def normalize_type(t: str) -> str:
    """SAGE numbers repeated types (chair_1, book2); strip the suffix."""
    return re.sub(r"_?\d+$", "", t.strip().lower())


def iter_objects(layout: dict):
    """SAGE-10k layout: rooms[].objects[] with id, type, description,
    dimensions{width,length,height}, mass (default 1.0), place_id."""
    for room in layout.get("rooms", []):
        for obj in room.get("objects", []):
            yield room, obj


def write_layout_copy(layout_path: str | Path, out_dir: str | Path, assign_fn) -> Path:
    """Copy a SAGE layout JSON into out_dir and add a `sage_phy` block to each object.
    `assign_fn(obj_dict) -> dict` returns the fields for one object."""
    layout_path, out_dir = Path(layout_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / layout_path.name
    if out.resolve() == layout_path.resolve():
        raise ValueError("refusing to overwrite the original layout")
    with open(layout_path, encoding="utf-8") as f:
        layout = json.load(f)
    for _, obj in iter_objects(layout):
        obj["sage_phy"] = assign_fn(obj)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)
    return out
