# data/

Small hand-built tables live here and are committed. Large downloads (ABO, SAGE-10k) are git-ignored.

| File | What | Built | Source |
|---|---|---|---|
| `materials.csv` | Friction and restitution ranges and density for each material class | Nov 25 | SceneSmith's table (19 classes). The header is in place; fill one row per class and cite it. |
| `material_keywords.json` | `{"keyword": "material_class"}` for the keyword map | Nov 25 | Hand-built |
| `type_map.json` | `{"SAGE object type": "ABO product_type"}` (about 15 categories) | Nov 24 | Hand-matched |
| `abo_densities.json` | Density P5 / P50 / P95 per ABO category | Nov 24 | Run `python -m sage_phy abo-densities` |
| `abo/` | ABO listings metadata (`listings/metadata/*.json.gz`) | Nov 22–23 | ABO download (git-ignored) |
| `sage/` | SAGE-10k `kits/` and the 15 candidate scenes | Nov 23 | SAGE-10k (git-ignored) |
| `layouts_grounded/` | Copies of the layouts written by the assigner | Nov 26 | Assigner output (git-ignored) |

Don't fill in material values from memory. Every row must trace back to the cited table.
