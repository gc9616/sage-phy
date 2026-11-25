"""Find and download SAGE-10k scenes by room type.

Each scene is a ~50 MB zip on Hugging Face. To find candidates cheaply, this reads
only the layout JSON inside each zip over HTTP range requests (~150 KB per scene),
then downloads and extracts the full zip only for scenes that match.

    python scripts/fetch_scenes.py --kits-only                 # SAGE loader -> data/sage/kits
    python scripts/fetch_scenes.py --rooms kitchen dining --want 15
    python scripts/fetch_scenes.py --scan-only --scan 400      # index only

All paths are relative to the repo root. Writes data/sage/scene_index.json (git-ignored):
scene -> room types and objects (id, type, mass, dims, description).
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KITS = ["isaacsim_utils.py", "tex_utils_local.py", "glb_utils.py", "utils.py", "models.py",
        "load_isaacsim_demo.py", "export_usd.py"]

REPO = "https://huggingface.co/datasets/nvidia/SAGE-10k"
API = "https://huggingface.co/api/datasets/nvidia/SAGE-10k/tree/main/scenes"
UA = {"User-Agent": "sage-phy-fetch/0.1"}


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def list_scenes(limit: int) -> list[dict]:
    out, url = [], API + "?limit=1000"
    while url and len(out) < limit:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            out += json.load(r)
            nxt = r.headers.get("Link", "")
        url = nxt.split(";")[0].strip("<> ") if 'rel="next"' in nxt else None
    return [x for x in out if x["path"].endswith(".zip")][:limit]


def read_layout(entry: dict) -> dict:
    url = f"{REPO}/resolve/main/{entry['path']}"
    with zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url, entry["size"]), 1 << 16)) as z:
        name = next(n for n in z.namelist() if n.endswith(".json") and "/" not in n.strip("/"))
        return json.loads(z.read(name))


def summarize(layout: dict) -> dict:
    return {"rooms": [r["room_type"] for r in layout["rooms"]],
            "objects": [{"id": o["id"], "type": o["type"], "mass": o.get("mass", 1.0),
                         "dims": o["dimensions"], "description": o.get("description", "")[:120]}
                        for r in layout["rooms"] for o in r["objects"]]}


def download(entry: dict, out: Path) -> Path:
    stem = Path(entry["path"]).stem
    dest = out / stem
    if dest.exists():
        return dest
    url = f"{REPO}/resolve/main/{entry['path']}"
    tmp = out / (stem + ".zip")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed HF URL)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(dest)
    tmp.unlink()
    return dest


def fetch_kits(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in KITS:
        if not (dest / f).exists():
            urllib.request.urlretrieve(f"{REPO}/resolve/main/kits/{f}", dest / f)  # noqa: S310
    print(f"SAGE kits -> {dest.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="*", default=["kitchen", "dining"])
    ap.add_argument("--want", type=int, default=15)
    ap.add_argument("--scan", type=int, default=400, help="max scenes to inspect")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--out", default="data/sage/scenes")
    ap.add_argument("--index", default="data/sage/scene_index.json")
    ap.add_argument("--kits-only", action="store_true")
    a = ap.parse_args()
    fetch_kits(ROOT / "data/sage/kits")
    if a.kits_only:
        return
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    idx_path = ROOT / a.index
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    got = 0
    for i, e in enumerate(list_scenes(a.scan)):
        key = Path(e["path"]).stem
        if key not in index:
            try:
                index[key] = summarize(read_layout(e))
            except Exception as ex:  # keep scanning
                print(f"skip {key}: {ex}")
                continue
            idx_path.write_text(json.dumps(index, indent=1))
        rooms = index[key]["rooms"]
        match = any(w in r.lower() for r in rooms for w in a.rooms)
        print(f"[{i}] {key} rooms={rooms}{'  <- match' if match else ''}")
        if match and not a.scan_only:
            print(f"    downloaded to {download(e, out).relative_to(ROOT)}")
            got += 1
            if got >= a.want:
                break
    print(f"index: {len(index)} scenes -> {idx_path}")


if __name__ == "__main__":
    main()
