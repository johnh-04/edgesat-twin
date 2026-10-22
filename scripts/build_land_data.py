"""Build the land data used by the simulator and the dashboard.

Input : Natural Earth 1:110m land polygons (public domain), e.g.
        https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_land.geojson
Output: edgesat/data/land_110m.json   compact polygon list [[lon, lat], ...] for the dashboard
        edgesat/data/landmask_0p25.npz equirectangular land/water raster for the simulator

Usage:  python scripts/build_land_data.py path/to/ne_110m_land.geojson
Requires Pillow (only for this one-off build step).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RES_DEG = 0.25
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "edgesat" / "data"


def main(src: str) -> None:
    data = json.loads(Path(src).read_text())
    rings: list[list[list[float]]] = []
    for feat in data["features"]:
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            # Natural Earth 110m land has no holes worth keeping; outer ring only.
            ring = [[round(x, 2), round(y, 2)] for x, y in poly[0]]
            rings.append(ring)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "land_110m.json").write_text(json.dumps(rings, separators=(",", ":")))

    w, h = int(360 / RES_DEG), int(180 / RES_DEG)
    img = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for ring in rings:
        pts = [((lon + 180.0) / RES_DEG, (90.0 - lat) / RES_DEG) for lon, lat in ring]
        draw.polygon(pts, fill=1)
    mask = np.array(img, dtype=bool)
    np.savez_compressed(OUT_DIR / "landmask_0p25.npz", mask=np.packbits(mask, axis=1),
                        shape=np.array(mask.shape), res_deg=np.array(RES_DEG))
    print(f"{len(rings)} rings, land fraction {mask.mean():.3f}, raster {mask.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
