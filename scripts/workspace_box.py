"""The campaign's end-effector box from end-effector positions recorded by hand.

Reads the poses that ``scripts/record_poses.py`` wrote (you moved the arm by
hand to the eight corners of the space the campaign may use), finds the
largest axis-aligned box inside the solid they span (``panda_sysid.workspace``)
and writes it, with a picture.

Outputs:
  <out>.json  lo, hi (m, base frame), size, volume, the recorded points, and the
              box of the convex hull alone for comparison
  <out>.png   the points, the hexahedron's edges and the box, from three sides

Example:
  uv run python scripts/workspace_box.py --poses runs/corners.json --out runs/workspace_box
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from panda_sysid import workspace


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--poses", required=True, help="record_poses.py output (JSON)")
  parser.add_argument("--out", required=True, help="output path without extension")
  args = parser.parse_args()
  points = workspace.recorded_positions(args.poses)
  lo, hi = workspace.largest_box(*workspace.halfspaces(points), points.mean(0))
  hull = ConvexHull(points)
  hull_lo, hull_hi = workspace.largest_box(hull.equations[:, :3], -hull.equations[:, 3], points.mean(0))
  box = {
    "poses": args.poses, "points": points.round(5).tolist(),
    "lo": lo.round(4).tolist(), "hi": hi.round(4).tolist(), "size": (hi - lo).round(4).tolist(),
    "volume_l": round(float(np.prod(hi - lo)) * 1e3, 2),
    "convex_hull_only": {"lo": hull_lo.round(4).tolist(), "hi": hull_hi.round(4).tolist(),
                         "volume_l": round(float(np.prod(hull_hi - hull_lo)) * 1e3, 2)},
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.with_suffix(".json").write_text(json.dumps(box, indent=1))
  print(json.dumps({k: v for k, v in box.items() if k != "points"}, indent=1))

  center = points.mean(0)
  corner = {tuple((p > center).astype(int)): p for p in points}
  edges = [(a, b) for a, b in itertools.combinations(corner, 2) if sum(x != y for x, y in zip(a, b)) == 1]
  box_corners = {c: workspace.corners(lo, hi)[i] for i, c in enumerate(itertools.product((0, 1), repeat=3))}
  fig = plt.figure(figsize=(16, 5.5))
  for i, (elev, azim, title) in enumerate(((20, -60, "perspective"), (90, -90, "top (x right, y up)"),
                                          (0, -90, "front (x right, z up)"))):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    for a, b in edges:
      ax.plot(*np.array([corner[a], corner[b]]).T, color="tab:blue", lw=1)
      ax.plot(*np.array([box_corners[a], box_corners[b]]).T, color="tab:red", lw=2)
    ax.scatter(*points.T, color="tab:blue", s=25)
    for k, p in enumerate(points):
      ax.text(*p, f" {k + 1}", fontsize=9)
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title=title)
    ax.view_init(elev, azim)
    if i:
      ax.set_proj_type("ortho")
    ax.set_box_aspect(np.ptp(points, axis=0))
  fig.suptitle(f"Recorded end-effector points (blue) and the largest axis-aligned box inside them (red): "
               f"{box['size']} m, {box['volume_l']} L")
  fig.tight_layout()
  fig.savefig(out.with_suffix(".png"), dpi=110)


if __name__ == "__main__":
  main()
