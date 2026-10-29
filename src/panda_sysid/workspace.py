"""The end-effector workspace of the identification campaign: an axis-aligned box.

Eight end-effector positions, recorded by hand-guiding the arm to roughly
the corners of the space it may use (``scripts/record_poses.py``), span a
solid. The campaign's end effector stays in the largest box with edges
along the base frame's axes that fits inside it.

The solid is taken conservatively: the convex hull of the points, and also,
for each of the hexahedron's six faces (each recorded point is the corner of
the box it is nearest to), the half-spaces of all four triangles of the face,
so the box fits inside the hexahedron however its non-planar faces are
folded. A box [lo, hi] lies in the half-space a.x <= b exactly when
a+ . hi - a- . lo <= b (a+ and a- the positive and negative parts of a), so
the largest box maximizes sum(log(hi - lo)) under linear constraints: a
concave problem with one optimum.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def recorded_positions(path: str | Path) -> np.ndarray:
  """End-effector positions (n, 3) from a ``record_poses.py`` file (O_T_EE as the robot reported it)."""
  return np.array([pose["position"] for pose in json.loads(Path(path).read_text())["poses"]])


def halfspaces(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """(A, b) with the solid = {x : A x <= b}: the convex hull, and every
  triangle of each face of the hexahedron the eight points span."""
  hull = ConvexHull(points)
  a = [hull.equations[:, :3]]
  b = [-hull.equations[:, 3]]
  if len(points) == 8:
    center = points.mean(0)
    corner = [tuple(c) for c in (points > center).astype(int)]
    if len(set(corner)) != 8:
      raise ValueError("the eight points are not one per corner of a box")
    index = {c: i for i, c in enumerate(corner)}
    for axis, side in itertools.product(range(3), (0, 1)):
      face = [index[c] for c in index if c[axis] == side]
      for tri in itertools.combinations(face, 3):
        p0, p1, p2 = points[list(tri)]
        n = np.cross(p1 - p0, p2 - p0)
        n /= np.linalg.norm(n)
        if n @ (center - p0) > 0:
          n = -n
        a.append(n[None])
        b.append(np.array([n @ p0]))
  return np.concatenate(a), np.concatenate(b)


def largest_box(a: np.ndarray, b: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The largest-volume box [lo, hi] (axis-aligned) in {x : a x <= b}; ``start`` is an interior point."""
  pos, neg = np.clip(a, 0, None), np.clip(-a, 0, None)

  def volume(x):
    size = x[3:] - x[:3]
    if np.any(size <= 0):  # a line search step past the degenerate box
      return 1e10, np.zeros(6)
    return -np.sum(np.log(size)), np.concatenate([1 / size, -1 / size])

  constraint = {"type": "ineq", "fun": lambda x: b - pos @ x[3:] + neg @ x[:3],
                "jac": lambda x: np.hstack([neg, -pos])}
  x0 = np.concatenate([start - 1e-3, start + 1e-3])
  result = minimize(volume, x0, jac=True, constraints=[constraint], method="SLSQP",
                    options={"ftol": 1e-12, "maxiter": 1000})
  if not result.success:
    raise RuntimeError(result.message)
  return result.x[:3], result.x[3:]


def corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
  return np.array([[(lo, hi)[s][i] for i, s in enumerate(c)] for c in itertools.product((0, 1), repeat=3)])


def inside(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: float = 0.0) -> np.ndarray:
  """Whether each point (n, 3) lies in the box shrunk by ``margin``."""
  return np.all((points >= lo + margin) & (points <= hi - margin), axis=-1)


def load_box(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
  box = json.loads(Path(path).read_text())
  return np.array(box["lo"]), np.array(box["hi"])
