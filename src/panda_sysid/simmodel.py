"""A simulation model of the Panda in one JSON file (``model.json``).

It holds everything a simulation of the arm needs besides the meshes:

- ``dynamics``: link inertials of links 1-7 and per-joint dry friction,
  damping and armature (``robot.Dynamics``);
- ``delays``: the signal path of the robot's torque controller
  (``delayed.Delays``);
- ``torque_bias``: center and half-width (N m per joint) of a constant joint
  torque that a simulation draws anew for each episode (uniform): what the
  robot's gravity compensation leaves, which changes with the pose
  (optional);
- ``notes``: where each part comes from.

``load("nominal")`` is the published values (``robot.Dynamics.nominal()``)
with the ideal controller (``delayed.IDEAL``) and no bias.

Example:
  from panda_sysid import model, simmodel
  sim = simmodel.load("results/model.json")
  mj_model = model.make_spec(40.0, 20.0, sim.dynamics).compile()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from panda_sysid import delayed, robot

# The model identified on our Panda (results/model.json in a checkout of this repository).
OURS = Path(__file__).resolve().parents[2] / "results" / "model.json"


@dataclass
class SimModel:
  dynamics: robot.Dynamics
  delays: delayed.Delays = delayed.IDEAL
  bias_center: tuple[float, ...] | None = None
  bias_half_width: tuple[float, ...] | None = None
  notes: dict = field(default_factory=dict)

  def draw_bias(self, rng: np.random.Generator) -> np.ndarray:
    """A torque bias (N m) for one episode, uniform in center +- half-width (zero without one)."""
    if self.bias_center is None:
      return np.zeros(7)
    center, half = np.array(self.bias_center), np.array(self.bias_half_width)
    return center + half * rng.uniform(-1.0, 1.0, 7)

  def to_dict(self) -> dict:
    out = {"dynamics": self.dynamics.to_dict(), "delays": self.delays.as_dict()}
    if self.bias_center is not None:
      out["torque_bias"] = {"center": list(self.bias_center), "half_width": list(self.bias_half_width)}
    out["notes"] = self.notes
    return out


def load(path: str | Path) -> SimModel:
  """A model.json, or "nominal"."""
  if str(path) == "nominal":
    return SimModel(robot.Dynamics.nominal(), delayed.IDEAL, notes={"source": "published values, ideal controller"})
  d = json.loads(Path(path).read_text())
  bias = d.get("torque_bias")
  return SimModel(robot.Dynamics.from_dict(d["dynamics"]), delayed.Delays(**d["delays"]),
                  tuple(bias["center"]) if bias else None, tuple(bias["half_width"]) if bias else None,
                  d.get("notes", {}))


def save(sim: SimModel, path: str | Path) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(sim.to_dict(), indent=1))
