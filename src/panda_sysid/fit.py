"""Recorded runs and the fit of the arm's joint parameters to them.

A recording (``load_log``) is a ``real/joint_impedance`` log on its 1 ms grid.
It is cut into chunks of equal length; ``panda_sysid.delayed`` rolls each
chunk out from its measured q and dq under the recorded q_d, and the
residual is the difference of the predicted and measured q and dq at every
millisecond, in units of ``SIGMA_Q`` and ``SIGMA_DQ``. Ticks the robot missed
carry no residual. The parameters are written into a copy of the compiled
model (``dof_armature``, ``dof_frictionloss``, ``dof_damping``, and a constant
torque in ``actuator_biasprm``), so a candidate costs no recompilation.

The rollouts tighten MuJoCo's solver tolerance from 1e-8 to ``TOLERANCE``:
the dry friction is a very stiff soft constraint (``robot.FRICTION_SOLREF``),
and at the default tolerance where the Newton solver stops varies from
candidate to candidate, so finite differences of the residual are noise
(their size grows as 1 / step). At 1e-12 they agree from 1e-2 to 1e-5
relative steps; the optimizer differences with ``FD_STEP`` (its default,
1.5e-8, is in the noise even then).

``mujoco.sysid`` provides the parameter containers, the Levenberg-Marquardt
optimizer and the confidence intervals.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mujoco import sysid

DT = 0.001
# Residual units: a model error of this size counts as 1.
SIGMA_Q = 1e-3  # rad
SIGMA_DQ = 1e-2  # rad/s
TOLERANCE = 1e-12
FD_STEP = 1e-4  # absolute; the parameters are 0.01-1

# Parameter name -> (MjModel field, bounds for joints 1-4, bounds for joints 5-7).
# "bias" is a constant joint torque (N m), the offset of each joint's position
# actuator (actuator_biasprm[:, 0]): what the robot's gravity compensation
# leaves near one pose.
FIELDS = {
  "armature": ("dof_armature", (0.01, 1.5), (0.005, 0.5)),
  "frictionloss": ("dof_frictionloss", (0.0, 3.0), (0.0, 1.5)),
  "damping": ("dof_damping", (0.0, 10.0), (0.0, 3.0)),
  "bias": ("actuator_biasprm", (-2.0, 2.0), (-1.0, 1.0)),
}
# Ticks of tau_J_d a chunk keeps from before its start (panda_sysid.delayed: the
# torques the robot was still applying).
HISTORY = 8


def field_values(model: mujoco.MjModel, field: str) -> np.ndarray:
  """The model's per-joint values of ``field`` (a writable view)."""
  values = getattr(model, FIELDS[field][0])
  return values[:, 0] if field == "bias" else values


@dataclass
class Recording:
  """A run on a uniform 1 ms grid. ``valid`` is False where the robot sent no state."""

  name: str
  q_d: np.ndarray  # (n, 7) targets, the controller's at each tick
  q: np.ndarray  # (n, 7)
  dq: np.ndarray  # (n, 7)
  valid: np.ndarray  # (n,)
  # (n, 7) the torque command the robot last received (held over missed ticks),
  # and (HISTORY - 1, 7) its values on the ticks before the first.
  tau_J_d: np.ndarray | None = None
  tau_pre: np.ndarray | None = None
  # (n, 7) Franka's gravity model at q (the logged gravity1..7), and a known joint
  # torque the rollouts add (``add_gravity_offset``).
  gravity: np.ndarray | None = None
  tau_offset: np.ndarray | None = None

  def __len__(self) -> int:
    return len(self.q)

  def chunks(self, length: float) -> list[Recording]:
    """Consecutive pieces of ``length`` seconds that start on a received state."""
    n = round(length / DT)
    out = []
    start = 0
    while start + n <= len(self):
      if not self.valid[start]:
        start += 1
        continue
      sl = slice(start, start + n)
      tau_J_d = tau_pre = None
      if self.tau_J_d is not None:
        tau_J_d = self.tau_J_d[sl]
        before = np.concatenate([np.repeat(self.tau_pre[-1:], HISTORY, 0), self.tau_J_d[:start]])
        tau_pre = before[-(HISTORY - 1):]
      gravity = None if self.gravity is None else self.gravity[sl]
      tau_offset = None if self.tau_offset is None else self.tau_offset[sl]
      out.append(Recording(f"{self.name}[{start * DT:.1f}s]", self.q_d[sl], self.q[sl],
                           self.dq[sl], self.valid[sl], tau_J_d, tau_pre, gravity, tau_offset))
      start += n
    return out


def find_log(run_dir: str | Path) -> Path:
  """A run's log: <run_dir>/log.csv.gz, or log.csv as the players write it."""
  for name in ("log.csv.gz", "log.csv"):
    if (Path(run_dir) / name).exists():
      return Path(run_dir) / name
  raise FileNotFoundError(f"no log.csv(.gz) in {run_dir}")


def load_log(path: str | Path, name: str | None = None) -> Recording:
  """A ``real/joint_impedance`` log on its robot-time grid (1 ms); ``path`` may be gzipped."""
  header = np.loadtxt(path, delimiter=",", max_rows=1, dtype=str).tolist()
  table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
  t = np.rint(table[:, header.index("t_ms")]).astype(int)
  t -= t[0]
  n = t[-1] + 1

  def columns(prefix: str) -> np.ndarray:
    out = np.full((n, 7), np.nan)
    out[t] = table[:, [header.index(f"{prefix}{j}") for j in range(1, 8)]]
    return out

  q_d, q, dq, tau_J_d = columns("qd"), columns("q"), columns("dq"), columns("tau_J_d")
  gravity = columns("gravity") if "gravity1" in header else None
  valid = np.zeros(n, dtype=bool)
  valid[t] = True
  # A missed tick keeps the previous target (the trajectory moves by at most a
  # policy step between ticks, and the robot holds its last command).
  for k in range(1, n):
    if not valid[k]:
      q_d[k] = q_d[k - 1]
      tau_J_d[k] = tau_J_d[k - 1]
      if gravity is not None:
        gravity[k] = gravity[k - 1]
  return Recording(name or str(path), q_d, q, dq, valid, tau_J_d, tau_J_d[:1].repeat(HISTORY - 1, 0), gravity)


def add_gravity_offset(rec: Recording, model: mujoco.MjModel) -> None:
  """Set ``rec.tau_offset`` to the torque the robot's gravity compensation adds
  beyond ``model``'s gravity: Franka's logged model minus ``model``'s gravity at the
  logged q. The sim compensates its own gravity exactly, the robot with Franka's
  model; with true link inertials in ``model`` this is what the robot really adds."""
  if rec.gravity is None:
    raise ValueError(f"{rec.name} has no gravity columns")
  data = mujoco.MjData(model)
  out = np.zeros((len(rec), 7))
  ours = np.zeros(model.nv)
  for k in range(len(rec)):
    if not rec.valid[k]:
      out[k] = out[k - 1]
      continue
    data.qpos[:7] = rec.q[k]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_rne(model, data, 0, ours)  # zero velocity: the gravity torque
    out[k] = rec.gravity[k] - ours[:7]
  rec.tau_offset = out


def make_params(fields: tuple[str, ...] = tuple(FIELDS), joints: tuple[int, ...] = tuple(range(7)),
                nominal: mujoco.MjModel | None = None, inside: bool = True) -> sysid.ParameterDict:
  """One parameter per joint and field, nominal at the model's current value.

  ``inside`` moves values on a bound 1% of the range inward (an optimizer's start).
  """
  params = sysid.ParameterDict()
  for field in fields:
    _, big, small = FIELDS[field]
    for j in joints:
      lo, hi = big if j < 4 else small
      value = float(field_values(nominal, field)[j]) if nominal is not None else 0.5 * (lo + hi)
      if inside:  # no damping is the model's value and the lower bound
        value = float(np.clip(value, lo + 0.01 * (hi - lo), hi - 0.01 * (hi - lo)))
      params.add(sysid.Parameter(f"{field}{j + 1}", value, lo, hi))
  return params


def values(params: sysid.ParameterDict) -> dict[str, float]:
  return {name: float(param.value[0]) for name, param in params.items()}


def apply_values(model: mujoco.MjModel, values: dict[str, float]) -> mujoco.MjModel:
  """A copy of ``model`` with ``values`` (parameter name -> value) written in."""
  out = copy.copy(model)
  for name, value in values.items():
    field = name.rstrip("0123456789")
    j = int(name[len(field):]) - 1
    field_values(out, field)[j] = value
  data = mujoco.MjData(out)
  mujoco.mj_setConst(out, data)
  return out


def solver_model(model: mujoco.MjModel) -> mujoco.MjModel:
  """A copy of ``model`` with the identification's solver tolerance."""
  out = copy.copy(model)
  out.opt.tolerance = TOLERANCE
  return out


Group = tuple[mujoco.MjModel, list[Recording]]


def solve(problem, params: sysid.ParameterDict, max_iters: int = 50):
  """Least squares from ``params`` on a residual function (``delayed.Problem``);
  returns the fitted parameters, the optimizer result, the 95% confidence
  half-widths and the problem."""
  opt_params, result = sysid.optimize(params, problem, optimizer="mujoco", max_iters=max_iters,
                                      verbose=False, x_scale="jac", eps=FD_STEP)
  r, _, _ = problem(opt_params.as_vector(), opt_params)
  _, intervals = sysid.calculate_intervals(r, result.jac)
  return opt_params, result, intervals, problem
