"""Rollouts with the real controller's signal path, for fitting the arm.

MuJoCo's position actuator computes the torque of a step from that step's
state and applies it at once. On the robot it does not. From the robot's
logs (``real/joint_impedance``):

- ``dq`` in the robot state lags the joint velocity (the derivative of ``q``)
  by 2.5-3.25 ms at gain 0.99, on every joint and run; the controller's
  damping acts on it, and the logged ``dq`` is that signal too.
- libfranka low-pass filters the torque (first order, 100 Hz unless
  ``--cutoff 1000``) and limits its rate to 1 N m per ms, both against
  ``tau_J_d``, the command the robot last received; ``tau_J_d`` of the next
  state equals that output to 0.016 N m.
- The robot applies a command some ticks after it was sent (``torque_lag``;
  ``tau_J`` follows ``tau_J_d`` by about 3 ms).

A fit that leaves them out makes up for them with other armature, damping
and friction. Here the controller runs in Python around ``mj_step`` with the
recorded ``q_d``: the position actuators keep only their bias and the gravity
compensation, and the torque enters as ``qfrc_applied``, held over the 1 ms
step as on the robot. The predicted ``dq`` is the robot's (lagged) one. A
recording's ``tau_offset`` (``fit.add_gravity_offset``) is added to the
torque. Chunks run in parallel processes (12 us per step).
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from mujoco import sysid

from panda_sysid import fit


@dataclass(frozen=True)
class Delays:
  dq_lag: int = 3  # ticks the robot's dq lags the joint velocity
  torque_lag: int = 1  # ticks from sending a torque to the first step it acts in
  cutoff: float = 100.0  # Hz, libfranka's low-pass; 1000 turns it off
  rate_limit: float = 1.0  # N m per tick (libfranka's kMaxTorqueRate); inf turns it off

  @property
  def alpha(self) -> float:
    if self.cutoff >= 1000.0:
      return 1.0
    return fit.DT / (fit.DT + 1.0 / (2 * np.pi * self.cutoff))

  def as_dict(self) -> dict:
    return asdict(self)


# MuJoCo's position actuator: no delay, no filter, no rate limit.
IDEAL = Delays(dq_lag=0, torque_lag=0, cutoff=1000.0, rate_limit=np.inf)
# The lab's Panda (libfranka 0.8.0, no real-time kernel): dq 3 ticks late, torque 2 ticks
# late, libfranka's 100 Hz low-pass and rate limit. Of torque lags 0-3, 2 fits best.
MEASURED = Delays(dq_lag=3, torque_lag=2)


def torque_model(model: mujoco.MjModel) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray]:
  """A copy of ``model`` whose actuators apply only their bias and gravity
  compensation, with the solver tolerance of identification; and kp, kd."""
  out = fit.solver_model(model)
  kp = out.actuator_gainprm[:, 0].copy()
  kd = -out.actuator_biasprm[:, 2].copy()
  out.actuator_gainprm[:, 0] = 0.0
  out.actuator_biasprm[:, 1:3] = 0.0
  return out, kp, kd


def rollout(model: mujoco.MjModel, data: mujoco.MjData, rec: fit.Recording, kp: np.ndarray,
            kd: np.ndarray, delays: Delays) -> tuple[np.ndarray, np.ndarray]:
  """q and the robot's dq at every tick of ``rec``, from its first state.

  ``model`` is a ``torque_model``. The joint velocity at the start is the
  logged dq ``dq_lag`` ticks later; the robot's dq before then is the logged
  one, and the torques sent before the start are the logged tau_J_d.
  """
  n, lag, hist = len(rec), delays.dq_lag, fit.HISTORY
  if delays.torque_lag >= hist:
    raise ValueError(f"torque_lag must be below {hist}")
  alpha, rate = delays.alpha, delays.rate_limit
  # sent[hist + k]: the torque sent at tick k (k < 0 from the log).
  sent = np.empty((hist + n, 7))
  sent[:hist - 1] = rec.tau_pre
  sent[hist - 1] = rec.tau_J_d[0]
  mujoco.mj_resetData(model, data)
  data.qpos[:7] = rec.q[0]
  data.qvel[:7] = rec.dq[lag] if rec.valid[lag] else rec.dq[0]
  q = np.empty((n, 7))
  v = np.empty((n, 7))
  q[0], v[0] = data.qpos[:7], data.qvel[:7]
  for k in range(n):
    if rec.valid[k]:
      dq_robot = v[k - lag] if k >= lag else rec.dq[k]
      command = kp * (rec.q_d[k] - q[k]) - kd * dq_robot
      prev = sent[hist + k - 1]
      sent[hist + k] = prev + np.clip(alpha * (command - prev), -rate, rate)
    else:  # no state, no command: the robot holds the last one
      sent[hist + k] = sent[hist + k - 1]
    if k == n - 1:
      break
    data.qfrc_applied[:7] = sent[hist + k - delays.torque_lag]
    if rec.tau_offset is not None:
      data.qfrc_applied[:7] += rec.tau_offset[k]
    mujoco.mj_step(model, data)
    q[k + 1], v[k + 1] = data.qpos[:7], data.qvel[:7]
  dq = np.empty_like(v)
  dq[lag:] = v[:n - lag]
  dq[:lag] = np.nan_to_num(rec.dq[:lag])
  return q, dq


# Worker state, inherited through fork: [(model, recordings, kp, kd)], delays.
_GROUPS: list = []
_DELAYS: list[Delays] = []


def _task(args) -> tuple[np.ndarray, np.ndarray]:
  values, g, i = args
  model, recordings, kp, kd = _GROUPS[g]
  candidate = fit.apply_values(model, values)
  return rollout(candidate, mujoco.MjData(candidate), recordings[i], kp, kd, _DELAYS[0])


def residuals(q: np.ndarray, dq: np.ndarray, rec: fit.Recording) -> np.ndarray:
  """Errors in units of fit.SIGMA_Q and fit.SIGMA_DQ at every valid tick after the first."""
  mask = rec.valid.astype(float)[:, None]
  mask[0] = 0.0
  r_q = (q - np.nan_to_num(rec.q)) / fit.SIGMA_Q * mask
  r_dq = (dq - np.nan_to_num(rec.dq)) / fit.SIGMA_DQ * mask
  return np.concatenate([r_q.ravel(), r_dq.ravel()])


class Problem:
  """The residual function for ``fit.solve``: (x, params) -> ([r], None, None).

  Each group pairs a model (e.g. one controller gain) with the recordings
  (chunks) made under it; every group gets the same parameters. Each chunk
  is rolled out with the controller of ``delays`` in a process of its own.
  """

  def __init__(self, groups: list[fit.Group], delays: Delays, n_workers: int = 100):
    self.delays = delays
    self.groups = []
    for model, recordings in groups:
      torques, kp, kd = torque_model(model)
      self.groups.append((torques, recordings, kp, kd))
    _GROUPS[:] = self.groups
    _DELAYS[:] = [delays]
    self.pool = mp.get_context("fork").Pool(n_workers)

  def close(self) -> None:
    self.pool.close()

  def _predict(self, value_sets: list[dict[str, float]]):
    tasks = [(values, g, i) for values in value_sets for g, (_, recordings, _, _) in enumerate(self.groups)
             for i in range(len(recordings))]
    return iter(self.pool.map(_task, tasks, chunksize=1))

  def __call__(self, x: np.ndarray, params: sysid.ParameterDict):
    columns = x if x.ndim == 2 else x[:, None]
    value_sets = []
    for i in range(columns.shape[1]):
      params.update_from_vector(columns[:, i])
      value_sets.append(fit.values(params))
    results = self._predict(value_sets)
    r = np.stack([
      np.concatenate([residuals(*next(results), rec) for _, recordings, _, _ in self.groups
                      for rec in recordings])
      for _ in value_sets
    ])
    return [r.T if x.ndim == 2 else r[0]], None, None

  def rms(self, params: sysid.ParameterDict, group: int | None = None) -> dict[str, np.ndarray]:
    """Per-joint RMS errors of q (deg) and the robot's dq (deg/s) under ``params``,
    over every group or one."""
    results = self._predict([fit.values(params)])
    errors: dict[str, list[np.ndarray]] = {"q_deg": [], "dq_deg_s": []}
    for g, (_, recordings, _, _) in enumerate(self.groups):
      for rec in recordings:
        q, dq = next(results)
        if group is not None and g != group:
          continue
        mask = rec.valid[:, None]
        errors["q_deg"].append(np.where(mask, q - np.nan_to_num(rec.q), np.nan))
        errors["dq_deg_s"].append(np.where(mask, dq - np.nan_to_num(rec.dq), np.nan))
    return {key: np.degrees(np.sqrt(np.nanmean(np.concatenate(err) ** 2, axis=0)))
            for key, err in errors.items()}
