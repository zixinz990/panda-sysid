"""The identification campaign: which runs, where, and how each is checked.

The campaign records everything the model needs from the arm, under joint
impedance (kp 40, kd 20, through ``real/joint_impedance``) unless noted:

- reference: a fixed one-minute session at the center pose P1, at the start,
  after the Fourier runs, after the pose sessions and at the end (friction
  drift as the joints warm up; run-to-run repeatability, the floor of any
  sim-to-real error); once more with libfranka's torque filter off
  (``--cutoff 1000``), which tests the delay model.
- fourier: three large periodic motions of every joint around P1 under
  Franka's own joint position controller (``real/joint_position``), run
  twice (cold and warm). Their torque sensor readings identify the link
  inertials, which the small motions cannot: the 43 base inertial parameters
  (Gaz et al., 2019) plus friction. Designed by random search for the
  smallest condition number of the base regressor (``regressor``) within
  speed, acceleration, joint limit, box, base plane and self-collision limits.
- pose sessions at six poses spread over the box (one at its center, four
  toward alternate corners, one low with the hand tilted and turned) and two
  held-out poses (H1, H2): graded steps from inside the dry-friction dead
  band to beyond it, sweeps at 5, 20 and 60 mrad/s, policy-like steps,
  multisine (``excitation.session("pose")``). Each run first moves slowly
  from the previous pose (the end effector on a straight line, joints at most
  0.08 rad/s), which maps the gravity error between the poses.

The end effector stays in the workspace box (``workspace``) with
``BOX_MARGIN`` to spare in the plan: each pose is as far toward its corner
as its whole session allows, the moves are straight lines between poses, and
each Fourier trajectory is scaled down until it fits.

Every impedance run is simulated with the delayed controller (``delayed``)
and checked against the player's abort thresholds with margin; every run
(planned and simulated motion) is checked for the box, joint limits,
self-collision, every link's clearance from the base plane (z = 0) and
singularities (Franka's Cartesian reflex); the Fourier runs also against
Franka's speed, acceleration and jerk limits.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from panda_sysid import checks, delayed, excitation, fit, regressor, robot, workspace

DT = excitation.DT
KP, KD = 40.0, 20.0
# real/joint_impedance's abort thresholds for these runs; the simulation must
# stay within MARGIN of each.
MAX_ERROR, MAX_SPEED, MAX_TORQUE_FRACTION = 0.2, 1.0, 0.5
MARGIN = 0.5
TRANSFER_SPEED = 0.08  # rad/s, peak joint speed of the moves between poses
BOX_MARGIN = 0.01  # m the planned end effector keeps from the box's faces
MIN_CLEARANCE = 0.05  # m, every link above the base plane
# Smallest singular value of the EE Jacobian along a run (Franka's Cartesian
# reflex reads J^-T tau_ext; the Fourier runs move faster, so model errors are larger).
MIN_SIGMA = {"impedance": 0.12, "position": 0.15}
# Fourier runs: period, harmonics, full periods; per-joint largest excursion
# (rad), speed (rad/s) and acceleration (rad/s^2) of one period.
PERIOD, HARMONICS, REPEATS, FADE = 10.0, 5, 4, 6.0
F_RANGE = np.array([0.7, 0.35, 0.6, 0.5, 0.9, 0.6, 1.2])
F_SPEED = np.array([0.6, 0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
F_ACCEL = np.array([2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
F_CANDIDATES = 3000
F_REFINE_ROUNDS, F_REFINE_SIZE = 8, 200
F_MIN_SCALE = 0.25  # smallest fraction of the limits a Fourier candidate may shrink to

# End-effector poses: where in the box (per axis, 0 at lo and 1 at hi) and a
# turn of the hand from pointing down (degrees about the world's x, y, z axes,
# applied in that order). Each pose moves from its anchor toward the box's
# center until its whole session fits (``place_poses``).
ANCHORS = {
  "P1": ((0.5, 0.5, 0.5), (0, 0, 0)),
  "P2": ((0, 1, 0), (0, 0, 0)),
  "P3": ((1, 1, 1), (0, 0, 0)),
  "P4": ((1, 0, 0), (0, 0, 0)),
  "P5": ((0, 0, 1), (0, 0, 0)),
  "P6": ((0.5, 0.5, 0), (12, 0, 25)),
  "H1": ((0.25, 0.75, 0.75), (0, 0, 0)),
  "H2": ((0.75, 0.25, 0.25), (0, -10, -30)),
}


@dataclass
class Run:
  name: str
  kind: str  # "impedance" or "position"
  pose: str
  purpose: str
  session: str = ""  # impedance: excitation.session name
  seed: int = 0
  fourier: int = -1  # position: index of the designed Fourier trajectory
  cutoff: float = 100.0
  q_d: np.ndarray = field(default=None, repr=False)
  spans: list = field(default_factory=list, repr=False)


def campaign_runs() -> list[Run]:
  runs = [Run("r1", "impedance", "P1", "reference, cold", "reference")]
  runs += [Run(f"f{i + 1}", "position", "P1", "rigid body, cold", fourier=i) for i in range(3)]
  runs += [Run("r2", "impedance", "P1", "reference", "reference"),
           Run("r2_nofilter", "impedance", "P1", "reference, libfranka filter off", "reference",
               cutoff=1000.0)]
  runs += [Run(f"s_{p.lower()}", "impedance", p, "pose session", "pose", seed=i + 1)
           for i, p in enumerate(["P1", "P2", "P3", "P4", "P5", "P6"])]
  runs += [Run(f"h_{p.lower()}", "impedance", p, "pose session, held out", "pose", seed=i + 11)
           for i, p in enumerate(["H1", "H2"])]
  runs += [Run("r3", "impedance", "P1", "reference", "reference")]
  runs += [Run(f"f{i + 1}b", "position", "P1", "rigid body, warm", fourier=i) for i in range(3)]
  runs += [Run("r4", "impedance", "P1", "reference, warm", "reference")]
  for i, run in enumerate(runs):
    run.name = f"{i + 1:02d}_{run.name}"
  return runs


def run_names(kind: str | None = None, session: str | None = None, pose_prefix: str | None = None,
              cutoff: float | None = None) -> list[str]:
  """Names of the campaign's runs of a kind ("impedance", "position"), session,
  pose (prefix "P", or "H" for held out) or libfranka cutoff, in campaign order."""
  return [r.name for r in campaign_runs()
          if (kind is None or r.kind == kind) and (session is None or r.session == session)
          and (pose_prefix is None or r.pose.startswith(pose_prefix)) and (cutoff is None or r.cutoff == cutoff)]


# The runs each fit uses (names as on the robot: 01_r1, 02_f1, ...).
REFERENCE = run_names("impedance", "reference", cutoff=100.0)  # the repeatability floor
POSE_SESSIONS = run_names("impedance", "pose")  # P1-P6, H1, H2
HELD_OUT = run_names("impedance", "pose", "H")  # H1, H2
JOINT_FIT = REFERENCE + run_names("impedance", "pose", "P")  # the joint fit
IMPEDANCE = run_names("impedance")  # every impedance run (the bias range)
# (name, Fourier trajectory, friction group: 0 cold, 1 warm)
FOURIER = [(r.name, r.fourier, int("warm" in r.purpose)) for r in campaign_runs() if r.kind == "position"]


def pose_rotation(turn_deg) -> np.ndarray:
  rot = np.eye(3)
  for axis, angle in zip(range(3), np.radians(turn_deg)):
    r = np.zeros(9)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.eye(3)[axis], angle)
    mujoco.mju_quat2Mat(r, quat)
    rot = r.reshape(3, 3) @ rot
  return rot @ checks.DOWN


def solve_pose(model: mujoco.MjModel, pos: np.ndarray, turn) -> np.ndarray | None:
  """IK in HOME's branch (``checks.HOME``) for the end effector at ``pos``, turned by ``turn``."""
  seed = checks.HOME.copy()
  seed[0] = np.arctan2(pos[1], pos[0])
  seed[6] += np.radians(turn[2])
  q, err_m, err_deg = checks.ik(model, pos, pose_rotation(turn), seed)
  return q if err_m < 1e-4 and err_deg < 0.05 else None


def fits(model: mujoco.MjModel, q: np.ndarray, box, margin: float, every: int = 20) -> bool:
  """Whether the end effector stays in the box (less ``margin``) along q, with
  every link clear of the base plane, no self-contact and no near-singularity."""
  if not workspace.inside(checks.ee_positions(model, q, every), *box, margin).all():
    return False
  lowest, _, contacts = checks.clearance(model, q, every * 5)
  return (not contacts and lowest > MIN_CLEARANCE
          and checks.jacobian_sigma_min(model, q, every * 5) > MIN_SIGMA["impedance"] + 0.02)


def place_poses(model: mujoco.MjModel, box, sessions: dict[str, list[tuple[str, int]]]) -> dict[str, dict]:
  """Each pose from its anchor toward the box's center, in steps of 2%, until
  every session planned there (name, seed) fits with ``BOX_MARGIN``."""
  lo, hi = box
  center = 0.5 * (lo + hi)
  out = {}
  for name, (anchor, turn) in ANCHORS.items():
    target = lo + np.array(anchor) * (hi - lo)
    offsets = [excitation.session(*session)[0] for session in sessions[name]]
    for t in np.arange(1.0, -1e-9, -0.02):
      pos = center + t * (target - center)
      q = solve_pose(model, pos, turn)
      if q is not None and all(fits(model, q + o, box, BOX_MARGIN) for o in offsets):
        out[name] = {"q": q, "position": pos, "turn": turn, "toward_anchor": round(float(t), 2)}
        break
    else:
      raise ValueError(f"no place for {name} in the box")
  return out


# -- Fourier design -------------------------------------------------------------

_DESIGN: dict = {}


def _fourier_period(coefficients: np.ndarray, t: np.ndarray):
  """Offsets, velocities and accelerations of one period at times t."""
  w = 2 * np.pi * np.arange(1, HARMONICS + 1) / PERIOD
  s, c = np.sin(w[None] * t[:, None]), np.cos(w[None] * t[:, None])
  a, b = coefficients[..., 0].T, coefficients[..., 1].T  # (L, 7)
  q = s @ a + c @ b
  dq = (c * w) @ a - (s * w) @ b
  ddq = -(s * w**2) @ a - (c * w**2) @ b
  return q, dq, ddq


def _feasible(q_abs: np.ndarray) -> bool:
  model = _DESIGN["check"]
  if checks.limit_margin(q_abs) < 0.15:
    return False
  if not workspace.inside(checks.ee_positions(model, q_abs, 5), *_DESIGN["box"], BOX_MARGIN).all():
    return False
  lowest, _, contacts = checks.clearance(model, q_abs, every=10)
  if contacts or lowest < MIN_CLEARANCE:
    return False
  return checks.jacobian_sigma_min(model, q_abs, every=10) > MIN_SIGMA["position"] + 0.02


def _candidate(seed: int, start: np.ndarray | None = None) -> tuple[float, int, np.ndarray | None]:
  """A random Fourier trajectory (or ``start`` randomly perturbed) scaled to the
  limits, then shrunk (bisection) until it fits the box and the checks;
  (condition number, seed, coefficients)."""
  rng = np.random.default_rng(seed)
  if start is None:
    coefficients = rng.normal(size=(7, HARMONICS, 2)) / np.arange(1, HARMONICS + 1)[None, :, None]
  else:
    coefficients = start + 0.15 * np.abs(start).mean() * rng.normal(size=start.shape)
  t = np.arange(0, PERIOD, 0.01)
  q, dq, ddq = _fourier_period(coefficients, t)
  scale = np.minimum.reduce([F_RANGE / np.abs(q).max(0), F_SPEED / np.abs(dq).max(0),
                             F_ACCEL / np.abs(ddq).max(0)])
  q, dq, ddq = q * scale, dq * scale, ddq * scale
  center = _DESIGN["center"]
  alpha = 1.0
  if not _feasible(center + q):
    low, high = 0.0, 1.0
    for _ in range(7):
      mid = 0.5 * (low + high)
      low, high = (mid, high) if _feasible(center + mid * q) else (low, mid)
    alpha = low
    if alpha < F_MIN_SCALE:
      return np.inf, seed, None
  coefficients *= (alpha * scale)[:, None, None]
  q, dq, ddq = alpha * q, alpha * dq, alpha * ddq
  y = regressor.regressor(_DESIGN["rigid"], center + q[::10], dq[::10], ddq[::10])
  return regressor.condition(y, _DESIGN["columns"]), seed, coefficients


def design_fourier(center: np.ndarray, box, n_candidates: int = F_CANDIDATES, n_best: int = 3,
                   workers: int = 48) -> tuple[list[np.ndarray], dict]:
  """The ``n_best`` random Fourier trajectories around ``center`` with the
  smallest condition numbers, and a summary."""
  rigid = regressor.rigid_model()
  _DESIGN.update(center=center, box=box, check=checks.check_model(), rigid=rigid,
                 columns=regressor.base_columns(rigid))
  # Forked workers only compute (this process has no GL context).
  with mp.get_context("fork").Pool(workers) as pool:
    results = pool.map(_candidate, range(n_candidates), chunksize=4)
    feasible = sorted((r for r in results if r[2] is not None), key=lambda r: r[0])
    best = feasible[:n_best]
    first = [round(r[0], 1) for r in best]
    # Hill climbing from each of the best: keep the best of random perturbations.
    for round_ in range(F_REFINE_ROUNDS):
      for i, (cond, seed, start) in enumerate(best):
        tries = pool.starmap(_candidate, [(n_candidates * (round_ + 1) + 1000 * i + k, start)
                                          for k in range(F_REFINE_SIZE)], chunksize=2)
        top = min(tries, key=lambda r: r[0])
        if top[0] < cond:
          best[i] = top
  t = np.arange(0, PERIOD, 0.1)
  stacked = []
  for _, _, coefficients in best:
    q, dq, ddq = _fourier_period(coefficients, t)
    stacked.append(regressor.regressor(rigid, center + q, dq, ddq))
  summary = {
    "candidates": n_candidates, "feasible": len(feasible), "condition_random_search": first,
    "condition_best": [round(r[0], 1) for r in best], "seeds": [r[1] for r in best],
    "condition_median_feasible": round(float(np.median([r[0] for r in feasible])), 1),
    "condition_all_three": round(regressor.condition(np.concatenate(stacked), _DESIGN["columns"]), 1),
    "base_parameters": len(_DESIGN["columns"]),
  }
  return [r[2] for r in best], summary


# -- building and checking runs ---------------------------------------------------

def build(run: Run, q_from: np.ndarray, poses: dict[str, np.ndarray], fouriers: list[np.ndarray],
          model: mujoco.MjModel) -> None:
  """Fill run.q_d (absolute, 1 kHz) and run.spans, starting at q_from."""
  if run.kind == "position":
    seg = excitation.fourier(fouriers[run.fourier], PERIOD, REPEATS, FADE)
    hold = np.zeros((round(0.5 / DT), 7))
    run.q_d = q_from + np.concatenate([hold, seg.offsets, hold])
    run.spans = [(f"fourier {run.fourier + 1}", 0.5, 0.5 + len(seg.offsets) * DT)]
    return
  q_pose = poses[run.pose]["q"]
  transfer = checks.line_path(model, q_from, q_pose, TRANSFER_SPEED)
  offsets, spans = excitation.session(run.session, run.seed)
  run.q_d = np.concatenate([transfer, q_pose + offsets])
  t0 = len(transfer) * DT
  run.spans = [(f"move to {run.pose}", 0.0, t0)] + [(n, t0 + a, t0 + b) for n, a, b in spans]


def simulate(run: Run, model: mujoco.MjModel, delays: delayed.Delays) -> dict[str, np.ndarray]:
  """The impedance run on the controller of ``delays`` (with the run's libfranka
  cutoff), from rest at its first target."""
  n = len(run.q_d)
  rec = fit.Recording(run.name, run.q_d, np.repeat(run.q_d[:1], n, 0), np.zeros((n, 7)),
                      np.ones(n, dtype=bool), np.zeros((n, 7)), np.zeros((fit.HISTORY - 1, 7)))
  delays = replace(delays, cutoff=run.cutoff)
  torques, kp, kd = delayed.torque_model(model)
  q, dq = delayed.rollout(torques, mujoco.MjData(torques), rec, kp, kd, delays)
  return {"q": q, "dq": dq, "tau": kp * (run.q_d - q) - kd * dq}


def check(run: Run, check_model: mujoco.MjModel, sim: dict | None, box) -> dict:
  q = run.q_d if sim is None else sim["q"]
  lowest, lowest_body, contacts = checks.clearance(check_model, np.concatenate([run.q_d, q]))
  ee = checks.ee_positions(check_model, np.concatenate([run.q_d, q]), every=10)
  to_faces = np.minimum(ee - box[0], box[1] - ee)
  report = {
    "duration_s": round(len(run.q_d) * DT, 1),
    "limit_margin_rad": round(checks.limit_margin(np.concatenate([run.q_d, q])), 3),
    "lowest_link_above_base_plane_m": round(lowest, 3), "lowest_link": lowest_body,
    "self_contacts": contacts[:5],
    "ee_min": ee.min(0).round(4).tolist(), "ee_max": ee.max(0).round(4).tolist(),
    "ee_to_box_faces_mm": round(float(to_faces.min()) * 1e3, 1),
  }
  report["sigma_min"] = round(checks.jacobian_sigma_min(check_model, q), 3)
  report["ee_speed_m_s"] = round(checks.ee_speed(check_model, q), 3)
  ok = {"box": report["ee_to_box_faces_mm"] >= 0.5 * BOX_MARGIN * 1e3,
        "joint limits": report["limit_margin_rad"] > checks.LIMIT_MARGIN,
        "clearance": lowest > MIN_CLEARANCE, "self-collision": not contacts,
        "singularity": report["sigma_min"] > MIN_SIGMA[run.kind]}
  if run.kind == "position":
    report.update({k: round(v, 3) for k, v in checks.derivative_fractions(run.q_d).items()})
    ok["speed, acceleration, jerk"] = max(report["speed_fraction"], report["acceleration_fraction"],
                                          report["jerk_fraction"]) < 0.5
  else:
    limits = np.array(robot.EFFORT_LIMITS)
    report["sim_tracking_error_rad"] = np.abs(run.q_d - sim["q"]).max(0).round(4).tolist()
    report["sim_speed_rad_s"] = np.abs(sim["dq"]).max(0).round(3).tolist()
    report["sim_torque_fraction"] = (np.abs(sim["tau"]).max(0) / limits).round(3).tolist()
    ok["tracking error"] = max(report["sim_tracking_error_rad"]) < MARGIN * MAX_ERROR
    ok["speed"] = max(report["sim_speed_rad_s"]) < MARGIN * MAX_SPEED
    ok["torque"] = max(report["sim_torque_fraction"]) < MARGIN * MAX_TORQUE_FRACTION
  report["passed"] = ok
  return report


def command(run: Run, out: Path, host: str) -> str:
  """The player's command line for ``run`` planned in ``out``."""
  d = out / run.name
  if run.kind == "position":
    return f"real/build/joint_position {host} {d / 'traj.csv'} {d / 'log.csv'}"
  cmd = (f"real/build/joint_impedance {host} {d / 'traj.csv'} {d / 'log.csv'} --kp {KP:g} "
         f"--kd {KD:g} --max-error {MAX_ERROR} --max-speed {MAX_SPEED} "
         f"--max-torque-fraction {MAX_TORQUE_FRACTION}")
  return cmd + (f" --cutoff {run.cutoff:g}" if run.cutoff != 100.0 else "")
