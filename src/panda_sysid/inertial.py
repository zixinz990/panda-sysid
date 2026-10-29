"""Link inertials of the arm from the joint torque sensors (Fourier runs).

The torque sensor of each joint sits between the gearbox and the link, so
tau_J sees the link side only: the rigid-body dynamics of the links beyond
it, Y(q, dq, ddq) pi (``panda_sysid.regressor``), plus the friction after the
sensor (Coulomb, viscous, a constant offset), not the rotor's inertia. The
runs are slow (harmonics up to 0.5 Hz), so q and tau_J are low-passed with
the same zero-phase filter and dq, ddq are differences of the filtered q.

A least-squares fit of the standard parameters can be physically
impossible (negative mass, inertia breaking the triangle inequality), and
the data fix only the 43 base combinations of them. So the fit adds, per
link, gamma times the log-det divergence of its pseudo-inertia from the
nominal one (Lee, Wensing and Park, T-RO 2020): it is finite only for
physically consistent links and keeps every direction the data cannot see
at the nominal values, and the whole problem stays convex (unique optimum,
found by damped Newton steps). The hand is rigid on link 7 and keeps the
Desk values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

from panda_sysid import regressor, robot

DT = 0.001
CUTOFF = 5.0  # Hz, zero-phase low-pass of q and tau_J
DECIMATE = 10  # keep every 10th sample (100 Hz) after filtering
TRIM = 1.0  # s cut at both ends (filter transients)
STILL = 0.02  # rad/s; a joint slower than this has undetermined static friction
FITTED = regressor.BODIES[:7]  # link1..link7; the hand stays at the Desk values
# Weight of the divergence from nominal (the data term is ~1e5 at unit-variance weights).
# Leaving one trajectory out, 10 to 1000 predict equally well (within 2% per joint), but
# below 1000 the weakly excited base parameters drift (a COM 26 cm out, rod-like
# inertias); 1000 keeps every link near the nominal values.
GAMMA = 1000.0
# kg m^2 added to the nominal second moments: Franka's link 4 is on the boundary of
# consistency (``robot._principal_inertia``), where the divergence is undefined.
MARGIN = 1e-4


@dataclass
class Samples:
  name: str
  q: np.ndarray  # (n, 7)
  dq: np.ndarray
  ddq: np.ndarray
  tau: np.ndarray  # tau_J
  group: int = 0  # friction group (cold 0, warm 1)


def load_run(path: str, name: str, group: int = 0) -> Samples:
  """A ``real/joint_position`` or ``real/joint_impedance`` log, filtered and decimated."""
  data = np.loadtxt(path, delimiter=",", skiprows=1)
  header = np.loadtxt(path, delimiter=",", max_rows=1, dtype=str).tolist()
  ms = np.rint(data[:, 0]).astype(int)
  ms -= ms[0]
  grid = np.arange(ms[-1] + 1)

  def column(prefix: str) -> np.ndarray:
    return np.column_stack([np.interp(grid, ms, data[:, header.index(f"{prefix}{j}")]) for j in range(1, 8)])

  b, a = butter(4, CUTOFF, fs=1.0 / DT)
  q = filtfilt(b, a, column("q"), axis=0)
  tau = filtfilt(b, a, column("tau_J"), axis=0)
  dq = np.gradient(q, DT, axis=0)
  ddq = np.gradient(dq, DT, axis=0)
  trim = round(TRIM / DT)
  keep = slice(trim, len(grid) - trim, DECIMATE)
  return Samples(name, q[keep], dq[keep], ddq[keep], tau[keep], group)


@dataclass
class Problem:
  """Everything the fits need from a set of runs, with the regressor evaluated once."""
  y: np.ndarray  # (n, 7, 70) link1..link7 standard-parameter columns
  known: np.ndarray  # (n, 7) torque of the bodies not fitted (hand, fingers) at nominal values
  friction: np.ndarray  # (n, 7, 21 * groups) Coulomb, viscous, offset per joint and group
  tau: np.ndarray  # (n, 7)
  mask: np.ndarray  # (n, 7) equations used (joint not still)
  run: np.ndarray  # (n,) index of the run


def build(samples: list[Samples], n_groups: int, offset: bool = False) -> Problem:
  """``offset`` adds a constant torque per joint. Off by default: Franka calibrates
  the torque sensors (tau_J at rest matches its gravity model to 0.1 N m), and
  over runs around one pose a constant would absorb the gravity error that the
  link masses must explain."""
  model = regressor.rigid_model()
  ys, knowns, frictions, taus, masks, runs = [], [], [], [], [], []
  pi0 = np.concatenate([regressor.get_standard(model, b) for b in FITTED])
  for i, s in enumerate(samples):
    full = regressor.regressor(model, s.q, s.dq, s.ddq)
    tau0 = regressor.inverse_dynamics(model, s.q, s.dq, s.ddq)
    y = full[:, :, :10 * len(FITTED)]
    f = np.zeros((len(s.q), 7, 21 * n_groups))
    f[:, :, 21 * s.group:21 * (s.group + 1)] = full[:, :, 10 * len(regressor.BODIES):]
    if not offset:
      f[:, :, 21 * s.group + 14:21 * (s.group + 1)] = 0.0
    ys.append(y)
    knowns.append(tau0 - y @ pi0)
    frictions.append(f)
    taus.append(s.tau)
    masks.append(np.abs(s.dq) > STILL)
    runs.append(np.full(len(s.q), i))
  return Problem(np.concatenate(ys), np.concatenate(knowns), np.concatenate(frictions),
                 np.concatenate(taus), np.concatenate(masks), np.concatenate(runs))


def subset(problem: Problem, runs: list[int]) -> Problem:
  keep = np.isin(problem.run, runs)
  return Problem(*(getattr(problem, f)[keep] for f in ("y", "known", "friction", "tau", "mask", "run")))


# -- physical consistency -------------------------------------------------------------

def pseudo_inertia(pi: np.ndarray) -> np.ndarray:
  """J = [[Sigma, m c], [m c^T, m]] of one body (Sigma = 0.5 tr(I_o) 1 - I_o, the
  second moment of its mass about its origin); realizable exactly when J > 0."""
  m, h = pi[0], pi[1:4]
  i_o = np.array([[pi[4], pi[7], pi[8]], [pi[7], pi[5], pi[9]], [pi[8], pi[9], pi[6]]])
  sigma = 0.5 * np.trace(i_o) * np.eye(3) - i_o
  return np.block([[sigma, h[:, None]], [h[None], np.array([[m]])]])


# J is linear in pi: J.ravel() = PSEUDO @ pi.
PSEUDO = np.stack([pseudo_inertia(e).ravel() for e in np.eye(10)], axis=1)


def consistent(pi: np.ndarray) -> bool:
  """Whether one body's standard parameters are physically realizable."""
  return bool(np.linalg.eigvalsh(pseudo_inertia(pi))[0] > 0)


@dataclass
class Fit:
  pi: np.ndarray  # (70,) standard parameters of link1..link7
  friction: np.ndarray  # (21 * groups,)
  weights: np.ndarray  # (7,) 1 / residual std per joint used in the fit


def residual(problem: Problem, pi: np.ndarray, friction: np.ndarray) -> np.ndarray:
  """tau_J minus the model, (n, 7), NaN where the joint is still."""
  r = problem.tau - problem.known - problem.y @ pi - problem.friction @ friction
  return np.where(problem.mask, r, np.nan)


def _rows(problem: Problem, weights: np.ndarray):
  w = np.broadcast_to(weights, problem.mask.shape)[problem.mask]
  a = np.concatenate([problem.y, problem.friction], axis=-1)[problem.mask] * w[:, None]
  b = (problem.tau - problem.known)[problem.mask] * w
  return a, b


def fit_friction(problem: Problem, pi: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
  """Friction only, with the inertials fixed at ``pi`` (e.g. nominal)."""
  w = np.ones(7) if weights is None else weights
  m = problem.mask
  ww = np.broadcast_to(w, m.shape)[m]
  a = problem.friction[m] * ww[:, None]
  b = (problem.tau - problem.known - problem.y @ pi)[m] * ww
  active = np.linalg.norm(a, axis=0) > 0
  out = np.zeros(a.shape[1])
  out[active] = np.linalg.lstsq(a[:, active], b, rcond=None)[0]
  return out


def fit_base(problem: Problem, columns: np.ndarray, pi0: np.ndarray, weights: np.ndarray) -> Fit:
  """Ordinary weighted least squares on the base columns (the other standard
  parameters stay at ``pi0``); the best linear fit, not necessarily physical."""
  a, b = _rows(problem, weights)
  n_inertial = problem.y.shape[-1]
  fixed = np.setdiff1d(np.arange(n_inertial), columns)
  b = b - a[:, fixed] @ pi0[fixed]
  cols = np.concatenate([columns, n_inertial + np.nonzero(np.linalg.norm(a[:, n_inertial:], axis=0) > 0)[0]])
  x = np.linalg.lstsq(a[:, cols], b, rcond=None)[0]
  pi = pi0.copy()
  pi[columns] = x[:len(columns)]
  friction = np.zeros(problem.friction.shape[-1])
  friction[cols[len(columns):] - n_inertial] = x[len(columns):]
  return Fit(pi, friction, weights)


def fit_physical(problem: Problem, pi0: np.ndarray, weights: np.ndarray, gamma: float = GAMMA,
                 iterations: int = 200) -> tuple[Fit, dict]:
  """Minimize 0.5 |weighted residual|^2 + gamma * sum over link1..link7 of
  d(J, J0) = tr(J0^-1 J) - log det(J0^-1 J) - 4 over (pi, friction)."""
  a, b = _rows(problem, weights)
  n_bodies = len(FITTED)
  n = 10 * n_bodies
  active = np.nonzero(np.linalg.norm(a[:, n:], axis=0) > 0)[0]
  a = np.hstack([a[:, :n], a[:, n:][:, active]])
  ata, atb, btb = a.T @ a, a.T @ b, b @ b
  bump = np.zeros(10)
  bump[4:7] = 2 * MARGIN  # adds MARGIN * 1 to Sigma
  start = pi0 + np.tile(bump, n_bodies)
  j0 = [pseudo_inertia(start[10 * k:10 * k + 10]) for k in range(n_bodies)]
  j0_inv = [np.linalg.inv(j) for j in j0]
  logdet0 = [np.linalg.slogdet(j)[1] for j in j0]

  def objective(x):
    reg = 0.0
    for k in range(n_bodies):
      j = (PSEUDO @ x[10 * k:10 * k + 10]).reshape(4, 4)
      try:
        logdet = 2 * np.log(np.diag(np.linalg.cholesky(j))).sum()
      except np.linalg.LinAlgError:
        return np.inf, np.inf
      reg += np.trace(j0_inv[k] @ j) - logdet + logdet0[k] - 4
    return 0.5 * x @ ata @ x - atb @ x + 0.5 * btb + gamma * reg, reg

  x = np.concatenate([start, fit_friction(problem, start, weights)[active]])
  f, _ = objective(x)
  for it in range(iterations):
    g = ata @ x - atb
    h = ata.copy()
    for k in range(n_bodies):
      sl = slice(10 * k, 10 * k + 10)
      j_inv = np.linalg.inv((PSEUDO @ x[sl]).reshape(4, 4))
      g[sl] += gamma * PSEUDO.T @ (j0_inv[k] - j_inv).ravel()
      h[sl, sl] += gamma * PSEUDO.T @ np.kron(j_inv, j_inv) @ PSEUDO
    # Directions the data cannot see make h nearly singular when gamma is small: damp it then.
    damping = 0.0
    while True:
      step = -np.linalg.solve(h + damping * np.eye(len(h)), g)
      decrement = float(-g @ step)  # Newton decrement squared
      if decrement > 0 or damping > 1e6:
        break
      damping = max(10 * damping, 1e-10 * np.trace(h) / len(h))
    if decrement < 1e-8:
      break
    t = 1.0
    while True:
      f_new, _ = objective(x + t * step)
      if f_new <= f - 0.25 * t * decrement or t < 1e-12:
        break
      t *= 0.5
    x, f = x + t * step, f_new
  pi = x[:n]
  friction = np.zeros(problem.friction.shape[-1])
  friction[active] = x[n:]
  _, reg = objective(x)
  info = {"objective": float(f), "divergence": float(reg), "gamma": gamma, "iterations": it + 1,
          "newton_decrement": decrement,
          "smallest_pseudo_inertia_eigenvalue": [float(np.linalg.eigvalsh(pseudo_inertia(pi[10 * k:10 * k + 10]))[0])
                                                 for k in range(n_bodies)]}
  return Fit(pi, friction, weights), info


def null_space(n: int = 2000, seed: int = 0, tol: float = 1e-9) -> np.ndarray:
  """(70, k) orthonormal directions of link1..link7's standard parameters that
  change no joint torque at any state (the hand fixed): what the data cannot see."""
  rng = np.random.default_rng(seed)
  q = rng.uniform(robot.Q_MIN, robot.Q_MAX, (n, 7))
  y = regressor.regressor(regressor.rigid_model(), q, rng.normal(0, 1, (n, 7)), rng.normal(0, 3, (n, 7)))
  _, s, vt = np.linalg.svd(y[:, :, :10 * len(FITTED)].reshape(n * 7, -1), full_matrices=True)
  return vt[(s > tol * s[0]).sum():].T


def closest_equivalent(pi: np.ndarray, pi0: np.ndarray, basis: np.ndarray,
                       iterations: int = 200) -> tuple[np.ndarray, dict]:
  """The standard parameters with the same torques as ``pi`` (pi + basis z) and the
  smallest summed log-det divergence from nominal: the same dynamics, split among
  the links as close to the nominal values as the data allow."""
  n_bodies = len(FITTED)
  bump = np.zeros(10)
  bump[4:7] = 2 * MARGIN
  j0_inv = [np.linalg.inv(pseudo_inertia((pi0 + np.tile(bump, n_bodies))[10 * k:10 * k + 10]))
            for k in range(n_bodies)]

  def divergence(z):
    x = pi + basis @ z
    total = 0.0
    for k in range(n_bodies):
      j = (PSEUDO @ x[10 * k:10 * k + 10]).reshape(4, 4)
      try:
        logdet = 2 * np.log(np.diag(np.linalg.cholesky(j))).sum()
      except np.linalg.LinAlgError:
        return np.inf
      total += np.trace(j0_inv[k] @ j) - logdet
    return total

  z = np.zeros(basis.shape[1])
  f = divergence(z)
  for it in range(iterations):
    x = pi + basis @ z
    g = np.zeros(len(pi))
    h = np.zeros((len(pi), len(pi)))
    for k in range(n_bodies):
      sl = slice(10 * k, 10 * k + 10)
      j_inv = np.linalg.inv((PSEUDO @ x[sl]).reshape(4, 4))
      g[sl] = PSEUDO.T @ (j0_inv[k] - j_inv).ravel()
      h[sl, sl] = PSEUDO.T @ np.kron(j_inv, j_inv) @ PSEUDO
    g, h = basis.T @ g, basis.T @ h @ basis
    step = -np.linalg.solve(h, g)
    decrement = float(-g @ step)
    if decrement < 1e-12:
      break
    t = 1.0
    while divergence(z + t * step) > f - 0.25 * t * decrement and t > 1e-12:
      t *= 0.5
    z = z + t * step
    f = divergence(z)
  return pi + basis @ z, {"iterations": it + 1, "divergence": float(f), "newton_decrement": decrement}


def rms(problem: Problem, pi: np.ndarray, friction: np.ndarray) -> np.ndarray:
  """RMS of tau_J minus the model per joint (N m), over the equations used."""
  return np.sqrt(np.nanmean(residual(problem, pi, friction) ** 2, axis=0))


def describe(pi: np.ndarray) -> dict:
  """Mass, center of mass and inertia about the center of mass (body frame) of one body."""
  m, c = pi[0], pi[1:4] / pi[0]
  i_o = np.array([[pi[4], pi[7], pi[8]], [pi[7], pi[5], pi[9]], [pi[8], pi[9], pi[6]]])
  i_c = i_o - m * (c @ c * np.eye(3) - np.outer(c, c))
  return {"mass": float(m), "com": c.tolist(), "inertia_com": i_c.tolist(),
          "principal_moments": np.linalg.eigvalsh(i_c).tolist()}
