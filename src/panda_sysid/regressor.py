"""The arm's rigid-body dynamics as a linear function of its inertial parameters.

Inverse dynamics is linear in each moving body's standard parameters
pi = (m, m c, I_o): mass, first moment, and the inertia about the body's
origin (xx, yy, zz, xy, xz, yz), all in the body frame. So
tau = Y(q, dq, ddq) pi, and the columns of Y follow exactly from MuJoCo's
inverse dynamics: tau(pi0 + h e_i) - tau(pi0) = h Y_i. Per joint, Coulomb
friction, viscous friction and a constant offset add three columns.

Not every parameter can be identified (link 1 only turns about z, the hand
is rigid on link 7, ...). The base parameters are an independent subset of
the columns, found by QR with column pivoting on a rich random sample
(``base_columns``). The condition number of a trajectory's base regressor,
columns normalized, measures how well the trajectory excites them (Swevers
et al. 1997; Gaz et al. 2019 for the Panda).
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import scipy.linalg

from panda_sysid import model as sysid_model
from panda_sysid import robot

BODIES = tuple(f"link{i}" for i in range(1, 8)) + ("hand",)
PARAMS = ("m", "mx", "my", "mz", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz")
FRICTION = ("coulomb", "viscous", "offset")
STEP = 1e-3
COULOMB_SPEED = 0.005  # rad/s; sign(dq) is smoothed as tanh(dq / COULOMB_SPEED)


def rigid_model() -> mujoco.MjModel:
  """The arm and the Franka Hand with nothing but rigid bodies: no armature,
  damping, friction or gravity compensation."""
  model = sysid_model.make_spec(40, 20).compile()
  model.dof_armature[:] = 0.0
  model.dof_damping[:] = 0.0
  model.dof_frictionloss[:] = 0.0
  model.body_gravcomp[:] = 0.0
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
  # The compiler marks bodies whose inertial frame is their own frame, and
  # kinematics then skips body_ipos / body_iquat; the regressor changes them.
  model.body_sameframe[:] = 0
  model.body_simple[:] = 0
  model.dof_simplenum[:] = 0
  return model


def names() -> list[str]:
  return ([f"{body}.{p}" for body in BODIES for p in PARAMS]
          + [f"{f}{j + 1}" for f in FRICTION for j in range(7)])


def get_standard(model: mujoco.MjModel, body: str) -> np.ndarray:
  b = model.body(body).id
  m, c = model.body_mass[b], model.body_ipos[b]
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, model.body_iquat[b])
  rot = rot.reshape(3, 3)
  i_o = rot @ np.diag(model.body_inertia[b]) @ rot.T + m * (c @ c * np.eye(3) - np.outer(c, c))
  return np.array([m, *(m * c), i_o[0, 0], i_o[1, 1], i_o[2, 2], i_o[0, 1], i_o[0, 2], i_o[1, 2]])


def set_standard(model: mujoco.MjModel, body: str, pi: np.ndarray) -> None:
  b = model.body(body).id
  m, c = pi[0], pi[1:4] / pi[0]
  i_o = np.array([[pi[4], pi[7], pi[8]], [pi[7], pi[5], pi[9]], [pi[8], pi[9], pi[6]]])
  moments, axes = np.linalg.eigh(i_o - m * (c @ c * np.eye(3) - np.outer(c, c)))
  if np.linalg.det(axes) < 0:
    axes[:, 0] *= -1
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, axes.flatten())
  model.body_mass[b] = m
  model.body_ipos[b] = c
  model.body_iquat[b] = quat
  model.body_inertia[b] = moments


def inverse_dynamics(model: mujoco.MjModel, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray,
                     data: mujoco.MjData | None = None) -> np.ndarray:
  data = mujoco.MjData(model) if data is None else data
  tau = np.empty((len(q), 7))
  for k in range(len(q)):
    data.qpos[:7], data.qvel[:7], data.qacc[:7] = q[k], dq[k], ddq[k]
    mujoco.mj_inverse(model, data)
    tau[k] = data.qfrc_inverse[:7]
  return tau


def regressor(model: mujoco.MjModel, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
  """Y (n, 7, 10 len(BODIES) + 21) at the samples (q, dq, ddq), from ``rigid_model``'s values."""
  n = len(q)
  columns = []
  # One copy, perturbed and restored body by body (a copy holds the meshes too).
  perturbed = copy.copy(model)
  data = mujoco.MjData(perturbed)
  tau0 = inverse_dynamics(perturbed, q, dq, ddq, data)
  for body in BODIES:
    pi0 = get_standard(model, body)
    for i in range(len(PARAMS)):
      pi = pi0.copy()
      pi[i] += STEP
      set_standard(perturbed, body, pi)
      columns.append((inverse_dynamics(perturbed, q, dq, ddq, data) - tau0) / STEP)
    b = model.body(body).id
    for name in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
      getattr(perturbed, name)[b] = getattr(model, name)[b]
  eye = np.eye(7)
  for f in (np.tanh(dq / COULOMB_SPEED), dq, np.ones_like(dq)):
    for j in range(7):
      columns.append(f[:, j:j + 1] * eye[j][None])
  return np.stack(columns, axis=-1).reshape(n, 7, -1)


def base_columns(model: mujoco.MjModel, n: int = 2000, seed: int = 0, tol: float = 1e-8) -> np.ndarray:
  """Indices of an independent set of regressor columns (the base parameters),
  from random states within the joint limits."""
  rng = np.random.default_rng(seed)
  q = rng.uniform(robot.Q_MIN, robot.Q_MAX, (n, 7))
  y = regressor(model, q, rng.normal(0, 1, (n, 7)), rng.normal(0, 3, (n, 7))).reshape(n * 7, -1)
  # Parameters with no effect at all (link 1's mass, ...) are round-off.
  norms = np.linalg.norm(y, axis=0)
  active = np.nonzero(norms > 1e-6 * norms.max())[0]
  _, r, perm = scipy.linalg.qr(y[:, active] / norms[active], mode="economic", pivoting=True)
  diag = np.abs(np.diag(r))
  return np.sort(active[perm[:int((diag > tol * diag[0]).sum())]])


def condition(y: np.ndarray, columns: np.ndarray) -> float:
  """Condition number of the base regressor with unit-norm columns."""
  w = y.reshape(-1, y.shape[-1])[:, columns]
  w = w / np.linalg.norm(w, axis=0).clip(1e-12)
  s = np.linalg.svd(w, compute_uv=False)
  return float(s[0] / s[-1])
