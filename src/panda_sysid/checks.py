"""Planning helpers for motions of the real Panda: robot state, IK, and safety checks.

Shared by ``scripts/move.py`` (moves between poses) and the campaign
(``panda_sysid.campaign``).
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline

from panda_sysid import model as sysid_model
from panda_sysid import robot

DT = 0.001
# Franka's joint limits ("Limits for Panda" in the FCI documentation).
DQ_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
DDQ_MAX = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
DDDQ_MAX = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000])
LIMIT_MARGIN = 0.1  # rad from every joint limit
# Hand pointing down, its x axis along the world's x axis (columns: x, y, z).
DOWN = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
HOME = sysid_model.HOME
# Bodies whose collision geoms must stay clear of the base plane.
CLEAR_BODIES = ("link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")


def read_state(path: str) -> dict:
  """The last state that ``echo_robot_state`` printed to ``path``."""
  with open(path) as f:
    lines = [line for line in f if line.startswith("{")]
  return json.loads(lines[-1])


def check_model() -> mujoco.MjModel:
  """The arm and hand with collision geoms and the base plane (z = 0, about where the table is)."""
  spec = sysid_model.make_spec(80, 20, contact=True)
  spec.worldbody.add_geom(name="base_plane", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(2, 2, 0.1),
                          contype=0, conaffinity=0)
  return spec.compile()


def ik(model: mujoco.MjModel, pos: np.ndarray, rot: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, float, float]:
  """Damped least squares on the end-effector pose; returns q, position and angle errors."""
  data = mujoco.MjData(model)
  site = model.site("ee").id
  lo = robot.Q_MIN + LIMIT_MARGIN
  hi = robot.Q_MAX - LIMIT_MARGIN
  q = seed.copy()
  jacp = np.zeros((3, model.nv))
  jacr = np.zeros((3, model.nv))
  for _ in range(500):
    data.qpos[:7] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    err_p = pos - data.site_xpos[site]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (rot @ data.site_xmat[site].reshape(3, 3).T).flatten())
    err_r = np.zeros(3)
    mujoco.mju_quat2Vel(err_r, quat, 1.0)
    err = np.concatenate([err_p, err_r])
    if np.linalg.norm(err) < 1e-9:
      break
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    jac = np.vstack([jacp, jacr])[:, :7]
    step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), err)
    # Pull toward the seed in the null space (keeps the seed's IK branch).
    null = np.eye(7) - np.linalg.pinv(jac) @ jac
    q = np.clip(q + step + 0.05 * null @ (seed - q), lo, hi)
  return q, float(np.linalg.norm(err_p)), float(np.degrees(np.linalg.norm(err_r)))


def ik_down(model: mujoco.MjModel, pos: np.ndarray, rot: np.ndarray = DOWN) -> tuple[np.ndarray, float, float]:
  """IK from ``HOME`` turned toward ``pos`` (HOME's IK branch)."""
  seed = HOME.copy()
  seed[0] = np.arctan2(pos[1], pos[0])
  return ik(model, pos, rot, seed)


def slerp(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
  """Rotation matrix a fraction ``s`` of the way from r0 to r1 (about their relative axis)."""
  rel = r1 @ r0.T
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, rel.flatten())
  vel = np.zeros(3)
  mujoco.mju_quat2Vel(vel, quat, 1.0)
  part = np.zeros(4)
  angle = np.linalg.norm(vel)
  mujoco.mju_axisAngle2Quat(part, vel / angle if angle > 1e-12 else np.array([0.0, 0, 1]), s * angle)
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, part)
  return mat.reshape(3, 3) @ r0


def line_path(model: mujoco.MjModel, q0: np.ndarray, q1: np.ndarray, max_speed: float,
              min_time: float = 3.0, step: float = 0.005) -> np.ndarray:
  """Absolute q at 1 kHz moving the end effector on the straight line from its
  pose at q0 to its pose at q1 (orientation turned about one axis), with a
  minimum-jerk profile along the line and no joint faster than ``max_speed``.

  IK every ``step`` m (and every degree), seeded on the straight joint-space
  line between q0 and q1, so the elbow moves as it would there and the path
  ends exactly at q1.
  """
  data = mujoco.MjData(model)
  site = model.site("ee").id

  def pose(q):
    data.qpos[:7] = q
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()

  (p0, r0), (p1, r1) = pose(q0), pose(q1)
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, (r1 @ r0.T).flatten())
  angle = 2 * np.arccos(np.clip(abs(quat[0]), 0, 1))
  n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)), int(np.ceil(np.degrees(angle)))) + 1
  knots = np.linspace(0, 1, n)
  path = [q0]
  for s in knots[1:-1]:
    q, err_p, err_deg = ik(model, p0 + s * (p1 - p0), slerp(r0, r1, s), q0 + s * (q1 - q0))
    if err_p > 1e-4 or err_deg > 0.05:
      raise ValueError(f"no IK on the line at s = {s:.3f}: {err_p * 1e3:.2f} mm, {err_deg:.2f} deg")
    path.append(q)
  path.append(q1)
  spline = CubicSpline(knots, np.array(path))
  # Joint speeds at unit duration: dq/ds at s(tau) times ds/dtau.
  tau = np.linspace(0, 1, 20 * n)
  speed = np.abs(spline(10 * tau**3 - 15 * tau**4 + 6 * tau**5, 1)).max(1) * 30 * tau**2 * (1 - tau)**2
  duration = max(min_time, speed.max() / max_speed)
  t = np.arange(round(duration / DT) + 1) * DT / duration
  return spline(10 * t**3 - 15 * t**4 + 6 * t**5)


def ee_positions(model: mujoco.MjModel, q: np.ndarray, every: int = 10) -> np.ndarray:
  data = mujoco.MjData(model)
  site = model.site("ee").id
  out = np.empty((len(range(0, len(q), every)), 3))
  for i, k in enumerate(range(0, len(q), every)):
    data.qpos[:7] = q[k]
    mujoco.mj_kinematics(model, data)
    out[i] = data.site_xpos[site]
  return out


def clearance(model: mujoco.MjModel, q: np.ndarray, every: int = 20) -> tuple[float, str, list]:
  """Smallest distance of the checked links from the base plane, and self-contacts."""
  data = mujoco.MjData(model)
  plane = model.geom("base_plane").id
  geoms = [g for g in range(model.ngeom)
           if model.body(model.geom_bodyid[g]).name in CLEAR_BODIES and model.geom_contype[g]]
  fromto = np.zeros(6)
  lowest, where, contacts = np.inf, "", []
  for k in range(0, len(q), every):
    data.qpos[:7] = q[k]
    mujoco.mj_forward(model, data)
    for g in geoms:
      dist = mujoco.mj_geomDistance(model, data, g, plane, 1.0, fromto)
      if dist < lowest:
        lowest, where = dist, model.body(model.geom_bodyid[g]).name
    for c in data.contact[:data.ncon]:
      names = {model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name}
      if "world" not in names:
        contacts.append((k * DT, *sorted(names)))
  return float(lowest), where, contacts


def jacobian_sigma_min(model: mujoco.MjModel, q: np.ndarray, every: int = 20) -> float:
  """Smallest singular value of the end-effector Jacobian along ``q`` (Franka's
  Cartesian reflex reads J^-T tau_ext, which grows near singularities)."""
  data = mujoco.MjData(model)
  site = model.site("ee").id
  jacp = np.zeros((3, model.nv))
  jacr = np.zeros((3, model.nv))
  worst = np.inf
  for k in range(0, len(q), every):
    data.qpos[:7] = q[k]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    worst = min(worst, float(np.linalg.svd(np.vstack([jacp, jacr])[:, :7], compute_uv=False)[-1]))
  return worst


def ee_speed(model: mujoco.MjModel, q: np.ndarray, every: int = 10) -> float:
  """Largest end-effector speed (m/s) along ``q`` at 1 kHz."""
  data = mujoco.MjData(model)
  site = model.site("ee").id
  pos = []
  for k in range(0, len(q), every):
    data.qpos[:7] = q[k]
    mujoco.mj_kinematics(model, data)
    pos.append(data.site_xpos[site].copy())
  return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).max() / (every * DT))


def derivative_fractions(q: np.ndarray) -> dict[str, float]:
  """Largest speed, acceleration and jerk of a 1 kHz path as fractions of Franka's limits."""
  dq = np.diff(q, axis=0) / DT
  ddq = np.diff(dq, axis=0) / DT
  dddq = np.diff(ddq, axis=0) / DT
  return {"speed_fraction": float((np.abs(dq) / DQ_MAX).max()),
          "acceleration_fraction": float((np.abs(ddq) / DDQ_MAX).max()),
          "jerk_fraction": float((np.abs(dddq) / DDDQ_MAX).max())}


def limit_margin(q: np.ndarray) -> float:
  return float(min((q - robot.Q_MIN).min(), (robot.Q_MAX - q).min()))
