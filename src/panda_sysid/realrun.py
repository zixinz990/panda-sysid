"""Summaries of a run on the real robot, written next to its log.

``impedance``: timing (missed ticks), tracking, external torque, and the
real q against a replay of the same q_d on a model with its delayed
controller. ``position`` (a Fourier run): timing, tracking, the real end
effector against the workspace box, and the measured joint torque against
Franka's gravity model and the nominal rigid-body model.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.signal import savgol_filter

from panda_sysid import checks, delayed, fit, regressor, robot, simmodel
from panda_sysid import model as sysid_model


def _table(path: Path):
  table = np.genfromtxt(path, delimiter=",", names=True)

  def col(prefix: str) -> np.ndarray:
    return np.column_stack([table[f"{prefix}{j}"] for j in range(1, 8)])

  ms = np.rint(table["t_ms"]).astype(int)
  return col, ms - ms[0]


def _timing(ms: np.ndarray) -> dict:
  gaps = np.diff(ms)
  return {"states": len(ms), "missed_ticks": int((gaps - 1).sum()),
          "missed_fraction": round(float((gaps - 1).sum() / ms[-1]), 4), "longest_gap_ms": int(gaps.max())}


def impedance(run_dir: str | Path, sim: simmodel.SimModel, kp: float = 40.0, kd: float = 20.0) -> dict:
  """Summary of a ``real/joint_impedance`` run (check.json, real_vs_sim.png)."""
  d = Path(run_dir)
  plan = json.loads((d / "plan.json").read_text())
  log = fit.find_log(d)
  col, ms = _table(log)
  rec = fit.load_log(log)
  model = sysid_model.make_spec(kp, kd, sim.dynamics).compile()
  torques, kp_, kd_ = delayed.torque_model(model)
  delays = sim.delays if sim.delays == delayed.IDEAL else replace(sim.delays, cutoff=plan["cutoff"])
  q_sim, _ = delayed.rollout(torques, mujoco.MjData(torques), rec, kp_, kd_, delays)
  t = np.arange(len(rec)) * fit.DT
  err = np.where(rec.valid[:, None], q_sim - rec.q, np.nan)
  summary = {
    **_timing(ms),
    "max_tracking_error_rad": np.abs(col("qd") - col("q")).max(0).round(4).tolist(),
    "max_abs_tau_ext": np.abs(col("tau_ext")).max(0).round(2).tolist(),
    "max_abs_dq": np.abs(col("dq")).max(0).round(3).tolist(),
    "max_tau_cmd_fraction": (np.abs(col("tau_cmd")).max(0) / np.array(robot.EFFORT_LIMITS)).round(3).tolist(),
    "rms_sim_minus_real_deg": np.degrees(np.sqrt(np.nanmean(err**2, 0))).round(3).tolist(),
  }
  hold = next((s for s in plan["segments"] if s["name"] == "hold"), None)
  if hold is not None:
    rows = np.isin(ms, np.flatnonzero((t > hold["start_s"] + 0.5) & (t < hold["end_s"])))
    summary["hold_tau_J_minus_gravity_minus_tau_J_d"] = (
      col("tau_J") - col("gravity") - col("tau_J_d"))[rows].mean(0).round(3).tolist()
  (d / "check.json").write_text(json.dumps(summary, indent=1))

  fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
  for j in range(7):
    base = rec.q_d[0, j]
    axes[j].plot(t, np.degrees(rec.q_d[:, j] - base), color="0.6", lw=0.6, label="q_d")
    axes[j].plot(t, np.degrees(rec.q[:, j] - base), color="C0", lw=0.8, label="real q")
    axes[j].plot(t, np.degrees(q_sim[:, j] - base), color="C3", lw=0.8, ls="--", label="sim q")
    axes[j].set_ylabel(f"q{j + 1} [deg]")
    for s in plan["segments"]:
      axes[j].axvspan(s["start_s"], s["end_s"], color="0.94", zorder=0)
  axes[0].legend(ncol=3, loc="upper left")
  axes[0].set_title(f"{d.name}: real vs sim (offsets from the start); sim - real RMS (deg) "
                    f"{summary['rms_sim_minus_real_deg']}")
  axes[-1].set_xlabel("t [s]")
  fig.tight_layout()
  fig.savefig(d / "real_vs_sim.png", dpi=90)
  plt.close(fig)
  return summary


def position(run_dir: str | Path, box: tuple[np.ndarray, np.ndarray]) -> dict:
  """Summary of a ``real/joint_position`` run (check.json, torque.png)."""
  d = Path(run_dir)
  col, ms = _table(fit.find_log(d))
  # On the full 1 kHz grid (missed ticks interpolated) for differentiation.
  grid = np.arange(ms[-1] + 1)
  q = np.column_stack([np.interp(grid, ms, c) for c in col("q").T])
  dq = savgol_filter(q, 41, 3, deriv=1, delta=1e-3, axis=0)
  ddq = savgol_filter(q, 41, 3, deriv=2, delta=1e-3, axis=0)
  keep = ms[(ms > 100) & (ms < ms[-1] - 100)]
  rows = np.isin(ms, keep)
  tau_J, grav = col("tau_J")[rows], col("gravity")[rows]
  tau_nominal = regressor.inverse_dynamics(regressor.rigid_model(), q[keep], dq[keep], ddq[keep])
  ee = checks.ee_positions(checks.check_model(), q, every=10)
  lo, hi = box
  summary = {
    **_timing(ms),
    "max_abs_q_cmd_minus_q_deg": np.degrees(np.abs(col("q_cmd") - col("q")).max(0)).round(3).tolist(),
    "max_abs_dq": np.abs(col("dq")).max(0).round(3).tolist(),
    "max_abs_tau_ext": np.abs(col("tau_ext")).max(0).round(2).tolist(),
    "max_abs_tau_J": np.abs(tau_J).max(0).round(2).tolist(),
    "real_ee_min": ee.min(0).round(4).tolist(), "real_ee_max": ee.max(0).round(4).tolist(),
    "real_ee_to_box_faces_mm": round(float(np.min(np.minimum(ee - lo, hi - ee)) * 1e3), 1),
    "rms_tau_J_minus_franka_gravity": np.sqrt(((tau_J - grav) ** 2).mean(0)).round(2).tolist(),
    "rms_tau_J_minus_nominal_rigid_body": np.sqrt(((tau_J - tau_nominal) ** 2).mean(0)).round(2).tolist(),
  }
  (d / "check.json").write_text(json.dumps(summary, indent=1))
  t = keep * 1e-3
  fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
  for j in range(7):
    axes[j].plot(t, tau_J[:, j], lw=0.7, label="tau_J (measured)")
    axes[j].plot(t, tau_nominal[:, j], lw=0.7, ls="--", label="nominal rigid-body model")
    axes[j].plot(t, grav[:, j], lw=0.7, color="0.5", label="Franka's gravity")
    axes[j].set_ylabel(f"joint {j + 1} [N m]")
  axes[0].legend(ncol=3, loc="upper left")
  axes[0].set_title(f"{d.name}: measured joint torque vs models (no friction in either)")
  axes[-1].set_xlabel("t [s]")
  fig.tight_layout()
  fig.savefig(d / "torque.png", dpi=90)
  plt.close(fig)
  return summary

