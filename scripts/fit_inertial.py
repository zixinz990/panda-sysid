"""Fit the link inertials of the real Panda to the Fourier runs' joint torques.

The six Fourier runs of the campaign (three trajectories, cold and warm,
Franka's joint position controller) are fitted with ``panda_sysid.inertial``:
measured tau_J = rigid-body dynamics of link1..link7 (the hand at the Desk
values) + friction after the torque sensor, cold and warm separately.

Three models are compared, per joint, by the RMS of tau_J minus the model:
  nominal    the sim's inertials, only the friction fitted
  base       ordinary least squares on the base parameters (best linear fit,
             possibly not physical)
  physical   the fit regularized toward nominal by the log-det divergence
             (always physically consistent; the one used)
on the runs fitted and, leaving one trajectory out at a time, on runs not
fitted. A static check compares the gravity torques with tau_J held still at
the eight poses of the pose sessions, which the Fourier runs did not visit.

Outputs in --out:
  params.json   nominal and fitted mass, COM and inertia per link, friction,
                base-parameter fit, RMS tables, cross-validation, static check
  model.json    the fitted link inertials with the nominal joint values
                (``panda_sysid.simmodel``), the start of identify.py
  fit.png       tau_J and the models on a run left out of the fit
  static.png    tau_J at rest at the eight poses minus each model's gravity

Example:
  uv run python scripts/fit_inertial.py --logs data/campaign --out results/inertial
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from panda_sysid import campaign, delayed, fit, inertial, regressor, robot, simmodel

LOGS = "data/campaign"
# (run, trajectory, friction group: 0 cold, 1 warm)
FOURIER = campaign.FOURIER
POSE_SESSIONS = campaign.POSE_SESSIONS
SETTLE = 0.5  # s into a hold before averaging


def base_columns(n: int = 2000, seed: int = 0, tol: float = 1e-8) -> np.ndarray:
  """Independent link1..link7 columns (the hand fixed), from random states."""
  rng = np.random.default_rng(seed)
  model = regressor.rigid_model()
  q = rng.uniform(robot.Q_MIN, robot.Q_MAX, (n, 7))
  y = regressor.regressor(model, q, rng.normal(0, 1, (n, 7)), rng.normal(0, 3, (n, 7)))
  y = y[:, :, :10 * len(inertial.FITTED)].reshape(n * 7, -1)
  norms = np.linalg.norm(y, axis=0)
  active = np.nonzero(norms > 1e-6 * norms.max())[0]
  _, r, perm = scipy.linalg.qr(y[:, active] / norms[active], mode="economic", pivoting=True)
  diag = np.abs(np.diag(r))
  return np.sort(active[perm[:int((diag > tol * diag[0]).sum())]])


def _load_and_build(args) -> inertial.Problem:
  path, name, group, offset = args
  return inertial.build([inertial.load_run(path, name, group)], n_groups=2, offset=offset)


def merge(problems: list[inertial.Problem]) -> inertial.Problem:
  fields = ("y", "known", "friction", "tau", "mask")
  out = {f: np.concatenate([getattr(p, f) for p in problems]) for f in fields}
  out["run"] = np.concatenate([np.full(len(p.tau), i) for i, p in enumerate(problems)])
  return inertial.Problem(**out)


def _hold(args) -> dict:
  """Mean q, tau_J and Franka's gravity over a pose session's first hold."""
  name, root = args
  plan = json.loads(Path(f"{root}/{name}/plan.json").read_text())
  path = fit.find_log(f"{root}/{name}")
  header = np.loadtxt(path, delimiter=",", max_rows=1, dtype=str).tolist()
  data = np.loadtxt(path, delimiter=",", skiprows=1)
  t = (data[:, 0] - data[0, 0]) * 1e-3
  hold = next(s for s in plan["segments"] if s["name"] == "hold")
  rows = (t > hold["start_s"] + SETTLE) & (t < hold["end_s"])

  def mean(prefix):
    return np.array([data[rows, header.index(f"{prefix}{j}")].mean() for j in range(1, 8)])

  return {"run": name, "q": mean("q"), "tau_J": mean("tau_J"), "franka_gravity": mean("gravity")}


def gravity(pi: np.ndarray, q: np.ndarray) -> np.ndarray:
  """Gravity torques (n, 7) of link1..link7 at ``pi`` plus the hand and fingers at nominal."""
  model = regressor.rigid_model()
  zeros = np.zeros_like(q)
  y = regressor.regressor(model, q, zeros, zeros)[:, :, :10 * len(inertial.FITTED)]
  pi0 = np.concatenate([regressor.get_standard(model, b) for b in inertial.FITTED])
  return regressor.inverse_dynamics(model, q, zeros, zeros) - y @ pi0 + y @ pi


def fitted_dynamics(links: dict) -> robot.Dynamics:
  """Nominal joint values with the fitted inertials (mass, COM and inertia about it, rounded)."""
  inertials = {}
  for body, link in links.items():
    i_c = np.array(link["fitted"]["inertia_com"])
    inertials[body] = (round(link["fitted"]["mass"], 4), tuple(np.round(link["fitted"]["com"], 5).tolist()),
                       tuple(np.round([i_c[0, 0], i_c[1, 1], i_c[2, 2], i_c[0, 1], i_c[0, 2], i_c[1, 2]], 8).tolist()))
  nominal = robot.Dynamics.nominal()
  return robot.Dynamics(inertials, nominal.friction, nominal.damping, nominal.armature)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--logs", default=LOGS, help="directory of <run>/log.csv.gz and plan.json")
  parser.add_argument("--gamma", type=float, default=inertial.GAMMA, help="weight of the divergence from nominal")
  parser.add_argument("--offset", action="store_true", help="also fit a constant torque per joint")
  parser.add_argument("--workers", type=int, default=6)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  model = regressor.rigid_model()
  pi0 = np.concatenate([regressor.get_standard(model, b) for b in inertial.FITTED])
  columns = base_columns()
  print(f"{len(columns)} base parameters among link1..link7's {len(pi0)} (hand fixed)")
  # Compute only: forked workers, no GL anywhere in this script.
  with mp.get_context("fork").Pool(args.workers) as pool:
    problems = pool.map(_load_and_build, [(fit.find_log(f"{args.logs}/{name}"), name, group, args.offset)
                                          for name, _, group in FOURIER])
    holds = pool.map(_hold, [(name, args.logs) for name in POSE_SESSIONS])
  everything = merge(problems)
  names = [name for name, _, _ in FOURIER]
  print("samples per run:", [len(p.tau) for p in problems],
        "| equations used:", int(everything.mask.sum()), "of", everything.mask.size)

  # Weights: 1 / residual std per joint of an unweighted base fit on everything.
  first = inertial.fit_base(everything, columns, pi0, np.ones(7))
  weights = 1.0 / inertial.rms(everything, first.pi, first.friction)
  print("joint weights (1 / N m):", weights.round(2))

  def fit_all(problem):
    nominal = inertial.Fit(pi0, inertial.fit_friction(problem, pi0, weights), weights)
    base = inertial.fit_base(problem, columns, pi0, weights)
    physical, info = inertial.fit_physical(problem, pi0, weights, args.gamma)
    return {"nominal": nominal, "base": base, "physical": physical}, info

  # Leave one trajectory out (its cold and warm runs).
  folds = [([i for i, (_, t, _) in enumerate(FOURIER) if t != k], [i for i, (_, t, _) in enumerate(FOURIER) if t == k])
           for k in range(3)]
  scan = {}
  for gamma in (10.0, 100.0, 1000.0, 10000.0):
    test = []
    for tr, te in folds:
      f, _ = inertial.fit_physical(inertial.subset(everything, tr), pi0, weights, gamma)
      test.append(inertial.rms(inertial.subset(everything, te), f.pi, f.friction))
    full, full_info = inertial.fit_physical(everything, pi0, weights, gamma)
    scan[f"{gamma:g}"] = {"cv_test_rms_mean": float(np.mean(test)), "divergence": full_info["divergence"],
                          "masses": full.pi[0::10].round(3).tolist()}
    print(f"gamma {gamma:g}: mean test RMS {np.mean(test):.4f} N m, divergence {full_info['divergence']:.1f}, "
          f"masses {full.pi[0::10].round(2)}")
  cv = {}
  for k in range(3):
    train = [i for i, (_, traj, _) in enumerate(FOURIER) if traj != k]
    test = [i for i, (_, traj, _) in enumerate(FOURIER) if traj == k]
    fits, info = fit_all(inertial.subset(everything, train))
    test_problem = inertial.subset(everything, test)
    cv[f"trajectory {k + 1} left out"] = {
      "test_runs": [names[i] for i in test],
      **{m: inertial.rms(test_problem, f.pi, f.friction).round(3).tolist() for m, f in fits.items()}}
    print(f"left out {[names[i] for i in test]}: test RMS (N m)")
    for m, f in fits.items():
      print(f"  {m:9s}", inertial.rms(test_problem, f.pi, f.friction).round(3))
    if k == 2:
      shown = (test_problem, fits)

  fits, info = fit_all(everything)
  train_rms = {m: inertial.rms(everything, f.pi, f.friction).round(3).tolist() for m, f in fits.items()}
  print("fit on all six runs: RMS (N m)")
  for m, r in train_rms.items():
    print(f"  {m:9s}", np.round(r, 3))
  print("physical fit:", info)
  physical, base = fits["physical"], fits["base"]
  base_ok = {body: inertial.consistent(base.pi[10 * k:10 * k + 10]) for k, body in enumerate(inertial.FITTED)}
  print("base fit physically consistent per link:", base_ok)

  # Static check at the pose sessions' holds.
  q_hold = np.array([h["q"] for h in holds])
  tau_hold = np.array([h["tau_J"] for h in holds])
  offset = 0.5 * (physical.friction[14:21] + physical.friction[35:42])
  offset_nominal = 0.5 * (fits["nominal"].friction[14:21] + fits["nominal"].friction[35:42])
  static = {
    "poses": [h["run"] for h in holds],
    "tau_J": tau_hold.round(3).tolist(),
    "minus_franka_gravity": (tau_hold - np.array([h["franka_gravity"] for h in holds])).round(3).tolist(),
    "minus_nominal": (tau_hold - gravity(pi0, q_hold) - offset_nominal).round(3).tolist(),
    "minus_physical": (tau_hold - gravity(physical.pi, q_hold) - offset).round(3).tolist(),
  }
  for key in ("minus_franka_gravity", "minus_nominal", "minus_physical"):
    static[f"rms_{key}"] = np.sqrt(np.mean(np.square(static[key]), 0)).round(3).tolist()
    print(f"static, tau_J {key.replace('_', ' ')}: RMS over 8 poses (N m)", np.round(static[f"rms_{key}"], 3))

  friction = {}
  for g, label in enumerate(("cold", "warm")):
    f = physical.friction[21 * g:21 * (g + 1)]
    friction[label] = {"coulomb": f[:7].round(4).tolist(), "viscous": f[7:14].round(4).tolist(),
                       "offset": f[14:21].round(4).tolist()}
  links = {}
  for k, body in enumerate(inertial.FITTED):
    sl = slice(10 * k, 10 * k + 10)
    links[body] = {"nominal": inertial.describe(pi0[sl]), "fitted": inertial.describe(physical.pi[sl])}
  result = {
    "runs": names, "logs": args.logs, "gamma": args.gamma, "cutoff_hz": inertial.CUTOFF,
    "decimate": inertial.DECIMATE, "still_rad_s": inertial.STILL, "offset": args.offset,
    "weights": weights.round(4).tolist(),
    "n_base": len(columns), "base_columns": [regressor.names()[i] for i in columns],
    "total_mass": {"nominal": float(sum(pi0[0::10])), "fitted": float(sum(physical.pi[0::10]))},
    "links": links, "friction_after_sensor": friction,
    "rms_all_runs": train_rms, "cross_validation": cv, "static": static,
    "base_fit_consistent": base_ok,
    "standard": {"names": regressor.names()[:len(pi0)], "nominal": pi0.tolist(),
                 "physical": physical.pi.tolist(), "base_ols": base.pi.tolist()},
    "optimizer": info, "gamma_scan": scan,
  }
  (out / "params.json").write_text(json.dumps(result, indent=1))
  simmodel.save(simmodel.SimModel(fitted_dynamics(links), delayed.MEASURED, notes={
    "dynamics": f"link inertials fitted to {', '.join(names)} (fit_inertial.py, gamma {args.gamma:g}); "
                "joint values nominal", "delays": "measured on our Panda (delayed.MEASURED)"}), out / "model.json")

  test_problem, test_fits = shown
  first_test = test_problem.run == test_problem.run.max()
  t = np.arange(first_test.sum()) * inertial.DECIMATE * inertial.DT + inertial.TRIM
  fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
  for j in range(7):
    axes[j].plot(t, test_problem.tau[first_test, j], color="0.3", lw=0.8, label="tau_J (measured)")
    for m, style in (("nominal", "C1"), ("physical", "C0")):
      f = test_fits[m]
      model_tau = (test_problem.known + test_problem.y @ f.pi + test_problem.friction @ f.friction)[first_test, j]
      rms = np.sqrt(np.nanmean(inertial.residual(test_problem, f.pi, f.friction)[first_test, j] ** 2))
      axes[j].plot(t, model_tau, color=style, lw=0.8, ls="--" if m == "nominal" else "-",
                   label=f"{m} ({rms:.2f} N m RMS)")
    axes[j].set_ylabel(f"joint {j + 1} [N m]")
    axes[j].legend(loc="upper right", fontsize=8)
  axes[0].set_title(f"{names[test[-1]]}, left out of the fit: measured joint torque vs models fitted to the other "
                    "two trajectories")
  axes[-1].set_xlabel("t [s]")
  fig.tight_layout()
  fig.savefig(out / "fit.png", dpi=90)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(12, 5))
  width = 0.27
  for i, (key, label) in enumerate((("minus_franka_gravity", "Franka's model"), ("minus_nominal", "our nominal model"),
                                    ("minus_physical", "fitted model"))):
    ax.bar(np.arange(7) + (i - 1) * width, static[f"rms_{key}"], width, label=label)
  ax.set(xticks=range(7), xticklabels=[f"joint {j + 1}" for j in range(7)], ylabel="RMS over 8 poses [N m]",
         title="tau_J held still at the eight pose-session poses minus gravity (+ offset)")
  ax.legend()
  fig.tight_layout()
  fig.savefig(out / "static.png", dpi=90)
  plt.close(fig)
  print("wrote", out / "params.json")


if __name__ == "__main__":
  main()
