"""Fit the arm's joint parameters to real runs (``real/joint_impedance`` logs).

Every log is cut into chunks of --chunk seconds, each rolled out from its
measured state under the logged q_d with the robot's controller
(``panda_sysid.delayed``: the robot's lagged dq, libfranka's torque filter and
rate limit, a torque delay), on the model with that log's gains (--kp: one
value, or one per log). All logs share the joint parameters (--fields:
armature, dry friction, damping, and a constant torque bias). The fit starts
from the joint values of --model, or of --init, and reports per parameter
the start and fitted values with 95% confidence half-widths, and the RMS
errors of q and dq at the start and after.

--gravity-offset (logs with gravity columns) adds what the robot's gravity
compensation adds beyond the model's gravity (``fit.add_gravity_offset``), so
the bias does not have to stand in for it. Use it with link inertials that
match the robot (fit_inertial.py's model.json).

Outputs in --out:
  params.json   start, fit, ci95 per parameter; RMS at the start and after; the logs
  model.json    --model with the fitted joint values (but --keep's start values)
                and the delays (``panda_sysid.simmodel``)

Example (the campaign's joint fit):
  uv run python scripts/identify.py --logs data/campaign --model results/inertial/model.json \\
      --fields armature,frictionloss,damping,bias --gravity-offset --init results/joints/init.json \\
      --keep armature1,armature2,armature3,armature4 --out results/joints
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(_var, "1")

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from panda_sysid import campaign, delayed, fit, simmodel
from panda_sysid import model as sysid_model


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--logs", default="data/campaign", help="directory of <run>/log.csv.gz")
  parser.add_argument("--runs", default=",".join(campaign.JOINT_FIT), help="runs in --logs")
  parser.add_argument("--model", default="nominal", help="model.json (or 'nominal'): link inertials, start values")
  parser.add_argument("--kp", default="40", help="one value, or one per run")
  parser.add_argument("--kd", type=float, default=20.0)
  parser.add_argument("--fields", default="armature,frictionloss,damping,bias")
  parser.add_argument("--chunk", type=float, default=4.0, help="seconds per rollout")
  parser.add_argument("--max-iters", type=int, default=50)
  parser.add_argument("--init", help="start from this params.json's fit instead of --model's joint values")
  parser.add_argument("--keep", default="", help="parameters model.json keeps at their start values")
  parser.add_argument("--dq-lag", type=int, default=delayed.MEASURED.dq_lag, help="ticks")
  parser.add_argument("--torque-lag", type=int, default=delayed.MEASURED.torque_lag, help="ticks")
  parser.add_argument("--cutoff", type=float, default=delayed.MEASURED.cutoff, help="Hz")
  parser.add_argument("--rate-limit", type=float, default=delayed.MEASURED.rate_limit, help="N m per tick")
  parser.add_argument("--gravity-offset", action="store_true", help="add Franka's gravity minus the model's")
  parser.add_argument("--workers", type=int, default=min(100, os.cpu_count() or 1), help="rollout processes")
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  runs = args.runs.split(",")
  gains = [float(k) for k in args.kp.split(",")]
  gains = gains * len(runs) if len(gains) == 1 else gains
  if len(gains) != len(runs):
    parser.error("--kp takes one value or one per run")
  start_model = simmodel.load(args.model)
  dynamics = start_model.dynamics
  by_gain: dict[float, list] = {}
  reference = sysid_model.make_spec(40, args.kd, dynamics).compile()
  for run, kp in zip(runs, gains):
    rec = fit.load_log(fit.find_log(f"{args.logs}/{run}"), run)
    if args.gravity_offset:
      fit.add_gravity_offset(rec, reference)
    by_gain.setdefault(kp, []).extend(rec.chunks(args.chunk))
  groups = [(sysid_model.make_spec(kp, args.kd, dynamics).compile(), chunks) for kp, chunks in by_gain.items()]
  for kp, chunks in by_gain.items():
    print(f"kp {kp:g}: {len(chunks)} chunks of {args.chunk} s")
  params = fit.make_params(tuple(args.fields.split(",")), nominal=groups[0][0])
  start = params.copy()
  start_values = {**dynamics.joint_values(), **{name: 0.0 for name in params.keys() if name.startswith("bias")}}
  if args.init:
    init = json.loads(Path(args.init).read_text())["fit"]
    start_values.update({name: init[name] for name in params.keys() if name in init})
    for name, param in start.items():
      if name not in init:
        continue
      lo, hi = param.get_bounds()
      margin = 0.01 * (hi[0] - lo[0])  # off the bounds, as make_params
      param.value[0] = np.clip(init[name], lo[0] + margin, hi[0] - margin)
  delays = delayed.Delays(args.dq_lag, args.torque_lag, args.cutoff, args.rate_limit)
  print(f"controller: {delays}")
  problem = delayed.Problem(groups, delays, args.workers)
  opt, _, intervals, problem = fit.solve(problem, start, max_iters=args.max_iters)
  r = problem(opt.as_vector(), opt)[0][0]
  cost = float(0.5 * r @ r)
  print(f"cost {cost:.6g} over {r.size} residuals")
  rms_start = problem.rms(start)
  rms_fit = problem.rms(opt)
  rms_by_gain = {f"{kp:g}": {"start": problem.rms(start, g)["q_deg"].round(4).tolist(),
                             "fit": problem.rms(opt, g)["q_deg"].round(4).tolist()}
                 for g, kp in enumerate(by_gain)}
  problem.close()

  print(f"{'param':>14} {'start':>9} {'fit':>9} {'+-95%':>9}")
  for (name, p_fit), ci in zip(opt.items(), intervals):
    print(f"{name:>14} {start[name].value[0]:9.4f} {p_fit.value[0]:9.4f} {ci:9.4f}")
  for label, rms in (("start", rms_start), ("fit", rms_fit)):
    print(f"RMS {label:>5}: q (deg) {np.round(rms['q_deg'], 4)}  dq (deg/s) {np.round(rms['dq_deg_s'], 3)}")
  keep = [k for k in args.keep.split(",") if k]
  fitted = {name: float(p.value[0]) for name, p in opt.items()}
  result = {
    "logs": args.logs, "runs": runs, "kp": gains, "kd": args.kd, "chunk_s": args.chunk,
    "delays": delays.as_dict(), "model": args.model, "init": args.init, "cost": cost,
    "gravity_offset": args.gravity_offset, "keep": keep,
    "start": {name: float(p.value[0]) for name, p in start.items()},
    "fit": fitted,
    "ci95": {name: float(ci) for name, ci in zip(opt.keys(), intervals)},
    "rms_start": {k: v.round(4).tolist() for k, v in rms_start.items()},
    "rms_fit": {k: v.round(4).tolist() for k, v in rms_fit.items()},
    "q_rms_by_gain_deg": rms_by_gain,
  }
  (out / "params.json").write_text(json.dumps(result, indent=2))
  # Joint values rounded to 1e-3 (finer digits mean nothing: the runs at other poses differ more).
  values = {name: (start_values[name] if name in keep else round(v, 3)) for name, v in fitted.items()}
  simmodel.save(replace(start_model, dynamics=dynamics.with_values(values), delays=delays, notes={
    **start_model.notes,
    "joints": f"fitted to {', '.join(runs)} (identify.py, fields {args.fields}, gravity offset "
              f"{args.gravity_offset}); kept at the start: {', '.join(keep) or 'none'}",
    "delays": f"assumed in the joint fit: {delays}"}), out / "model.json")
  print("wrote", out / "params.json", out / "model.json")


if __name__ == "__main__":
  main()
