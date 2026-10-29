"""Fit a constant torque bias to each run, with the model's joint values fixed.

The bias stands for what the model leaves out at a pose and stays nearly
constant through a run (a session at one pose). Each run is cut into chunks
that start from the measured state, as identify.py fits, and replayed with
the model's delayed controller and Franka's gravity compensation minus the
model's added (``fit.add_gravity_offset``); only the seven biases are fitted.

A simulation compensates gravity exactly, so the bias it must draw is the
fitted one plus the run's mean gravity offset ("total"). The range of the
totals over the runs, widened by --margin on each side, gives a center and a
half-width per joint, which the output model.json holds as its torque bias.

Outputs:
  <out>/bias_per_run.json   per run: bias and ci95, gravity offset (mean, min, max),
                            total, q RMS before and after the bias (deg); the range
  --model-out               --model with the torque bias range (default <out>/model.json)

Example:
  uv run python scripts/bias_per_run.py --logs data/campaign --model results/joints/model.json \\
      --out results/bias --model-out results/model.json
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

KP, KD = campaign.KP, campaign.KD


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--model", required=True, help="model.json: inertials, joint values, delays")
  parser.add_argument("--runs", default=",".join(campaign.IMPEDANCE), help="impedance runs of the campaign")
  parser.add_argument("--logs", default="data/campaign", help="directory of <run>/log.csv.gz and plan.json")
  parser.add_argument("--chunk", type=float, default=4.0, help="seconds per rollout")
  parser.add_argument("--max-iters", type=int, default=15)
  parser.add_argument("--margin", type=float, default=0.1, help="N m the range is widened by on each side")
  parser.add_argument("--workers", type=int, default=min(60, os.cpu_count() or 1))
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", help="where to write the model with the bias range")
  args = parser.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  sim = simmodel.load(args.model)
  model = sysid_model.make_spec(KP, KD, sim.dynamics).compile()
  runs = {}
  for run in args.runs.split(","):
    plan = json.loads(Path(f"{args.logs}/{run}/plan.json").read_text())
    rec = fit.load_log(fit.find_log(f"{args.logs}/{run}"), run)
    fit.add_gravity_offset(rec, model)
    offset = rec.tau_offset[rec.valid]
    problem = delayed.Problem([(model, rec.chunks(args.chunk))], replace(sim.delays, cutoff=plan["cutoff"]),
                              args.workers)
    params = fit.make_params(("bias",), nominal=model)
    opt, _, ci, _ = fit.solve(problem, params, max_iters=args.max_iters)
    before, after = problem.rms(params)["q_deg"], problem.rms(opt)["q_deg"]
    problem.close()
    bias = np.array([p.value[0] for p in opt.values()])
    runs[run] = {"bias": bias.round(4).tolist(), "ci95": np.round(ci, 4).tolist(),
                 "gravity_offset": {"mean": offset.mean(0).round(4).tolist(),
                                    "min": offset.min(0).round(4).tolist(),
                                    "max": offset.max(0).round(4).tolist()},
                 "total": (bias + offset.mean(0)).round(4).tolist(),
                 "q_rms_deg": {"before": before.round(4).tolist(), "after": after.round(4).tolist()}}
    print(f"{run}: bias {bias.round(3)}  total {np.round(runs[run]['total'], 3)}\n"
          f"  q RMS (deg) {before.round(3)} -> {after.round(3)}", flush=True)

  totals = np.array([r["total"] for r in runs.values()])
  lo, hi = totals.min(0) - args.margin, totals.max(0) + args.margin
  after = np.array([r["q_rms_deg"]["after"] for r in runs.values()])
  result = {"model": args.model, "delays": sim.delays.as_dict(), "chunk_s": args.chunk, "margin": args.margin,
            "runs": runs,
            "total": {"min": totals.min(0).round(3).tolist(), "max": totals.max(0).round(3).tolist()},
            "center": ((lo + hi) / 2).round(2).tolist(), "half_width": ((hi - lo) / 2).round(2).tolist(),
            "q_rms_after_deg": {"mean": after.mean(0).round(3).tolist(), "max": after.max(0).round(3).tolist()}}
  for key in ("total", "center", "half_width", "q_rms_after_deg"):
    print(key, result[key])
  (out / "bias_per_run.json").write_text(json.dumps(result, indent=1))
  model_out = Path(args.model_out) if args.model_out else out / "model.json"
  simmodel.save(replace(sim, bias_center=tuple(result["center"]), bias_half_width=tuple(result["half_width"]),
                        notes={**sim.notes, "torque_bias": f"range of the per-run totals over {len(runs)} runs "
                               f"(bias_per_run.py), widened by {args.margin:g} N m on each side"}), model_out)
  print("wrote", out / "bias_per_run.json", model_out)


if __name__ == "__main__":
  main()
