"""Replay real runs whole in several sim models and compare with the run-to-run floor.

Each --model is label:model[:offset[:bias]], where model is a model.json (or
"nominal": the published values with an ideal controller), offset (0 or 1,
default 0) adds what the robot's gravity compensation adds beyond the model's
gravity (``fit.add_gravity_offset``), and bias is "center" (the middle of
model.json's bias range) or a bias_per_run.json whose bias for the run is added
(each run its own; bias_per_run.json@<run> adds that run's bias to every run). Each log is rolled out from its
first state over its whole length with the model's controller
(``panda_sysid.delayed``; the run's libfranka cutoff from its plan.json) and
compared with the measured q.

The floor (the robot's repeatability): the reference runs (--floor) repeat
one session; the RMS difference of their measured q, aligned at the session's
start, is what no model can beat. Errors are given over the whole run and over its session
(after the move to the pose), which is what the floor covers.

Outputs in --out:
  validate.json   RMS of sim minus real q per model, run and joint (deg), and the floor
  validate.png    the errors per joint against the floor, and one run's q over time

Example:
  uv run python scripts/validate.py --logs data/campaign --model "before identification:nominal" \\
      --model "identified, bias from the run:results/model.json:1:results/bias/bias_per_run.json" \\
      --out results/validate
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from panda_sysid import campaign, delayed, fit, simmodel
from panda_sysid import model as sysid_model

KP, KD = campaign.KP, campaign.KD


def session_start(plan: dict) -> int:
  """Tick where the run's session starts (its first hold, after the move to the pose; 0 if none)."""
  hold = next((s for s in plan["segments"] if s["name"] == "hold"), None)
  return round(hold["start_s"] / fit.DT) if hold else 0


def _replay(args) -> tuple[str, str, np.ndarray]:
  (label, path, offset, run_bias), run, root = args
  plan = json.loads(Path(f"{root}/{run}/plan.json").read_text())
  rec = fit.load_log(fit.find_log(f"{root}/{run}"), run)
  sim = simmodel.load(path)
  model = sysid_model.make_spec(KP, KD, sim.dynamics).compile()
  if run_bias == "center":
    model = fit.apply_values(model, {f"bias{j + 1}": b for j, b in enumerate(sim.bias_center)})
  elif run_bias != "-":
    path, _, source = run_bias.partition("@")
    bias = json.loads(Path(path).read_text())["runs"][source or run]["bias"]
    model = fit.apply_values(model, {f"bias{j + 1}": b for j, b in enumerate(bias)})
  if offset:
    fit.add_gravity_offset(rec, model)
  torques, kp, kd = delayed.torque_model(model)
  delays = sim.delays if sim.delays == delayed.IDEAL else replace(sim.delays, cutoff=plan["cutoff"])
  q, _ = delayed.rollout(torques, mujoco.MjData(torques), rec, kp, kd, delays)
  return label, run, np.where(rec.valid[:, None], q - rec.q, np.nan)


def _real_q(args) -> np.ndarray:
  run, root = args
  plan = json.loads(Path(f"{root}/{run}/plan.json").read_text())
  rec = fit.load_log(fit.find_log(f"{root}/{run}"), run)
  return rec.q[session_start(plan):]


def rms_deg(err: np.ndarray) -> np.ndarray:
  return np.degrees(np.sqrt(np.nanmean(err ** 2, axis=0)))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--model", action="append", required=True, help="label:model[:offset[:bias]]")
  parser.add_argument("--runs", default=",".join(campaign.HELD_OUT), help="runs to replay")
  parser.add_argument("--floor", default=",".join(campaign.REFERENCE), help="reference runs (same session)")
  parser.add_argument("--logs", default="data/campaign", help="directory of <run>/log.csv.gz and plan.json")
  parser.add_argument("--workers", type=int, default=16)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  models = []
  for spec in args.model:
    label, path, *rest = spec.split(":")
    models.append((label, path, bool(rest) and rest[0] == "1", rest[1] if len(rest) > 1 else "-"))
  runs = args.runs.split(",")
  refs = args.floor.split(",")

  # Compute only (forked workers, no GL).
  with mp.get_context("fork").Pool(args.workers) as pool:
    replays = pool.map(_replay, [(m, run, args.logs) for m in models for run in runs], chunksize=1)
    real = pool.map(_real_q, [(run, args.logs) for run in refs])
  starts = {run: session_start(json.loads(Path(f"{args.logs}/{run}/plan.json").read_text())) for run in runs}
  n = min(len(q) for q in real)
  pairs = {f"{a} vs {b}": rms_deg(real[i][:n] - real[j][:n]).round(3).tolist()
           for (i, a), (j, b) in combinations(enumerate(refs), 2)}
  floor = np.mean([v for v in pairs.values()], axis=0)
  print(f"floor (real vs real, mean of {len(pairs)} pairs of {refs}), deg:", floor.round(3))

  result = {"models": {label: {"model": p, "gravity_offset": o, "run_bias": b} for label, p, o, b in models},
            "floor_deg": floor.round(3).tolist(), "floor_pairs_deg": pairs, "runs": {}}
  errors = {}
  for label, run, err in replays:
    errors[(label, run)] = err
    result["runs"].setdefault(run, {})[label] = {
      "whole_deg": rms_deg(err).round(3).tolist(), "session_deg": rms_deg(err[starts[run]:]).round(3).tolist()}
  for run in runs:
    print(run)
    for label, *_ in models:
      r = result["runs"][run][label]
      print(f"  {label:12s} session {np.round(r['session_deg'], 3)}  whole {np.round(r['whole_deg'], 3)}")
  (out / "validate.json").write_text(json.dumps(result, indent=1))

  names = {run: f"Held-out pose {campaign.HELD_OUT.index(run) + 1}" if run in campaign.HELD_OUT else run
           for run in runs}
  colors = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]  # the videos' red and green
  fig, axes = plt.subplots(len(runs) + 1, 1, figsize=(13, 4 * len(runs) + 10))
  width = 0.8 / len(models)
  for ax, run in zip(axes, runs):
    for i, (label, *_) in enumerate(models):
      ax.bar(np.arange(7) + (i - (len(models) - 1) / 2) * width, result["runs"][run][label]["session_deg"],
             width, label=label, color=colors[i % len(colors)])
    ax.hlines(floor, np.arange(7) - 0.4, np.arange(7) + 0.4, colors="k", lw=2, zorder=3,
              label="repeatability: same commands, 4 runs on the robot")  # across each joint's bars
    ax.set(xticks=range(7), xticklabels=[f"joint {j + 1}" for j in range(7)], ylabel="angle error, RMS (deg)",
           title=f"{names[run]}: simulated minus measured joint angles over the session")
    ax.legend(fontsize=8)
  run = runs[0]
  t = np.arange(len(errors[(models[0][0], run)])) * fit.DT
  ax = axes[-1]
  for i, (label, *_) in enumerate(models):
    ax.plot(t, np.degrees(errors[(label, run)][:, 1]), lw=0.7, label=label, color=colors[i % len(colors)])
  ax.axhline(0, color="0.5", lw=0.5)
  ax.set(xlabel="time (s)", ylabel="angle error (deg)", title=f"{names[run]}: joint 2, simulated minus measured")
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(out / "validate.png", dpi=90)
  plt.close(fig)
  print("wrote", out / "validate.json")


if __name__ == "__main__":
  main()
