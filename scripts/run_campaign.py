"""Run the campaign on the real Panda, one run after another.

For each run: read the robot's state (it must be idle), plan the run again
from that state (plan_campaign.py --only), require every check to pass, read
the state again (the arm must still be within 0.2 deg of the planned start),
run the player, require it to finish, then summarize the log
(``panda_sysid.realrun``: check.json and a figure) and compress it. Any
failure stops the sequence before the next motion.

Before the first run, review the whole campaign in simulation
(plan_campaign.py --video) and move the arm to P1 (move.py --to-pose P1).
Keep the user stop at hand the whole time. With --confirm, each player asks
for Enter before it moves.

Outputs per run, in --out/<run>/:
  state_plan.out, state_before.out   the states the run was planned from and started at
  plan.out                           plan_campaign.py's output (plan.json, traj.csv as usual)
  run.out                            the player's output
  log.csv.gz                         every robot state of the run
  check.json, real_vs_sim.png or torque.png

Example:
  uv run python scripts/run_campaign.py --box runs/workspace_box.json --out runs/campaign
  uv run python scripts/run_campaign.py --box runs/workspace_box.json --out runs/campaign \\
      --runs 07_s_p1,08_s_p2 --confirm
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from panda_sysid import campaign, checks, realrun, simmodel, workspace

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "real" / "build" / "stream_state"
MAX_START_OFFSET_DEG = 0.2


def log(msg: str) -> None:
  print(time.strftime("%H:%M:%S"), msg, flush=True)


def stop(msg: str) -> None:
  log(f"STOP: {msg}")
  sys.exit(1)


def read_state(host: str, path: Path) -> dict:
  with open(path, "w") as f:
    subprocess.run([str(STREAM), host, "1", "1"], stdout=f, stderr=subprocess.STDOUT, timeout=30, check=True)
  return checks.read_state(str(path))


def tee(cmd: list[str], path: Path) -> int:
  """Run ``cmd`` with this terminal's stdin, showing and saving its output."""
  with open(path, "w") as f, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              text=True) as proc:
    for line in proc.stdout:
      print("  | " + line, end="", flush=True)
      f.write(line)
  return proc.returncode


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--box", required=True, help="workspace_box.py output (JSON)")
  parser.add_argument("--out", required=True, help="plan_campaign.py's output directory")
  parser.add_argument("--runs", default=",".join(r.name for r in campaign.campaign_runs()),
                      help="runs to do, in order (default: all)")
  parser.add_argument("--host", default="172.16.0.2", help="the robot's FCI address")
  parser.add_argument("--model", default=str(simmodel.OURS) if simmodel.OURS.exists() else "nominal",
                      help="model.json the plans and the summaries simulate")
  parser.add_argument("--confirm", action="store_true", help="the player asks for Enter before each motion")
  args = parser.parse_args()
  out = Path(args.out)
  if not STREAM.exists():
    sys.exit(f"{STREAM} is missing; build the programs in real/ first (see real/CMakeLists.txt)")
  box = workspace.load_box(args.box)
  sim = simmodel.load(args.model)

  for name in args.runs.split(","):
    log(f"== {name}")
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    state = read_state(args.host, d / "state_plan.out")
    if state["robot_mode"] != "Idle":
      stop(f"robot mode is {state['robot_mode']} before {name}")
    planned = subprocess.run([sys.executable, str(ROOT / "scripts" / "plan_campaign.py"), "--only", name,
                              "--state", str(d / "state_plan.out"), "--box", args.box, "--model", args.model,
                              "--host", args.host, "--out", str(out)], capture_output=True, text=True)
    (d / "plan.out").write_text(planned.stdout + planned.stderr)
    if planned.returncode != 0:
      stop(f"planning {name} failed:\n{planned.stderr[-2000:]}")
    log(planned.stdout.strip().splitlines()[-2])
    plan = json.loads((d / "plan.json").read_text())
    failed = [k for k, v in plan["checks"]["passed"].items() if not v]
    if failed:
      stop(f"{name} fails the checks {failed}")
    q0 = np.loadtxt(d / "traj.csv", delimiter=",", skiprows=1, max_rows=1)[1:]
    state = read_state(args.host, d / "state_before.out")
    offset = float(np.degrees(np.abs(np.array(state["q"]) - q0)).max())
    if state["robot_mode"] != "Idle" or offset > MAX_START_OFFSET_DEG:
      stop(f"before {name}: mode {state['robot_mode']}, {offset:.3f} deg from the planned start")
    cmd = plan["command"].split() + ([] if args.confirm else ["--yes"])
    log(" ".join(cmd))
    code = tee(cmd, d / "run.out")
    if code != 0:
      stop(f"{name} did not finish (exit {code}); its log is in {d}")
    try:
      summary = realrun.impedance(d, sim) if plan["kind"] == "impedance" else realrun.position(d, box)
      keys = ("missed_fraction", "longest_gap_ms", "max_abs_tau_ext", "max_tracking_error_rad",
              "rms_sim_minus_real_deg", "max_abs_q_cmd_minus_q_deg", "real_ee_to_box_faces_mm")
      log("summary " + json.dumps({k: summary[k] for k in keys if k in summary}))
    except Exception as e:  # the run itself is done; a failed summary does not stop the campaign
      log(f"(summary of {name} failed: {e!r})")
    with open(d / "log.csv", "rb") as src, gzip.open(d / "log.csv.gz", "wb") as dst:
      shutil.copyfileobj(src, dst)
    (d / "log.csv").unlink()
  log("ALL DONE")


if __name__ == "__main__":
  main()
