"""Plan the identification campaign on the real Panda and show it in MuJoCo.

The runs, poses, Fourier design and checks are ``panda_sysid.campaign``'s.
Every impedance run is simulated on --model (default: the model identified
on our Panda, ``results/model.json``) with its delayed controller.

Outputs in --out (one directory per run, in order):
  NN_name/traj.csv   t, q1..q7 (absolute q_d) for the player
  NN_name/plan.json  run, segments, checks, command
  NN_name/sim.png    q_d and the simulated q; the impedance torque
  NN_name/sim.mp4    the simulated run in real time (with --video)
  poses.json         the poses (planned once; delete it to plan them again)
  fourier.json       the Fourier design (designed once; delete it to design again)
  campaign.json      every run, its command and checks
  overview.mp4       the whole campaign, sped up
  poses.png          the arm at every pose

The plan starts at P1; ``scripts/move.py --to-pose P1`` takes the arm there.
At run time, plan each run again from the robot's measured state (the arm
ends each impedance run up to about 2 deg from its target, since dry
friction holds it there; the players refuse to start more than 0.5 deg off);
``scripts/run_campaign.py`` does that before every run:
  uv run python scripts/plan_campaign.py --box runs/workspace_box.json --state state.txt \\
      --only 07_s_p1 --out runs/campaign

Example:
  uv run python scripts/plan_campaign.py --box runs/workspace_box.json --video --out runs/campaign
"""

from __future__ import annotations

import os

# One BLAS thread per process: the design and the simulations run in process pools.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(_var, "1")
# Offscreen rendering, never through the display's X server.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import itertools
import json
import multiprocessing as mp
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from panda_sysid import campaign, checks, robot, simmodel, video, workspace
from panda_sysid import model as sysid_model
from panda_sysid.campaign import DT, KD, KP, Run

FPS = 30

# -- pictures -------------------------------------------------------------------


def render_model(box, positions: dict[str, list[float]]) -> mujoco.MjModel:
  """The arm with the base plane, the box's edges (red) and the poses (blue; held out pink)."""
  spec = sysid_model.make_spec(KP, KD)
  sysid_model.add_render_settings(spec, 960, 720)
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.75, 0.75, 0.0005), pos=(0.3, 0, -0.0005),
                          rgba=(0.5, 0.6, 0.7, 0.3), contype=0, conaffinity=0)
  corners = workspace.corners(*box)
  for i, j in itertools.combinations(range(8), 2):
    if np.count_nonzero(corners[i] != corners[j]) == 1:
      spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=(*corners[i], *corners[j]),
                              size=(0.002, 0, 0), rgba=(0.9, 0.1, 0.1, 0.7), contype=0, conaffinity=0)
  for name, pos in positions.items():
    spec.worldbody.add_site(name=f"pose_{name}", pos=pos, size=(0.008, 0, 0),
                            rgba=(0.2, 0.5, 1.0, 0.6) if name.startswith("P") else (1.0, 0.3, 0.6, 0.6))
  return spec.compile()


def cameras() -> list[mujoco.MjvCamera]:
  cams = []
  for azimuth, distance, lookat in ((135, 1.6, (0.35, 0.0, 0.3)), (200, 1.1, (0.43, -0.03, 0.22))):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, -18
    cams.append(cam)
  return cams


def segment_at(spans, t: float) -> str:
  for name, start, end in spans:
    if start <= t < end:
      return name
  return "hold"


def frames_of(run: Run, q: np.ndarray, poses: dict, box, step: int, renderer, model,
              header: str) -> Iterator[np.ndarray]:
  data = mujoco.MjData(model)
  site = model.site("ee").id
  font_big = video.font(28)
  font_small = video.font(22)
  q_pose = np.array(poses[run.pose]["q"])
  data.qpos[:7] = q_pose
  mujoco.mj_kinematics(model, data)
  ee_pose = data.site_xpos[site].copy()
  for k in range(0, len(q), step):
    data.qpos[:7] = q[k]
    mujoco.mj_forward(model, data)
    panels = []
    for cam in cameras():
      renderer.update_scene(data, camera=cam)
      panels.append(renderer.render().copy())
    img = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, img.width, 92], fill=(0, 0, 0, 160))
    t = k * DT
    draw.text((20, 10), f"{header}   {run.name} ({run.kind}, {run.purpose})   t = {t:6.1f} s   "
              f"{segment_at(run.spans, t)}", font=font_big, fill=(255, 255, 255))
    ee = data.site_xpos[site]
    off = (ee - ee_pose) * 1e3
    dev = np.degrees(np.abs(q[k] - q_pose).max())
    to_faces = np.minimum(ee - box[0], box[1] - ee).min() * 1e3
    draw.text((20, 54), f"EE from {run.pose}: x {off[0]:+6.1f}  y {off[1]:+6.1f}  z {off[2]:+6.1f} mm   "
              f"to the box's faces {to_faces:5.0f} mm   largest joint offset {dev:5.1f} deg",
              font=font_small, fill=(255, 255, 255))
    yield np.asarray(img)


def _positions(poses: dict) -> dict[str, list[float]]:
  return {name: list(p["position"]) for name, p in poses.items()}


def _render_run(args) -> str:
  run, q, poses, box, out = args
  model = render_model(box, _positions(poses))
  renderer = mujoco.Renderer(model, 720, 960)
  with mediapy.VideoWriter(out / run.name / "sim.mp4", (720, 1920), fps=FPS) as writer:
    for frame in frames_of(run, q, poses, box, round(1.0 / (FPS * DT)), renderer, model, "real time"):
      writer.add_image(frame)
  renderer.close()
  return run.name


def _render_overview_part(args) -> Path:
  run, q, poses, box, speedup, header, path = args
  model = render_model(box, _positions(poses))
  renderer = mujoco.Renderer(model, 720, 960)
  with mediapy.VideoWriter(path, (480, 1280), fps=FPS) as writer:
    for frame in frames_of(run, q, poses, box, round(speedup / (FPS * DT)), renderer, model, header):
      writer.add_image(np.asarray(Image.fromarray(frame).resize((1280, 480))))
  renderer.close()
  return path


def concatenate(parts: list[Path], path: Path) -> None:
  """Join the overview's parts (one per run) into one video and delete them."""
  with mediapy.VideoWriter(path, (480, 1280), fps=FPS) as writer:
    for part in parts:
      with mediapy.VideoReader(part) as reader:
        for frame in reader:
          writer.add_image(frame)
      part.unlink()


def render_poses(poses: dict, box, out: Path) -> None:
  model = render_model(box, _positions(poses))
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, 720, 960)
  cam = cameras()[0]
  font = video.font(40)
  tiles = []
  for name, pose in poses.items():
    data.qpos[:7] = pose["q"]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    img = Image.fromarray(renderer.render().copy())
    x, y, z = pose["position"]
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, img.width, 70], fill=(0, 0, 0, 160))
    draw.text((20, 15), f"{name}  ({x:.2f}, {y:+.2f}, {z:.2f}) m  turn {tuple(pose['turn'])}", font=font,
              fill=(255, 255, 255))
    tiles.append(np.asarray(img.resize((480, 360))))
  renderer.close()
  rows = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)]
  Image.fromarray(np.concatenate(rows, axis=0)).save(out / "poses.png")


def plot_run(run: Run, sim: dict | None, out: Path) -> None:
  t = np.arange(len(run.q_d)) * DT
  q = run.q_d if sim is None else sim["q"]
  ref = run.q_d[0] if run.kind == "position" else None
  fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
  for j in range(7):
    base = run.q_d[-1, j] if ref is None else ref[j]
    line, = axes[0].plot(t, np.degrees(q[:, j] - base), lw=0.8, label=f"q{j + 1}")
    axes[0].plot(t, np.degrees(run.q_d[:, j] - base), "--", lw=0.6, color=line.get_color())
    if sim is not None:
      axes[1].plot(t, sim["tau"][:, j] / robot.EFFORT_LIMITS[j], lw=0.6, label=f"joint {j + 1}")
  axes[0].set(ylabel="from the pose [deg]", title=f"{run.name}: q_d (dashed) and simulated q")
  axes[0].legend(ncol=7, fontsize=8)
  axes[1].set(xlabel="t [s]", ylabel="torque / limit", title="Impedance torque (delayed controller)")
  for ax in axes:
    for _, a, b in run.spans:
      ax.axvspan(a, b, color="0.92", zorder=0)
  fig.tight_layout()
  fig.savefig(out / run.name / "sim.png", dpi=100)
  plt.close(fig)


# -- main -------------------------------------------------------------------------

_SIM: list = []


def _simulate_task(run: Run) -> dict[str, np.ndarray]:
  sim = _SIM[0]
  return campaign.simulate(run, sysid_model.make_spec(KP, KD, sim.dynamics).compile(), sim.delays)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  source = parser.add_mutually_exclusive_group()
  source.add_argument("--state", help="stream_state output; its last state is the start (default: P1)")
  source.add_argument("--q-start", help="seven comma-separated joint angles (rad)")
  parser.add_argument("--box", required=True, help="workspace_box.py output (JSON)")
  parser.add_argument("--model", default=str(simmodel.OURS) if simmodel.OURS.exists() else "nominal",
                      help="model.json (or 'nominal') the impedance runs are simulated on")
  parser.add_argument("--host", default="172.16.0.2", help="the robot's FCI address, for the commands")
  parser.add_argument("--only", help="plan just this run (e.g. 07_s_p1) from the start state")
  parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1),
                      help="processes of the Fourier design and the simulations")
  parser.add_argument("--video", action="store_true", help="real-time video of every run")
  parser.add_argument("--speedup", type=float, default=10.0, help="of the overview video")
  parser.add_argument("--no-render", action="store_true", help="plan and check only")
  parser.add_argument("--render-workers", type=int, default=8, help="processes that render")
  parser.add_argument("--render-gpus", help="EGL devices the renderers use in turn (e.g. 0,2; default: MuJoCo's)")
  parser.add_argument("--render-runs", help="render only these runs (comma-separated, e.g. 01_r1,02_f1)")
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  box = workspace.load_box(args.box)
  sim_model = simmodel.load(args.model)
  _SIM[:] = [sim_model]
  check_model = checks.check_model()
  runs = campaign.campaign_runs()
  sessions: dict[str, list] = {name: [] for name in campaign.ANCHORS}
  for run in runs:
    if run.kind == "impedance":
      sessions[run.pose].append((run.session, run.seed))
  poses_path = out / "poses.json"
  if poses_path.exists():
    poses = {k: {**v, "q": np.array(v["q"])} for k, v in json.loads(poses_path.read_text()).items()}
  else:
    poses = campaign.place_poses(check_model, box, sessions)
    poses_path.write_text(json.dumps({k: {**v, "q": v["q"].tolist(), "position": v["position"].round(4).tolist()}
                                      for k, v in poses.items()}, indent=1))
  for name, pose in poses.items():
    print(f"{name}: EE {np.round(pose['position'], 3)}  turn {pose['turn']}  "
          f"{pose['toward_anchor']:.0%} of the way to its anchor")
  if args.state:
    state = checks.read_state(args.state)
    q_start = np.array(state["q"])
    print("robot_mode:", state["robot_mode"])
  elif args.q_start:
    q_start = np.array([float(v) for v in args.q_start.split(",")])
  else:
    q_start = poses["P1"]["q"]

  design_path = out / "fourier.json"
  if design_path.exists():
    design = json.loads(design_path.read_text())
    fouriers = [np.array(c) for c in design["coefficients"]]
  else:
    fouriers, summary = campaign.design_fourier(poses["P1"]["q"], box, workers=args.workers)
    design = {**summary, "center": poses["P1"]["q"].tolist(), "period_s": campaign.PERIOD,
              "repeats": campaign.REPEATS, "coefficients": [c.tolist() for c in fouriers]}
    design_path.write_text(json.dumps(design, indent=1))
  print("Fourier design:", {k: v for k, v in design.items() if k not in ("coefficients", "center")})

  if args.only:
    runs = [r for r in runs if r.name == args.only]
    if not runs:
      raise SystemExit(f"no run {args.only!r}")
    campaign.build(runs[0], q_start, poses, fouriers, check_model)
  else:
    # Each run starts where the one before ends: an impedance run at its pose,
    # a Fourier run where it began.
    q_from = q_start
    for run in runs:
      campaign.build(run, q_from, poses, fouriers, check_model)
      q_from = run.q_d[-1]

  impedance = [r for r in runs if r.kind == "impedance"]
  # Forked workers only compute (this process has no GL context).
  with mp.get_context("fork").Pool(min(len(impedance), args.workers) or 1) as pool:
    sims = dict(zip([r.name for r in impedance], pool.map(_simulate_task, impedance)))

  summary = {"q_start": q_start.tolist(), "box": {"lo": box[0].tolist(), "hi": box[1].tolist()},
             "poses": {k: {"position": np.round(v["position"], 4).tolist(), "turn": v["turn"],
                           "q": v["q"].tolist()} for k, v in poses.items()},
             "sim_model": args.model, "runs": []}
  total = 0.0
  for run in runs:
    d = out / run.name
    d.mkdir(exist_ok=True)
    sim = sims.get(run.name)
    t = np.arange(len(run.q_d)) * DT
    np.savetxt(d / "traj.csv", np.column_stack([t, run.q_d]), delimiter=",", fmt="%.10f",
               header="t," + ",".join(f"q{j + 1}" for j in range(7)), comments="")
    report = campaign.check(run, check_model, sim, box)
    plan = {"run": run.name, "kind": run.kind, "pose": run.pose, "purpose": run.purpose,
            "session": run.session, "seed": run.seed, "fourier": run.fourier, "cutoff": run.cutoff,
            "q_start": run.q_d[0].tolist(), "segments": [{"name": n, "start_s": a, "end_s": b}
                                                       for n, a, b in run.spans],
            "checks": report, "command": campaign.command(run, out, args.host)}
    (d / "plan.json").write_text(json.dumps(plan, indent=2))
    plot_run(run, sim, out)
    summary["runs"].append({k: plan[k] for k in ("run", "kind", "pose", "purpose", "command")}
                           | {"duration_s": report["duration_s"], "passed": all(report["passed"].values()),
                              "failed": [k for k, v in report["passed"].items() if not v]})
    total += report["duration_s"]
    status = "ok" if all(report["passed"].values()) else f"FAILED {summary['runs'][-1]['failed']}"
    extra = (f"err {max(report['sim_tracking_error_rad']):.3f} rad, torque "
             f"{max(report['sim_torque_fraction']):.3f}" if sim is not None else
             f"speed {report['speed_fraction']:.2f} acc {report['acceleration_fraction']:.2f}")
    extra += f"  sigma {report['sigma_min']:.2f}  ee {report['ee_speed_m_s']:.2f} m/s"
    print(f"{run.name:16s} {run.kind:9s} {report['duration_s']:6.1f} s  box "
          f"{report['ee_to_box_faces_mm']:5.1f} mm  low {report['lowest_link_above_base_plane_m']:.3f} m  "
          f"{extra}  {status}")
  summary["total_motion_s"] = round(total, 1)
  print(f"total motion {total / 60:.1f} min")
  if args.only:
    (out / f"{args.only}.replan.json").write_text(json.dumps(summary, indent=1))
    return
  (out / "campaign.json").write_text(json.dumps(summary, indent=1))
  if args.no_render:
    return

  qs = {r.name: (r.q_d if r.kind == "position" else sims[r.name]["q"]) for r in runs}
  if args.render_runs:
    runs = [r for r in runs if r.name in args.render_runs.split(",")]
  parts_dir = out / "overview_parts"
  parts_dir.mkdir(exist_ok=True)
  with video.render_pool(args.render_workers, video.parse_gpus(args.render_gpus)) as pool:
    poses_done = pool.apply_async(render_poses, (poses, box, out))
    parts = pool.map(_render_overview_part, [(r, qs[r.name], poses, box, args.speedup, f"{args.speedup:g}x",
                                              parts_dir / f"{r.name}.mp4") for r in runs])
    # The real-time videos render while this process joins the overview.
    videos = (pool.imap_unordered(_render_run, [(r, qs[r.name], poses, box, out) for r in runs])
              if args.video else [])
    concatenate(parts, out / "overview.mp4")
    parts_dir.rmdir()
    poses_done.get()
    print("overview", flush=True)
    for name in videos:
      print("video", name, flush=True)


if __name__ == "__main__":
  main()
