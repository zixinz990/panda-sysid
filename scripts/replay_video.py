"""The sim-to-real comparison on the campaign's large moves: a short video and the errors.

Each run's first segment, the slow straight-line move from the previous
pose (end effector 17-38 cm, joints 22-63 deg on our robot), is replayed
from its first measured state in two models: --model-a (default: the
published values with an ideal controller) and --model-b (default: ours,
with its delays and the torque bias at the center of its range).

Outputs in --out:
  real_vs_sim.mp4  one move (--video-run): the robot's measured joint angles
                   and --model-b's simulation (green) on the same commands,
                   at true scale, with both end-effector paths
  real_vs_sim.gif  the same, smaller (for the README)
  moves.png        mean end-effector error per move, both models
  moves.json       end-effector errors per move (mm)

The process computes in forked workers; one spawned process renders (EGL).

Example (ours: the move to P1 is in the aborted first try of 15_r3):
  uv run python scripts/replay_video.py --logs data/campaign --out results/replay \\
      --runs 08_s_p2,09_s_p3,10_s_p4,11_s_p5,12_s_p6,13_h_h1,14_h_h2,15_r3_aborted
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import multiprocessing as mp
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from panda_sysid import campaign, simmodel

FPS = 30
W, H = 1280, 720
COLORS = {"a": (0.90, 0.20, 0.15), "b": (0.10, 0.70, 0.30)}
HOLD = 2.0  # s kept after each move
# The moves P1 -> P2 -> ... -> P6 -> H1 -> H2 -> P1 (the first segments of 08_s_p2 ... 15_r3).
DEFAULT_RUNS = [r.name for r in campaign.campaign_runs() if r.kind == "impedance"][4:12]


def ee_positions(q: np.ndarray) -> np.ndarray:
  import mujoco

  from panda_sysid import model as sysid_model
  model = sysid_model.make_spec(40, 20).compile()
  data = mujoco.MjData(model)
  site = model.site("ee").id
  out = np.empty((len(q), 3))
  for k in range(len(q)):
    data.qpos[:7] = q[k]
    mujoco.mj_kinematics(model, data)
    out[k] = data.site_xpos[site]
  return out


def replay(args) -> dict:
  import mujoco

  from panda_sysid import delayed, fit
  from panda_sysid import model as sysid_model
  run, logs, models = args
  plan = json.loads(Path(f"{logs}/{run}/plan.json").read_text())
  move = plan["segments"][0]
  rec = fit.load_log(fit.find_log(f"{logs}/{run}"), run).chunks(move["end_s"] + HOLD)[0]
  real = np.where(rec.valid[:, None], rec.q, np.nan)
  for j in range(7):
    bad = np.isnan(real[:, j])
    real[bad, j] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), real[~bad, j])
  out = {"run": run, "move": move["name"], "real": real}
  e_real = ee_positions(real)
  for key, path in models.items():
    sim = simmodel.load(path)
    model = sysid_model.make_spec(40, 20, sim.dynamics).compile()
    if sim.bias_center is not None:
      model = fit.apply_values(model, {f"bias{j + 1}": b for j, b in enumerate(sim.bias_center)})
    delays = sim.delays if sim.delays == delayed.IDEAL else replace(sim.delays, cutoff=plan["cutoff"])
    torques, kp, kd = delayed.torque_model(model)
    out[key], _ = delayed.rollout(torques, mujoco.MjData(torques), rec, kp, kd, delays)
    out[f"ee_{key}"] = np.linalg.norm(ee_positions(out[key]) - e_real, axis=1) * 1000
  out["travel_mm"] = float(np.linalg.norm(e_real[-1] - e_real[0]) * 1000)
  out["joint_deg"] = float(np.degrees(np.abs(real[-1] - real[0]).max()))
  return out


def make_video(args) -> None:
  import mediapy
  import mujoco
  from PIL import Image, ImageDraw
  from scipy import ndimage

  from panda_sysid import model as sysid_model
  from panda_sysid import video

  move, label, name, out, speed = args
  real, sim = move["real"], move["b"]
  ee_real, ee_sim = ee_positions(real), ee_positions(sim)
  err = np.linalg.norm(ee_sim - ee_real, axis=1) * 1000

  spec = sysid_model.make_spec(40, 20)
  sysid_model.add_render_settings(spec, W, H)
  spec.visual.headlight.ambient = (0.25, 0.25, 0.25)
  spec.visual.headlight.diffuse = (0.45, 0.45, 0.45)
  spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=(0.96, 0.97, 0.99), rgb2=(0.80, 0.84, 0.90), width=512, height=3072)
  spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=(0.97, 0.97, 0.98), rgb2=(0.90, 0.91, 0.93), width=300, height=300)
  floor = spec.add_material(name="floor", texrepeat=(8, 8), reflectance=0.0)
  floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1.2, 1.2, 0.01), pos=(0.3, 0, 0),
                          material="floor", contype=0, conaffinity=0)
  model = spec.compile()
  renderer = mujoco.Renderer(model, H, W, max_geom=4000)
  spec.visual.quality.offsamples = 0  # multisampling would blend the ids at the edges
  segmenter = mujoco.Renderer(spec.compile(), H, W, max_geom=4000)
  segmenter.enable_segmentation_rendering()
  d_real, d_sim = mujoco.MjData(model), mujoco.MjData(model)
  arm = np.flatnonzero(model.geom_bodyid > 0)  # everything but the floor
  cam = mujoco.MjvCamera()
  cam.lookat[:] = (0.7 * ee_real[:, 0].mean(), ee_real[:, 1].mean(), 0.40)
  cam.distance, cam.azimuth, cam.elevation = 1.6, 225, -28

  def path(scn, points, color, radius):
    for a, b in zip(points[:-1], points[1:]):
      if scn.ngeom >= scn.maxgeom:
        return
      g = scn.geoms[scn.ngeom]
      mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                          np.array(color, dtype=np.float32))
      mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
      g.objtype = mujoco.mjtObj.mjOBJ_SITE  # tells the path apart in the segmentation
      g.segid = scn.ngeom
      scn.ngeom += 1

  dark, green = (0.15, 0.15, 0.17, 1.0), (*COLORS["b"], 1.0)
  big, small = video.font(30), video.font(22)
  lines = [("Real Panda, its measured joint angles: white arm, dark path", dark[:3]),
           (f"Simulation of the {name} model on the same commands: green", green[:3]),
           (f"Move {label.replace('->', ' to ')}, which no fit used: end effector {move['travel_mm'] / 10:.0f} cm, "
            f"joints up to {move['joint_deg']:.0f} deg. True scale, {speed:g}x speed.", (0.35, 0.35, 0.38))]
  step = round(0.025 / 0.001)  # path points every 25 ms
  ticks = np.arange(0, len(real), speed / FPS / 0.001).astype(int)
  with mediapy.VideoWriter(out / "real_vs_sim.mp4", (H, W), fps=FPS) as writer:
    for k in list(ticks) + [len(real) - 1] * FPS:
      # The real arm and its path; then the simulated arm and its path alone, laid over it:
      # the arm as a green tint inside a green outline, the path opaque (drawing both in one
      # scene makes their coincident surfaces flicker and hides one path inside the other).
      d_real.qpos[:7] = real[k]
      mujoco.mj_forward(model, d_real)
      renderer.update_scene(d_real, camera=cam)
      upto = np.r_[np.arange(0, k + 1, step), k]
      path(renderer.scene, ee_real[upto], dark, 0.006)
      frame = renderer.render().astype(float)
      d_sim.qpos[:7] = sim[k]
      mujoco.mj_forward(model, d_sim)
      renderer.update_scene(d_sim, camera=cam)
      scn = renderer.scene
      for i in range(scn.ngeom):
        if scn.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM and scn.geoms[i].objid in arm:
          scn.geoms[i].rgba[:] = green
          scn.geoms[i].matid = -1
      path(scn, ee_sim[upto], green, 0.003)
      ghost = renderer.render().astype(float)
      segmenter.update_scene(d_sim, camera=cam)
      path(segmenter.scene, ee_sim[upto], green, 0.003)
      seg = segmenter.render()
      mask = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], arm)
      edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
      frame[mask] = 0.8 * frame[mask] + 0.2 * ghost[mask]
      frame[edge] = np.array(green[:3]) * 255
      line = seg[..., 1] == mujoco.mjtObj.mjOBJ_SITE
      frame[line] = ghost[line]
      image = Image.fromarray(frame.astype(np.uint8))
      draw = ImageDraw.Draw(image)
      for i, (text, color) in enumerate(lines):
        draw.text((36, 28 + 44 * i), text, font=big if i < 2 else small,
                  fill=tuple(int(255 * c) for c in color))
      draw.text((36, H - 64), f"End-effector distance, simulation to real: {err[k]:4.1f} mm", font=big,
                fill=tuple(int(255 * c) for c in dark[:3]))
      writer.add_image(np.asarray(image))
  renderer.close()
  segmenter.close()


def summary_figure(moves, labels, fitted, names, out) -> dict:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(12, 5.5), dpi=100, facecolor="white")
  ax = fig.add_axes([0.07, 0.12, 0.58, 0.76])
  x = np.arange(len(moves))
  means = {key: [m[f"ee_{key}"].mean() for m in moves] for key in ("a", "b")}
  ax.bar(x - 0.19, means["a"], 0.38, color=COLORS["a"], label=names["a"])
  ax.bar(x + 0.19, means["b"], 0.38, color=COLORS["b"], label=names["b"])
  ax.set(xticks=x, xticklabels=labels, ylabel="mean end-effector error (mm)")
  for i in range(len(moves)):
    if not fitted[i]:
      ax.axvspan(i - 0.5, i + 0.5, color="0.94", zorder=0)
  ax.legend(fontsize=11, loc="upper left")
  ax.set_title("Simulated minus real end-effector position per move (gray: no fit used it)", fontsize=12)
  fit_mask = np.array(fitted)
  summary = {"runs": [m["run"] for m in moves], "labels": labels, "fitted": fitted,
             "travel_mm": [round(m["travel_mm"]) for m in moves], "joint_deg": [round(m["joint_deg"]) for m in moves]}
  for key in ("a", "b"):
    per_move = np.array(means[key])
    summary[key] = {"model": names[key], "mean_per_move_mm": per_move.round(2).tolist(),
                    "mean_fitted_mm": float(per_move[fit_mask].mean()) if fit_mask.any() else None,
                    "mean_held_out_mm": float(per_move[~fit_mask].mean()) if (~fit_mask).any() else None,
                    "max_mm": float(max(m[f"ee_{key}"].max() for m in moves))}
  y = 0.80
  fig.text(0.69, y, "Mean end-effector error", fontsize=14, weight="bold")
  for label, field in (("moves in the fit", "mean_fitted_mm"), ("moves no fit used", "mean_held_out_mm")):
    if summary["a"][field] is not None:
      y -= 0.08
      fig.text(0.69, y, f"{label}: {summary['a'][field]:.1f} -> {summary['b'][field]:.1f} mm", fontsize=12)
  fig.text(0.69, y - 0.08, f"largest: {summary['a']['max_mm']:.1f} -> {summary['b']['max_mm']:.1f} mm",
           fontsize=12, color="0.3")
  fig.savefig(out / "moves.png")
  plt.close(fig)
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--logs", default="data/campaign", help="directory of <run>/log.csv.gz and plan.json")
  parser.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="runs whose first segment is a move")
  parser.add_argument("--fitted", default=",".join(campaign.JOINT_FIT), help="runs the joint fit used")
  parser.add_argument("--model-a", default="nominal", help="model.json, or 'nominal'")
  parser.add_argument("--name-a", default="published values, no delays")
  parser.add_argument("--model-b", default=str(simmodel.OURS))
  parser.add_argument("--name-b", default="identified")
  parser.add_argument("--video-run", default=campaign.HELD_OUT[-1], help="the run whose move the video shows")
  parser.add_argument("--speed", type=float, default=3.0, help="playback speed of the video")
  parser.add_argument("--gpu", help="EGL device of the renderer")
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  runs = args.runs.split(",")
  models = {"a": args.model_a, "b": args.model_b}
  # Forked workers only compute (this process has no GL context).
  with mp.get_context("fork").Pool(len(runs)) as pool:
    moves = pool.map(replay, [(run, args.logs, models) for run in runs])
  poses = [json.loads(Path(f"{args.logs}/{run}/plan.json").read_text())["pose"] for run in runs]
  order = campaign.campaign_runs()
  before = {b.name: a.pose for a, b in zip(order[:-1], order[1:])}
  starts = [before.get(runs[0])] + poses[:-1]
  labels = [f"{a}->{b}" if a else m["move"] for a, b, m in zip(starts, poses, moves)]
  fitted = [run in args.fitted.split(",") for run in runs]
  summary = summary_figure(moves, labels, fitted, {"a": args.name_a, "b": args.name_b}, out)
  (out / "moves.json").write_text(json.dumps(summary, indent=1))
  print(json.dumps(summary, indent=1))
  i = runs.index(args.video_run)
  from panda_sysid import video
  with video.render_pool(1, video.parse_gpus(args.gpu)) as pool:
    pool.apply(make_video, ((moves[i], labels[i], args.name_b, out, args.speed),))
  # GitHub shows a GIF in a README, not an MP4 from the repository.
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / "real_vs_sim.mp4"), "-vf",
                  "fps=15,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=4", str(out / "real_vs_sim.gif")], check=True)


if __name__ == "__main__":
  main()
