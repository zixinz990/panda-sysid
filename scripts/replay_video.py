"""A video of the sim-to-real gap of two models on the campaign's large moves.

Each run's first segment, the slow straight-line move from the previous
pose (end effector 17-38 cm, joints 22-63 deg on our robot), is replayed
from its first measured state in two models: --model-a (default: the
published values with an ideal controller) and --model-b (default: ours,
with its delays and the torque bias at the center of its range). The video
shows the real arm (white) with both simulated arms as ghosts, their errors
magnified, and the end-effector error over time; it ends with the mean
error per move.

Outputs in --out:
  moves.mp4    the video
  moves.png    its last frame (mean error per move)
  moves.json   end-effector errors per move (mm)

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
from dataclasses import replace
from pathlib import Path

import numpy as np

from panda_sysid import campaign, simmodel

FPS = 30
W, H = 1920, 1080
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


def make_video(args) -> dict:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import mediapy
  import mujoco

  from panda_sysid import model as sysid_model

  moves, labels, fitted, names, out, speed, magnify = args
  real = np.concatenate([m["real"] for m in moves])
  ghosts_q = {key: np.concatenate([m[key] for m in moves]) for key in ("a", "b")}
  err = {key: np.concatenate([m[f"ee_{key}"] for m in moves]) for key in ("a", "b")}
  bounds = np.cumsum([0] + [len(m["real"]) for m in moves])
  n = len(real)
  t = np.arange(n) * 0.001
  ticks = np.arange(0, n, speed / FPS / 0.001).astype(int)
  rgb = {key: COLORS[key] for key in ("a", "b")}

  def canvas(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

  # The legend strip over the 3-D view.
  fig = plt.figure(figsize=(9.6, 1.0), dpi=100)
  fig.patch.set_alpha(0.0)
  fig.text(0.03, 0.62, "white: robot", fontsize=17, color="0.15", va="center")
  fig.text(0.25, 0.62, f"red: {names['a']}", fontsize=17, color=rgb["a"], va="center")
  fig.text(0.25, 0.22, f"green: {names['b']}", fontsize=17, color=rgb["b"], va="center")
  fig.text(0.80, 0.62, f"errors x{magnify:g}", fontsize=14, color="0.3", va="center")
  fig.text(0.80, 0.22, f"{speed:g}x speed", fontsize=14, color="0.3", va="center")
  fig.canvas.draw()
  strip = np.asarray(fig.canvas.buffer_rgba()).astype(float) / 255
  plt.close(fig)
  alpha = strip[..., 3:]

  spec = sysid_model.make_spec(40, 20)
  sysid_model.add_render_settings(spec, 960, H)
  spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=(0.96, 0.97, 0.99), rgb2=(0.80, 0.84, 0.90), width=512, height=3072)
  spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=(0.97, 0.97, 0.98), rgb2=(0.90, 0.91, 0.93), width=300, height=300)
  floor = spec.add_material(name="floor", texrepeat=(8, 8), reflectance=0.0)
  floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1.2, 1.2, 0.01), pos=(0.3, 0, 0),
                          material="floor", contype=0, conaffinity=0)
  model = spec.compile()
  renderer = mujoco.Renderer(model, H, 960)
  d_real, d_ghost = mujoco.MjData(model), mujoco.MjData(model)
  opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
  cam = mujoco.MjvCamera()
  cam.lookat[:] = (0.33, 0.0, 0.33)
  cam.distance, cam.azimuth, cam.elevation = 1.8, 125, -20
  ghosts = [(real + magnify * (ghosts_q[key] - real), rgb[key]) for key in ("a", "b")]

  fig = plt.figure(figsize=(9.6, H / 100), dpi=100, facecolor="white")
  ax = fig.add_axes([0.11, 0.42, 0.85, 0.50])
  dec = slice(None, None, 50)
  ymax = max(err["a"].max(), err["b"].max()) * 1.12
  for i, a in enumerate(bounds[:-1]):
    if not fitted[i]:
      ax.axvspan(a / 1000, bounds[i + 1] / 1000, color="0.94", zorder=0)
    ax.axvline(a / 1000, color="0.75", lw=0.8)
    ax.text((a + bounds[i + 1]) / 2000, ymax * 0.97, labels[i], ha="center", va="top", fontsize=11, color="0.3")
  lines = {key: ax.plot([], [], color=rgb[key], lw=1.4, label=names[key])[0] for key in ("a", "b")}
  cursor = ax.axvline(0, color="0.2", lw=1)
  ax.set(xlim=(0, t[-1]), ylim=(0, ymax), xlabel="time (s)", ylabel="end-effector error (mm)")
  ax.set_title("Simulated minus real end-effector position (gray: moves no fit saw)", fontsize=15)
  ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=13)
  seg_text = fig.text(0.5, 0.965, "", ha="center", fontsize=17)
  now = {key: fig.text(0.08, y, "", fontsize=22, color=rgb[key]) for key, y in (("a", 0.27), ("b", 0.20))}
  avg = fig.text(0.08, 0.08, "", fontsize=17, color="0.25")
  fig.text(0.08, 0.33, "end-effector error now", fontsize=17, color="0.3")

  with mediapy.VideoWriter(out / "moves.mp4", (H, W), fps=FPS) as writer:
    for k in ticks:
      d_real.qpos[:7] = real[k]
      mujoco.mj_forward(model, d_real)
      renderer.update_scene(d_real, camera=cam, scene_option=opt)
      scn = renderer.scene
      for q, color in ghosts:
        d_ghost.qpos[:7] = q[k]
        mujoco.mj_kinematics(model, d_ghost)
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(model, d_ghost, opt, pert, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scn)
        for i in range(n0, scn.ngeom):
          scn.geoms[i].rgba[:] = (*color, 0.42)
          scn.geoms[i].matid = -1
      left = renderer.render().copy()
      band = 0.3 * left[:100].astype(float) + 0.7 * 255
      left[:100] = (band * (1 - alpha) + strip[..., :3] * 255 * alpha).astype(np.uint8)
      i = np.searchsorted(bounds, k, side="right") - 1
      seg_text.set_text(f"move {i + 1}: {labels[i]}" + ("" if fitted[i] else " (no fit saw it)"))
      for key in ("a", "b"):
        lines[key].set_data(t[:k + 1][dec], err[key][:k + 1][dec])
        now[key].set_text(f"{names[key]}  {err[key][k]:5.1f} mm")
      cursor.set_xdata([t[k], t[k]])
      avg.set_text(f"mean so far: {err['a'][:k + 1].mean():.1f} mm and {err['b'][:k + 1].mean():.1f} mm")
      writer.add_image(np.concatenate([left, canvas(fig)], axis=1))
    plt.close(fig)
    renderer.close()

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    fig.text(0.5, 0.93, "Simulated minus real end-effector position on large moves", ha="center", fontsize=32,
             weight="bold")
    ax = fig.add_axes([0.06, 0.14, 0.52, 0.66])
    x = np.arange(len(moves))
    means = {key: [m[f"ee_{key}"].mean() for m in moves] for key in ("a", "b")}
    ax.bar(x - 0.19, means["a"], 0.38, color=rgb["a"], label=names["a"])
    ax.bar(x + 0.19, means["b"], 0.38, color=rgb["b"], label=names["b"])
    ax.set(xticks=x, xticklabels=labels, ylabel="mean end-effector error (mm)")
    for i in range(len(moves)):
      if not fitted[i]:
        ax.axvspan(i - 0.5, i + 0.5, color="0.94", zorder=0)
    ax.legend(fontsize=14, loc="upper left")
    ax.set_title("Mean error per move (gray: moves no fit saw)", fontsize=18)
    fit_mask = np.array(fitted)
    summary = {"runs": [m["run"] for m in moves], "labels": labels, "fitted": fitted,
               "travel_mm": [round(m["travel_mm"]) for m in moves], "joint_deg": [round(m["joint_deg"]) for m in moves]}
    for key in ("a", "b"):
      per_move = np.array(means[key])
      summary[key] = {"model": names[key], "mean_per_move_mm": per_move.round(2).tolist(),
                      "mean_fitted_mm": float(per_move[fit_mask].mean()) if fit_mask.any() else None,
                      "mean_held_out_mm": float(per_move[~fit_mask].mean()) if (~fit_mask).any() else None,
                      "max_mm": float(err[key].max())}
    y = 0.72
    fig.text(0.63, y, "Mean end-effector error", fontsize=24, weight="bold")
    for label, field in (("moves in the fit", "mean_fitted_mm"), ("moves no fit saw", "mean_held_out_mm")):
      if summary["a"][field] is not None:
        y -= 0.08
        fig.text(0.63, y, f"{label}: {summary['a'][field]:.1f} mm -> {summary['b'][field]:.1f} mm", fontsize=21)
    fig.text(0.63, y - 0.1, f"largest: {summary['a']['max_mm']:.1f} mm -> {summary['b']['max_mm']:.1f} mm",
             fontsize=19, color="0.3")
    fig.text(0.63, y - 0.18, f"(red: {names['a']}; green: {names['b']})", fontsize=15, color="0.4")
    end = canvas(fig)
    plt.close(fig)
    for _ in range(FPS * 6):
      writer.add_image(end)
  mediapy.write_image(out / "moves.png", end)
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
  parser.add_argument("--speed", type=float, default=6.0, help="playback speed")
  parser.add_argument("--magnify", type=float, default=5.0, help="of the simulated arms' errors")
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
  from panda_sysid import video
  with video.render_pool(1, video.parse_gpus(args.gpu)) as pool:
    summary = pool.apply(make_video, ((moves, labels, fitted, {"a": args.name_a, "b": args.name_b}, out,
                                       args.speed, args.magnify),))
  (out / "moves.json").write_text(json.dumps(summary, indent=1))
  print(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
