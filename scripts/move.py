"""Plan a move of the real Panda to a new pose and show it in MuJoCo first.

The target is one of:
  --to-pose P1 --poses runs/campaign/poses.json   a campaign pose; the end effector
                                                  moves on a straight line (``checks.line_path``)
  --to x,y,z      an end-effector position with the hand pointing down, solved by IK
                  from ``checks.HOME`` turned toward it (the campaign's IK branch)
  --to-q q1,...   seven joint angles
The last two move on a straight line in joint space. Every path has a
minimum-jerk profile, slow enough that no joint is faster than --max-speed.
``real/joint_position`` plays it with Franka's internal joint position
controller, which tracks to hundredths of a degree, so the video shows the
motion itself.

Checks: joint limits with margin, Franka's joint velocity, acceleration and
jerk limits, self-collision along the path, the distance of every link from
the base plane (z = 0, about where the table is), singularities, and with
--box the workspace box.

Outputs in --out:
  traj.csv    t, q1..q7 (absolute) for real/joint_position
  plan.json   start, target, checks
  sim.mp4     the move from two sides, with the paths of the hand and the elbow

Example:
  real/build/stream_state 172.16.0.2 1 1 > state.txt
  uv run python scripts/move.py --state state.txt --to-pose P1 --poses runs/campaign/poses.json \\
      --box runs/workspace_box.json --out runs/moves/to_p1
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
from pathlib import Path

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from panda_sysid import checks, excitation, video, workspace
from panda_sysid import model as sysid_model

DT = checks.DT
FPS = 30
MIN_SIGMA = 0.14  # smallest singular value of the EE Jacobian along the path


def add_sphere(scene: mujoco.MjvScene, pos, rgba, size) -> None:
  if scene.ngeom < scene.maxgeom:
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
                        np.asarray(pos), np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def render(q: np.ndarray, title: str, out: Path) -> None:
  spec = sysid_model.make_spec(80, 20)
  sysid_model.add_render_settings(spec)
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.7, 0.7, 0.0005), pos=(0.3, 0, -0.0005),
                          rgba=(0.5, 0.6, 0.7, 0.25), contype=0, conaffinity=0)
  model = spec.compile()
  data = mujoco.MjData(model)
  width, height = 960, 720
  renderer = mujoco.Renderer(model, height, width)
  cams = []
  for azimuth in (135, 225):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.25, 0.0, 0.35]
    cam.distance, cam.azimuth, cam.elevation = 2.1, azimuth, -15
    cams.append(cam)
  hand_path, elbow_path = [], []
  for qk in q[::50]:
    data.qpos[:7] = qk
    mujoco.mj_kinematics(model, data)
    hand_path.append(data.site_xpos[model.site("ee").id].copy())
    elbow_path.append(data.xpos[model.body("link4").id].copy())
  font = video.font(28)
  frames = []
  for k in range(0, len(q), round(1.0 / (FPS * DT))):
    data.qpos[:7] = q[k]
    mujoco.mj_forward(model, data)
    panels = []
    for cam in cams:
      renderer.update_scene(data, camera=cam)
      for p in hand_path:
        add_sphere(renderer.scene, p, [1.0, 0.8, 0.1, 0.6], 0.004)
      for p in elbow_path:
        add_sphere(renderer.scene, p, [0.2, 0.6, 1.0, 0.6], 0.004)
      panels.append(renderer.render().copy())
    img = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, 2 * width, 60], fill=(0, 0, 0, 150))
    draw.text((20, 14), f"{title}   t = {k * DT:5.2f} s   yellow: hand path   blue: elbow path   "
              "gray: base plane z = 0", font=font, fill=(255, 255, 255))
    frames.append(np.asarray(img))
  renderer.close()
  mediapy.write_video(out, frames, fps=FPS)


def _render(args) -> None:
  render(*args)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--state", help="stream_state output; its last state is the start")
  source.add_argument("--q-start", help="seven comma-separated joint angles (rad)")
  target = parser.add_mutually_exclusive_group(required=True)
  target.add_argument("--to-pose", help="a pose of --poses (e.g. P1)")
  target.add_argument("--to", help="end-effector x,y,z (m), hand pointing down")
  target.add_argument("--to-q", help="seven comma-separated joint angles (rad)")
  parser.add_argument("--poses", help="poses.json of plan_campaign.py (with --to-pose)")
  parser.add_argument("--box", help="workspace_box.py output: the end effector must stay inside")
  parser.add_argument("--max-speed", type=float, default=0.25, help="rad/s, every joint")
  parser.add_argument("--min-clearance", type=float, default=0.05, help="m above the base plane")
  parser.add_argument("--host", default="172.16.0.2", help="the robot's FCI address, for the command")
  parser.add_argument("--no-video", action="store_true")
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  if args.state:
    q_start = np.array(checks.read_state(args.state)["q"])
  else:
    q_start = np.array([float(v) for v in args.q_start.split(",")])
  model = checks.check_model()
  err_mm = err_deg = 0.0
  if args.to_pose:
    if not args.poses:
      parser.error("--to-pose needs --poses")
    q_goal = np.array(json.loads(Path(args.poses).read_text())[args.to_pose]["q"])
    path = checks.line_path(model, q_start, q_goal, args.max_speed)
    hold = round(0.5 / DT)
    q = np.concatenate([np.repeat(path[:1], hold, 0), path, np.repeat(path[-1:], hold, 0)])
  else:
    if args.to:
      q_goal, err_m, err_deg = checks.ik_down(model, np.array([float(v) for v in args.to.split(",")]))
      err_mm = err_m * 1e3
    else:
      q_goal = np.array([float(v) for v in args.to_q.split(",")])
    q = excitation.min_jerk(q_start, q_goal, args.max_speed, hold=0.5)
  t = np.arange(len(q)) * DT
  np.savetxt(out / "traj.csv", np.column_stack([t, q]), delimiter=",", fmt="%.10f",
             header="t," + ",".join(f"q{j + 1}" for j in range(7)), comments="")
  lowest, lowest_body, contacts = checks.clearance(model, q)
  ee = checks.ee_positions(model, q, every=5)
  report = {
    "ik_error_mm": err_mm, "ik_error_deg": err_deg,
    "limit_margin_rad": round(checks.limit_margin(q), 3),
    **{k: round(v, 4) for k, v in checks.derivative_fractions(q).items()},
    "max_ee_speed_m_s": round(checks.ee_speed(model, q), 3),
    "jacobian_sigma_min": round(checks.jacobian_sigma_min(model, q), 3),
    "lowest_link_above_base_plane_m": round(lowest, 4), "lowest_link": lowest_body,
    "self_contacts": contacts[:10],
  }
  ok = {
    "ik": err_mm < 1.0 and err_deg < 0.5,
    "joint limits": report["limit_margin_rad"] > checks.LIMIT_MARGIN - 1e-6,
    "speed, acceleration, jerk": max(report["speed_fraction"], report["acceleration_fraction"],
                                     report["jerk_fraction"]) < 0.5,
    "singularity": report["jacobian_sigma_min"] > MIN_SIGMA,
    "clearance": lowest > args.min_clearance,
    "self-collision": not contacts,
  }
  if args.box:
    lo, hi = workspace.load_box(args.box)
    report["ee_to_box_faces_mm"] = round(float(np.min(np.minimum(ee - lo, hi - ee))) * 1e3, 1)
    ok["inside the box"] = bool(workspace.inside(ee, lo, hi).all())
  report["passed"] = ok
  plan = {
    "q_start": q_start.tolist(), "q_goal": q_goal.tolist(), "duration_s": float(t[-1]),
    "ee_start_m": ee[0].round(4).tolist(), "ee_goal_m": ee[-1].round(4).tolist(),
    "joint_change_deg": np.degrees(q_goal - q_start).round(1).tolist(), "checks": report,
    "command": f"real/build/joint_position {args.host} {out / 'traj.csv'} {out / 'log.csv'}",
  }
  (out / "plan.json").write_text(json.dumps(plan, indent=2))
  print(json.dumps(plan, indent=2))
  if not all(ok.values()):
    print("FAILED:", [k for k, v in ok.items() if not v])
  if not args.no_video:
    # Rendered in a spawned process, never in this one (see panda_sysid.video).
    with video.render_pool(1) as pool:
      pool.apply(_render, ((q, f"move {out.name}", out / "sim.mp4"),))
    print("video:", out / "sim.mp4")
  print(f"Run on the robot (after review):\n  {plan['command']}")


if __name__ == "__main__":
  main()
