"""Record poses of the real Panda while you move it by hand, with a live view in the browser.

Press the user stop (white light), move the arm with the guiding buttons at the wrist, let go,
and record the pose: the "Record pose" button in the browser, or Enter in this terminal
("d" + Enter deletes the last pose, "q" + Enter quits). A pose is recorded only while the arm
is still. The script only reads the robot's state (``real/build/stream_state``); it never
commands a motion. It needs the optional "viewer" dependencies (viser, mjviser).

The view (viser, http://localhost:8080) shows the arm with the Franka Hand at the measured
joint angles, the end-effector frame that the robot reports (O_T_EE: the end effector set in
Desk, in the robot's base frame) and the recorded poses as numbered frames.

Each pose holds the joint angles q, O_T_EE (row-major 4x4, and as position and quaternion wxyz)
and the robot's whole state. The file is written after every change, so nothing is lost when
the script stops.

Example:
  uv run --extra viewer python scripts/record_poses.py --out runs/corners.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjviser.scene import ViserMujocoScene

from panda_sysid import model as sysid_model

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "real" / "build" / "stream_state"
STILL = 0.01  # rad/s: the fastest joint speed at which a pose is recorded
MAX_AGE = 0.5  # s: an older state counts as lost


def transform(column_major) -> np.ndarray:
  """libfranka's column-major 4x4 transform as a matrix."""
  return np.asarray(column_major, dtype=float).reshape(4, 4).T


def quat_wxyz(rot: np.ndarray) -> np.ndarray:
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, rot.flatten())
  return quat


def rpy_deg(rot: np.ndarray) -> np.ndarray:
  """Roll, pitch and yaw (rotations about the fixed x, y and z axes, in this order)."""
  return np.degrees([np.arctan2(rot[2, 1], rot[2, 2]), -np.arcsin(np.clip(rot[2, 0], -1, 1)),
                     np.arctan2(rot[1, 0], rot[0, 0])])


class Robot:
  """The latest state that ``stream_state`` printed, read in a thread."""

  def __init__(self, host: str) -> None:
    if not STREAM.exists():
      sys.exit(f"{STREAM} is missing; build it: cmake --build real/build")
    self.proc = subprocess.Popen([str(STREAM), host, "20"], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
    self.latest: tuple[float, dict] | None = None
    self.error = ""
    threading.Thread(target=self._read, daemon=True).start()

  def _read(self) -> None:
    for line in self.proc.stdout:
      if line.startswith("{"):
        self.latest = (time.monotonic(), json.loads(line))
    self.error = self.proc.stderr.read().strip() or f"stream_state exited ({self.proc.wait()})"

  def state(self) -> dict | None:
    latest = self.latest
    return latest[1] if latest and time.monotonic() - latest[0] < MAX_AGE else None

  def close(self) -> None:
    self.proc.send_signal(signal.SIGINT)
    try:
      self.proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
      self.proc.kill()


class Recorder:
  def __init__(self, args: argparse.Namespace) -> None:
    self.args = args
    self.out = Path(args.out)
    self.out.parent.mkdir(parents=True, exist_ok=True)
    self.poses: list[dict] = []
    self.lock = threading.Lock()
    self.quit = False
    self.robot = Robot(args.host)

    spec = sysid_model.make_spec(80, 20)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.7, 0.7, 0.0005), pos=(0.3, 0, -0.0005),
                            rgba=(0.5, 0.6, 0.7, 0.25), contype=0, conaffinity=0)
    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    self.server = viser.ViserServer(port=args.port)
    self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)
    self.scene.camera_tracking_enabled = False
    self.ee = self.server.scene.add_frame("/ee", axes_length=0.1, axes_radius=0.004)
    self.pose_handles: list = []
    self._gui()

    @self.server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
      client.camera.position = (1.5, -1.1, 0.9)
      client.camera.look_at = (0.3, 0.0, 0.3)
      client.camera.up_direction = (0.0, 0.0, 1.0)

  def _gui(self) -> None:
    g = self.server.gui
    g.add_markdown(f"**Record {self.args.count} poses** of the Panda at {self.args.host}  \n"
                   f"File: `{self.out}`")
    self.g_status = g.add_markdown("")
    self.g_record = g.add_button("", color="green", icon=viser.Icon.CAMERA)
    self.g_delete = g.add_button("Delete the last pose", icon=viser.Icon.TRASH)
    self.g_message = g.add_markdown("")
    self.g_poses = g.add_markdown("")
    self._refresh()

    @self.g_record.on_click
    def _(_) -> None:
      self.say(self.record())

    @self.g_delete.on_click
    def _(_) -> None:
      self.say(self.delete_last())

  def say(self, text: str) -> None:
    print(text, flush=True)
    self.g_message.content = text

  # -- poses -----------------------------------------------------------------

  def record(self) -> str:
    with self.lock:
      state = self.robot.state()
      if state is None:
        return f"No robot state. {self.robot.error}"
      if len(self.poses) >= self.args.count:
        return f"All {self.args.count} poses are recorded; delete one to record it again."
      speed = np.abs(state["dq"]).max()
      if speed > STILL:
        return f"The arm is moving ({speed:.3f} rad/s); let go of it and record again."
      tf = transform(state["O_T_EE"])
      self.poses.append({
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "q": state["q"],
        "O_T_EE": tf.tolist(),
        "position": tf[:3, 3].tolist(),
        "quat_wxyz": quat_wxyz(tf[:3, :3]).tolist(),
        "state": state,
      })
      self._changed()
      n = len(self.poses)
      text = f"Recorded pose {n} of {self.args.count}: EE at {np.round(tf[:3, 3] * 1e3, 1)} mm."
      return text + (f" Done; saved to {self.out}." if n == self.args.count else "")

  def delete_last(self) -> str:
    with self.lock:
      if not self.poses:
        return "No pose to delete."
      self.poses.pop()
      self._changed()
      return f"Deleted pose {len(self.poses) + 1}."

  def _changed(self) -> None:
    self._save()
    self._draw_poses()
    self._refresh()

  def _save(self) -> None:
    doc = {
      "robot": self.args.host,
      "frame": "O_T_EE: the end effector set in Desk (F_T_EE, EE_T_K in each state) in the robot's base frame",
      "poses": self.poses,
    }
    tmp = self.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(self.out)

  def _draw_poses(self) -> None:
    for h in self.pose_handles:
      h.remove()
    self.pose_handles = []
    sc = self.server.scene
    for i, pose in enumerate(self.poses, start=1):
      # Removing the frame removes its label too.
      self.pose_handles.append(sc.add_frame(f"/poses/{i}", axes_length=0.06, axes_radius=0.003,
                                            position=pose["position"], wxyz=pose["quat_wxyz"]))
      sc.add_label(f"/poses/{i}/label", str(i), position=(0, 0, -0.02))
    if len(self.poses) > 1:
      p = np.array([pose["position"] for pose in self.poses])
      self.pose_handles.append(sc.add_line_segments("/poses/path", np.stack([p[:-1], p[1:]], axis=1),
                                                    (120, 120, 120), thickness=0.002))

  def _refresh(self) -> None:
    n, count = len(self.poses), self.args.count
    self.g_record.label = f"Record pose {n + 1} of {count}" if n < count else f"All {count} poses recorded"
    self.g_record.disabled = n >= count
    self.g_delete.disabled = n == 0
    rows = [f"{i}. q (deg) {' '.join(f'{v:.1f}' for v in np.degrees(pose['q']))}"
            for i, pose in enumerate(self.poses, start=1)]
    self.g_poses.content = "<small>" + "  \n".join(rows) + "</small>"

  # -- loop ------------------------------------------------------------------

  def terminal(self) -> None:
    for line in sys.stdin:
      cmd = line.strip().lower()
      if cmd == "":
        self.say(self.record())
      elif cmd == "d":
        self.say(self.delete_last())
      elif cmd == "q":
        self.quit = True
        return
      else:
        print("Enter: record, d + Enter: delete the last pose, q + Enter: quit", flush=True)

  def status(self, state: dict | None) -> str:
    if state is None:
      return f"**No robot state.** {self.robot.error}"
    tf = transform(state["O_T_EE"])
    speed = np.abs(state["dq"]).max()
    # The model's end-effector site at the measured q against the pose the robot reports.
    fk_error = np.linalg.norm(self.data.site_xpos[self.model.site("ee").id] - tf[:3, 3]) * 1e3
    return "  \n".join([
      f"Robot mode: **{state['robot_mode']}**, "
      + ("still" if speed <= STILL else f"**moving** ({speed:.3f} rad/s)"),
      f"q (deg): {' '.join(f'{v:.1f}' for v in np.degrees(state['q']))}",
      f"EE (mm): {' '.join(f'{v:.1f}' for v in tf[:3, 3] * 1e3)}",
      f"EE roll pitch yaw (deg): {' '.join(f'{v:.1f}' for v in rpy_deg(tf[:3, :3]))}",
      f"<small>Model vs. robot EE position: {fk_error:.1f} mm</small>",
    ])

  def run(self) -> None:
    print(f"Open http://localhost:{self.args.port}. Enter: record, d + Enter: delete the last pose, "
          "q + Enter or Ctrl+C: quit.", flush=True)
    threading.Thread(target=self.terminal, daemon=True).start()
    shown = 0.0
    while not self.quit:
      state = self.robot.state()
      if state is not None:
        self.data.qpos[:7] = state["q"]
        mujoco.mj_kinematics(self.model, self.data)
        self.scene.update_from_mjdata(self.data)
        tf = transform(state["O_T_EE"])
        self.ee.position, self.ee.wxyz = tf[:3, 3], quat_wxyz(tf[:3, :3])
      if time.monotonic() - shown > 0.2:
        self.g_status.content = self.status(state)
        shown = time.monotonic()
      time.sleep(1 / 30)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--host", default="172.16.0.2", help="the robot's Control (FCI) address")
  parser.add_argument("--count", type=int, default=8, help="number of poses to record")
  parser.add_argument("--out", default=f"runs/poses/poses_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
  parser.add_argument("--port", type=int, default=8080, help="viser port")
  args = parser.parse_args()
  recorder = Recorder(args)
  try:
    recorder.run()
  except KeyboardInterrupt:
    pass
  finally:
    recorder.robot.close()
  print(f"{len(recorder.poses)} poses in {recorder.out}" if recorder.poses else "No pose recorded.")


if __name__ == "__main__":
  main()
