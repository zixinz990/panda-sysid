"""Fast checks of the model, the parameter files and the planning pieces (no robot, no logs)."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from panda_sysid import campaign, delayed, excitation, fit, inertial, regressor, robot, simmodel, workspace
from panda_sysid import model as sysid_model

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_model_compiles_with_and_without_contacts():
  for contact in (False, True):
    model = sysid_model.make_spec(40, 20, contact=contact).compile()
    names = [model.body(i).name for i in range(model.nbody)]
    assert names == ["world", "link0", *robot.LINKS, "hand", "left_finger", "right_finger"]
    assert model.nu == 7 and model.site("ee").id >= 0


def test_forward_kinematics_match_the_robot():
  """The end-effector site at the recorded joint angles is where the robot reported O_T_EE."""
  corners = json.loads((RESULTS / "workspace" / "corners.json").read_text())
  model = sysid_model.make_spec(40, 20).compile()
  data = mujoco.MjData(model)
  for pose in corners["poses"]:
    pos, rot = sysid_model.ee_pose(model, data, np.array(pose["q"]))
    o_t_ee = np.array(pose["O_T_EE"])
    assert np.abs(pos - o_t_ee[:3, 3]).max() < 1e-5
    assert np.abs(rot - o_t_ee[:3, :3]).max() < 1e-4


def test_dynamics_round_trip_and_values():
  nominal = robot.Dynamics.nominal()
  assert robot.Dynamics.from_dict(json.loads(json.dumps(nominal.to_dict()))) == nominal
  changed = nominal.with_values({"frictionloss3": 0.5, "armature7": 0.1, "bias2": 0.3})
  assert changed.friction[2] == 0.5 and changed.armature[6] == 0.1
  assert changed.joint_values()["frictionloss3"] == 0.5
  model = sysid_model.make_spec(40, 20, changed).compile()
  assert model.dof_frictionloss[2] == 0.5 and model.dof_armature[6] == 0.1


def test_our_model_file():
  ours = simmodel.load(RESULTS / "model.json")
  assert ours.delays == delayed.MEASURED
  assert len(ours.bias_center) == 7 and min(ours.bias_half_width) > 0
  bias = ours.draw_bias(np.random.default_rng(0))
  assert np.all(np.abs(bias - ours.bias_center) <= ours.bias_half_width)
  nominal = simmodel.load("nominal")
  assert nominal.delays == delayed.IDEAL and not nominal.draw_bias(np.random.default_rng(0)).any()


def test_regressor_is_linear_in_the_inertials():
  """Changing the inertials of the regressor's bodies by d pi changes MuJoCo's inverse dynamics by Y d pi
  (the fingers are not among them and stay in the constant part)."""
  rigid = regressor.rigid_model()
  rng = np.random.default_rng(1)
  q = rng.uniform(robot.Q_MIN, robot.Q_MAX, (5, 7))
  dq, ddq = rng.normal(0, 1, (5, 7)), rng.normal(0, 3, (5, 7))
  y = regressor.regressor(rigid, q, dq, ddq)
  pi0 = regressor.get_standard(rigid, "link2")
  pi1 = pi0 * 1.1  # link 2 10% heavier, same center of mass
  other = rigid.__copy__()
  regressor.set_standard(other, "link2", pi1)
  change = regressor.inverse_dynamics(other, q, dq, ddq) - regressor.inverse_dynamics(rigid, q, dq, ddq)
  np.testing.assert_allclose(change, y[..., 10:20] @ (pi1 - pi0), atol=1e-6)
  assert inertial.consistent(pi0) and inertial.consistent(pi1)


def test_delayed_rollout_holds_still():
  """At rest at its target with exact gravity compensation the arm does not move."""
  model = sysid_model.make_spec(40, 20).compile()
  torques, kp, kd = delayed.torque_model(model)
  n = 500
  q0 = sysid_model.HOME
  rec = fit.Recording("still", np.repeat(q0[None], n, 0), np.repeat(q0[None], n, 0), np.zeros((n, 7)),
                      np.ones(n, dtype=bool), np.zeros((n, 7)), np.zeros((fit.HISTORY - 1, 7)))
  for delays in (delayed.IDEAL, delayed.MEASURED):
    q, _ = delayed.rollout(torques, mujoco.MjData(torques), rec, kp, kd, delays)
    assert np.abs(q - q0).max() < 1e-9


def test_delayed_rollout_steps_toward_the_target():
  model = sysid_model.make_spec(40, 20, robot.Dynamics.nominal().with_values(
    {f"frictionloss{j}": 0.0 for j in range(1, 8)})).compile()
  torques, kp, kd = delayed.torque_model(model)
  n = 3000
  q0 = sysid_model.HOME
  target = q0 + 0.05
  rec = fit.Recording("step", np.repeat(target[None], n, 0), np.repeat(q0[None], n, 0), np.zeros((n, 7)),
                      np.ones(n, dtype=bool), np.zeros((n, 7)), np.zeros((fit.HISTORY - 1, 7)))
  # Without friction and with the nominal armature, joint 7 is unstable under the measured
  # delays at kd 20 (the wrist loops sit near their delay stability limit), so: no delays.
  q, _ = delayed.rollout(torques, mujoco.MjData(torques), rec, kp, kd, delayed.IDEAL)
  assert np.abs(q[-1] - target).max() < 1e-3


def test_sessions_start_and_end_at_the_pose():
  for name in ("reference", "pose"):
    offsets, spans = excitation.session(name, seed=1)
    assert np.abs(offsets[0]).max() == 0 and np.abs(offsets[-1]).max() == 0
    assert all(a < b for _, a, b in spans)


def test_largest_box_in_a_cube():
  cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 2) for z in (0, 3)], dtype=float)
  lo, hi = workspace.largest_box(*workspace.halfspaces(cube), cube.mean(0))
  np.testing.assert_allclose(lo, [0, 0, 0], atol=1e-5)
  np.testing.assert_allclose(hi, [1, 2, 3], atol=1e-5)


def test_campaign_runs_are_the_fits_runs():
  names = [r.name for r in campaign.campaign_runs()]
  assert names[:3] == ["01_r1", "02_f1", "03_f2"] and names[-1] == "19_r4" and len(names) == 19
  assert campaign.REFERENCE == ["01_r1", "05_r2", "15_r3", "19_r4"]
  assert campaign.HELD_OUT == ["13_h_h1", "14_h_h2"]
  assert [f[0] for f in campaign.FOURIER] == ["02_f1", "03_f2", "04_f3", "16_f1b", "17_f2b", "18_f3b"]


@pytest.mark.parametrize("path", sorted((RESULTS).glob("*/params.json")))
def test_result_files_parse(path):
  assert json.loads(path.read_text())
