"""The real robot's model: the Panda with the Franka Hand under joint impedance.

The arm is ``robot.make_arm_spec`` (link inertials, dry friction, damping,
armature of a ``robot.Dynamics``). The Franka Hand is Menagerie's model, whose
inertials are the end-effector values that Desk holds for it (0.73 kg),
mounted as on the real robot (Franka's F_T_EE: the hand turned by -45 deg
about the flange axis). Its fingers are rigid at a fixed opening, since the
gripper does not move during identification.

Each joint has a position actuator, tau = kp (q_d - q) - kd dq, with the
joint's torque limit, and gravity compensation counted against that limit.
The controller on the robot (``real/joint_impedance.cpp``) computes the same
torque from its measured q and dq; the robot adds gravity. So ``ctrl`` here
is the logged q_d. ``panda_sysid.delayed`` rolls the same controller out with
the robot's delays.
"""

from __future__ import annotations

import mujoco
import numpy as np

from panda_sysid import robot

# Franka's flange-to-end-effector transform for the Franka Hand (F_T_EE in the
# robot state): the hand turned by -45 deg about the flange axis, and the
# end-effector frame 0.1034 m beyond the flange.
HAND_YAW = -np.pi / 4
EE_OFFSET = 0.1034
FINGER_OPENING = 0.04  # m per finger: the hand fully open (79.9 mm measured)

# The hand pointing straight down, its x axis along the world's x axis, the
# elbow up: the seed of the IK (``checks.ik_down``) and so the IK branch of
# every planned pose.
HOME = np.array([0.0, 0.4813, 0.0, -1.8121, 0.0, 2.2935, -HAND_YAW])


def _franka_hand() -> mujoco.MjSpec:
  hand = mujoco.MjSpec.from_file(str(robot.HAND_XML))
  for act in list(hand.actuators):
    hand.delete(act)
  for eq in list(hand.equalities):
    hand.delete(eq)
  for tendon in list(hand.tendons):
    hand.delete(tendon)
  for exclude in list(hand.excludes):  # make_spec adds them back (with contacts)
    hand.delete(exclude)
  for name in ("left_finger", "right_finger"):
    finger = hand.body(name)
    for joint in list(finger.joints):
      hand.delete(joint)
    # Each finger slides along its own y axis (the right one is turned by 180 deg).
    finger.pos = np.array(finger.pos) + frame_y(finger.quat) * FINGER_OPENING
  hand.body("hand").quat = (1.0, 0.0, 0.0, 0.0)
  for body in hand.bodies[1:]:
    body.gravcomp = 1.0
  return hand


def frame_y(quat) -> np.ndarray:
  """The y axis of a body frame with orientation ``quat``, in its parent's frame."""
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
  return mat.reshape(3, 3)[:, 1]


def add_position_actuator(spec: mujoco.MjSpec, joint: str, kp: float, kd: float, limit: float) -> None:
  """A position actuator on ``joint``: force kp (ctrl - q) - kd dq, clamped to +-limit.

  ``ctrl`` is not limited (a target beyond the joint range still pulls).
  """
  act = spec.add_actuator(name=joint, target=joint)
  act.trntype = mujoco.mjtTrn.mjTRN_JOINT
  act.dyntype = mujoco.mjtDyn.mjDYN_NONE
  act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
  act.gainprm[0] = kp
  act.biasprm[1] = -kp
  act.biasprm[2] = -kd
  act.inheritrange = 0.0
  act.ctrllimited = False
  act.forcelimited = True
  act.forcerange[:] = np.array([-limit, limit])
  # Informational: beyond this range of ctrl the force saturates at any q.
  lo, hi = spec.joint(joint).range
  act.ctrlrange[:] = np.array([lo - limit / kp, hi + limit / kp])


def make_spec(kp: float | np.ndarray, kd: float | np.ndarray, dynamics: robot.Dynamics | None = None,
              contact: bool = False) -> mujoco.MjSpec:
  """The arm and the Franka Hand under joint impedance (kp, kd).

  ``dynamics`` defaults to ``robot.Dynamics.nominal()``. ``contact`` keeps the
  collision geoms of every link and the hand (for checking a planned motion);
  identification runs without contacts.
  """
  links = tuple(f"link{i}" for i in range(8)) if contact else ()
  spec = robot.make_arm_spec(dynamics, collide_links=links)
  spec.option.timestep = 0.001
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
  spec.option.iterations = 100
  spec.option.ls_iterations = 50
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = 10.0
  if not contact:
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

  hand = _franka_hand()
  link7 = spec.body("link7")
  frame = link7.add_frame(pos=(0, 0, robot.FLANGE_OFFSET),
                          quat=(np.cos(HAND_YAW / 2), 0, 0, np.sin(HAND_YAW / 2)))
  # The hand's elements keep the prefix (its materials share the arm's names); its bodies drop it.
  frame.attach_body(hand.body("hand"), "hand/", "")
  for body in spec.bodies:
    if body.name.startswith("hand/"):
      body.name = body.name[len("hand/"):]
  spec.body("hand").add_site(name="ee", pos=(0, 0, EE_OFFSET), size=(0.004, 0, 0))
  if contact:
    # The hand's own exclusions; adjacent links never collide (MuJoCo's parent filter).
    spec.add_exclude(bodyname1="hand", bodyname2="left_finger")
    spec.add_exclude(bodyname1="hand", bodyname2="right_finger")
    spec.add_exclude(bodyname1="link7", bodyname2="hand")
  else:
    for geom in spec.geoms:
      geom.contype = 0
      geom.conaffinity = 0

  kp = np.broadcast_to(np.asarray(kp, dtype=float), (7,))
  kd = np.broadcast_to(np.asarray(kd, dtype=float), (7,))
  for name, k, d, limit in zip(robot.JOINTS, kp, kd, robot.EFFORT_LIMITS):
    add_position_actuator(spec, name, float(k), float(d), limit)
    spec.add_sensor(name=f"{name}_pos", type=mujoco.mjtSensor.mjSENS_JOINTPOS, objname=name,
                    objtype=mujoco.mjtObj.mjOBJ_JOINT)
  for name in robot.JOINTS:
    spec.add_sensor(name=f"{name}_vel", type=mujoco.mjtSensor.mjSENS_JOINTVEL, objname=name,
                    objtype=mujoco.mjtObj.mjOBJ_JOINT)
  return spec


def add_render_settings(spec: mujoco.MjSpec, width: int = 960, height: int = 720) -> None:
  spec.visual.global_.offwidth = width
  spec.visual.global_.offheight = height
  spec.worldbody.add_light(pos=(0.6, -0.4, 1.5), dir=(0.0, 0.25, -1.0),
                           type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=(0.4, 0.4, 0.4))


def ee_pose(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Position and rotation matrix of the end-effector frame at joint angles q."""
  data.qpos[:7] = q
  mujoco.mj_kinematics(model, data)
  site = model.site("ee").id
  return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()
