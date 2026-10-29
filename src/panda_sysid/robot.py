"""The Franka Emika Panda arm in MuJoCo, with published or identified dynamics.

Meshes and kinematics are MuJoCo Menagerie's (``assets/franka_emika_panda``:
the arm without its hand, ``panda_nohand.xml``, and the Franka Hand,
``hand.xml``). A ``Dynamics`` holds what this package identifies: the
inertials of links 1-7 and, per joint, dry friction, viscous damping and
armature. ``Dynamics.nominal()`` holds the published values that the
identification starts from:

- link inertials: Franka's own values for the arm alone (franka_description,
  ``robots/fer/inertials.yaml``). Menagerie's Panda uses the inertials of
  Gaz et al. (RA-L 2019), which were identified with the Franka Hand mounted
  and include its 0.7 kg in the wrist links;
- dry friction: the Coulomb level of the friction model that Gaz et al.
  identified on a real Panda, as MuJoCo's ``frictionloss``;
- armature: the effective rotor inertia that Menagerie's FR3 identified under
  torque control. The FR3 has the Panda's drives (motor inertias and gear
  ratios in franka_description); the values are about a third of the full
  reflected rotor inertia;
- no viscous damping (Gaz et al. found viscous friction negligible).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets" / "franka_emika_panda"
PANDA_XML = ASSETS / "panda_nohand.xml"
HAND_XML = ASSETS / "hand.xml"

JOINTS = tuple(f"joint{i}" for i in range(1, 8))
LINKS = tuple(f"link{i}" for i in range(1, 8))
# Joint torque limits (N m), speed limits (rad/s) and position limits (rad): Franka's
# datasheet ("Limits for Panda" in the FCI documentation).
EFFORT_LIMITS = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
VELOCITY_LIMITS = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# The flange face (Franka's "joint8" frame), along the link-7 z axis.
FLANGE_OFFSET = 0.107

# Link inertials of the arm alone (franka_description, robots/fer/inertials.yaml):
# mass (kg), center of mass in the link frame (m), and the inertia about it,
# (xx, yy, zz, xy, xz, yz) in kg m^2.
FRANKA_INERTIALS = {
  "link1": (2.7907, (0.00033, -0.02204, -0.04762),
            (0.01564782655, 0.01439883526, 0.00500443991, 0.00000531236, -0.00003676721,
             -0.00248480843)),
  "link2": (2.5420, (0.00038, -0.09211, 0.01908),
            (0.01662427728, 0.00462501261, 0.01545124526, 0.00003086077, 0.00000835940,
             0.00355899830)),
  "link3": (2.2513, (0.05152, 0.01696, -0.02971),
            (0.00631741992, 0.00866285493, 0.00657804051, 0.00097309955, 0.00292525834,
             0.00093469224)),
  "link4": (2.2037, (-0.05113, 0.05825, 0.01698),
            (0.00784585566, 0.00649543136, 0.01003079917, -0.00339591957, 0.00158704074,
             -0.00181477884)),
  "link5": (2.2855, (-0.00005, 0.03730, -0.09280),
            (0.02297014781, 0.02095060919, 0.00430606551, -0.00000949345, -0.00002063156,
             0.00382345782)),
  "link6": (1.353, (0.06572, -0.00371, 0.00153),
            (0.00087964522, 0.00277796968, 0.00286701969, -0.00021487814, -0.00011911662,
             0.00001274322)),
  "link7": (0.35973, (0.00089, -0.00044, 0.05491),
            (0.00019541063, 0.00019210361, 0.00017936256, 0.00000165231, 0.00000148826,
             -0.00000131132)),
}
# Joint dry friction, N m: half the span phi_1 of the sigmoid friction model
# identified by Gaz et al. (RA-L 2019, supplementary Table I), which saturates
# at about +-phi_1 / 2.
NOMINAL_FRICTION = (0.273, 0.436, 0.320, 0.640, 0.420, 0.152, 0.282)
# Joint armature, kg m^2: MuJoCo Menagerie's FR3 (franka_fr3/fr3.xml, 2025-04-25,
# "identified using torque control").
NOMINAL_ARMATURE = (0.195, 0.195, 0.195, 0.195, 0.074, 0.074, 0.074)
# MuJoCo's dry friction is a soft constraint; with its default impedance (0.9)
# a joint loaded below its friction creeps, at joint impedance gains, all the
# way to its target within seconds. The stiffest impedance MuJoCo allows and a
# time constant of two steps hold it: loaded to half its friction, every joint
# moves less than 4% of the way to its target in 3 s.
FRICTION_SOLREF = (0.002, 1.0)
FRICTION_SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2.0)

# Names of the per-joint values in a fit ("frictionloss3", ...) and Dynamics' fields.
JOINT_FIELDS = {"frictionloss": "friction", "damping": "damping", "armature": "armature"}


@dataclass(frozen=True)
class Dynamics:
  """Link inertials of links 1-7 and joint values of the arm.

  ``inertials`` maps each link to (mass, center of mass in the link frame,
  inertia about it as (xx, yy, zz, xy, xz, yz)), as ``FRANKA_INERTIALS``.
  """

  inertials: dict
  friction: tuple[float, ...]  # N m, MuJoCo frictionloss
  damping: tuple[float, ...]  # N m s/rad
  armature: tuple[float, ...]  # kg m^2

  @classmethod
  def nominal(cls) -> Dynamics:
    return cls(dict(FRANKA_INERTIALS), NOMINAL_FRICTION, (0.0,) * 7, NOMINAL_ARMATURE)

  @classmethod
  def from_dict(cls, d: dict) -> Dynamics:
    inertials = {link: (float(v["mass"]), tuple(v["com"]), tuple(v["inertia"])) for link, v in d["inertials"].items()}
    return cls(inertials, *(tuple(float(x) for x in d["joints"][k]) for k in ("friction", "damping", "armature")))

  def to_dict(self) -> dict:
    return {
      "inertials": {link: {"mass": m, "com": list(com), "inertia": list(full)}
                    for link, (m, com, full) in self.inertials.items()},
      "joints": {"friction": list(self.friction), "damping": list(self.damping), "armature": list(self.armature)},
    }

  def with_values(self, values: dict[str, float]) -> Dynamics:
    """A copy with fitted joint values ("frictionloss3": 0.55, ...) written in; other names are ignored."""
    fields = {f: list(getattr(self, f)) for f in JOINT_FIELDS.values()}
    for name, value in values.items():
      field = name.rstrip("0123456789")
      if field in JOINT_FIELDS:
        fields[JOINT_FIELDS[field]][int(name[len(field):]) - 1] = float(value)
    return replace(self, **{f: tuple(v) for f, v in fields.items()})

  def joint_values(self) -> dict[str, float]:
    """The joint values under their fit names."""
    return {f"{name}{j + 1}": getattr(self, field)[j] for name, field in JOINT_FIELDS.items() for j in range(7)}


def _principal_inertia(full: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
  """Principal moments and frame (quaternion) of an inertia (xx, yy, zz, xy, xz, yz).

  Franka's link-4 inertia violates the triangle inequality (its two smaller
  principal moments sum to 91% of the largest); both are raised by the same
  amount until they sum to the largest, the smallest change that makes the
  link physical.
  """
  xx, yy, zz, xy, xz, yz = full
  moments, axes = np.linalg.eigh(np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]))
  deficit = moments[2] - moments[0] - moments[1]
  if deficit > 0:
    moments[:2] += deficit / 2
  if np.linalg.det(axes) < 0:
    axes[:, 0] *= -1
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, axes.flatten())
  return moments, quat


def make_arm_spec(dynamics: Dynamics | None = None, collide_links: tuple[str, ...] = ()) -> mujoco.MjSpec:
  """The arm alone, with ``dynamics`` (default: nominal) and nothing at its flange.

  Only ``collide_links`` keep collision geoms. Every body is gravity
  compensated, through the joints' actuators (as the robot adds gravity to
  the torque command).
  """
  dynamics = Dynamics.nominal() if dynamics is None else dynamics
  arm = mujoco.MjSpec.from_file(str(PANDA_XML))
  for act in list(arm.actuators):
    arm.delete(act)
  for key in list(arm.keys):
    arm.delete(key)
  for light in list(arm.lights):
    arm.delete(light)
  for site in list(arm.sites):
    arm.delete(site)
  # Menagerie's link0-link1 exclude is moot (neither link collides here).
  for exclude in list(arm.excludes):
    arm.delete(exclude)
  arm.delete(arm.body("attachment"))

  for body in arm.bodies[1:]:
    collide = body.name in collide_links
    for geom in body.geoms:
      # Menagerie's collision geoms are group 3; the visual ones never collide.
      if geom.group == 3:
        geom.contype = 1 if collide else 0
        geom.conaffinity = 1 if collide else 0
    body.gravcomp = 1.0
  for name, (mass, com, full) in dynamics.inertials.items():
    body = arm.body(name)
    moments, quat = _principal_inertia(full)
    body.mass = mass
    body.ipos = com
    body.inertia = moments
    body.iquat = quat
    body.fullinertia = [np.nan] * 6
    body.explicitinertial = True
  for name, friction, damping, armature in zip(JOINTS, dynamics.friction, dynamics.damping, dynamics.armature):
    joint = arm.joint(name)
    joint.actgravcomp = True
    joint.armature = armature
    joint.damping = (damping, 0.0, 0.0)
    joint.frictionloss = friction
    joint.solref_friction = FRICTION_SOLREF
    joint.solimp_friction = FRICTION_SOLIMP
  return arm
