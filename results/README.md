# Results for a Franka Emika Panda

The robot for these results is a Franka Emika Panda with the Franka Hand.
Franka made the robot in 2019. The robot uses FCI server version 4 and
libfranka 0.8.0. We set the mass of the hand in Desk to 0.73 kg. A PC
without a real-time kernel controls the robot.

The joint impedance controller has kp 40 and kd 20 on all joints. We
recorded the campaign on 2026-10-28. The campaign has 19 runs and 36 minutes
of movement. The script `run_all.sh` makes all files in this directory from
the logs of the campaign.

## The two models

This text compares two models of the robot. The tables and the figures use
the labels "before identification" and "identified".

The model before identification has the parameters that a Panda simulation
usually starts from:

- The inertial parameters of the links from Franka (`franka_description`)
- The dry friction from Gaz and other authors (RA-L 2019)
- No damping
- The armature of the FR3 in MuJoCo Menagerie
- A controller without delays.

The identified model is the model in `model.json`. We calculated its
parameters from the runs of the campaign.

## Inertial parameters of the links (`inertial/`)

We calculated the inertial parameters of the links from six Fourier runs. In
a Fourier run, all joints turn at the same time, and the angles are large.
The movement of each joint is a sum of sine waves. The joint position
controller of Franka moves the robot in these runs. The campaign has three
Fourier movements, and the robot did each movement two times.

For a test of the inertial parameters, we calculated them from two of the
three movements. Then we compared the measured joint torques of the
remaining movement with the torques of the model. With the identified model,
the RMS error is 0.05 to 0.23 N m for each joint. With the inertial
parameters of Franka, the largest RMS errors are 3.0 N m (joint 2) and 2.6 N
m (joint 4).

We also did a test at the eight poses of the sessions, when the robot does
not move. At these poses, the RMS error of the identified model is 0.38 to
0.52 N m. The internal model of the robot from Franka has almost the same
error (0.39 to 0.52 N m).

| Link | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mass (kg), before identification | 2.79 | 2.54 | 2.25 | 2.20 | 2.29 | 1.35 | 0.36 |
| Mass (kg), identified | 2.69 | 1.65 | 2.30 | 2.51 | 2.47 | 1.55 | 0.83 |

The torque data give only 43 linear combinations of the 70 inertial
parameters. Thus, the data do not give one correct value for the mass of
each link. For the combinations that the data do not give, the inertial fit
keeps the values of Franka. The data give the mass matrix and the gravity
torques of the robot. The mass matrix is the relation between the joint
torques and the joint accelerations.

## Joint parameters (`joints/`, `model.json`)

We calculated the joint parameters from 10 runs with the joint impedance
controller:

- Four runs of the reference session (one minute at the center pose)
- One session at each of six poses (small step inputs, slow sweeps, and sums
  of sine waves on all joints).

The joint fit uses the identified inertial parameters and the controller
delays in this section. The joint fit starts from the values of the model
before identification.

| Joint | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dry friction (N m) | 0.723 | 1.104 | 0.637 | 0.829 | 0.611 | 0.587 | 0.445 |
| Damping (N m s/rad) | 3.19 | 1.69 | 4.94 | 0.58 | 0 | 0 | 0 |
| Armature (kg m^2) | 0.195* | 0.195* | 0.195* | 0.195* | 0.088 | 0.083 | 0.088 |
| Torque bias, middle of range (N m) | 0.19 | 0.35 | 0.19 | 0.06 | 0.03 | -0.03 | 0.13 |
| Torque bias, limits: middle +- this value (N m) | 0.34 | 0.21 | 0.30 | 0.30 | 0.20 | 0.18 | 0.16 |

\* In these slow runs, the armature of joints 1 to 4 has almost no effect. A
value from 0.02 to 0.4 changes the simulated joint angles by less than 0.003
deg. Thus, the armature of these joints stays at the value of the FR3.

**Torque bias**: The gravity compensation of the robot is not fully
accurate. The remaining error has the same effect as a small constant torque
on each joint. The name of this torque is the torque bias. The torque bias
changes with the pose. Thus, the model gives a range for the torque bias,
not one value.

**Controller delays**: The joint velocity that the robot measures has a delay
of 3 ms. A torque command has its effect 2 ms after the controller sends it.
Also, libfranka applies a low-pass filter (100 Hz) to the command. The
maximum rate of change of the command is 1 N m for each ms.

## Test at two held-out poses (`validate/`)

We did a test of the models with full sessions (three minutes each) at two
held-out poses. No fit used the data from these poses. Each simulation
starts from the measured joint angles and velocities at the start of the
session. Then the simulation gets the same commands as the robot. The table
gives the RMS error of the simulated joint angles during the session, in
degrees.

Repeatability is the difference between four runs of the reference session
on the robot. No model can have a smaller error than this difference.

The table and the figure compare these models:

- Before identification: the model before identification.
- Identified, bias in the middle of its range: the identified model with the
  torque bias in the middle of its range. A simulation that does not know
  the torque bias of the run gets this result.
- Identified, bias from the run: the identified model with a torque bias that
  we calculated from that run. This bias has seven values. We calculated
  all other parameters from other runs.

| Joint | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Repeatability: the robot compared with the robot | 0.17 | 0.25 | 0.11 | 0.08 | 0.14 | 0.10 | 0.14 |
| Pose 1, before identification | 0.40 | 0.62 | 0.36 | 0.21 | 0.35 | 0.39 | 0.28 |
| Pose 1, identified, bias in the middle of its range | 0.34 | 0.49 | 0.38 | 0.19 | 0.22 | 0.17 | 0.12 |
| Pose 1, identified, bias from the run | 0.14 | 0.26 | 0.09 | 0.12 | 0.21 | 0.12 | 0.12 |
| Pose 2, before identification | 0.74 | 0.83 | 0.69 | 0.23 | 0.22 | 0.36 | 0.29 |
| Pose 2, identified, bias in the middle of its range | 0.28 | 0.49 | 0.41 | 0.24 | 0.13 | 0.10 | 0.09 |
| Pose 2, identified, bias from the run | 0.12 | 0.36 | 0.10 | 0.15 | 0.14 | 0.10 | 0.09 |

![Validation](validate/validate.png)

With the torque bias in the middle of its range, the identified model has
errors near the repeatability on joints 5 to 7. But on joints 1 to 3, the
errors are 0.3 to 0.5 deg. The torque bias causes most of this error. With
the torque bias from the run, the errors on joints 1 to 3 are only 0.09 to
0.36 deg. These errors are near the repeatability.

One torque bias is not correct for all poses. Thus, we recommend that a
simulation selects a random torque bias in the range of `model.json` for
each episode. This range includes the torque biases of all 13 impedance runs
at eight poses, plus 0.1 N m on each side. The 13 runs include the runs at
the two held-out poses.

## Large movements (`replay/`)

Between two sessions, the robot moves slowly from one pose to the next pose.
In these movements, the end effector moves 17 to 38 cm along a straight
line. In each movement, one joint turns 22 to 63 deg, and the other joints
turn less. We simulated these movements from the measured joint angles and
velocities at their start, with the torque bias in the middle of its range.

On the movements that the fit used, the identified model has an average
end-effector error of 3.2 mm. On the three movements that no fit used, the
average error is 2.9 mm. The model before identification has average errors
of 10.0 mm and 9.8 mm. The largest error is 10.3 mm with the identified
model and 24.0 mm with the model before identification.

![Large movements](replay/moves.png)

The videos `replay/move_1`, `move_2`, and `move_3` (MP4 and GIF) show the
three movements that no fit used. The white arm shows the measured joint
angles of the robot. The green arm shows the identified model, and the red
arm shows the model before identification. The two simulations get the same
commands as the robot. The videos show the errors of the simulated arms
multiplied by five.

## Files

| Path | Contents |
| --- | --- |
| `model.json` | The identified model: inertial parameters of the links, joint parameters, controller delays, range of the torque bias (`panda_sysid.simmodel`) |
| `workspace/` | The eight corners that we recorded when we moved the robot by hand (`corners.json`), and the box that the campaign stays in |
| `inertial/` | The inertial fit (`params.json`, `model.json`), its torques on a movement that the fit did not use (`fit.png`), and its torques at the eight poses (`static.png`) |
| `joints/` | The joint fit: start values, calculated values, and confidence intervals (`params.json`), the log of the fit, and `model.json` |
| `bias/` | The torque bias of each impedance run, with the joint parameters from the joint fit (`bias_per_run.json`) |
| `validate/` | The test at the two held-out poses, with the repeatability (`validate.json`, `validate.png`) |
| `replay/` | The end-effector errors on the large movements (`moves.json`, `moves.png`), and one video for each movement that no fit used |
| `run_all.sh` | All fits in this directory, in sequence, from the logs of the campaign in `data/campaign` |
