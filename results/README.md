# Results on our Panda

Our robot: a Franka Emika Panda (production 2019, system version with FCI
server version 4, libfranka 0.8.0) with the Franka Hand (0.73 kg in Desk),
controlled from a PC without a real-time kernel. Joint impedance kp 40, kd
20 on every joint. The campaign ran on 2026-10-28 (19 runs, 36 minutes of
motion); `run_all.sh` makes every file here from its logs.

Two models are compared throughout:

- **Before identification**: the values a Panda simulation usually starts
  from. Link inertials from Franka (franka_description), dry friction from
  Gaz et al. (RA-L 2019), no damping, armature of the FR3 in MuJoCo
  Menagerie, and a controller without delays.
- **Identified**: the model in `model.json`, fitted to our runs.

## Link inertials (`inertial/`)

Fitted on six Fourier runs: three large periodic motions of every joint
at once, each run twice, under Franka's joint position controller. Fitted
on two of the motions, the identified inertials predict the measured joint
torques of the third to 0.05-0.23 N m RMS per joint; Franka's inertials miss
by up to 3.0 N m (joint 2) and 2.6 N m (joint 4). With the arm at rest at
the eight session poses, the identified inertials match the measured
torques as well as Franka's own (closed-source) model does.

| Link | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mass (kg), before identification | 2.79 | 2.54 | 2.25 | 2.20 | 2.29 | 1.35 | 0.36 |
| Mass (kg), identified | 2.69 | 1.65 | 2.30 | 2.51 | 2.47 | 1.55 | 0.83 |

The torques fix only 43 combinations of the 70 inertial parameters, so
single masses are not unique: the fit keeps what the data cannot see at
Franka's values. What the data do fix is how the arm accelerates under a
torque and the torque gravity puts on each joint.

## Joints (`joints/`, `model.json`)

Fitted on 10 runs under joint impedance: four repeats of a reference
session (one minute at the center pose) and one session at each of six
poses (small steps, slow sweeps and mixed sine waves on every joint). The
fit uses the identified inertials and the delays below and starts from the
values before identification.

| Joint | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dry friction (N m) | 0.723 | 1.104 | 0.637 | 0.829 | 0.611 | 0.587 | 0.445 |
| Damping (N m s/rad) | 3.19 | 1.69 | 4.94 | 0.58 | 0 | 0 | 0 |
| Armature (kg m^2) | 0.195* | 0.195* | 0.195* | 0.195* | 0.088 | 0.083 | 0.088 |
| Torque bias, middle of range (N m) | 0.19 | 0.35 | 0.19 | 0.06 | 0.03 | -0.03 | 0.13 |
| Torque bias, +- half of range (N m) | 0.34 | 0.21 | 0.30 | 0.30 | 0.20 | 0.18 | 0.16 |

\* These runs are too slow to show it (any value from 0.02 to 0.4 moves
the simulated joints by less than 0.003 deg), so it stays at the FR3's
value.

**Torque bias**: the robot's gravity compensation is not exact, and what it
leaves acts like a small constant torque on each joint. It changes with the
pose, so the model gives a range for it, not one value.

**Controller delays**: the robot's measured joint velocity is 3 ms late, a
torque command acts 2 ms after it is sent, and libfranka low-pass filters
the command (100 Hz) and limits its rate (1 N m per ms).

## Validation (`validate/`)

Whole sessions (three minutes each) at two held-out poses, which no fit
used, simulated from their first measured state on the robot's commands.
The table gives the error of the simulated joint angles over the session
(RMS, deg). Repeatability is how far four repeats of the reference session
land from each other on the robot; no model can be expected to do better.

| Joint | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Repeatability: the robot against itself | 0.17 | 0.25 | 0.11 | 0.08 | 0.14 | 0.10 | 0.14 |
| Pose 1, before identification | 0.40 | 0.62 | 0.36 | 0.21 | 0.35 | 0.39 | 0.28 |
| Pose 1, identified, bias in the middle of its range | 0.34 | 0.49 | 0.38 | 0.19 | 0.22 | 0.17 | 0.12 |
| Pose 1, identified, bias fitted to the run | 0.14 | 0.26 | 0.09 | 0.12 | 0.21 | 0.12 | 0.12 |
| Pose 2, before identification | 0.74 | 0.83 | 0.69 | 0.23 | 0.22 | 0.36 | 0.29 |
| Pose 2, identified, bias in the middle of its range | 0.28 | 0.49 | 0.41 | 0.24 | 0.13 | 0.10 | 0.09 |
| Pose 2, identified, bias fitted to the run | 0.12 | 0.36 | 0.10 | 0.15 | 0.14 | 0.10 | 0.09 |

![Validation](validate/validate.png)

With the bias in the middle of its range, the identified model brings
joints 5-7 near repeatability, but joints 1-3 still miss by 0.3-0.5 deg. Most of that is the bias: fitting only the seven bias values to the
run, with everything else taken from other runs, brings joints 1-3 down to
0.09-0.36 deg. No single bias fits every pose, so a simulation should draw
the bias for each episode from the range in `model.json`. The range covers
the biases of all 13 impedance runs at eight poses (the held-out ones
included) plus 0.1 N m on each side.

## Large moves (`replay/`)

Between sessions the arm moves slowly from one pose to the next (end
effector 17-38 cm on a straight line, joints turning up to 22-63 deg).
Simulated from their first measured state, with the bias in the middle of
its range, the identified model's end effector is on average 3.2 mm from
the robot's on the moves in the fit and 2.9 mm on the three moves no fit
used; before identification, 10.0 and 9.8 mm. The largest error is 10.3 mm
identified and 24.0 mm before identification.

![Large moves](replay/moves.png)

`replay/move_1`, `move_2` and `move_3` (MP4 and GIF) show the three moves
no fit used: the robot's measured joint angles (white) and the two
simulations on the same commands, identified (green) and before
identification (red), their errors drawn five times larger.

## Files

| Path | Content |
| --- | --- |
| `model.json` | The identified model: link inertials, joint values, delays, torque bias range (`panda_sysid.simmodel`) |
| `workspace/` | The eight hand-guided corners (`corners.json`) and the box the campaign stays in |
| `inertial/` | The inertial fit (`params.json`, `model.json`), its torques on a motion left out (`fit.png`) and at rest (`static.png`) |
| `joints/` | The joint fit: start, fitted values and confidence intervals (`params.json`), its log, `model.json` |
| `bias/` | The bias each impedance run needs with the joint values fixed (`bias_per_run.json`) |
| `validate/` | The held-out sessions against repeatability (`validate.json`, `validate.png`) |
| `replay/` | End-effector errors on the large moves (`moves.json`, `moves.png`) and a video of each move no fit used |
| `run_all.sh` | Every fit above, in order, from the campaign's logs in `data/campaign` |
