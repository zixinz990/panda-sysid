"""Excitation trajectories for identification: q_d offsets from the start pose at 1 kHz.

Every segment starts and ends at offset zero with zero target velocity, so
segments can be joined and a session ends where it started.

- ``hold``: the start pose; the controller holds still.
- ``policy_steps``: targets that jump every 20 ms, like those of a learned
  policy at 50 Hz (measured q plus a scaled action), here a mean-reverting
  random walk so that the targets stay near the start.
- ``multisine``: smooth sums of sines, a different set of frequencies per
  joint, faded in and out.
- ``graded_steps``: every joint steps from rest by a ladder of sizes, from
  inside the dry-friction dead band (friction / kp, 0.8-1.9 deg at kp 40)
  to well beyond it.
- ``sweeps``: every joint at once sweeps back and forth at one constant
  speed (triangle waves; friction against speed).
- ``fourier``: large, fast periodic motion of every joint for Franka's own
  joint position controller (link inertials from the torque sensors).

``min_jerk`` is a move between two poses (absolute angles, not an offset).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT = 0.001
POLICY_DT = 0.02


@dataclass
class Segment:
  name: str
  offsets: np.ndarray  # (n, 7), rad, relative to the start pose


def hold(duration: float) -> Segment:
  return Segment("hold", np.zeros((round(duration / DT), 7)))


def policy_steps(duration: float, sigma: float | np.ndarray, bound: float | np.ndarray,
                 correlation_time: float = 1.0, seed: int = 0) -> Segment:
  """Targets held for 20 ms each, as a mean-reverting random walk around zero.

  Each 20 ms step adds a Gaussian jump of std ``sigma`` (rad) and pulls back
  toward zero with time constant ``correlation_time``; targets are clipped
  to +-``bound``. The last second returns to zero in steps of at most
  ``sigma``.
  """
  rng = np.random.default_rng(seed)
  sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (7,))
  bound = np.broadcast_to(np.asarray(bound, dtype=float), (7,))
  rho = np.exp(-POLICY_DT / correlation_time)
  n_steps = round(duration / POLICY_DT)
  n_return = round(1.0 / POLICY_DT)
  targets = np.zeros((n_steps, 7))
  for k in range(1, n_steps - n_return):
    targets[k] = np.clip(rho * targets[k - 1] + sigma * rng.standard_normal(7), -bound, bound)
  for k in range(max(1, n_steps - n_return), n_steps):
    targets[k] = targets[k - 1] - np.clip(targets[k - 1], -sigma, sigma)
  targets[-1] = 0.0
  per_step = round(POLICY_DT / DT)
  return Segment("policy_steps", np.repeat(targets, per_step, axis=0))


def multisine(duration: float, amplitude: float | np.ndarray, f_min: float = 0.1,
              f_max: float = 1.5, fade: float = 2.0, seed: int = 0) -> Segment:
  """Per joint, a sum of sines at its own harmonics of 1 / duration.

  The harmonics in [f_min, f_max] are dealt to the joints in turn, so no two
  joints share a frequency. Phases are Schroeder's (low peak for the power),
  then each joint is scaled to its peak ``amplitude`` and faded in and out
  over ``fade`` seconds with a raised cosine.
  """
  rng = np.random.default_rng(seed)
  amplitude = np.broadcast_to(np.asarray(amplitude, dtype=float), (7,))
  n = round(duration / DT)
  t = np.arange(n) * DT
  f0 = 1.0 / duration
  harmonics = np.arange(int(np.ceil(f_min / f0)), int(np.floor(f_max / f0)) + 1)
  order = rng.permutation(7)
  offsets = np.zeros((n, 7))
  for slot, j in enumerate(order):
    ks = harmonics[slot::7]
    m = len(ks)
    phases = -np.pi * np.arange(m) * (np.arange(m) + 1) / m
    signal = np.sin(2 * np.pi * f0 * ks[:, None] * t[None] + phases[:, None]).sum(0)
    offsets[:, j] = signal / np.abs(signal).max()
  window = np.ones(n)
  n_fade = round(fade / DT)
  ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_fade) / n_fade)
  window[:n_fade] = ramp
  window[n - n_fade:] = ramp[::-1]
  offsets *= window[:, None]
  return Segment("multisine", offsets * amplitude / np.abs(offsets).max(0).clip(1e-12))



def graded_steps(sizes: list[float], hold: float = 1.2, seed: int = 0) -> Segment:
  """Every joint steps from rest by each of ``sizes`` (rad), both ways.

  Per size the targets are 0 -> +s -> 0 -> -s -> 0 (or the mirror image),
  each held ``hold`` seconds, so every step starts from rest. Each joint
  takes the sizes in its own random order, so the joints do not step by the
  same amount at once.
  """
  rng = np.random.default_rng(seed)
  sizes = np.asarray(sizes, dtype=float)
  pattern = np.array([1.0, 0.0, -1.0, 0.0])
  targets = np.zeros((4 * len(sizes), 7))
  for j in range(7):
    signs = rng.choice([-1.0, 1.0], len(sizes))
    targets[:, j] = (rng.permutation(sizes)[:, None] * signs[:, None] * pattern[None]).ravel()
  return Segment("graded_steps", np.repeat(targets, round(hold / DT), axis=0))


def sweeps(speed: float, amplitude: float, cycles: int = 2, spread: float = 0.3) -> Segment:
  """Every joint at once: 0 -> +a -> -a -> ... -> 0 at constant ``speed`` (rad/s).

  Joint j's amplitude is ``amplitude`` (1 + spread j / 6), so the joints turn
  at different times; joints that finish early hold zero. The corners are
  rounded over 0.1 s.
  """
  paths = []
  for j in range(7):
    a = amplitude * (1 + spread * j / 6)
    knots = [0.0] + [a, -a] * cycles + [a, 0.0]
    t_knots = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(knots)) / speed)])
    t = np.arange(round(t_knots[-1] / DT)) * DT
    path = np.interp(t, t_knots, knots)
    k = round(0.1 / DT)
    paths.append(np.convolve(np.pad(path, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k])
  offsets = np.zeros((max(len(p) for p in paths), 7))
  for j, path in enumerate(paths):
    offsets[:len(path), j] = path
  return Segment(f"sweeps {speed:g} rad/s", offsets)


def smoothstep(x: np.ndarray) -> np.ndarray:
  """0 -> 1 with zero slope and curvature at both ends (quintic)."""
  x = np.clip(x, 0.0, 1.0)
  return 10 * x**3 - 15 * x**4 + 6 * x**5


def fourier(coefficients: np.ndarray, period: float, repeats: int, fade: float = 4.0) -> Segment:
  """A periodic excitation: per joint j, sum over harmonics l = 1..L of
  a_jl sin(2 pi l t / period) + b_jl cos(2 pi l t / period), with
  ``coefficients`` (7, L, 2) = (a, b) in rad.

  It runs ``repeats`` full periods, faded in and out over ``fade`` seconds
  (``smoothstep``, so speed and acceleration start and end at zero, as
  Franka's motion generators require).
  """
  n_fade = round(fade / DT)
  n = round(repeats * period / DT) + 2 * n_fade
  t = np.arange(n) * DT
  harmonics = np.arange(1, coefficients.shape[1] + 1)
  phase = 2 * np.pi * harmonics[None] * t[:, None] / period  # (n, L)
  offsets = np.sin(phase) @ coefficients[..., 0].T + np.cos(phase) @ coefficients[..., 1].T
  window = np.minimum(smoothstep(t / fade), smoothstep((t[-1] - t) / fade))
  return Segment("fourier", offsets * window[:, None])


def min_jerk(q0: np.ndarray, q1: np.ndarray, max_speed: float, min_time: float = 3.0,
             hold: float = 0.0) -> np.ndarray:
  """Absolute q at 1 kHz: hold, minimum-jerk move (peak speed 1.875 dq / T), hold."""
  duration = max(min_time, float(np.max(1.875 * np.abs(q1 - q0) / max_speed)))
  s = smoothstep(np.arange(round(duration / DT) + 1) * DT / duration)
  path = q0 + s[:, None] * (q1 - q0)
  n_hold = round(hold / DT)
  return np.vstack([np.repeat(q0[None], n_hold, 0), path, np.repeat(q1[None], n_hold, 0)])


def join(segments: list[Segment], gap: float = 1.0) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
  """Concatenate segments with ``gap`` seconds of holding between them.

  Returns the offsets and (name, start time, end time) per segment.
  """
  pause = np.zeros((round(gap / DT), 7))
  parts, spans, t = [pause], [], gap
  for seg in segments:
    parts.append(seg.offsets)
    spans.append((seg.name, t, t + len(seg.offsets) * DT))
    t += len(seg.offsets) * DT + gap
    parts.append(pause)
  return np.concatenate(parts), spans


def session(name: str, seed: int = 0) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
  """The planned sessions.

  - ``reference``: the drift and repeatability check of the campaign (always
    seed 0, so every repeat is identical): hold, graded steps, sweeps at
    0.02 rad/s, policy-like steps. About a minute.
  - ``pose``: the campaign's session at each pose: hold, graded steps
    (4-48 mrad), sweeps at 5, 20 and 60 mrad/s, policy-like steps,
    multisine. About two and a half minutes.
  """
  if name == "reference":
    return join([
      hold(2.0),
      graded_steps([0.005, 0.01, 0.02, 0.04]),
      sweeps(0.02, 0.04),
      policy_steps(15.0, sigma=0.01, bound=0.04, correlation_time=0.3),
    ])
  if name == "pose":
    return join([
      hold(3.0),
      graded_steps([0.004, 0.008, 0.012, 0.016, 0.024, 0.032, 0.048], seed=seed),
      sweeps(0.005, 0.015, cycles=1),
      sweeps(0.02, 0.04),
      sweeps(0.06, 0.07),
      policy_steps(30.0, sigma=0.01, bound=0.04, correlation_time=0.3, seed=seed),
      multisine(30.0, amplitude=[0.05, 0.04, 0.05, 0.05, 0.06, 0.06, 0.08], f_max=1.5, seed=seed),
    ])
  raise ValueError(f"unknown session {name!r}")
