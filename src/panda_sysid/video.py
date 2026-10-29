"""Offscreen rendering for the planning videos: fonts and render processes.

Rendering uses MuJoCo's EGL backend (``MUJOCO_GL=egl``). Every process that
renders is spawned, never forked, and the process that starts them never
renders: a forked child shares its parent's GPU driver connection, and many
such children rendering at once crashed the NVIDIA kernel driver
(570.133.07) and hung the machine.
"""

from __future__ import annotations

import multiprocessing as mp
import os

from PIL import ImageFont

FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf")


def font(size: int) -> ImageFont.FreeTypeFont:
  for path in FONTS:
    try:
      return ImageFont.truetype(path, size)
    except OSError:
      continue
  return ImageFont.load_default(size)


def _init(counter, gpus: list[int]) -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  with counter.get_lock():
    i = counter.value
    counter.value += 1
  if gpus:
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpus[i % len(gpus)])


def render_pool(workers: int, gpus: list[int] | None = None):
  """Spawned processes that render, each with its own EGL display, on ``gpus``
  (EGL device indices) in turn; ``None`` leaves the device to MuJoCo."""
  ctx = mp.get_context("spawn")
  return ctx.Pool(workers, initializer=_init, initargs=(ctx.Value("i", 0), list(gpus or [])))


def parse_gpus(text: str | None) -> list[int] | None:
  return [int(g) for g in text.split(",")] if text else None
