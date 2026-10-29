#!/bin/bash
# Every fit of results/, in order, from the campaign's logs in data/campaign.
set -e
cd "$(dirname "$0")/.."
uv run python scripts/fit_inertial.py --logs data/campaign --out results/inertial
uv run python scripts/identify.py --logs data/campaign --model results/inertial/model.json \
  --gravity-offset --fix armature1,armature2,armature3,armature4 --out results/joints
uv run python scripts/bias_per_run.py --logs data/campaign --model results/joints/model.json \
  --out results/bias --model-out results/model.json
uv run python scripts/validate.py --logs data/campaign --model published:nominal \
  --model identified:results/model.json:1 \
  --model "identified, own bias:results/model.json:1:results/bias/bias_per_run.json" \
  --model "identified, V1's bias:results/model.json:1:results/bias/bias_per_run.json@13_v_v1" \
  --model "identified, V2's bias:results/model.json:1:results/bias/bias_per_run.json@14_v_v2" --out results/validate
# The video of the large moves (renders with EGL in a spawned process).
uv run python scripts/replay_video.py --logs data/campaign --out results/replay \
  --runs 08_s_p2,09_s_p3,10_s_p4,11_s_p5,12_s_p6,13_v_v1,14_v_v2,15_r3_aborted
