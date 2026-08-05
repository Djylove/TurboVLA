# TurboVLA GR3 profiles

The primary training profile is
`xpolicylab.gr3_anygrasp_lerobot_v3`. It reads the existing GR3 AnyGrasp
LeRobot v3 dataset in place: one center-cropped top camera, the canonical 33D
GR3 state, canonical 37D actions at 30 Hz, and per-clip subtask text. Source
videos and action parquet files remain read-only and are not copied into the
training workspace.

The canonical dimensions above describe storage. The TurboVLA GR3 model uses
only the first 31 named joint axes for both state and action. The two base-state
axes are excluded before normalization, and inference pads six zero base-action
axes when returning the external 37D action contract.

`xpolicylab.gr3_dagger_v2` remains a legacy compatibility profile for the old
recorder format. Its intervention and expert-safe-label semantics must not be
applied to AnyGrasp data.

Training is not launched directly. The supported production path is
Interactive Training's restricted XPolicyLab runner, which validates and
hashes the dataset before invoking `policy/TurboVLA/train.sh`.

The CPU-only data check may be run directly because it does not train or load a
policy:

```bash
python -m experiments.gr3.preflight \
  --dataset-manifest /path/to/dataset-manifest.json
```

Create the local policy environment with the GR3 data and checkpoint extras:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e '.[gr3]'
uv pip install --python .venv/bin/python -e /home/ubuntu/xpolicylabdagger
```

The XPolicyLab install is required by the recorded-replay policy server and
client. Training must still be launched through Interactive Training; the
environment setup command does not launch training.

A short curated clip selection is sufficient only for a loader/overfit smoke
test. It is not sufficient evidence for simulation or robot deployment quality.
