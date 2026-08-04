import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.gr3.train import _load_normalization
from turbovla.data.gr3_anygrasp import Gr3AnygraspDataset


def _manifest(tmp_path: Path, *, instruction_source: str, sample_stride: int) -> Path:
    dataset_root = tmp_path / "dataset"
    batch = dataset_root / "task_1" / "batch_0"
    (batch / "meta").mkdir(parents=True)
    (batch / "data" / "chunk-000").mkdir(parents=True)
    video = batch / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    (batch / "meta" / "info.json").write_text(
        json.dumps(
            {
                "robot_type": "gr3qnexo",
                "fps": 30,
                "features": {
                    "observation.state": {"shape": [33]},
                    "action": {"shape": [37]},
                },
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "observation.state": [np.zeros(33, dtype=np.float32).tolist()] * 5,
                "action": [np.zeros(37, dtype=np.float32).tolist()] * 5,
                "episode_index": [0] * 5,
                "frame_index": list(range(5)),
            }
        ),
        batch / "data" / "chunk-000" / "file-000.parquet",
    )
    clips = tmp_path / "clips.parquet"
    pq.write_table(
        pa.table(
            {
                "clip_id": ["clip-0"],
                "task_id": ["task-1"],
                "batch_path": ["task_1/batch_0"],
                "episode_index": [0],
                "subtask": ["atomic instruction"],
                "prompt": ["deployment instruction"],
                "clip_frame_start": [0],
                "video_frame_start": [0],
                "num_clip_frames": [5],
                "source_video": [
                    "task_1/batch_0/videos/observation.images.top/chunk-000/file-000.mp4"
                ],
            }
        ),
        clips,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "xpolicy_dataset.v1",
                "profile_id": "xpolicylab.gr3_anygrasp_lerobot_v3",
                "dataset_id": "test",
                "episodes": [
                    {
                        "source_schema": "lerobot_v3_gr3qnexo_top",
                        "train_eligible_after_filters": True,
                        "path": str(tmp_path),
                        "profile_data": {
                            "robot_type": "gr3qnexo",
                            "state_dim": 33,
                            "action_dim": 37,
                            "fps": 30,
                            "camera_key": "observation.images.top",
                            "dataset_root": str(dataset_root),
                            "clips_file": clips.name,
                            "clip_limit": 1,
                            "selected_task_ids": ["task-1"],
                            "instruction_source": instruction_source,
                            "sample_stride": sample_stride,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_deployment_prompt_and_sample_stride_are_manifest_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "turbovla.data.gr3_anygrasp._decode_av1_frame",
        lambda _path, _index: np.zeros((12, 12, 3), dtype=np.uint8),
    )
    dataset = Gr3AnygraspDataset(
        _manifest(tmp_path, instruction_source="prompt", sample_stride=2),
        horizon=3,
        decode_threads=2,
    )

    assert dataset.instruction_source == "prompt"
    assert dataset.sample_stride == 2
    assert dataset.decode_threads == 2
    assert len(dataset) == 3
    assert {sample.instruction for sample in dataset.samples} == {
        "deployment instruction"
    }
    batch = dataset.__getitems__([0, 2])
    assert len(batch) == 2
    assert all(sample["lang"] == "deployment instruction" for sample in batch)


def test_subtask_remains_the_backward_compatible_default(tmp_path):
    manifest = _manifest(tmp_path, instruction_source="subtask", sample_stride=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    profile = payload["episodes"][0]["profile_data"]
    profile.pop("instruction_source")
    profile.pop("sample_stride")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    dataset = Gr3AnygraspDataset(manifest, horizon=3)

    assert dataset.instruction_source == "subtask"
    assert dataset.sample_stride == 1
    assert len(dataset) == 5
    assert {sample.instruction for sample in dataset.samples} == {
        "atomic instruction"
    }


def test_preflight_normalization_is_bound_to_dataset_id(tmp_path):
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": "dataset-a",
                "normalization": {
                    "state_mean": [0.0] * 33,
                    "state_std": [1.0] * 33,
                    "action_low": [-1.0] * 37,
                    "action_high": [1.0] * 37,
                },
            }
        ),
        encoding="utf-8",
    )

    stats, source = _load_normalization(path, expected_dataset_id="dataset-a")

    assert stats.state_mean.shape == (33,)
    assert source["path"] == str(path.resolve())
    assert len(source["sha256"]) == 64
    with pytest.raises(ValueError, match="dataset_id"):
        _load_normalization(path, expected_dataset_id="dataset-b")
