"""Read-only GR3 AnyGrasp loader for the curated LeRobot v3 dataset."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gr3_common import (
    GR3_ACTION_DIM,
    GR3_STATE_DIM,
    Gr3NormalizationStats,
    prepare_gr3_rgb,
)

GR3_ANYGRASP_PROFILE_ID = "xpolicylab.gr3_anygrasp_lerobot_v3"
GR3_ANYGRASP_SOURCE_SCHEMA = "lerobot_v3_gr3qnexo_top"


def load_gr3_anygrasp_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("AnyGrasp manifest cannot be a symlink")
    source = source.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "xpolicy_dataset.v1":
        raise ValueError("unsupported XPolicyLab dataset manifest version")
    if payload.get("profile_id") != GR3_ANYGRASP_PROFILE_ID:
        raise ValueError(f"expected profile_id={GR3_ANYGRASP_PROFILE_ID}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ValueError("AnyGrasp profile requires one curated-selection binding")
    binding = episodes[0]
    if binding.get("source_schema") != GR3_ANYGRASP_SOURCE_SCHEMA:
        raise ValueError(f"expected source_schema={GR3_ANYGRASP_SOURCE_SCHEMA}")
    if binding.get("train_eligible_after_filters") is not True:
        raise ValueError("AnyGrasp selection is not training eligible")
    profile = binding.get("profile_data", {})
    expected = {
        "robot_type": "gr3qnexo",
        "state_dim": GR3_STATE_DIM,
        "action_dim": GR3_ACTION_DIM,
        "fps": 30,
        "camera_key": "observation.images.top",
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError(f"AnyGrasp profile mismatch for {key}: {profile.get(key)!r}")
    return payload


@dataclass(frozen=True)
class _Sample:
    clip_id: str
    task_id: str
    batch_path: str
    video_path: Path
    row_index: int
    episode_index: int
    frame_index: int
    frame_offset: int
    clip_frames: int
    instruction: str


@dataclass(frozen=True)
class _BatchArrays:
    states: np.ndarray
    actions: np.ndarray
    episode_indices: np.ndarray
    frame_indices: np.ndarray


def _fixed_list_numpy(column: Any, width: int) -> np.ndarray:
    values = column.combine_chunks()
    flat = np.asarray(values.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    if flat.size != len(values) * width:
        raise ValueError(f"invalid fixed-list column width: expected {width}")
    return flat.reshape(len(values), width)


def _decode_av1_frame(path: Path, frame_index: int) -> np.ndarray:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("GR3 AnyGrasp video decoding requires the 'av' package") from exc

    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        if rate <= 0:
            raise ValueError(f"video has no positive frame rate: {path}")
        start_time = int(stream.start_time or 0)
        target_pts = start_time + int((frame_index / rate) / float(stream.time_base))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            decoded_index = int(round(float((frame.pts - start_time) * stream.time_base) * rate))
            if decoded_index == frame_index:
                return frame.to_ndarray(format="rgb24")
            if decoded_index > frame_index:
                break
    raise RuntimeError(f"could not decode frame {frame_index} from {path}")


class Gr3AnygraspDataset:
    """Samples the curated AnyGrasp clip index without copying source media."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        horizon: int = 50,
        image_size: int = 224,
        stats: Gr3NormalizationStats | None = None,
        batch_cache_size: int = 2,
        decode_threads: int = 1,
    ) -> None:
        if (
            horizon < 1
            or image_size < 16
            or batch_cache_size < 1
            or decode_threads < 1
        ):
            raise ValueError("horizon/image_size/cache size must be positive")
        self.manifest = load_gr3_anygrasp_manifest(manifest_path)
        self.horizon = int(horizon)
        self.image_size = int(image_size)
        self.batch_cache_size = int(batch_cache_size)
        self.decode_threads = int(decode_threads)
        self._batch_cache: OrderedDict[str, _BatchArrays] = OrderedDict()
        self._batch_cache_lock = threading.Lock()
        self._decode_pool = (
            ThreadPoolExecutor(
                max_workers=self.decode_threads,
                thread_name_prefix="gr3-anygrasp-decode",
            )
            if self.decode_threads > 1
            else None
        )

        binding = self.manifest["episodes"][0]
        profile = binding["profile_data"]
        self.instruction_source = str(profile.get("instruction_source", "subtask"))
        if self.instruction_source not in {"subtask", "prompt"}:
            raise ValueError(
                "AnyGrasp instruction_source must be 'subtask' or 'prompt'"
            )
        self.sample_stride = int(profile.get("sample_stride", 1))
        if self.sample_stride < 1:
            raise ValueError("AnyGrasp sample_stride must be positive")
        self.dataset_root = Path(profile["dataset_root"]).expanduser().resolve()
        selection_root = Path(binding["path"]).expanduser().resolve()
        clips_path = selection_root / str(profile["clips_file"])
        if not self.dataset_root.is_dir() or not clips_path.is_file():
            raise FileNotFoundError("AnyGrasp dataset root or clips manifest is unavailable")

        import pyarrow.parquet as pq

        columns = [
            "clip_id",
            "task_id",
            "batch_path",
            "episode_index",
            "subtask",
            "prompt",
            "clip_frame_start",
            "video_frame_start",
            "num_clip_frames",
            "source_video",
        ]
        clips = pq.read_table(clips_path, columns=columns)
        selected_task_ids = {str(task_id) for task_id in profile.get("selected_task_ids", [])}
        selected_clips = clips.to_pylist()
        if selected_task_ids:
            selected_clips = [
                clip for clip in selected_clips if str(clip["task_id"]) in selected_task_ids
            ]
        clip_limit = int(profile.get("clip_limit") or len(selected_clips))
        if clip_limit < 1 or clip_limit > len(selected_clips):
            raise ValueError("AnyGrasp clip_limit is outside the clips manifest")
        self.samples: list[_Sample] = []
        for clip in selected_clips[:clip_limit]:
            clip_frames = int(clip["num_clip_frames"])
            batch_path = str(clip["batch_path"])
            video_path = self.dataset_root / str(clip["source_video"])
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            instruction = str(clip.get(self.instruction_source) or "").strip()
            if not instruction:
                raise ValueError(
                    f"clip {clip['clip_id']} has no {self.instruction_source} text"
                )
            for offset in range(0, clip_frames, self.sample_stride):
                self.samples.append(
                    _Sample(
                        clip_id=str(clip["clip_id"]),
                        task_id=str(clip["task_id"]),
                        batch_path=batch_path,
                        video_path=video_path,
                        row_index=int(clip["video_frame_start"]) + offset,
                        episode_index=int(clip["episode_index"]),
                        frame_index=int(clip["clip_frame_start"]) + offset,
                        frame_offset=offset,
                        clip_frames=clip_frames,
                        instruction=instruction,
                    )
                )
        if not self.samples:
            raise ValueError("AnyGrasp selection contains no training samples")
        self.stats = stats or self.compute_stats()

    def _load_batch(self, batch_path: str) -> _BatchArrays:
        with self._batch_cache_lock:
            cached = self._batch_cache.pop(batch_path, None)
            if cached is not None:
                self._batch_cache[batch_path] = cached
                return cached

        import pyarrow.parquet as pq

        batch_root = self.dataset_root / batch_path
        info = json.loads((batch_root / "meta/info.json").read_text(encoding="utf-8"))
        if info.get("robot_type") != "gr3qnexo" or int(info.get("fps", 0)) != 30:
            raise ValueError(f"unsupported AnyGrasp batch metadata: {batch_root}")
        features = info.get("features", {})
        if features.get("observation.state", {}).get("shape") != [GR3_STATE_DIM]:
            raise ValueError(f"AnyGrasp batch has a non-33D state: {batch_root}")
        if features.get("action", {}).get("shape") != [GR3_ACTION_DIM]:
            raise ValueError(f"AnyGrasp batch has a non-37D action: {batch_root}")
        table = pq.read_table(
            batch_root / "data/chunk-000/file-000.parquet",
            columns=["observation.state", "action", "episode_index", "frame_index"],
        )
        arrays = _BatchArrays(
            states=_fixed_list_numpy(table["observation.state"], GR3_STATE_DIM),
            actions=_fixed_list_numpy(table["action"], GR3_ACTION_DIM),
            episode_indices=np.asarray(
                table["episode_index"].to_numpy(), dtype=np.int64
            ),
            frame_indices=np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
        )
        with self._batch_cache_lock:
            self._batch_cache[batch_path] = arrays
            while len(self._batch_cache) > self.batch_cache_size:
                self._batch_cache.popitem(last=False)
        return arrays

    def preload_batches(self) -> int:
        """Cache all referenced state/action batches once for random training."""
        batch_paths = sorted({sample.batch_path for sample in self.samples})
        if self.batch_cache_size < len(batch_paths):
            raise ValueError(
                "batch_cache_size must cover every selected batch when preloading"
            )
        for batch_path in batch_paths:
            self._load_batch(batch_path)
        return len(batch_paths)

    def _validate_row(self, sample: _Sample, arrays: _BatchArrays) -> None:
        if sample.row_index >= len(arrays.states):
            raise IndexError(f"AnyGrasp row is outside batch parquet: {sample.row_index}")
        if (
            int(arrays.episode_indices[sample.row_index]) != sample.episode_index
            or int(arrays.frame_indices[sample.row_index]) != sample.frame_index
        ):
            raise ValueError(f"AnyGrasp clip/parquet index mismatch for {sample.clip_id}")

    def compute_stats(self) -> Gr3NormalizationStats:
        grouped: dict[str, list[_Sample]] = defaultdict(list)
        for sample in self.samples:
            grouped[sample.batch_path].append(sample)
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for batch_path, samples in grouped.items():
            arrays = self._load_batch(batch_path)
            indices = np.asarray(sorted({sample.row_index for sample in samples}), dtype=np.int64)
            states.append(arrays.states[indices])
            actions.append(arrays.actions[indices])
        state_values = np.concatenate(states, axis=0)
        action_values = np.concatenate(actions, axis=0)
        state_std = np.maximum(state_values.std(axis=0), 1e-4)
        action_low = np.quantile(action_values, 0.01, axis=0).astype(np.float32)
        action_high = np.quantile(action_values, 0.99, axis=0).astype(np.float32)
        stagnant = action_high - action_low < 1e-5
        action_low[stagnant] -= 1e-3
        action_high[stagnant] += 1e-3
        return Gr3NormalizationStats(
            state_mean=state_values.mean(axis=0),
            state_std=state_std,
            action_low=action_low,
            action_high=action_high,
        )

    def normalization_diagnostics(self) -> dict[str, list[str]]:
        return {
            "near_constant_state_axes": [
                str(index) for index, std in enumerate(self.stats.state_std) if std <= 1.01e-4
            ],
            "near_constant_action_axes": [
                str(index)
                for index, (low, high) in enumerate(zip(self.stats.action_low, self.stats.action_high, strict=True))
                if high - low <= 2.01e-3
            ],
        }

    def __len__(self) -> int:
        return len(self.samples)

    def raw_observation(self, index: int) -> dict[str, Any]:
        """Return one unnormalized RGB/state observation for replay runtimes."""
        sample = self.samples[index]
        arrays = self._load_batch(sample.batch_path)
        self._validate_row(sample, arrays)
        return {
            "image": _decode_av1_frame(sample.video_path, sample.row_index),
            "lang": sample.instruction,
            "state": arrays.states[sample.row_index].copy(),
            "metadata": {
                "clip_id": sample.clip_id,
                "task_id": sample.task_id,
                "batch_path": sample.batch_path,
                "episode_index": sample.episode_index,
                "frame_index": sample.frame_index,
                "video_frame_index": sample.row_index,
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        arrays = self._load_batch(sample.batch_path)
        self._validate_row(sample, arrays)
        valid_steps = min(self.horizon, sample.clip_frames - sample.frame_offset)
        action_end = sample.row_index + valid_steps
        actions = arrays.actions[sample.row_index:action_end]
        if len(actions) != valid_steps:
            raise ValueError(f"AnyGrasp action chunk crosses batch boundary for {sample.clip_id}")
        padded = np.repeat(actions[-1:], self.horizon, axis=0)
        padded[:valid_steps] = actions
        mask = np.zeros(self.horizon, dtype=np.float32)
        mask[:valid_steps] = 1.0
        observation = self.raw_observation(index)
        return {
            "image": prepare_gr3_rgb(observation["image"], self.image_size),
            "lang": observation["lang"],
            "state": self.stats.normalize_state(observation["state"]),
            "action": self.stats.normalize_action(padded),
            "action_mask": mask,
            "metadata": observation["metadata"],
        }

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        """Decode a batch concurrently without multiprocessing/CPFS sockets."""
        if self._decode_pool is None:
            return [self[index] for index in indices]
        return list(self._decode_pool.map(self.__getitem__, indices))
