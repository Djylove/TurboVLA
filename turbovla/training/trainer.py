import argparse
import glob
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.libero_rlds import (
    LiberoRLDSDataset,
    vla_collate_fn,
)
from ..models.turbovla import (
    build_turbovla,
)


class DummyArgs:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TurboVLA with DINOv3 ViT-Base visual tokens and cached BERT text features."
    )

    parser.add_argument("--dataset_dir", type=str, default="./data/libero/libero_10_no_noops/1.0.0")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dinov3_path", type=str, required=True)
    parser.add_argument("--text_cache_path", type=str, required=True)
    parser.add_argument(
        "--pretrained_gdino_ckpt",
        type=str,
        required=True,
    )
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--checkpoint_prefix", type=str, default="turbovla_step")
    parser.add_argument("--resume_mode", type=str, default="none", choices=["none", "model", "all"])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--dinov3_lr", type=float, default=5e-5)
    parser.add_argument("--head_weight_decay", type=float, default=1e-10)
    parser.add_argument("--dinov3_weight_decay", type=float, default=1e-10)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16_amp"])
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument(
        "--lr_schedule_steps",
        type=int,
        default=None,
        help="Cosine LR horizon. Defaults to max_steps; set separately for multi-fidelity searches.",
    )
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--min_lr_ratio", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.set_defaults(save_final=True)
    parser.add_argument("--save_final", dest="save_final", action="store_true")
    parser.add_argument("--no_save_final", dest="save_final", action="store_false")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=512)
    parser.add_argument("--step_mix_buffer_size", type=int, default=64)
    parser.add_argument("--expected_image_size", type=int, default=256)

    parser.add_argument("--text_hidden_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument("--vla_feature_enhancer_layers", type=int, default=6)
    parser.add_argument("--enhancer_inner_dim", type=int, default=1024)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--chunk_size", type=int, default=12)
    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--num_state_tokens", type=int, default=2)
    parser.add_argument("--text_dropout", type=float, default=0.0)
    parser.add_argument("--fusion_dropout", type=float, default=0.0)
    parser.add_argument("--fusion_droppath", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.set_defaults(allow_hf_download=False)
    parser.add_argument("--allow_hf_download", dest="allow_hf_download", action="store_true")
    parser.add_argument("--no_allow_hf_download", dest="allow_hf_download", action="store_false")

    parser.set_defaults(shuffle_steps_within_episode=True)
    parser.add_argument("--shuffle_steps_within_episode", dest="shuffle_steps_within_episode", action="store_true")
    parser.add_argument("--no_shuffle_steps_within_episode", dest="shuffle_steps_within_episode", action="store_false")

    parser.set_defaults(freeze_backbones=False)
    parser.add_argument("--freeze_backbones", dest="freeze_backbones", action="store_true")
    parser.add_argument("--no_freeze_backbones", dest="freeze_backbones", action="store_false")

    parser.set_defaults(require_feature_enhancer_preload=True)
    parser.add_argument("--require_feature_enhancer_preload", dest="require_feature_enhancer_preload", action="store_true")
    parser.add_argument("--no_require_feature_enhancer_preload", dest="require_feature_enhancer_preload", action="store_false")

    parser.set_defaults(load_text_proj_from_gdino=True)
    parser.add_argument("--load_text_proj_from_gdino", dest="load_text_proj_from_gdino", action="store_true")
    parser.add_argument("--no_load_text_proj_from_gdino", dest="load_text_proj_from_gdino", action="store_false")

    parser.set_defaults(require_text_proj_preload=True)
    parser.add_argument("--require_text_proj_preload", dest="require_text_proj_preload", action="store_true")
    parser.add_argument("--no_require_text_proj_preload", dest="require_text_proj_preload", action="store_false")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        use_cuda = torch.cuda.is_available()
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False
    return is_distributed, rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class CachedTextBank:
    def __init__(self, payload, path):
        if not isinstance(payload, dict):
            raise ValueError(f"text cache at {path} must be a dict")
        for key in ["instructions", "last_hidden_state", "attention_mask"]:
            if key not in payload:
                raise ValueError(f"text cache at {path} missing required key: {key}")

        self.path = path
        self.instructions = [str(item) for item in payload["instructions"]]
        raw_mapping = payload.get("instruction_to_index")
        if raw_mapping is None:
            raw_mapping = {instruction: idx for idx, instruction in enumerate(self.instructions)}
        self.instruction_to_index = {str(key): int(value) for key, value in raw_mapping.items()}
        self.stripped_to_index = {}
        for instruction, idx in self.instruction_to_index.items():
            self.stripped_to_index.setdefault(instruction.strip(), idx)

        self.last_hidden_state = payload["last_hidden_state"].detach().cpu().contiguous()
        self.attention_mask = payload["attention_mask"].detach().cpu().bool().contiguous()
        self.text_self_attention_masks = payload.get("text_self_attention_masks")
        if self.text_self_attention_masks is not None:
            self.text_self_attention_masks = self.text_self_attention_masks.detach().cpu().bool().contiguous()

        if self.last_hidden_state.ndim != 3:
            raise ValueError(f"last_hidden_state should be [N, L, H], got {self.last_hidden_state.shape}")
        if self.attention_mask.ndim != 2:
            raise ValueError(f"attention_mask should be [N, L], got {self.attention_mask.shape}")
        if tuple(self.last_hidden_state.shape[:2]) != tuple(self.attention_mask.shape):
            raise ValueError(
                f"text cache shape mismatch: last_hidden_state={self.last_hidden_state.shape}, "
                f"attention_mask={self.attention_mask.shape}"
            )
        if len(self.instructions) != self.last_hidden_state.shape[0]:
            raise ValueError(
                f"instruction count {len(self.instructions)} does not match cache rows "
                f"{self.last_hidden_state.shape[0]}"
            )

    @classmethod
    def load(cls, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"text cache not found: {path}. Build it first with build_libero_instruction_text_cache.py"
            )
        payload = torch.load(path, map_location="cpu")
        return cls(payload, path)

    @property
    def text_hidden_dim(self):
        return int(self.last_hidden_state.shape[-1])

    def lookup(self, instructions):
        indices = []
        for instruction in instructions:
            instruction = str(instruction)
            idx = self.instruction_to_index.get(instruction)
            if idx is None:
                idx = self.stripped_to_index.get(instruction.strip())
            if idx is None:
                known = "\n".join(f"- {item}" for item in self.instructions[:20])
                raise KeyError(
                    f"instruction not found in text cache: {instruction!r}\nKnown cached instructions:\n{known}"
                )
            indices.append(idx)

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        result = {
            "last_hidden_state": self.last_hidden_state.index_select(0, idx_tensor),
            "attention_mask": self.attention_mask.index_select(0, idx_tensor),
        }
        if self.text_self_attention_masks is not None:
            result["text_self_attention_masks"] = self.text_self_attention_masks.index_select(0, idx_tensor)
        return result


def build_model_architecture(args):
    model_args = DummyArgs()
    model_args.LOCAL_DINOV3_PATH = args.dinov3_path
    model_args.text_hidden_dim = args.text_hidden_dim
    model_args.hidden_dim = args.hidden_dim
    model_args.nheads = args.nheads
    model_args.dim_feedforward = args.dim_feedforward
    model_args.max_text_len = args.max_text_len
    model_args.vla_feature_enhancer_layers = args.vla_feature_enhancer_layers
    model_args.enhancer_inner_dim = args.enhancer_inner_dim
    model_args.text_dropout = args.text_dropout
    model_args.fusion_dropout = args.fusion_dropout
    model_args.fusion_droppath = args.fusion_droppath
    model_args.action_dim = args.action_dim
    model_args.chunk_size = args.chunk_size
    model_args.state_dim = args.state_dim
    model_args.num_state_tokens = args.num_state_tokens
    model_args.local_files_only = not args.allow_hf_download
    model_args.freeze_vision_encoder = args.freeze_backbones
    return build_turbovla(model_args)


def freeze_backbones(model):
    for name, param in model.named_parameters():
        if name.startswith("dinov3"):
            param.requires_grad = False


def get_latest_checkpoint(ckpt_dir, prefix):
    ckpts = glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.pth"))
    if not ckpts:
        return None

    def extract_step(path):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return -1

    return max(ckpts, key=extract_step)


def masked_l1_loss(pred, target, mask):
    l1 = torch.abs(pred - target)
    mask = mask.unsqueeze(-1).float()
    l1 = l1 * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return l1.sum() / denom


def build_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            warmup_scale = float(step + 1) / float(max(1, warmup_steps))
            return 0.1 + 0.9 * warmup_scale

        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _use_weight_decay(name, param):
    if param.ndim <= 1:
        return False
    lowered = name.lower()
    if lowered.endswith(".bias"):
        return False
    if "norm" in lowered or "layernorm" in lowered:
        return False
    return True


def build_param_group_optimizer(model, args):
    head_lr = args.lr if args.head_lr is None else args.head_lr
    head_wd = args.weight_decay if args.head_weight_decay is None else args.head_weight_decay
    grouped = {
        ("dinov3_decay", args.dinov3_lr, args.dinov3_weight_decay): [],
        ("dinov3_no_decay", args.dinov3_lr, 0.0): [],
        ("head_decay", head_lr, head_wd): [],
        ("head_no_decay", head_lr, 0.0): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_dino = name.startswith("dinov3")
        decay = _use_weight_decay(name, param)
        if is_dino and decay:
            key = ("dinov3_decay", args.dinov3_lr, args.dinov3_weight_decay)
        elif is_dino:
            key = ("dinov3_no_decay", args.dinov3_lr, 0.0)
        elif decay:
            key = ("head_decay", head_lr, head_wd)
        else:
            key = ("head_no_decay", head_lr, 0.0)
        grouped[key].append(param)

    param_groups = []
    summary = []
    for (group_name, lr, weight_decay), params in grouped.items():
        if not params:
            continue
        count = sum(p.numel() for p in params)
        param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay, "name": group_name})
        summary.append({"name": group_name, "lr": lr, "weight_decay": weight_decay, "params": count})
    return AdamW(param_groups), summary


def reduce_mean(value, device, is_distributed, world_size):
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor.item()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["model_state_dict", "model", "state_dict"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("unsupported checkpoint format")


def load_feature_enhancer_from_groundingdino(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    target_state = model.state_dict()
    mapped = {}
    skipped_missing = 0
    skipped_shape = 0
    missing_keys = []
    shape_mismatch_keys = []

    prefix_pairs = [
        ("transformer.encoder.fusion_layers.", "feature_enhancer.fusion_layers."),
        ("transformer.encoder.text_layers.", "feature_enhancer.text_layers."),
    ]

    for src_key, tensor in source_state.items():
        tgt_key = None
        for src_prefix, tgt_prefix in prefix_pairs:
            if src_key.startswith(src_prefix):
                tgt_key = tgt_prefix + src_key[len(src_prefix):]
                break
        if tgt_key is None:
            continue
        if tgt_key not in target_state:
            skipped_missing += 1
            missing_keys.append((src_key, tgt_key))
            continue
        if target_state[tgt_key].shape != tensor.shape:
            skipped_shape += 1
            shape_mismatch_keys.append((src_key, tgt_key, tuple(tensor.shape), tuple(target_state[tgt_key].shape)))
            continue
        mapped[tgt_key] = tensor

    if not mapped:
        raise RuntimeError("no feature-enhancer parameters were mapped from checkpoint")

    target_state.update(mapped)
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    return {
        "mapped": len(mapped),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "load_missing_after_update": len(missing),
        "load_unexpected_after_update": len(unexpected),
        "missing_keys": missing_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def load_text_proj_from_groundingdino(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    target_state = model.state_dict()
    mapped = {}
    skipped_missing = 0
    skipped_shape = 0
    missing_keys = []
    shape_mismatch_keys = []

    for src_key, tensor in source_state.items():
        tgt_key = None
        if src_key.startswith("feat_map."):
            tgt_key = "text_proj." + src_key[len("feat_map."):]
        else:
            continue

        if tgt_key not in target_state:
            skipped_missing += 1
            missing_keys.append((src_key, tgt_key))
            continue
        if target_state[tgt_key].shape != tensor.shape:
            skipped_shape += 1
            shape_mismatch_keys.append((src_key, tgt_key, tuple(tensor.shape), tuple(target_state[tgt_key].shape)))
            continue
        mapped[tgt_key] = tensor

    if not mapped:
        raise RuntimeError("no text projection parameters were mapped from checkpoint")

    target_state.update(mapped)
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    return {
        "mapped": len(mapped),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "load_missing_after_update": len(missing),
        "load_unexpected_after_update": len(unexpected),
        "missing_keys": missing_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def move_samples_to_device(samples, device):
    if isinstance(samples, dict):
        return {k: v.to(device, non_blocking=True) for k, v in samples.items()}
    return samples.to(device, non_blocking=True)


def train_model():
    args = parse_args()
    if args.head_lr is None:
        args.head_lr = args.lr
    if args.head_weight_decay is None:
        args.head_weight_decay = args.weight_decay
    if args.precision == "bf16_amp":
        if not torch.cuda.is_available():
            raise ValueError("bf16_amp precision requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16_amp requested but current CUDA device does not report bf16 support")
    torch.set_float32_matmul_precision("high")

    if args.lr_schedule_steps is None:
        args.lr_schedule_steps = args.max_steps
    if args.lr_schedule_steps <= args.warmup_steps:
        raise ValueError(
            f"lr_schedule_steps={args.lr_schedule_steps} must be greater than warmup_steps={args.warmup_steps}"
        )
    set_seed(args.seed)

    is_distributed, rank, world_size, local_rank, device = setup_distributed()

    try:
        text_cache = CachedTextBank.load(args.text_cache_path)
        args.text_hidden_dim = text_cache.text_hidden_dim
        model = build_model_architecture(args)

        need_any_gdino_preload = (
            args.require_feature_enhancer_preload
            or args.require_text_proj_preload
            or args.load_text_proj_from_gdino
        )
        if args.pretrained_gdino_ckpt is None and need_any_gdino_preload:
            raise ValueError(
                "`--pretrained_gdino_ckpt` is required for requested preload options "
                "(feature-enhancer and/or text_proj)"
            )

        if args.pretrained_gdino_ckpt is not None:
            if args.require_feature_enhancer_preload:
                fe_report = load_feature_enhancer_from_groundingdino(model, args.pretrained_gdino_ckpt)
                if rank == 0:
                    print("feature-enhancer preload from GroundingDINO:")
                    print(f"  ckpt: {args.pretrained_gdino_ckpt}")
                    print(f"  mapped={fe_report['mapped']}")
                    print(f"  skipped_missing={fe_report['skipped_missing']}")
                    print(f"  skipped_shape={fe_report['skipped_shape']}")
                    print("missing_keys:", fe_report["missing_keys"])
                    print("shape_mismatch_keys:", fe_report["shape_mismatch_keys"])

            if args.load_text_proj_from_gdino:
                text_proj_report = load_text_proj_from_groundingdino(model, args.pretrained_gdino_ckpt)
                if rank == 0:
                    print("text projection preload from GroundingDINO:")
                    print(f"  ckpt: {args.pretrained_gdino_ckpt}")
                    print(f"  mapped={text_proj_report['mapped']}")
                    print(f"  skipped_missing={text_proj_report['skipped_missing']}")
                    print(f"  skipped_shape={text_proj_report['skipped_shape']}")
                    print("missing_keys:", text_proj_report["missing_keys"])
                    print("shape_mismatch_keys:", text_proj_report["shape_mismatch_keys"])
            elif args.require_text_proj_preload:
                raise ValueError(
                    "text projection preload is required, but `--no_load_text_proj_from_gdino` was set"
                )

        if args.freeze_backbones:
            freeze_backbones(model)

        if rank == 0:
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
            print(f"device={device}, distributed={is_distributed}, world_size={world_size}")
            print(f"dinov3_path={args.dinov3_path}")
            print(f"text_cache_path={args.text_cache_path}")
            print(f"text_cache_rows={len(text_cache.instructions)}, text_hidden_dim={args.text_hidden_dim}")
            print(f"pretrained_gdino_ckpt={args.pretrained_gdino_ckpt}")
            print(f"load_text_proj_from_gdino={args.load_text_proj_from_gdino}")
            print(f"max_steps={args.max_steps}, lr_schedule_steps={args.lr_schedule_steps}")
            print(
                f"precision={args.precision}, head_lr={args.head_lr}, dinov3_lr={args.dinov3_lr}, "
                f"head_wd={args.head_weight_decay}, dinov3_wd={args.dinov3_weight_decay}"
            )
            print(f"warmup_steps={args.warmup_steps}, min_lr_ratio={args.min_lr_ratio}")
            print(
                f"shuffle: episode_buffer={args.shuffle_buffer}, "
                f"within_episode={args.shuffle_steps_within_episode}"
            )
            print(f"trainable params: {len(trainable)}")
            print(f"frozen params: {len(frozen)}")
            print("first 30 trainable:", trainable[:30])

        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if is_distributed:
            dist.barrier()

        global_step = 0
        ckpt = None
        latest_ckpt_path = get_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix)
        if latest_ckpt_path is not None and args.resume_mode != "none":
            if rank == 0:
                print(f"resume from {latest_ckpt_path} with mode={args.resume_mode}")
            ckpt = torch.load(latest_ckpt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if rank == 0:
                print("resume missing keys:", len(missing))
                print("resume unexpected keys:", len(unexpected))
            if args.resume_mode == "all":
                global_step = int(ckpt.get("global_step", 0))
            else:
                global_step = 0
                if rank == 0:
                    print("optimizer/scheduler reset due to resume_mode=model")
        elif rank == 0:
            print("no checkpoint resumed, train from current initialization")

        model.to(device)
        if is_distributed:
            if device.type == "cuda":
                model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            else:
                model = DDP(model, find_unused_parameters=True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer, optimizer_summary = build_param_group_optimizer(unwrap_model(model), args)
        if rank == 0:
            print("optimizer param groups:", optimizer_summary)
        scheduler = build_scheduler(
            optimizer,
            max_steps=args.lr_schedule_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

        if ckpt is not None and args.resume_mode == "all":
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        model_to_load = unwrap_model(model)

        dataset = LiberoRLDSDataset(
            dataset_dir=args.dataset_dir,
            LOCAL_DINOV3_PATH=args.dinov3_path,
            rank=rank,
            world_size=world_size,
            chunk_size=args.chunk_size,
            split=args.dataset_split,
            shuffle_buffer=args.shuffle_buffer,
            shuffle_steps_within_episode=args.shuffle_steps_within_episode,
            step_mix_buffer_size=args.step_mix_buffer_size,
            seed=args.seed,
            local_files_only=not args.allow_hf_download,
            expected_image_size=args.expected_image_size,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=None,
            collate_fn=vla_collate_fn,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        data_iter = iter(dataloader)

        model.train()
        if args.freeze_backbones:
            model_to_load.dinov3.eval()

        if rank == 0:
            per_gpu_effective_bs = args.batch_size * args.grad_accum_steps
            global_effective_bs = per_gpu_effective_bs * world_size
            print(f"batch_size(per_gpu)={args.batch_size}, grad_accum_steps={args.grad_accum_steps}")
            print(f"effective_batch_size(per_gpu)={per_gpu_effective_bs}")
            print(f"effective_batch_size(global)={global_effective_bs}")
            pbar = tqdm(total=args.max_steps, initial=global_step, desc="training")
        else:
            pbar = None

        loss_window = []
        while global_step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for _ in range(args.grad_accum_steps):
                try:
                    samples, instructions, states, gt_actions, action_chunk_masks = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    samples, instructions, states, gt_actions, action_chunk_masks = next(data_iter)

                samples = move_samples_to_device(samples, device)
                states = states.to(device, non_blocking=True)
                gt_actions = gt_actions.to(device, non_blocking=True)
                action_chunk_masks = action_chunk_masks.to(device, non_blocking=True)

                cached_text = text_cache.lookup(instructions)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.precision == "bf16_amp",
                ):
                    pred_actions = model(cached_text, samples, states)
                    if pred_actions.shape != gt_actions.shape:
                        raise ValueError(
                            f"pred_actions.shape={pred_actions.shape}, gt_actions.shape={gt_actions.shape} mismatch"
                        )

                    loss = masked_l1_loss(pred_actions, gt_actions, action_chunk_masks)
                (loss / args.grad_accum_steps).backward()
                loss_accum += loss.detach().item()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            local_loss = loss_accum / args.grad_accum_steps
            global_loss = reduce_mean(local_loss, device, is_distributed, world_size)

            loss_window.append(global_loss)
            if len(loss_window) > args.log_freq:
                loss_window.pop(0)

            global_step += 1

            if rank == 0:
                avg_window_loss = sum(loss_window) / len(loss_window)
                pbar.update(1)
                if global_step % args.log_freq == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{global_loss:.5f}",
                            "avg": f"{avg_window_loss:.5f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )

            should_save = global_step % args.save_steps == 0 or (
                args.save_final and global_step == args.max_steps
            )
            if rank == 0 and should_save:
                save_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_{global_step}.pth")
                torch.save(
                    {
                        "global_step": global_step,
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": global_loss,
                        "args": vars(args),
                    },
                    save_path,
                )
                print(f"saved: {save_path}")

        if pbar is not None:
            pbar.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    train_model()
