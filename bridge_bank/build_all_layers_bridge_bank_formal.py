"""Build a stronger all-layer OPRD-Bridge ps_bank.pt from TCOD buffers.

This is an OPRD-style bridge-construction script for ALFWorld turn-level
experiences.  It uses existing TCOD SQLite buffers as the prompt/response
source, recomputes student and teacher hidden states from token ids, then fits:

    z_t = P_T (h_t - mean_t)  with frozen teacher PCA bases
    z_s = P_S h_s             with trainable student projectors

Unlike the earlier lightweight builder, this script supports train/val split,
multiple ranks, PCA row subsampling, and validation metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from experimental.oprd_hidden.rep_distillation_utils import (
    all_layer_response_token_hidden,
    fit_teacher_pca_from_rows,
    proportional_layer_pairs,
)


@dataclass(frozen=True)
class TokenExperience:
    token_ids: list[int]
    prompt_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal OPRD-Bridge construction from TCOD token buffers."
    )
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8960)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--max-total-response-rows", type=int, default=65536)
    parser.add_argument("--max-pca-rows", type=int, default=16384)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-layer-mode", default="all", choices=["all", "last"])
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def load_token_experiences(args: argparse.Namespace) -> list[TokenExperience]:
    con = sqlite3.connect(args.db_path)
    cur = con.cursor()

    rng = random.Random(args.seed)
    row_ids = [row_id for (row_id,) in cur.execute("select rowid from pipeline_input")]
    rng.shuffle(row_ids)

    experiences: list[TokenExperience] = []
    total_response_rows = 0
    scanned_rows = 0
    skipped = {
        "unpickle": 0,
        "missing_tokens": 0,
        "bad_lengths": 0,
        "too_long": 0,
    }
    for row_id in row_ids:
        if len(experiences) >= args.max_pairs:
            break
        if total_response_rows >= args.max_total_response_rows:
            break
        row = cur.execute(
            "select experience_bytes from pipeline_input where rowid = ?",
            (row_id,),
        ).fetchone()
        scanned_rows += 1
        if row is None:
            continue
        (blob,) = row
        try:
            exp = pickle.loads(blob)
        except Exception:
            skipped["unpickle"] += 1
            continue

        tokens = getattr(exp, "tokens", None)
        prompt_length = int(getattr(exp, "prompt_length", 0) or 0)
        if tokens is None:
            skipped["missing_tokens"] += 1
            continue
        token_ids = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        response_len = len(token_ids) - prompt_length
        if prompt_length <= 0 or response_len <= 0:
            skipped["bad_lengths"] += 1
            continue
        if len(token_ids) > args.max_model_len:
            skipped["too_long"] += 1
            continue
        if response_len > args.max_response_tokens:
            token_ids = token_ids[: prompt_length + args.max_response_tokens]
            response_len = args.max_response_tokens
        experiences.append(TokenExperience(token_ids=token_ids, prompt_length=prompt_length))
        total_response_rows += response_len

    con.close()
    if not experiences:
        raise RuntimeError(f"No usable token experiences found in {args.db_path}")
    print(
        json.dumps(
            {
                "candidate_rows": len(row_ids),
                "scanned_rows": scanned_rows,
                "loaded_pairs": len(experiences),
                "loaded_response_rows": total_response_rows,
                "skipped": skipped,
            },
            indent=2,
        ),
        flush=True,
    )
    return experiences


def split_experiences(
    experiences: list[TokenExperience],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[TokenExperience], list[TokenExperience]]:
    rng = random.Random(seed)
    shuffled = list(experiences)
    rng.shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_fraction)))
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    if not train:
        raise RuntimeError("Validation split consumed all experiences.")
    return train, val


@torch.inference_mode()
def collect_model_rows(
    experiences: list[TokenExperience],
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    rows = []
    for exp in tqdm(experiences, desc=desc):
        input_ids = torch.tensor([exp.token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        response_len = len(exp.token_ids) - int(exp.prompt_length)
        response_mask = torch.ones((1, response_len), dtype=torch.bool, device=device)
        hidden = all_layer_response_token_hidden(outputs.hidden_states, response_mask)
        hidden = hidden[0, :, :response_len, :].detach().float().cpu()
        rows.append(hidden.transpose(0, 1).contiguous())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(rows, dim=0)


def select_rows(rows: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    if rows.shape[0] <= max_rows:
        return rows
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(rows.shape[0], generator=generator)[:max_rows]
    return rows[idx].contiguous()


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return float((a * b).sum(dim=-1).mean().item())


def eval_projector(
    projector: nn.Linear,
    h_student: torch.Tensor,
    z_teacher: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    projector.eval()
    losses = []
    cosines = []
    with torch.no_grad():
        for start in range(0, h_student.shape[0], batch_size):
            end = min(start + batch_size, h_student.shape[0])
            hs = h_student[start:end].to(device)
            zt = z_teacher[start:end].to(device)
            zs = projector(hs)
            losses.append(float(F.mse_loss(zs.float(), zt.float()).cpu().item()))
            cosines.append(cosine_mean(zs.cpu(), zt.cpu()))
    return {
        "mse": float(sum(losses) / max(len(losses), 1)),
        "cosine": float(sum(cosines) / max(len(cosines), 1)),
    }


def train_one_projector(
    train_student: torch.Tensor,
    train_teacher: torch.Tensor,
    val_student: torch.Tensor,
    val_teacher: torch.Tensor,
    *,
    rank: int,
    teacher_weights_full: torch.Tensor,
    teacher_mean: torch.Tensor,
    fitted_rank_full: int,
    pca_rows: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, dict]:
    teacher_weights = teacher_weights_full[:rank].contiguous()
    z_train = (train_teacher.float() - teacher_mean) @ teacher_weights.T
    z_val = (val_teacher.float() - teacher_mean) @ teacher_weights.T

    projector = nn.Linear(train_student.shape[-1], rank, bias=False, device=device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(train_student.float(), z_train.float())
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    history = []
    best_state = None
    best_val = math.inf
    best_epoch = 0
    for epoch in range(1, int(epochs) + 1):
        projector.train()
        losses = []
        for h_s, z_t in loader:
            h_s = h_s.to(device)
            z_t = z_t.to(device)
            loss = F.mse_loss(projector(h_s).float(), z_t.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        train_loss = float(sum(losses) / max(len(losses), 1))
        val_metrics = eval_projector(
            projector,
            val_student.float(),
            z_val.float(),
            batch_size=batch_size,
            device=device,
        )
        row = {
            "epoch": epoch,
            "train_epoch_mse": train_loss,
            "val_mse": val_metrics["mse"],
            "val_cosine": val_metrics["cosine"],
        }
        history.append(row)
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in projector.state_dict().items()}
        print(json.dumps(row), flush=True)

    if best_state is not None:
        projector.load_state_dict(best_state)
    train_metrics = eval_projector(
        projector,
        train_student.float(),
        z_train.float(),
        batch_size=batch_size,
        device=device,
    )
    val_metrics = eval_projector(
        projector,
        val_student.float(),
        z_val.float(),
        batch_size=batch_size,
        device=device,
    )
    metrics = {
        "fitted_rank": int(min(rank, fitted_rank_full)),
        "pca_rows": int(pca_rows),
        "best_epoch": int(best_epoch),
        "train_mse": train_metrics["mse"],
        "train_cosine": train_metrics["cosine"],
        "val_mse": val_metrics["mse"],
        "val_cosine": val_metrics["cosine"],
        "history": history,
    }
    return projector, teacher_weights, teacher_mean, metrics


def save_bank(output_dir: Path, payload: dict, metrics: dict) -> None:
    rank_dir = output_dir / f"rank_{metrics['rank']}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    payload["metrics"] = metrics
    torch.save(payload, rank_dir / "ps_bank.pt")
    with open(rank_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {rank_dir / 'ps_bank.pt'}", flush=True)


def build_for_rank(
    *,
    rank: int,
    args: argparse.Namespace,
    train_student: torch.Tensor,
    train_teacher: torch.Tensor,
    val_student: torch.Tensor,
    val_teacher: torch.Tensor,
    layer_pairs: list[tuple[int, int]],
    teacher_pca_cache: dict[str, dict[str, torch.Tensor | int]],
    device: torch.device,
) -> None:
    payload = {
        "subspace_mode": "full",
        "projector_type": "linear",
        "rank": int(rank),
        "layer_pairs": [
            {"student_layer": int(s), "teacher_layer": int(t)}
            for s, t in layer_pairs
        ],
        "state_dict": {},
        "frozen_pt_weights": {},
        "frozen_pt_means": {},
    }

    pair_metrics = {}
    val_losses = []
    val_cosines = []
    for pair_idx, (student_layer_idx, teacher_layer_idx) in enumerate(layer_pairs):
        key = f"s{student_layer_idx}_t{teacher_layer_idx}"
        print(
            f"rank {rank}: training pair {pair_idx + 1}/{len(layer_pairs)} {key}",
            flush=True,
        )
        pca = teacher_pca_cache[key]
        projector, teacher_weights, teacher_mean, metrics = train_one_projector(
            train_student[:, student_layer_idx, :],
            train_teacher[:, teacher_layer_idx, :],
            val_student[:, student_layer_idx, :],
            val_teacher[:, teacher_layer_idx, :],
            rank=rank,
            teacher_weights_full=pca["teacher_weights"],
            teacher_mean=pca["teacher_mean"],
            fitted_rank_full=int(pca["fitted_rank"]),
            pca_rows=int(pca["pca_rows"]),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + pair_idx + rank * 1000,
            device=device,
        )
        payload["state_dict"][f"projectors.{key}.weight"] = projector.weight.detach().float().cpu()
        payload["frozen_pt_weights"][key] = teacher_weights.detach().float().cpu()
        payload["frozen_pt_means"][key] = teacher_mean.detach().float().cpu()
        pair_metrics[key] = metrics
        val_losses.append(metrics["val_mse"])
        val_cosines.append(metrics["val_cosine"])
        print(
            f"rank {rank} pair {key}: val_mse={metrics['val_mse']:.6f} "
            f"val_cosine={metrics['val_cosine']:.6f} best_epoch={metrics['best_epoch']}",
            flush=True,
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_db_path": args.db_path,
        "student_model_path": args.student_model_path,
        "teacher_model_path": args.teacher_model_path,
        "student_dim": int(train_student.shape[-1]),
        "teacher_dim": int(train_teacher.shape[-1]),
        "student_layers": int(train_student.shape[1]),
        "teacher_layers": int(train_teacher.shape[1]),
        "num_layer_pairs": int(len(layer_pairs)),
        "rank": int(rank),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_pca_rows": int(args.max_pca_rows),
        "train_rows": int(train_student.shape[0]),
        "val_rows": int(val_student.shape[0]),
        "mean_val_mse": float(sum(val_losses) / max(len(val_losses), 1)),
        "mean_val_cosine": float(sum(val_cosines) / max(len(val_cosines), 1)),
        "pair_metrics": pair_metrics,
    }
    save_bank(out, payload, summary)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    experiences = load_token_experiences(args)
    train_exp, val_exp = split_experiences(
        experiences,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "train_pairs": len(train_exp),
                "val_pairs": len(val_exp),
                "args": {k: v for k, v in vars(args).items() if k != "ranks"},
                "ranks": args.ranks,
            },
            indent=2,
        ),
        flush=True,
    )

    train_student = collect_model_rows(
        train_exp,
        args.student_model_path,
        dtype=dtype,
        device=device,
        desc="student train hidden",
    )
    val_student = collect_model_rows(
        val_exp,
        args.student_model_path,
        dtype=dtype,
        device=device,
        desc="student val hidden",
    )
    train_teacher = collect_model_rows(
        train_exp,
        args.teacher_model_path,
        dtype=dtype,
        device=device,
        desc="teacher train hidden",
    )
    val_teacher = collect_model_rows(
        val_exp,
        args.teacher_model_path,
        dtype=dtype,
        device=device,
        desc="teacher val hidden",
    )

    print(
        json.dumps(
            {
                "train_student_shape": list(train_student.shape),
                "train_teacher_shape": list(train_teacher.shape),
                "val_student_shape": list(val_student.shape),
                "val_teacher_shape": list(val_teacher.shape),
            },
            indent=2,
        ),
        flush=True,
    )

    if train_student.dim() != 3 or train_teacher.dim() != 3:
        raise RuntimeError("Expected [rows,layers,dim] hidden tensors.")
    if train_student.shape[0] != train_teacher.shape[0]:
        raise RuntimeError("Student/teacher train row counts differ.")
    if val_student.shape[0] != val_teacher.shape[0]:
        raise RuntimeError("Student/teacher val row counts differ.")

    layer_pairs = proportional_layer_pairs(train_student.size(1), train_teacher.size(1))
    if args.limit_layer_mode == "last":
        layer_pairs = [layer_pairs[-1]]
    print(
        json.dumps(
            {
                "layer_pairs": [
                    {"student_layer": int(s), "teacher_layer": int(t)}
                    for s, t in layer_pairs
                ]
            },
            indent=2,
        ),
        flush=True,
    )

    run_summary = {
        "token_experience_args": vars(args),
        "train_pairs": len(train_exp),
        "val_pairs": len(val_exp),
        "train_rows": int(train_student.shape[0]),
        "val_rows": int(val_student.shape[0]),
        "layer_pairs": [
            {"student_layer": int(s), "teacher_layer": int(t)}
            for s, t in layer_pairs
        ],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    max_rank = max(int(rank) for rank in args.ranks)
    teacher_pca_cache: dict[str, dict[str, torch.Tensor | int]] = {}
    for pair_idx, (student_layer_idx, teacher_layer_idx) in enumerate(layer_pairs):
        key = f"s{student_layer_idx}_t{teacher_layer_idx}"
        print(
            f"fitting teacher PCA {pair_idx + 1}/{len(layer_pairs)} {key} max_rank={max_rank}",
            flush=True,
        )
        pca_rows = select_rows(
            train_teacher[:, teacher_layer_idx, :],
            args.max_pca_rows,
            args.seed + pair_idx,
        )
        teacher_weights, teacher_mean, fitted_rank = fit_teacher_pca_from_rows(
            pca_rows,
            max_rank,
        )
        teacher_pca_cache[key] = {
            "teacher_weights": teacher_weights.detach().float().cpu(),
            "teacher_mean": teacher_mean.detach().float().cpu(),
            "fitted_rank": int(fitted_rank),
            "pca_rows": int(pca_rows.shape[0]),
        }
        print(
            f"teacher PCA {key}: rows={pca_rows.shape[0]} fitted_rank={fitted_rank}",
            flush=True,
        )

    for rank in args.ranks:
        build_for_rank(
            rank=rank,
            args=args,
            train_student=train_student,
            train_teacher=train_teacher,
            val_student=val_student,
            val_teacher=val_teacher,
            layer_pairs=layer_pairs,
            teacher_pca_cache=teacher_pca_cache,
            device=device,
        )


if __name__ == "__main__":
    main()
