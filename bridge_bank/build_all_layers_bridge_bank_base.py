"""Build an all-layer OPRD-Bridge ps_bank.pt from TCOD experiences.

This is the multi-layer counterpart of build_last_layer_bridge_bank.py.
It uses exact sampled token ids from TCOD experiences, recomputes student
hidden states for every decoder layer over response tokens, and reads teacher
all-layer hidden states from Experience.info["teacher_all_hidden_repr"].
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from experimental.oprd_hidden.rep_distillation_utils import (
    all_layer_response_token_hidden,
    fit_teacher_pca_from_rows,
    proportional_layer_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def load_experiences(db_path: str, max_pairs: int) -> list:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        "select experience_bytes from pipeline_input order by id limit ?",
        (int(max_pairs),),
    ).fetchall()
    con.close()
    experiences = []
    for (blob,) in rows:
        exp = pickle.loads(blob)
        if exp.tokens is None:
            continue
        if "teacher_all_hidden_repr" not in exp.info:
            continue
        if len(exp.tokens) <= int(exp.prompt_length):
            continue
        experiences.append(exp)
    if not experiences:
        raise RuntimeError(f"No valid all-layer experiences found in {db_path}")
    return experiences


@torch.inference_mode()
def collect_all_layer_hidden(
    experiences: list,
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    student_rows = []
    teacher_rows = []
    for exp in tqdm(experiences, desc="student all-layer forward"):
        token_ids = exp.tokens.tolist()
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        response_len = len(token_ids) - int(exp.prompt_length)
        response_mask = torch.ones((1, response_len), dtype=torch.bool, device=device)
        student_hidden = all_layer_response_token_hidden(
            outputs.hidden_states,
            response_mask,
        )[0, :, :response_len, :].detach().float().cpu()
        teacher_hidden = torch.tensor(
            exp.info["teacher_all_hidden_repr"],
            dtype=torch.float32,
        )
        if teacher_hidden.dim() == 2:
            teacher_hidden = teacher_hidden.unsqueeze(1)
        teacher_hidden = teacher_hidden[:, :response_len, :]
        common_len = min(student_hidden.size(1), teacher_hidden.size(1))
        if common_len <= 0:
            continue
        student_hidden = student_hidden[:, :common_len, :]
        teacher_hidden = teacher_hidden[:, :common_len, :]
        student_hidden = student_hidden.transpose(0, 1).contiguous()
        teacher_hidden = teacher_hidden.transpose(0, 1).contiguous()
        student_rows.append(student_hidden)
        teacher_rows.append(teacher_hidden)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(student_rows, dim=0), torch.cat(teacher_rows, dim=0)


def train_projector_for_pair(
    student_rows: torch.Tensor,
    teacher_rows: torch.Tensor,
    *,
    rank: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, dict]:
    teacher_weights, teacher_mean, fitted_rank = fit_teacher_pca_from_rows(teacher_rows, rank)
    z_teacher = (teacher_rows.float() - teacher_mean) @ teacher_weights.T

    projector = nn.Linear(student_rows.shape[-1], rank, bias=False, device=device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)
    dataset = TensorDataset(student_rows.float(), z_teacher.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    history = []
    for _ in range(int(epochs)):
        losses = []
        for h_s, z_t in loader:
            h_s = h_s.to(device)
            z_t = z_t.to(device)
            loss = torch.nn.functional.mse_loss(projector(h_s), z_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append(sum(losses) / max(len(losses), 1))

    with torch.no_grad():
        final_loss = torch.nn.functional.mse_loss(
            projector(student_rows.float().to(device)),
            z_teacher.float().to(device),
        )
    metrics = {
        "fitted_rank": int(fitted_rank),
        "initial_epoch_mse": float(history[0]) if history else None,
        "final_mse": float(final_loss.detach().cpu().item()),
    }
    return projector, teacher_weights, teacher_mean, metrics


def save_bank(
    output_dir: str,
    payload: dict,
    metrics: dict,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rank_dir = out / f"rank_{metrics['rank']}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    bank_path = rank_dir / "ps_bank.pt"
    payload["metrics"] = metrics
    torch.save(payload, bank_path)
    with open(rank_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {bank_path}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return bank_path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    experiences = load_experiences(args.db_path, args.max_pairs)
    print(f"loaded {len(experiences)} all-layer experiences", flush=True)
    student_hidden, teacher_hidden = collect_all_layer_hidden(
        experiences,
        args.student_model_path,
        dtype=dtype,
        device=device,
    )
    if student_hidden.dim() != 3 or teacher_hidden.dim() != 3:
        raise RuntimeError(
            f"Expected [N,L,D] tensors, got student={tuple(student_hidden.shape)} "
            f"teacher={tuple(teacher_hidden.shape)}"
        )

    layer_pairs = proportional_layer_pairs(student_hidden.size(1), teacher_hidden.size(1))
    payload = {
        "subspace_mode": "full",
        "projector_type": "linear",
        "rank": int(args.rank),
        "layer_pairs": [
            {"student_layer": int(s), "teacher_layer": int(t)}
            for s, t in layer_pairs
        ],
        "state_dict": {},
        "frozen_pt_weights": {},
        "frozen_pt_means": {},
    }

    pair_metrics = {}
    final_losses = []
    for pair_idx, (student_layer_idx, teacher_layer_idx) in enumerate(layer_pairs):
        print(
            f"training pair {pair_idx + 1}/{len(layer_pairs)}: "
            f"s{student_layer_idx}_t{teacher_layer_idx}",
            flush=True,
        )
        projector, teacher_weights, teacher_mean, metrics = train_projector_for_pair(
            student_hidden[:, student_layer_idx, :],
            teacher_hidden[:, teacher_layer_idx, :],
            rank=args.rank,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        key = f"s{student_layer_idx}_t{teacher_layer_idx}"
        payload["state_dict"][f"projectors.{key}.weight"] = (
            projector.weight.detach().float().cpu()
        )
        payload["frozen_pt_weights"][key] = teacher_weights.detach().float().cpu()
        payload["frozen_pt_means"][key] = teacher_mean.detach().float().cpu()
        pair_metrics[key] = metrics
        final_losses.append(metrics["final_mse"])
        print(
            f"pair {key}: initial={metrics['initial_epoch_mse']:.6f} "
            f"final={metrics['final_mse']:.6f}",
            flush=True,
        )

    metrics = {
        "num_pairs": len(experiences),
        "student_dim": int(student_hidden.shape[-1]),
        "teacher_dim": int(teacher_hidden.shape[-1]),
        "student_layers": int(student_hidden.shape[1]),
        "teacher_layers": int(teacher_hidden.shape[1]),
        "num_layer_pairs": int(len(layer_pairs)),
        "rank": int(args.rank),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "mean_final_mse": float(sum(final_losses) / max(len(final_losses), 1)),
        "pair_metrics": pair_metrics,
    }
    save_bank(args.output_dir, payload, metrics)


if __name__ == "__main__":
    main()
