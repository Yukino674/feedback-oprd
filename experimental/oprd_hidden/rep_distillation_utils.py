"""Small OPRD-style representation distillation helpers.

This intentionally implements the narrow first experiment:
- last decoder layer only
- last valid response token only
- normalized MSE
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def last_response_token_hidden(
    hidden_states: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return hidden state at the last valid response token.

    Args:
        hidden_states: [batch, seq_len, hidden_dim], full prompt+response states.
        response_mask: [batch, response_len], true for generated response tokens.
    """

    response_len = response_mask.size(1)
    response_hidden = hidden_states[:, -response_len:, :]
    last_idx = response_mask.long().sum(dim=1).clamp_min(1) - 1
    batch_idx = torch.arange(response_hidden.size(0), device=response_hidden.device)
    return response_hidden[batch_idx, last_idx]


def all_layer_last_response_token_hidden(
    hidden_states: tuple[torch.Tensor, ...],
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return every decoder layer hidden state at the last valid response token.

    HF ``hidden_states[0]`` is embeddings; ``hidden_states[1:]`` are block outputs.
    Returns shape ``[batch, num_layers, hidden_dim]``.
    """

    layer_reprs = [
        last_response_token_hidden(layer_hidden, response_mask)
        for layer_hidden in hidden_states[1:]
    ]
    if not layer_reprs:
        raise ValueError("Expected hidden_states to contain decoder layers.")
    return torch.stack(layer_reprs, dim=1)


def response_token_hidden(
    hidden_states: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return hidden states over the full response segment.

    Args:
        hidden_states: [batch, seq_len, hidden_dim], full prompt+response states.
        response_mask: [batch, response_len], true for generated response tokens.
    """

    response_len = response_mask.size(1)
    return hidden_states[:, -response_len:, :]


def all_layer_response_token_hidden(
    hidden_states: tuple[torch.Tensor, ...],
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return every decoder layer hidden states over the full response segment.

    HF ``hidden_states[0]`` is embeddings; ``hidden_states[1:]`` are block outputs.
    Returns shape ``[batch, num_layers, response_len, hidden_dim]``.
    """

    layer_reprs = [
        response_token_hidden(layer_hidden, response_mask)
        for layer_hidden in hidden_states[1:]
    ]
    if not layer_reprs:
        raise ValueError("Expected hidden_states to contain decoder layers.")
    return torch.stack(layer_reprs, dim=1)


def proportional_layer_pairs(
    num_student_layers: int,
    num_teacher_layers: int,
) -> list[tuple[int, int]]:
    """Pair each student layer with a proportionally spaced teacher layer."""

    if num_student_layers <= 0 or num_teacher_layers <= 0:
        raise ValueError("Layer counts must be positive.")
    if num_student_layers == 1:
        return [(0, num_teacher_layers - 1)]
    teacher_indices = torch.linspace(0, num_teacher_layers - 1, steps=num_student_layers)
    return [(idx, int(round(float(teacher_indices[idx])))) for idx in range(num_student_layers)]


def normalized_mse_loss(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    position_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MSE between L2-normalized student and teacher representations."""

    student_repr = F.normalize(student_repr.float(), p=2, dim=-1, eps=eps)
    teacher_repr = F.normalize(teacher_repr.float(), p=2, dim=-1, eps=eps)
    per_position_mse = ((student_repr - teacher_repr) ** 2).mean(dim=-1)
    if position_mask is None:
        return per_position_mse.mean()
    mask = position_mask.to(per_position_mse.device, dtype=per_position_mse.dtype)
    while mask.dim() < per_position_mse.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(per_position_mse)
    return (per_position_mse * mask).sum() / mask.sum().clamp_min(eps)


@torch.no_grad()
def fit_teacher_pca_from_rows(
    teacher_rows: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fit OPRD-Bridge teacher PCA basis P_T from teacher hidden rows.

    Returns:
        components: [rank, teacher_dim], padded with zeros if there are too few rows.
        mean: [teacher_dim]
        fitted_rank: number of non-zero PCA directions actually estimated.
    """

    if teacher_rows.dim() != 2:
        raise ValueError(f"teacher_rows must be [rows, dim], got {tuple(teacher_rows.shape)}")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")

    rows = teacher_rows.float()
    mean = rows.mean(dim=0)
    centered = rows - mean
    fitted_rank = min(int(rank), max(rows.shape[0] - 1, 1), rows.shape[1])
    components = rows.new_zeros(rank, rows.shape[1])
    if rows.shape[0] <= rows.shape[1]:
        # SVD on the sample matrix is much cheaper than a DxD covariance eigendecomp
        # for LLM hidden sizes where rows << hidden_dim.
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        components[:fitted_rank] = vh[:fitted_rank]
    else:
        cov = (centered.T @ centered) / max(rows.shape[0] - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        components[:fitted_rank] = eigvecs[:, order].T[:fitted_rank]
    return components, mean, fitted_rank


class LastLayerLowRankBridge(nn.Module):
    """OPRD-Bridge style low-rank bridge for one layer.

    It follows the OPRD-Bridge Stage-2 projection form:
        z_s = P_S h_s
        z_t = P_T (h_t - mu_t)
        loss = MSE(z_s, stop_grad(z_t))

    This lightweight TCOD experiment initializes P_T from the first on-policy
    teacher batch when no offline ps_bank checkpoint is available.
    """

    def __init__(
        self,
        *,
        student_dim: int,
        teacher_dim: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        freeze_ps: bool = False,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.rank = int(rank)
        self.student_projector = nn.Linear(student_dim, rank, bias=False, device=device, dtype=dtype)
        self.register_buffer("teacher_weights", torch.zeros(rank, teacher_dim, device=device))
        self.register_buffer("teacher_mean", torch.zeros(teacher_dim, device=device))
        self.teacher_pt_initialized = False
        self.loaded_from_checkpoint = False
        self.loaded_ps = False
        self.loaded_pt = False
        self.fitted_rank = 0
        self.ps_frozen = bool(freeze_ps)
        if self.ps_frozen:
            for param in self.student_projector.parameters():
                param.requires_grad_(False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self.ps_frozen:
            return []
        return list(self.student_projector.parameters())

    def freeze_student_projectors(self) -> None:
        self.ps_frozen = True
        for param in self.student_projector.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def maybe_init_teacher_pca_from_batch(self, teacher_repr: torch.Tensor) -> None:
        if self.teacher_pt_initialized:
            return
        rows = teacher_repr.reshape(-1, teacher_repr.shape[-1]).float()
        components, mean, fitted_rank = fit_teacher_pca_from_rows(rows, self.rank)
        self.teacher_weights.copy_(components.to(self.teacher_weights.device))
        self.teacher_mean.copy_(mean.to(self.teacher_mean.device))
        self.fitted_rank = int(fitted_rank)
        self.teacher_pt_initialized = True

    def project_student(self, student_repr: torch.Tensor) -> torch.Tensor:
        return self.student_projector(student_repr)

    def project_teacher(self, teacher_repr: torch.Tensor) -> torch.Tensor:
        centered = teacher_repr - self.teacher_mean.to(teacher_repr.dtype)
        return centered @ self.teacher_weights.to(teacher_repr.dtype).T

    def project_pair(
        self,
        student_repr: torch.Tensor,
        teacher_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.teacher_pt_initialized:
            self.maybe_init_teacher_pca_from_batch(teacher_repr)
        return self.project_student(student_repr), self.project_teacher(teacher_repr).detach()

    def load_from_ps_bank(self, checkpoint_path: str | Path) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get("state_dict", {})
        frozen_weights = checkpoint.get("frozen_pt_weights", {})
        frozen_means = checkpoint.get("frozen_pt_means", {})

        ps_weight = state_dict.get("projectors.s0_t0.weight")
        if ps_weight is None:
            ps_weight = state_dict.get("projectors.0_0.weight")
        if ps_weight is not None:
            if ps_weight.shape != self.student_projector.weight.shape:
                raise ValueError(
                    "P_S shape mismatch: "
                    f"checkpoint={tuple(ps_weight.shape)} expected="
                    f"{tuple(self.student_projector.weight.shape)}"
                )
            self.student_projector.weight.data.copy_(
                ps_weight.to(
                    device=self.student_projector.weight.device,
                    dtype=self.student_projector.weight.dtype,
                )
            )
            self.loaded_ps = True

        pt_weight = frozen_weights.get("s0_t0")
        if pt_weight is None:
            pt_weight = frozen_weights.get("0_0")
        pt_mean = frozen_means.get("s0_t0")
        if pt_mean is None:
            pt_mean = frozen_means.get("0_0")
        if pt_weight is not None:
            if pt_weight.shape != self.teacher_weights.shape:
                raise ValueError(
                    "P_T shape mismatch: "
                    f"checkpoint={tuple(pt_weight.shape)} expected={tuple(self.teacher_weights.shape)}"
                )
            self.teacher_weights.copy_(pt_weight.to(self.teacher_weights.device))
            self.loaded_pt = True
        if pt_mean is not None:
            if pt_mean.shape != self.teacher_mean.shape:
                raise ValueError(
                    "teacher mean shape mismatch: "
                    f"checkpoint={tuple(pt_mean.shape)} expected={tuple(self.teacher_mean.shape)}"
                )
            self.teacher_mean.copy_(pt_mean.to(self.teacher_mean.device))

        self.loaded_from_checkpoint = self.loaded_ps or self.loaded_pt
        self.teacher_pt_initialized = self.loaded_pt
        self.fitted_rank = self.rank if self.loaded_pt else 0

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        return {
            "actor/oprd_bridge_rank": float(self.rank),
            "actor/oprd_bridge_fitted_rank": float(self.fitted_rank),
            "actor/oprd_bridge_teacher_pt_initialized": float(self.teacher_pt_initialized),
            "actor/oprd_bridge_loaded_from_checkpoint": float(self.loaded_from_checkpoint),
            "actor/oprd_bridge_loaded_ps": float(self.loaded_ps),
            "actor/oprd_bridge_loaded_pt": float(self.loaded_pt),
            "actor/oprd_bridge_ps_frozen": float(self.ps_frozen),
            "actor/oprd_bridge_ps_weight_norm": float(
                self.student_projector.weight.detach().float().norm().item()
            ),
            "actor/oprd_bridge_pt_weight_norm": float(
                self.teacher_weights.detach().float().norm().item()
            ),
            "actor/oprd_bridge_teacher_mean_norm": float(
                self.teacher_mean.detach().float().norm().item()
            ),
        }


class MultiLayerLowRankBridge(nn.Module):
    """OPRD-Bridge style low-rank bridge over proportionally paired layers."""

    def __init__(
        self,
        *,
        student_dim: int,
        teacher_dim: int,
        student_layers: int,
        teacher_layers: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        freeze_ps: bool = False,
        layer_pairs: list[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.student_layers = int(student_layers)
        self.teacher_layers = int(teacher_layers)
        self.rank = int(rank)
        self.layer_pairs = layer_pairs or proportional_layer_pairs(
            self.student_layers,
            self.teacher_layers,
        )
        self.student_projectors = nn.ModuleList(
            [
                nn.Linear(student_dim, rank, bias=False, device=device, dtype=dtype)
                for _ in self.layer_pairs
            ]
        )
        self.register_buffer(
            "teacher_weights",
            torch.zeros(len(self.layer_pairs), rank, teacher_dim, device=device),
        )
        self.register_buffer(
            "teacher_means",
            torch.zeros(len(self.layer_pairs), teacher_dim, device=device),
        )
        self.teacher_pt_initialized = False
        self.loaded_from_checkpoint = False
        self.loaded_ps = False
        self.loaded_pt = False
        self.fitted_ranks = [0 for _ in self.layer_pairs]
        self.ps_frozen = bool(freeze_ps)
        if self.ps_frozen:
            self.freeze_student_projectors()

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self.ps_frozen:
            return []
        return list(self.student_projectors.parameters())

    def freeze_student_projectors(self) -> None:
        self.ps_frozen = True
        for param in self.student_projectors.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def maybe_init_teacher_pca_from_batch(self, teacher_repr: torch.Tensor) -> None:
        if self.teacher_pt_initialized:
            return
        for pair_idx, (_, teacher_layer_idx) in enumerate(self.layer_pairs):
            rows = teacher_repr[:, teacher_layer_idx, ...].reshape(-1, teacher_repr.shape[-1]).float()
            components, mean, fitted_rank = fit_teacher_pca_from_rows(rows, self.rank)
            self.teacher_weights[pair_idx].copy_(components.to(self.teacher_weights.device))
            self.teacher_means[pair_idx].copy_(mean.to(self.teacher_means.device))
            self.fitted_ranks[pair_idx] = int(fitted_rank)
        self.teacher_pt_initialized = True

    def project_pair(
        self,
        student_repr: torch.Tensor,
        teacher_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if student_repr.dim() not in {3, 4} or teacher_repr.dim() not in {3, 4}:
            raise ValueError(
                "MultiLayerLowRankBridge expects [batch, layers, dim] or "
                "[batch, layers, tokens, dim], got "
                f"student={tuple(student_repr.shape)} teacher={tuple(teacher_repr.shape)}"
            )
        if not self.teacher_pt_initialized:
            self.maybe_init_teacher_pca_from_batch(teacher_repr)

        projected_student = []
        projected_teacher = []
        for pair_idx, (student_layer_idx, teacher_layer_idx) in enumerate(self.layer_pairs):
            h_s = student_repr[:, student_layer_idx, ...]
            h_t = teacher_repr[:, teacher_layer_idx, ...]
            z_s = self.student_projectors[pair_idx](h_s)
            centered = h_t - self.teacher_means[pair_idx].to(h_t.dtype)
            z_t = centered @ self.teacher_weights[pair_idx].to(h_t.dtype).T
            projected_student.append(z_s)
            projected_teacher.append(z_t.detach())
        return torch.stack(projected_student, dim=1), torch.stack(projected_teacher, dim=1)

    def project_student(self, student_repr: torch.Tensor) -> torch.Tensor:
        """Project student layers with P_S only.

        Used when teacher targets were already projected with P_T before being
        written to the experience buffer.
        """

        if student_repr.dim() not in {3, 4}:
            raise ValueError(
                "MultiLayerLowRankBridge.project_student expects [batch, layers, dim] "
                f"or [batch, layers, tokens, dim], got {tuple(student_repr.shape)}"
            )
        projected_student = []
        for pair_idx, (student_layer_idx, _) in enumerate(self.layer_pairs):
            h_s = student_repr[:, student_layer_idx, ...]
            projected_student.append(self.student_projectors[pair_idx](h_s))
        return torch.stack(projected_student, dim=1)

    def load_from_ps_bank(self, checkpoint_path: str | Path) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        ckpt_pairs = checkpoint.get("layer_pairs")
        if ckpt_pairs:
            loaded_pairs = [
                (int(pair["student_layer"]), int(pair["teacher_layer"]))
                for pair in ckpt_pairs
            ]
            if loaded_pairs != self.layer_pairs:
                raise ValueError(
                    f"layer_pairs mismatch: checkpoint={loaded_pairs[:5]}... "
                    f"expected={self.layer_pairs[:5]}..."
                )

        state_dict = checkpoint.get("state_dict", {})
        frozen_weights = checkpoint.get("frozen_pt_weights", {})
        frozen_means = checkpoint.get("frozen_pt_means", {})

        loaded_ps_count = 0
        loaded_pt_count = 0
        for pair_idx, (student_layer_idx, teacher_layer_idx) in enumerate(self.layer_pairs):
            key = f"s{student_layer_idx}_t{teacher_layer_idx}"
            ps_weight = state_dict.get(f"projectors.{key}.weight")
            if ps_weight is not None:
                if ps_weight.shape != self.student_projectors[pair_idx].weight.shape:
                    raise ValueError(
                        f"P_S shape mismatch for {key}: checkpoint={tuple(ps_weight.shape)} "
                        f"expected={tuple(self.student_projectors[pair_idx].weight.shape)}"
                    )
                self.student_projectors[pair_idx].weight.data.copy_(
                    ps_weight.to(
                        device=self.student_projectors[pair_idx].weight.device,
                        dtype=self.student_projectors[pair_idx].weight.dtype,
                    )
                )
                loaded_ps_count += 1

            pt_weight = frozen_weights.get(key)
            pt_mean = frozen_means.get(key)
            if pt_weight is not None:
                if pt_weight.shape != self.teacher_weights[pair_idx].shape:
                    raise ValueError(
                        f"P_T shape mismatch for {key}: checkpoint={tuple(pt_weight.shape)} "
                        f"expected={tuple(self.teacher_weights[pair_idx].shape)}"
                    )
                self.teacher_weights[pair_idx].copy_(pt_weight.to(self.teacher_weights.device))
                loaded_pt_count += 1
            if pt_mean is not None:
                if pt_mean.shape != self.teacher_means[pair_idx].shape:
                    raise ValueError(
                        f"teacher mean shape mismatch for {key}: checkpoint={tuple(pt_mean.shape)} "
                        f"expected={tuple(self.teacher_means[pair_idx].shape)}"
                    )
                self.teacher_means[pair_idx].copy_(pt_mean.to(self.teacher_means.device))

        self.loaded_ps = loaded_ps_count == len(self.layer_pairs)
        self.loaded_pt = loaded_pt_count == len(self.layer_pairs)
        self.loaded_from_checkpoint = loaded_ps_count > 0 or loaded_pt_count > 0
        self.teacher_pt_initialized = self.loaded_pt
        if self.loaded_pt:
            self.fitted_ranks = [self.rank for _ in self.layer_pairs]

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        return {
            "actor/oprd_bridge_rank": float(self.rank),
            "actor/oprd_bridge_num_layer_pairs": float(len(self.layer_pairs)),
            "actor/oprd_bridge_student_layers": float(self.student_layers),
            "actor/oprd_bridge_teacher_layers": float(self.teacher_layers),
            "actor/oprd_bridge_fitted_rank": float(
                sum(self.fitted_ranks) / max(len(self.fitted_ranks), 1)
            ),
            "actor/oprd_bridge_teacher_pt_initialized": float(self.teacher_pt_initialized),
            "actor/oprd_bridge_loaded_from_checkpoint": float(self.loaded_from_checkpoint),
            "actor/oprd_bridge_loaded_ps": float(self.loaded_ps),
            "actor/oprd_bridge_loaded_pt": float(self.loaded_pt),
            "actor/oprd_bridge_ps_frozen": float(self.ps_frozen),
            "actor/oprd_bridge_ps_weight_norm": float(
                torch.stack([
                    projector.weight.detach().float().norm()
                    for projector in self.student_projectors
                ]).mean().item()
            ),
            "actor/oprd_bridge_pt_weight_norm": float(
                self.teacher_weights.detach().float().norm(dim=(1, 2)).mean().item()
            ),
            "actor/oprd_bridge_teacher_mean_norm": float(
                self.teacher_means.detach().float().norm(dim=-1).mean().item()
            ),
        }
