"""Backbone-only OPRD-Bridge hidden projection."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from verl.utils.device import is_cuda_available, is_npu_available
from verl.workers.actor.dp_actor_oprd_bridge_hidden_only import (
    DataParallelPPOActor as HiddenOnlyOPRDBridgePPOActor,
)

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


class DataParallelPPOActor(HiddenOnlyOPRDBridgePPOActor):
    @contextmanager
    def _oprd_bridge_hidden_forward_context(self):
        wrapped_module = getattr(self.actor_module, "_fsdp_wrapped_module", self.actor_module)
        model_cls = wrapped_module.__class__
        original_forward = model_cls.forward

        def hidden_forward(model_self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["output_hidden_states"] = True
            kwargs["use_cache"] = False
            backbone = getattr(model_self, "model", None)
            if backbone is None:
                base_model = getattr(model_self, "base_model", None)
                if base_model is not None:
                    backbone = getattr(base_model, "model", None) or getattr(base_model, "language_model", None)
            if backbone is None:
                raise RuntimeError(
                    "Backbone-only OPRD-Bridge requires a model with 'model' or 'language_model' submodule"
                )
            return backbone(*args, **kwargs)

        model_cls.forward = hidden_forward
        try:
            yield
        finally:
            model_cls.forward = original_forward

    def _oprd_bridge_project_micro_batch(
        self,
        micro_batch,
        *,
        checkpoint_path: str,
        mode: str,
        return_mask: bool = False,
    ):
        if mode not in {"student", "teacher"}:
            raise ValueError(f"mode must be 'student' or 'teacher', got {mode!r}")

        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        response_len = int(micro_batch["responses"].size(-1))
        response_last_k = int(self.config.get("oprd_bridge_response_last_k", 0))

        batch_size, seqlen = input_ids.shape
        use_rmpad = (
            self.use_remove_padding
            and not self.use_ulysses_sp
            and position_ids.dim() != 3
        )

        with self._oprd_bridge_hidden_forward_context():
            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                if use_rmpad:
                    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices,
                    ).transpose(0, 1)
                    outputs = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                else:
                    if position_ids.dim() == 3:
                        position_ids = position_ids.transpose(0, 1)
                    outputs = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        output_hidden_states=True,
                    )

        hidden_states = outputs.hidden_states[1:]
        dtype = hidden_states[-1].dtype
        bank = self._get_oprd_bridge_bank(checkpoint_path, input_ids.device, dtype)

        def slice_response_hidden(layer_hidden: torch.Tensor) -> torch.Tensor:
            if use_rmpad:
                layer_hidden = pad_input(
                    hidden_states=layer_hidden.squeeze(0),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
            return layer_hidden[:, -response_len:, :]

        selected_idx, selected_mask = self._oprd_bridge_get_response_selection(
            micro_batch,
            response_len=response_len,
            response_last_k=response_last_k,
        )

        projected = []
        for pair_idx in self._oprd_bridge_layer_indices(bank):
            student_layer_idx, teacher_layer_idx = bank["layer_pairs"][pair_idx]
            if mode == "student":
                layer_idx = student_layer_idx
                if layer_idx >= len(hidden_states):
                    raise RuntimeError(
                        f"Student layer {layer_idx} is out of range for {len(hidden_states)} layers"
                    )
                h = slice_response_hidden(hidden_states[layer_idx])
                if selected_idx is not None:
                    h = torch.gather(h, dim=1, index=selected_idx[:, :, None].expand(-1, -1, h.size(-1)))
                weight = bank["student_weights"][pair_idx]
                z = F.linear(h, weight)
            else:
                layer_idx = teacher_layer_idx
                if layer_idx >= len(hidden_states):
                    raise RuntimeError(
                        f"Teacher layer {layer_idx} is out of range for {len(hidden_states)} layers"
                    )
                h = slice_response_hidden(hidden_states[layer_idx])
                if selected_idx is not None:
                    h = torch.gather(h, dim=1, index=selected_idx[:, :, None].expand(-1, -1, h.size(-1)))
                weight = bank["teacher_weights"][pair_idx]
                mean = bank["teacher_means"][pair_idx]
                z = F.linear(h - mean.to(h.dtype), weight)
            projected.append(z)

        projected = torch.stack(projected, dim=1)
        if return_mask:
            return projected, selected_mask
        return projected
