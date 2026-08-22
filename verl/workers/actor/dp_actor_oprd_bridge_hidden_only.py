"""Hidden-only actor update for OPRD-Bridge supervision."""

import torch

from verl import DataProto
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.dp_actor_oprd_bridge import DataParallelPPOActor as OPRDBridgePPOActor


class DataParallelPPOActor(OPRDBridgePPOActor):
    def update_policy(self, data: DataProto):
        if not self.config.get("oprd_bridge_hidden_only_update", False):
            return super().update_policy(data=data)

        self.actor_module.train()
        multi_turn = data.meta_info.get("multi_turn", False)
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "teacher_oprd_bridge_repr"]
        if "teacher_oprd_bridge_mask" in data.batch.keys():
            select_keys.append("teacher_oprd_bridge_mask")
        if multi_turn and "loss_mask" in data.batch.keys():
            select_keys.append("loss_mask")
        if "response_mask" in data.batch.keys():
            select_keys.append("response_mask")

        batch = data.select(batch_keys=select_keys).batch
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            raise NotImplementedError("hidden-only OPRD-Bridge actor does not support multi-modal inputs yet")

        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {"actor/oprd_bridge_hidden_only_update": 1.0}
        bridge_path = self.config.get("oprd_bridge_checkpoint_path", "")
        if not bridge_path:
            raise RuntimeError(
                "actor.oprd_bridge_hidden_only_update=True requires actor.oprd_bridge_checkpoint_path"
            )

        for _ in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                    grad_denom = max(len(micro_batches), 1)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    grad_denom = self.gradient_accumulation

                self.actor_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_torch_device().current_device())
                    bridge_loss, bridge_metrics = self._oprd_bridge_hidden_loss(
                        micro_batch,
                        checkpoint_path=bridge_path,
                        coef=self.config.get("oprd_bridge_hidden_loss_coef", 1.0),
                    )
                    loss = bridge_loss / grad_denom
                    loss.backward()
                    append_to_dict(metrics, bridge_metrics)
                    append_to_dict(metrics, {
                        "actor/pg_loss": 0.0,
                        "actor/pg_clipfrac": 0.0,
                        "actor/ppo_kl": 0.0,
                        "actor/pg_clipfrac_lower": 0.0,
                    })

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
