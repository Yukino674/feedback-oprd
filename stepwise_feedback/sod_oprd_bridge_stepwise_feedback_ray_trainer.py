"""Cached OPRD-Bridge trainer with step-wise teacher feedback rollout."""

from __future__ import annotations

import numpy as np
import torch

from agent_system.multi_turn_rollout import adjust_batch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import _timer, compute_response_mask
from verl.trainer.ppo.sod_oprd_bridge_cached_ray_trainer import (
    SODOPRDBridgeCachedRayTrainer,
)


class SODOPRDBridgeStepwiseFeedbackRayTrainer(SODOPRDBridgeCachedRayTrainer):
    @staticmethod
    def _add_stepwise_action_metrics(batch: DataProto, metrics):
        metric_keys = {
            "original_action_tag_found": "episode/original_action_tag_parse_rate",
            "original_action_in_admissible": "episode/original_action_in_admissible_rate",
            "original_action_no_think_valid": "episode/original_action_no_think_valid_rate",
            "teacher_mentioned_action_found": "episode/teacher_mentioned_action_rate",
            "teacher_mentioned_action_in_admissible": "episode/teacher_mentioned_action_in_admissible_rate",
            "teacher_mentioned_action_all_in_admissible": "episode/teacher_mentioned_action_all_in_admissible_rate",
            "action_tag_found": "episode/action_tag_parse_rate",
            "action_in_admissible": "episode/action_in_admissible_rate",
            "action_no_think_valid": "episode/action_no_think_valid_rate",
        }
        for key, metric_name in metric_keys.items():
            if key not in batch.non_tensor_batch:
                continue
            values = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
            if values.size > 0:
                metrics[metric_name] = float(values.mean())

    def _collect_rollout_with_teacher_hidden(self, gen_batch, timing_raw, metrics):
        with _timer("gen", timing_raw):
            if self.ref_policy_wg is None:
                raise RuntimeError(
                    "Step-wise feedback rollout requires a RefPolicy worker for "
                    "teacher feedback generation."
                )
            batch = self.traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                ref_policy_wg=self.ref_policy_wg,
                envs=self.envs,
                is_train=True,
            )
            ref_cfg = self.config.actor_rollout_ref.ref
            if (
                ref_cfg.get("feedback_backend", "hf") == "vllm"
                and bool(ref_cfg.get("feedback_release_after_rollout", False))
            ):
                self.ref_policy_wg.release_teacher_feedback_rollout()

        batch = adjust_batch(self.config, batch)
        self._add_stepwise_action_metrics(batch, metrics)
        batch.batch["response_mask"] = compute_response_mask(batch)
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)
        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"],
            dim=-1,
        ).tolist()

        with _timer("teacher_forward", timing_raw):
            teacher_hidden = self._compute_teacher_oprd_bridge_hidden(batch)
            batch = batch.union(teacher_hidden)
            metrics["sod_bridge/enabled"] = 1.0
            metrics["sod_bridge/logprob_opd_disabled"] = 1.0
            metrics["sod_bridge/teacher_target_attached_to_experience"] = 1.0
            metrics["sod_bridge/stepwise_feedback_enabled"] = 1.0
            metrics["sod_bridge/teacher_target_shape_layers"] = float(
                batch.batch["teacher_oprd_bridge_repr"].shape[1]
            )
            metrics["sod_bridge/teacher_target_shape_tokens"] = float(
                batch.batch["teacher_oprd_bridge_repr"].shape[2]
            )
            metrics["sod_bridge/teacher_target_shape_rank"] = float(
                batch.batch["teacher_oprd_bridge_repr"].shape[-1]
            )
            if "teacher_oprd_bridge_mask" in batch.batch.keys():
                metrics["sod_bridge/teacher_target_valid_tokens"] = float(
                    batch.batch["teacher_oprd_bridge_mask"].sum().detach().item()
                )
        return batch
