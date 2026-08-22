"""
SOD trainer variant that replaces log-prob OPD supervision with OPRD-Bridge
hidden-state supervision.

The SOD training loop is reused as-is. We only override teacher-logprob
computation: it now computes teacher hidden targets, attaches them to the batch,
and returns student old_log_probs as a dummy teacher logprob. Under
sod_mode=uniform and opd_only=true this makes the OPD advantage exactly zero,
so the actor gradient comes from OPRD-Bridge hidden loss.
"""

from omegaconf import open_dict

from verl import DataProto
from verl.trainer.ppo.sod_ray_trainer import SODRayTrainer


class SODOPRDBridgeRayTrainer(SODRayTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sod_cfg = self.config.algorithm.get("sod", {})
        hidden_cfg = sod_cfg.get("hidden_signal", {})
        self.use_oprd_bridge_hidden_signal = hidden_cfg.get("enabled", False)
        self.oprd_bridge_checkpoint_path = hidden_cfg.get("bridge_checkpoint_path", "")
        self.oprd_bridge_hidden_loss_coef = hidden_cfg.get("loss_coef", 1.0)
        self.oprd_bridge_hidden_micro_batch_size = hidden_cfg.get("micro_batch_size", 32)
        self.oprd_bridge_disable_logprob_opd = hidden_cfg.get("disable_logprob_opd", True)

        if self.use_oprd_bridge_hidden_signal and not self.oprd_bridge_checkpoint_path:
            raise ValueError(
                "algorithm.sod.hidden_signal.enabled=true requires "
                "algorithm.sod.hidden_signal.bridge_checkpoint_path"
            )
        if not self.use_oprd_bridge_hidden_signal:
            raise ValueError("SODOPRDBridgeRayTrainer requires algorithm.sod.hidden_signal.enabled=true")
        if not self.oprd_bridge_disable_logprob_opd:
            raise ValueError(
                "This variant replaces log-prob OPD; set "
                "algorithm.sod.hidden_signal.disable_logprob_opd=true"
            )

        with open_dict(self.config.actor_rollout_ref.actor):
            self.config.actor_rollout_ref.actor.use_oprd_bridge_hidden_loss = True
            self.config.actor_rollout_ref.actor.oprd_bridge_checkpoint_path = (
                self.oprd_bridge_checkpoint_path
            )
            self.config.actor_rollout_ref.actor.oprd_bridge_hidden_loss_coef = (
                self.oprd_bridge_hidden_loss_coef
            )

    def _compute_teacher_oprd_bridge_hidden(self, batch: DataProto) -> DataProto:
        batch.meta_info["oprd_bridge_checkpoint_path"] = self.oprd_bridge_checkpoint_path
        batch.meta_info["oprd_bridge_hidden_micro_batch_size"] = (
            self.oprd_bridge_hidden_micro_batch_size
        )
        if self.ref_policy_wg is not None:
            return self.ref_policy_wg.compute_ref_oprd_bridge_hidden(batch)
        return self.actor_rollout_wg.compute_ref_oprd_bridge_hidden(batch)

    def _compute_teacher_log_probs(self, batch: DataProto):
        teacher_hidden = self._compute_teacher_oprd_bridge_hidden(batch)
        batch.union(teacher_hidden)

        # Return a dummy teacher logprob equal to the stored student logprob.
        # The inherited SOD loop will compute teacher - student = 0, so the
        # log-prob OPD advantage is disabled without changing rollout/batch code.
        return batch.batch["old_log_probs"].detach()
