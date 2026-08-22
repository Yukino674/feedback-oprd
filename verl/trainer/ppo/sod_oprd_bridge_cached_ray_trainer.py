"""SOD OPRD-Bridge variant that attaches teacher targets to rollout experience."""

from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.sod_ray_trainer import SODRayTrainer
from verl.trainer.ppo.sod_oprd_bridge_ray_trainer import SODOPRDBridgeRayTrainer
from verl.utils.metric import reduce_metrics

from agent_system.multi_turn_rollout import adjust_batch


class SODOPRDBridgeCachedRayTrainer(SODOPRDBridgeRayTrainer):
    def _collect_rollout_with_teacher_hidden(self, gen_batch, timing_raw, metrics):
        with _timer("gen", timing_raw):
            batch = self.traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.envs,
                is_train=True,
            )

        # Align the final experience batch before computing teacher targets so
        # duplicated/balanced samples and teacher representations stay paired.
        batch = adjust_batch(self.config, batch)
        batch.batch["response_mask"] = compute_response_mask(batch)
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)
        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"], dim=-1
        ).tolist()

        with _timer("teacher_forward", timing_raw):
            teacher_hidden = self._compute_teacher_oprd_bridge_hidden(batch)
            batch = batch.union(teacher_hidden)
            metrics["sod_bridge/enabled"] = 1.0
            metrics["sod_bridge/logprob_opd_disabled"] = 1.0
            metrics["sod_bridge/teacher_target_attached_to_experience"] = 1.0
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sod_cfg = self.config.algorithm.get("sod", {})
        hidden_cfg = sod_cfg.get("hidden_signal", {})
        self.use_oprd_bridge_hidden_signal = hidden_cfg.get("enabled", False)
        self.oprd_bridge_checkpoint_path = hidden_cfg.get("bridge_checkpoint_path", "")
        self.oprd_bridge_hidden_loss_coef = hidden_cfg.get("loss_coef", 1.0)
        self.oprd_bridge_hidden_micro_batch_size = hidden_cfg.get("micro_batch_size", 32)
        self.oprd_bridge_response_last_k = int(hidden_cfg.get("response_last_k", 0))
        self.oprd_bridge_layer_stride = int(hidden_cfg.get("layer_stride", 1))
        self.oprd_bridge_max_layer_pairs = int(hidden_cfg.get("max_layer_pairs", 0))
        self.oprd_bridge_disable_logprob_opd = hidden_cfg.get("disable_logprob_opd", True)

        if not self.use_oprd_bridge_hidden_signal:
            raise ValueError("SODOPRDBridgeRayTrainer requires algorithm.sod.hidden_signal.enabled=true")
        if not self.oprd_bridge_checkpoint_path:
            raise ValueError(
                "algorithm.sod.hidden_signal.enabled=true requires "
                "algorithm.sod.hidden_signal.bridge_checkpoint_path"
            )
        if not self.oprd_bridge_disable_logprob_opd:
            raise ValueError(
                "This variant replaces log-prob OPD; set "
                "algorithm.sod.hidden_signal.disable_logprob_opd=true"
            )

        with open_dict(self.config.actor_rollout_ref.actor):
            self.config.actor_rollout_ref.actor.use_oprd_bridge_hidden_loss = True
            self.config.actor_rollout_ref.actor.oprd_bridge_checkpoint_path = self.oprd_bridge_checkpoint_path
            self.config.actor_rollout_ref.actor.oprd_bridge_hidden_loss_coef = self.oprd_bridge_hidden_loss_coef
            self.config.actor_rollout_ref.actor.oprd_bridge_hidden_only_update = True
            self.config.actor_rollout_ref.actor.oprd_bridge_response_last_k = self.oprd_bridge_response_last_k
            self.config.actor_rollout_ref.actor.oprd_bridge_layer_stride = self.oprd_bridge_layer_stride
            self.config.actor_rollout_ref.actor.oprd_bridge_max_layer_pairs = self.oprd_bridge_max_layer_pairs
        with open_dict(self.config.actor_rollout_ref.ref):
            self.config.actor_rollout_ref.ref.oprd_bridge_response_last_k = self.oprd_bridge_response_last_k
            self.config.actor_rollout_ref.ref.oprd_bridge_layer_stride = self.oprd_bridge_layer_stride
            self.config.actor_rollout_ref.ref.oprd_bridge_max_layer_pairs = self.oprd_bridge_max_layer_pairs

    def _compute_teacher_oprd_bridge_hidden(self, batch: DataProto) -> DataProto:
        batch.meta_info["oprd_bridge_checkpoint_path"] = self.oprd_bridge_checkpoint_path
        batch.meta_info["oprd_bridge_hidden_micro_batch_size"] = self.oprd_bridge_hidden_micro_batch_size
        batch.meta_info["oprd_bridge_response_last_k"] = self.oprd_bridge_response_last_k
        batch.meta_info["oprd_bridge_layer_stride"] = self.oprd_bridge_layer_stride
        batch.meta_info["oprd_bridge_max_layer_pairs"] = self.oprd_bridge_max_layer_pairs
        if self.ref_policy_wg is not None:
            return self.ref_policy_wg.compute_ref_oprd_bridge_hidden(batch)
        return self.actor_rollout_wg.compute_ref_oprd_bridge_hidden(batch)

    def _launch_teacher_oprd_bridge_hidden(self, batch: DataProto):
        batch.meta_info["oprd_bridge_checkpoint_path"] = self.oprd_bridge_checkpoint_path
        batch.meta_info["oprd_bridge_hidden_micro_batch_size"] = self.oprd_bridge_hidden_micro_batch_size
        batch.meta_info["oprd_bridge_response_last_k"] = self.oprd_bridge_response_last_k
        batch.meta_info["oprd_bridge_layer_stride"] = self.oprd_bridge_layer_stride
        batch.meta_info["oprd_bridge_max_layer_pairs"] = self.oprd_bridge_max_layer_pairs
        if self.ref_policy_wg is None:
            return self._compute_teacher_oprd_bridge_hidden(batch)
        teacher_batches = batch.chunk(self.ref_policy_wg.world_size)
        return self.ref_policy_wg.execute_all_async("ref_compute_ref_oprd_bridge_hidden", teacher_batches)

    def _materialize_teacher_oprd_bridge_hidden(self, teacher_future):
        if isinstance(teacher_future, list):
            teacher_outputs = ray.get(teacher_future)
            return DataProto.concat(teacher_outputs)
        return teacher_future

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="SOD OPRD-Bridge Hidden-Only Training",
        )
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    del batch
                    batch = self._collect_rollout_with_teacher_hidden(
                        gen_batch=gen_batch,
                        timing_raw=timing_raw,
                        metrics=metrics,
                    )

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )
                        metrics["sod_bridge/grpo_adv_abs_mean_before_zero"] = batch.batch["advantages"].abs().mean().detach().item()
                        batch.batch["advantages"] = torch.zeros_like(batch.batch["advantages"])
                        metrics["sod_bridge/advantages_zeroed"] = 1.0

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            batch.meta_info["oprd_bridge_checkpoint_path"] = self.oprd_bridge_checkpoint_path
                            batch.meta_info["oprd_bridge_hidden_micro_batch_size"] = self.oprd_bridge_hidden_micro_batch_size
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or (
                                self.global_steps >= test_start_step
                                and self.global_steps % self.config.trainer.test_freq == 0
                            )
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
