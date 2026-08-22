"""Step-wise feedback worker additions for OPRD-Bridge.

This module subclasses the backbone-only OPRD-Bridge worker so existing actor,
critic, reward, and hidden-target paths stay unchanged.  It only adds teacher
generation from the RefPolicy FSDP model for short step feedback prompts.
"""

from __future__ import annotations

import gc
import inspect

import torch
from tensordict import TensorDict
from transformers import GenerationConfig

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import get_response_mask
from verl.workers.rollout import HFRollout
from verl.workers.fsdp_workers_oprd_bridge_backbone_only import (
    ActorRolloutRefWorker as _BackboneActorRolloutRefWorker,
    AsyncActorRolloutRefWorker as _BackboneAsyncActorRolloutRefWorker,
    CriticWorker,
    RewardModelWorker,
)


class _TeacherVLLMFeedbackManager:
    """Wake/sleep a frozen teacher vLLM engine without syncing FSDP weights."""

    def __init__(self, inference_engine, *, enable_sleep_mode: bool = True):
        self.inference_engine = inference_engine
        self.enable_sleep_mode = enable_sleep_mode

    def __enter__(self):
        get_torch_device().empty_cache()
        if not self.enable_sleep_mode:
            return self
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["weights"])
            self.inference_engine.wake_up(tags=["kv_cache"])
        else:
            self.inference_engine.wake_up()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enable_sleep_mode:
            self.inference_engine.sleep(level=1)
        get_torch_device().empty_cache()

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data


class ActorRolloutRefWorker(_BackboneActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self.role == "ref":
            self._build_teacher_feedback_rollout()

    def _build_teacher_feedback_rollout(self):
        """Build a frozen teacher rollout used only for step feedback text."""

        feedback_cfg = self.config.ref.get("feedback_vllm", None)
        if feedback_cfg is not None and not bool(feedback_cfg.get("enabled", True)):
            self.teacher_feedback_rollout = None
            self.teacher_feedback_sharding_manager = None
            return

        from omegaconf import OmegaConf, open_dict

        rollout_config = OmegaConf.create(OmegaConf.to_container(self.config.rollout, resolve=True))
        response_length = int(self.config.ref.get("feedback_response_length", rollout_config.response_length))
        max_num_batched_tokens = int(
            self.config.ref.get("feedback_max_num_batched_tokens", rollout_config.max_num_batched_tokens)
        )
        max_num_seqs = int(self.config.ref.get("feedback_max_num_seqs", rollout_config.max_num_seqs))
        with open_dict(rollout_config):
            rollout_config.response_length = response_length
            rollout_config.max_num_batched_tokens = max_num_batched_tokens
            rollout_config.max_num_seqs = max_num_seqs
            rollout_config.n = 1
            rollout_config.do_sample = bool(self.config.ref.get("feedback_do_sample", True))
            rollout_config.temperature = float(self.config.ref.get("feedback_temperature", 0.2))
            rollout_config.top_p = float(self.config.ref.get("feedback_top_p", 0.9))
            rollout_config.top_k = int(self.config.ref.get("feedback_top_k", -1))

        feedback_backend = self.config.ref.get("feedback_backend", "hf")
        if feedback_backend == "vllm":
            from torch.distributed.device_mesh import init_device_mesh
            from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode

            if vllm_mode != "spmd":
                raise NotImplementedError("Teacher feedback vLLM backend currently supports only spmd vLLM mode.")
            infer_tp = int(self.config.ref.get("feedback_tensor_model_parallel_size", 1))
            if infer_tp != 1:
                raise NotImplementedError("Teacher feedback vLLM currently supports tensor_model_parallel_size=1 only.")
            assert self.world_size % infer_tp == 0, (
                f"ref world_size: {self.world_size} is not divisible by teacher feedback infer_tp: {infer_tp}"
            )
            with open_dict(rollout_config):
                rollout_config.tensor_model_parallel_size = infer_tp
                rollout_config.gpu_memory_utilization = float(
                    self.config.ref.get("feedback_gpu_memory_utilization", 0.45)
                )
                rollout_config.load_format = self.config.ref.get("feedback_load_format", "auto")
                rollout_config.free_cache_engine = False
                rollout_config.enable_sleep_mode = bool(
                    self.config.ref.get("feedback_enable_sleep_mode", True)
                )

            dp = self.world_size // infer_tp
            rollout_device_mesh = init_device_mesh(
                get_device_name(),
                mesh_shape=(dp, infer_tp),
                mesh_dim_names=["dp", "infer_tp"],
            )
            ref_path = self.config.ref.model.path if self.config.ref.get("model", None) is not None else self.config.model.path
            local_path = copy_to_local(ref_path, use_shm=self.config.model.get("use_shm", False))
            wrapped_ref = getattr(self.ref_module_fsdp, "_fsdp_wrapped_module", self.ref_module_fsdp)
            ref_model_config = getattr(wrapped_ref, "config", None)
            if ref_model_config is None:
                raise RuntimeError("Could not find ref model config for teacher feedback vLLM.")

            if self.rank == 0:
                print(
                    "[StepFeedback] Building teacher feedback vLLM rollout "
                    f"from {local_path} with tp={infer_tp}, "
                    f"gpu_memory_utilization={rollout_config.gpu_memory_utilization}, "
                    f"enable_sleep_mode={rollout_config.enable_sleep_mode}"
                )
            self.teacher_feedback_rollout = vLLMRollout(
                model_path=local_path,
                config=rollout_config,
                tokenizer=self.tokenizer,
                model_hf_config=ref_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
            )
            self.teacher_feedback_sharding_manager = _TeacherVLLMFeedbackManager(
                self.teacher_feedback_rollout.inference_engine,
                enable_sleep_mode=bool(rollout_config.enable_sleep_mode),
            )
            if self.rank == 0:
                print("[StepFeedback] Teacher feedback vLLM rollout is ready.")
            return

        if feedback_backend != "hf":
            raise ValueError(f"Unsupported teacher feedback backend: {feedback_backend}")

        if self.rank == 0:
            print(
                "[StepFeedback] Building teacher feedback HF rollout "
                f"with response_length={response_length}"
            )
        self.teacher_feedback_rollout = HFRollout(
            module=self.ref_module_fsdp,
            config=rollout_config,
        )
        self.teacher_feedback_sharding_manager = None
        if self.rank == 0:
            print("[StepFeedback] Teacher feedback HF rollout is ready.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_teacher_feedback_rollout(self):
        """Release teacher vLLM feedback engine before hidden forward/update."""

        if self.role != "ref":
            return
        if getattr(self, "teacher_feedback_rollout", None) is None:
            return

        engine = getattr(self.teacher_feedback_rollout, "inference_engine", None)
        for target in (engine, getattr(engine, "llm_engine", None)):
            shutdown = getattr(target, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    if self.rank == 0:
                        print(f"[StepFeedback] teacher feedback vLLM shutdown skipped: {exc}")

        self.teacher_feedback_rollout = None
        self.teacher_feedback_sharding_manager = None
        gc.collect()
        get_torch_device().empty_cache()
        if self.rank == 0:
            print("[StepFeedback] Teacher feedback rollout released.")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_oprd_bridge_hidden(self, data: DataProto):
        output = super().compute_ref_oprd_bridge_hidden(data)
        if self._is_ref and self.config.ref.fsdp_config.get("param_offload", False):
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)
            get_torch_device().empty_cache()
            if self.rank == 0:
                print("[StepFeedback] Ref teacher offloaded after OPRD-Bridge hidden forward.")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @torch.no_grad()
    def generate_ref_sequences(self, prompts: DataProto):
        """Generate short natural-language feedback with the teacher model."""

        assert self._is_ref
        if (
            self.config.ref.get("feedback_backend", "hf") == "vllm"
            and getattr(self, "teacher_feedback_rollout", None) is None
        ):
            self._build_teacher_feedback_rollout()
        if getattr(self, "teacher_feedback_rollout", None) is not None:
            prompts = prompts.to(get_torch_device().current_device())
            prompts.meta_info.update(
                {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "do_sample": bool(prompts.meta_info.get("do_sample", True)),
                    "temperature": float(prompts.meta_info.get("temperature", 0.2)),
                    "top_p": float(prompts.meta_info.get("top_p", 0.9)),
                    "top_k": int(prompts.meta_info.get("top_k", -1)),
                }
            )
            if getattr(self, "teacher_feedback_sharding_manager", None) is not None:
                with self.teacher_feedback_sharding_manager:
                    rollout_input = self.teacher_feedback_sharding_manager.preprocess_data(prompts)
                    output = self.teacher_feedback_rollout.generate_sequences(prompts=rollout_input)
                    output = self.teacher_feedback_sharding_manager.postprocess_data(output)
            else:
                output = self.teacher_feedback_rollout.generate_sequences(prompts=prompts)
            get_torch_device().empty_cache()
            return output.to("cpu")

        prompts = prompts.to(get_torch_device().current_device())
        eos_token_id = (
            self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id
        )
        pad_token_id = (
            self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id
        )
        prompts.meta_info.update(
            {
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
            }
        )

        do_sample = bool(prompts.meta_info.get("do_sample", True))
        temperature = float(prompts.meta_info.get("temperature", 0.2))
        response_length = int(prompts.meta_info.get("response_length", 192))
        top_p = float(prompts.meta_info.get("top_p", 0.9))
        top_k = int(prompts.meta_info.get("top_k", -1))

        generation_config = GenerationConfig(
            do_sample=do_sample,
            num_beams=1,
            top_p=top_p,
            top_k=max(0, top_k),
            temperature=temperature,
            num_return_sequences=1,
        )

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        prompt_length = input_ids.size(1)

        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.ref_module_fsdp)

        try:
            self.ref_module_fsdp.eval()
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                output = self.ref_module_fsdp.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
        finally:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.ref_module_fsdp)

        seq = output.sequences
        expected_length = prompt_length + response_length
        if seq.shape[1] < expected_length:
            pad = torch.full(
                (seq.shape[0], expected_length - seq.shape[1]),
                pad_token_id,
                dtype=seq.dtype,
                device=seq.device,
            )
            seq = torch.cat((seq, pad), dim=1)
        elif seq.shape[1] > expected_length:
            seq = seq[:, :expected_length]

        responses = seq[:, prompt_length:]
        batch = TensorDict(
            {
                "responses": responses,
            },
            batch_size=seq.shape[0],
        )
        get_torch_device().empty_cache()
        return DataProto(batch=batch).to("cpu")


class AsyncActorRolloutRefWorker(_BackboneAsyncActorRolloutRefWorker):
    pass
