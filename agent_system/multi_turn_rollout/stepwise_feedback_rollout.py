"""Step-wise teacher-feedback rollout for ALFWorld OPRD-Bridge probes.

Each active turn is collected as:
student original response -> teacher feedback -> student refined response ->
environment step with the refined response.  Downstream training sees the
refined response in the normal ``responses`` tensor, so existing OPRD-Bridge
hidden loss is computed on the executed turn.
"""

from __future__ import annotations

import json
import re
import sys
import time
import uuid
from typing import Dict, List

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy
from torch import Tensor
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask


def _extract_between(text: str, left: str, right: str = "\n") -> str:
    if left not in text:
        return ""
    segment = text.split(left, 1)[1]
    if right and right in segment:
        segment = segment.split(right, 1)[0]
    return segment.strip()


def _extract_after_until(text: str, left: str, stops: List[str]) -> str:
    if left not in text:
        return ""
    segment = text.split(left, 1)[1]
    stop_positions = [segment.find(stop) for stop in stops if stop in segment]
    if stop_positions:
        segment = segment[: min(pos for pos in stop_positions if pos >= 0)]
    return segment.strip()


TEACHER_STEP_FEEDBACK_TEMPLATE = """You are a teacher model giving one round of actionable feedback to a student ALFWorld agent.

You will inspect ONLY the student's proposed response for the current step.

Current step context:
{current_step_context}

Student proposed response:
{student_original_response}

Generate feedback that helps the student rewrite this current step.

Your feedback should include:
1. whether the proposed action is executable in the current admissible actions,
2. whether the proposed action is appropriate for the current observation,
3. a high-level hint for revising the current step.

The current admissible actions are the only actions executable at this step.
When judging executability, compare the proposed action by exact string match against Actions.
Plausible ALFWorld commands that are not written verbatim in Actions are not executable.
If the proposed action is not an exact string match in Actions, treat that as a mistake.
The action hint should guide the student where to focus, but should not decide the final action for the student.
Keep each field to one short sentence.

Output exactly this format:
Mistake: <the main issue, or "None." only when the proposed action exactly matches Actions and is appropriate>
Reason: <whether the proposed action exactly matches Actions, and whether it fits the current observation>
Action hint: <a high-level hint for choosing from the current admissible actions, not a full action command>
"""


STUDENT_STEP_REGENERATION_TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment.

You are revising your response for ONLY the current step using the teacher feedback.

The previous response has not been executed. Only your rewritten response will be executed.

Current step context:
{current_step_context}

Your previous response:
{student_original_response}

Teacher feedback:
{teacher_feedback}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation, using the teacher feedback to correct mistakes in your previous response while preserving useful reasoning when appropriate.
Your final action must be copied exactly from Actions; do not invent, paraphrase, or add objects/prepositions.
If the teacher says your previous action does not exactly match Actions, choose another action from Actions.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


class StepwiseFeedbackTrajectoryCollector(TrajectoryCollector):
    """Trajectory collector that executes refined step responses."""

    @staticmethod
    def _extract_line_containing(text: str, marker: str) -> str:
        for line in text.splitlines():
            if marker in line:
                return line.strip()
        return ""

    def _compact_step_context(self, prompt_text: str) -> str:
        task_description = _extract_between(prompt_text, "Your task is to:", "\n")
        recent_history = _extract_after_until(
            prompt_text,
            "observations and the corresponding actions you took:",
            ["\nYou are now at step"],
        )
        current_step_line = self._extract_line_containing(prompt_text, "You are now at step")
        current_observation = _extract_between(prompt_text, "your current observation is:", "\n")
        admissible_actions = _extract_after_until(
            prompt_text,
            "Your admissible actions of the current situation are:",
            ["\n\nNow it's your turn", "\nNow it's your turn"],
        ).strip()
        if task_description:
            task_line = f"Task: {task_description}"
        else:
            task_line = ""
        history_line = f"Recent history: {recent_history}" if recent_history else ""
        if current_step_line:
            obs_line = current_step_line
        else:
            obs_line = f"Observation: {current_observation}" if current_observation else ""
        actions_line = f"Actions: {admissible_actions}" if admissible_actions else ""
        return "\n".join(line for line in [task_line, history_line, obs_line, actions_line] if line)

    def _debug_log(self, message: str) -> None:
        feedback_cfg = self.config.algorithm.sod.get("stepwise_feedback", {})
        if bool(feedback_cfg.get("debug", False)):
            print(f"[StepwiseFeedbackDebug] {message}", flush=True, file=sys.stderr)

    @staticmethod
    def _clip_text(text, max_chars: int) -> str:
        text = "" if text is None else str(text)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...<truncated>"

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): StepwiseFeedbackTrajectoryCollector._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [StepwiseFeedbackTrajectoryCollector._json_safe(v) for v in value]
        return value

    @staticmethod
    def _selected_info(info: Dict) -> Dict:
        selected = {}
        preferred_keys = {
            "action",
            "executed_action",
            "is_action_valid",
            "tool_calling",
            "won",
            "done",
            "success",
            "score",
            "reward",
        }
        for key, value in info.items():
            key_str = str(key)
            if key_str in preferred_keys or "action" in key_str:
                selected[key_str] = StepwiseFeedbackTrajectoryCollector._json_safe(value)
        return selected

    @staticmethod
    def _extract_action_tag(response: str) -> str:
        text = "" if response is None else str(response)
        lowered = text.lower()
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lowered.find(start_tag)
        end_idx = lowered.find(end_tag)
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return ""
        return lowered[start_idx + len(start_tag) : end_idx].strip()

    @staticmethod
    def _extract_admissible_actions(prompt_text: str) -> List[str]:
        action_block = _extract_after_until(
            prompt_text,
            "Your admissible actions of the current situation are:",
            ["\n\nNow it's your turn", "\nNow it's your turn"],
        )
        return [
            match.group(1).strip().lower()
            for match in re.finditer(r"'([^']+)'", action_block)
            if match.group(1).strip().lower() != "help"
        ]

    def _action_quality(self, prompt_text: str, response: str) -> Dict:
        parsed_action = self._extract_action_tag(response)
        admissible_actions = self._extract_admissible_actions(prompt_text)
        admissible_set = set(admissible_actions)
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", "" if response is None else str(response)))
        action_tag_found = bool(parsed_action)
        action_in_admissible = action_tag_found and parsed_action in admissible_set
        return {
            "parsed_action": parsed_action,
            "action_tag_found": action_tag_found,
            "action_in_admissible": action_in_admissible,
            "action_no_think_valid": action_tag_found and action_in_admissible and not has_chinese,
            "action_has_chinese": has_chinese,
            "admissible_action_count": len(admissible_actions),
        }

    def _teacher_mentioned_action_quality(self, prompt_text: str, feedback: str) -> Dict:
        admissible_actions = self._extract_admissible_actions(prompt_text)
        admissible_set = set(admissible_actions)
        text = "" if feedback is None else str(feedback)
        mentioned_actions = []
        for match in re.finditer(r'"([^"]+)"|<action>(.*?)</action>', text, flags=re.IGNORECASE | re.DOTALL):
            action = (match.group(1) or match.group(2) or "").strip().lower()
            if action and action not in mentioned_actions:
                mentioned_actions.append(action)
        in_admissible = [action in admissible_set for action in mentioned_actions]
        return {
            "teacher_mentioned_actions": mentioned_actions,
            "teacher_mentioned_action_found": bool(mentioned_actions),
            "teacher_mentioned_action_in_admissible": bool(mentioned_actions) and any(in_admissible),
            "teacher_mentioned_action_all_in_admissible": bool(mentioned_actions) and all(in_admissible),
            "teacher_mentioned_action_count": len(mentioned_actions),
        }

    def _should_log_sample_turn(self, turn: int, feedback_cfg: Dict) -> bool:
        if not bool(feedback_cfg.get("log_samples", False)):
            return False
        turns = feedback_cfg.get("log_sample_turns", None)
        if turns is not None:
            return turn in {int(item) for item in turns}
        every = int(feedback_cfg.get("log_sample_every", 0) or 0)
        return every > 0 and turn % every == 0

    def _log_quality_samples(
        self,
        *,
        turn: int,
        active_indices,
        current_step_prompts: List[str],
        original_responses: List[str],
        teacher_feedbacks: List[str],
        refined_responses: List[str],
        rewards,
        dones,
        infos: List[Dict],
    ) -> None:
        feedback_cfg = self.config.algorithm.sod.get("stepwise_feedback", {})
        if not self._should_log_sample_turn(turn, feedback_cfg):
            return

        sample_count = int(feedback_cfg.get("log_sample_count", 2) or 2)
        max_chars = int(feedback_cfg.get("log_sample_max_chars", 1200) or 1200)
        rewards_np = torch_to_numpy(rewards)
        for local_idx in range(min(sample_count, len(refined_responses))):
            global_idx = int(active_indices[local_idx])
            original_action_quality = self._action_quality(
                current_step_prompts[local_idx],
                original_responses[local_idx],
            )
            refined_action_quality = self._action_quality(
                current_step_prompts[local_idx],
                refined_responses[local_idx],
            )
            teacher_action_quality = self._teacher_mentioned_action_quality(
                current_step_prompts[local_idx],
                teacher_feedbacks[local_idx],
            )
            sample = {
                "turn": int(turn),
                "env_index": global_idx,
                "context": self._clip_text(
                    self._compact_step_context(current_step_prompts[local_idx]),
                    max_chars,
                ),
                "student_original_response": self._clip_text(
                    original_responses[local_idx],
                    max_chars,
                ),
                "teacher_feedback": self._clip_text(
                    teacher_feedbacks[local_idx],
                    max_chars,
                ),
                "refined_response": self._clip_text(
                    refined_responses[local_idx],
                    max_chars,
                ),
                "reward": self._json_safe(rewards_np[global_idx]),
                "done": self._json_safe(dones[global_idx]),
                "info": self._selected_info(infos[global_idx]),
                "original_action_parse": self._json_safe(original_action_quality),
                "teacher_action_parse": self._json_safe(teacher_action_quality),
                "action_parse": self._json_safe(refined_action_quality),
            }
            print(
                "[StepwiseFeedbackSample] "
                + json.dumps(sample, ensure_ascii=False, default=str),
                flush=True,
                file=sys.stderr,
            )

    def _log_action_stats(
        self,
        turn: int,
        active_indices,
        action_quality: List[Dict],
        label: str = "refined",
    ) -> None:
        feedback_cfg = self.config.algorithm.sod.get("stepwise_feedback", {})
        if not bool(feedback_cfg.get("log_action_stats", True)):
            return
        if not action_quality:
            return
        tag_found = np.array([item["action_tag_found"] for item in action_quality], dtype=np.float32)
        in_admissible = np.array(
            [item["action_in_admissible"] for item in action_quality],
            dtype=np.float32,
        )
        no_think_valid = np.array(
            [item["action_no_think_valid"] for item in action_quality],
            dtype=np.float32,
        )
        stats = {
            "turn": int(turn),
            "label": label,
            "active_count": int(len(active_indices)),
            "action_tag_rate": float(tag_found.mean()),
            "action_in_admissible_rate": float(in_admissible.mean()),
            "action_no_think_valid_rate": float(no_think_valid.mean()),
        }
        print(
            "[StepwiseFeedbackActionStats] "
            + json.dumps(stats, ensure_ascii=False, default=str),
            flush=True,
            file=sys.stderr,
        )

    def _log_teacher_action_stats(self, turn: int, active_indices, teacher_quality: List[Dict]) -> None:
        feedback_cfg = self.config.algorithm.sod.get("stepwise_feedback", {})
        if not bool(feedback_cfg.get("log_action_stats", True)):
            return
        if not teacher_quality:
            return
        found = np.array(
            [item["teacher_mentioned_action_found"] for item in teacher_quality],
            dtype=np.float32,
        )
        any_in_admissible = np.array(
            [item["teacher_mentioned_action_in_admissible"] for item in teacher_quality],
            dtype=np.float32,
        )
        all_in_admissible = np.array(
            [item["teacher_mentioned_action_all_in_admissible"] for item in teacher_quality],
            dtype=np.float32,
        )
        stats = {
            "turn": int(turn),
            "label": "teacher_feedback",
            "active_count": int(len(active_indices)),
            "teacher_mentioned_action_rate": float(found.mean()),
            "teacher_mentioned_action_in_admissible_rate": float(any_in_admissible.mean()),
            "teacher_mentioned_action_all_in_admissible_rate": float(all_in_admissible.mean()),
        }
        print(
            "[StepwiseFeedbackActionStats] "
            + json.dumps(stats, ensure_ascii=False, default=str),
            flush=True,
            file=sys.stderr,
        )

    @staticmethod
    def _meta_with_optional_response_length(meta_info: Dict, response_length) -> Dict:
        meta = dict(meta_info)
        if response_length is not None:
            response_length = int(response_length)
            if response_length > 0:
                meta["response_length"] = response_length
        return meta

    def _make_prompt_batch(
        self,
        prompt_texts: List[str],
        *,
        data_sources: np.ndarray | List[object],
        meta_info: Dict,
    ) -> DataProto:
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        processed_samples = []
        for item, prompt_text in enumerate(prompt_texts):
            chat = np.array([{"role": "user", "content": prompt_text}])
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            raw_prompt_ids = self.tokenizer.encode(
                prompt_with_chat_template,
                add_special_tokens=False,
            )
            if len(raw_prompt_ids) > self.config.data.max_prompt_length:
                if self.config.data.truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
                elif self.config.data.truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
                elif self.config.data.truncation == "middle":
                    left_half = self.config.data.max_prompt_length // 2
                    right_half = self.config.data.max_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif self.config.data.truncation == "error":
                    raise RuntimeError(
                        f"Prompt length {len(raw_prompt_ids)} is longer than "
                        f"{self.config.data.max_prompt_length}."
                    )

            row = {
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": compute_position_id_with_mask(attention_mask)[0],
                "raw_prompt_ids": raw_prompt_ids,
                "index": item,
                "data_source": data_sources[item],
            }
            if self.config.data.get("return_raw_chat", False):
                row["raw_prompt"] = chat.tolist()
            processed_samples.append(row)

        return DataProto.from_single_dict(
            data=collate_fn(processed_samples),
            meta_info=meta_info,
        )

    def _generate_with_worker(
        self,
        *,
        batch_input: DataProto,
        worker_group,
        method_name: str,
        reuse_rollout_weights: bool,
    ) -> DataProto:
        batch_input_padded, pad_size = pad_dataproto_to_divisor(
            batch_input,
            worker_group.world_size,
        )
        if method_name == "actor":
            if reuse_rollout_weights:
                output_padded = worker_group.generate_sequences_in_rollout(batch_input_padded)
            else:
                output_padded = worker_group.generate_sequences(batch_input_padded)
        elif method_name == "ref":
            output_padded = worker_group.generate_ref_sequences(batch_input_padded)
        else:
            raise ValueError(f"Unsupported generation method: {method_name}")
        return unpad_dataproto(output_padded, pad_size=pad_size)

    @staticmethod
    def _gather_by_indices(values, indices):
        return [values[i] for i in indices]

    @staticmethod
    def _scatter_to_full_batch(batch_size: int, active_indices, active_values, pad_value):
        full_values = [pad_value] * batch_size
        for idx, value in zip(active_indices, active_values):
            full_values[idx] = value
        return full_values

    def _build_teacher_feedback_prompts(
        self,
        *,
        current_step_prompts: List[str],
        original_responses: List[str],
    ) -> List[str]:
        return [
            TEACHER_STEP_FEEDBACK_TEMPLATE.format(
                current_step_context=self._compact_step_context(prompt),
                student_original_response=response.strip(),
            )
            for prompt, response in zip(current_step_prompts, original_responses)
        ]

    def _build_regeneration_prompts(
        self,
        *,
        current_step_prompts: List[str],
        original_responses: List[str],
        teacher_feedbacks: List[str],
    ) -> List[str]:
        return [
            STUDENT_STEP_REGENERATION_TEMPLATE.format(
                current_step_context=self._compact_step_context(prompt),
                student_original_response=response.strip(),
                teacher_feedback=feedback.strip(),
            )
            for prompt, response, feedback in zip(
                current_step_prompts,
                original_responses,
                teacher_feedbacks,
            )
        ]

    def stepwise_feedback_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        ref_policy_wg,
        envs: EnvironmentManagerBase,
    ):
        batch_size = len(gen_batch.batch)
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop("env_kwargs", None))

        obs_len = len(obs["text"]) if obs["text"] is not None else len(obs["image"])
        assert batch_size == obs_len, f"gen_batch size {batch_size} does not match obs size {obs_len}"

        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(batch_size)], dtype=object)

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        feedback_cfg = self.config.algorithm.sod.get("stepwise_feedback", {})
        student_max_tokens = feedback_cfg.get("student_max_tokens", None)
        student_generation_meta = self._meta_with_optional_response_length(
            gen_batch.meta_info,
            student_max_tokens,
        )

        reuse_weights = self.config.actor_rollout_ref.rollout.get("reuse_weights_across_turns", False)
        if reuse_weights:
            actor_rollout_wg.rollout_mode_enter()
        try:
            self._debug_log(
                "limits "
                f"student_max_tokens={student_max_tokens} "
                f"teacher_max_tokens={feedback_cfg.get('teacher_max_tokens', 192)}"
            )
            for _step in range(self.config.env.max_steps):
                active_masks = np.logical_not(is_done)
                active_indices = np.flatnonzero(active_masks)
                self._debug_log(
                    f"turn={_step} active={int(active_masks.sum())}/{batch_size} "
                    f"done={int(is_done.sum())}/{batch_size}"
                )
                if active_indices.size == 0:
                    self._debug_log(f"turn={_step} all_done_break")
                    break
                current_step_prompts = self._gather_by_indices(obs["text"], active_indices)
                data_sources = gen_batch.non_tensor_batch["data_source"][active_indices]

                original_prompt_batch = self._make_prompt_batch(
                    current_step_prompts,
                    data_sources=data_sources,
                    meta_info=gen_batch.meta_info,
                )
                original_input = original_prompt_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "raw_prompt"]
                    if "raw_prompt" in original_prompt_batch.non_tensor_batch
                    else ["raw_prompt_ids"],
                )
                original_input.meta_info = student_generation_meta
                stage_t0 = time.perf_counter()
                original_output = self._generate_with_worker(
                    batch_input=original_input,
                    worker_group=actor_rollout_wg,
                    method_name="actor",
                    reuse_rollout_weights=reuse_weights,
                )
                original_s = time.perf_counter() - stage_t0
                self._debug_log(f"turn={_step} original_generation_done elapsed_s={original_s:.3f}")
                original_responses = self.tokenizer.batch_decode(
                    original_output.batch["responses"],
                    skip_special_tokens=True,
                )
                assert len(original_responses) == active_indices.size, (
                    f"original response count {len(original_responses)} does not match "
                    f"active env count {active_indices.size}"
                )

                teacher_prompt_batch = self._make_prompt_batch(
                    self._build_teacher_feedback_prompts(
                        current_step_prompts=current_step_prompts,
                        original_responses=original_responses,
                    ),
                    data_sources=data_sources,
                    meta_info={
                        **gen_batch.meta_info,
                        "response_length": int(feedback_cfg.get("teacher_max_tokens", 192)),
                        "temperature": float(feedback_cfg.get("teacher_temperature", 0.2)),
                        "top_p": float(feedback_cfg.get("teacher_top_p", 0.9)),
                        "top_k": int(feedback_cfg.get("teacher_top_k", -1)),
                        "do_sample": bool(feedback_cfg.get("teacher_do_sample", True)),
                    },
                )
                teacher_input = teacher_prompt_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "raw_prompt"]
                    if "raw_prompt" in teacher_prompt_batch.non_tensor_batch
                    else ["raw_prompt_ids"],
                )
                teacher_input.meta_info = teacher_prompt_batch.meta_info
                stage_t0 = time.perf_counter()
                teacher_feedback_output = self._generate_with_worker(
                    batch_input=teacher_input,
                    worker_group=ref_policy_wg,
                    method_name="ref",
                    reuse_rollout_weights=False,
                )
                teacher_s = time.perf_counter() - stage_t0
                self._debug_log(f"turn={_step} teacher_feedback_done elapsed_s={teacher_s:.3f}")
                teacher_feedbacks = self.tokenizer.batch_decode(
                    teacher_feedback_output.batch["responses"],
                    skip_special_tokens=True,
                )
                assert len(teacher_feedbacks) == active_indices.size, (
                    f"teacher feedback count {len(teacher_feedbacks)} does not match "
                    f"active env count {active_indices.size}"
                )

                regeneration_prompt_batch = self._make_prompt_batch(
                    self._build_regeneration_prompts(
                        current_step_prompts=current_step_prompts,
                        original_responses=original_responses,
                        teacher_feedbacks=teacher_feedbacks,
                    ),
                    data_sources=data_sources,
                    meta_info=gen_batch.meta_info,
                )
                regen_input = regeneration_prompt_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "raw_prompt"]
                    if "raw_prompt" in regeneration_prompt_batch.non_tensor_batch
                    else ["raw_prompt_ids"],
                )
                regen_input.meta_info = student_generation_meta
                stage_t0 = time.perf_counter()
                refined_output = self._generate_with_worker(
                    batch_input=regen_input,
                    worker_group=actor_rollout_wg,
                    method_name="actor",
                    reuse_rollout_weights=reuse_weights,
                )
                regen_s = time.perf_counter() - stage_t0
                self._debug_log(f"turn={_step} regeneration_done elapsed_s={regen_s:.3f}")

                refined_batch = regeneration_prompt_batch.union(refined_output)
                refined_responses = self.tokenizer.batch_decode(
                    refined_batch.batch["responses"],
                    skip_special_tokens=True,
                )
                assert len(refined_responses) == active_indices.size, (
                    f"refined response count {len(refined_responses)} does not match "
                    f"active env count {active_indices.size}"
                )
                original_action_quality = [
                    self._action_quality(prompt, response)
                    for prompt, response in zip(current_step_prompts, original_responses)
                ]
                teacher_action_quality = [
                    self._teacher_mentioned_action_quality(prompt, feedback)
                    for prompt, feedback in zip(current_step_prompts, teacher_feedbacks)
                ]
                action_quality = [
                    self._action_quality(prompt, response)
                    for prompt, response in zip(current_step_prompts, refined_responses)
                ]
                self._log_action_stats(
                    turn=_step,
                    active_indices=active_indices,
                    action_quality=original_action_quality,
                    label="original",
                )
                self._log_teacher_action_stats(
                    turn=_step,
                    active_indices=active_indices,
                    teacher_quality=teacher_action_quality,
                )
                self._log_action_stats(
                    turn=_step,
                    active_indices=active_indices,
                    action_quality=action_quality,
                    label="refined",
                )
                full_refined_responses = self._scatter_to_full_batch(
                    batch_size=batch_size,
                    active_indices=active_indices,
                    active_values=refined_responses,
                    pad_value="pass",
                )
                stage_t0 = time.perf_counter()
                next_obs, rewards, dones, infos = envs.step(full_refined_responses)
                env_s = time.perf_counter() - stage_t0
                self._debug_log(f"turn={_step} env_step_done elapsed_s={env_s:.3f}")

                if len(rewards.shape) == 2:
                    rewards = rewards.squeeze(1)
                if len(dones.shape) == 2:
                    dones = dones.squeeze(1)
                assert len(rewards) == batch_size, (
                    f"env should return rewards for all environments, got {len(rewards)} "
                    f"rewards for {batch_size} environments"
                )
                assert len(dones) == batch_size, (
                    f"env should return dones for all environments, got {len(dones)} "
                    f"dones for {batch_size} environments"
                )
                assert len(infos) == batch_size, (
                    f"env should return infos for all environments, got {len(infos)} "
                    f"infos for {batch_size} environments"
                )
                self._debug_log(
                    f"turn={_step} dones_after_step={int(np.asarray(dones).sum())}/{batch_size}"
                )
                self._log_quality_samples(
                    turn=_step,
                    active_indices=active_indices,
                    current_step_prompts=current_step_prompts,
                    original_responses=original_responses,
                    teacher_feedbacks=teacher_feedbacks,
                    refined_responses=refined_responses,
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                )

                refined_batch.non_tensor_batch["uid"] = uid_batch[active_indices]
                refined_batch.non_tensor_batch["traj_uid"] = traj_uid[active_indices]
                refined_batch.non_tensor_batch["student_original_response"] = np.array(
                    original_responses,
                    dtype=object,
                )
                refined_batch.non_tensor_batch["teacher_feedback"] = np.array(
                    teacher_feedbacks,
                    dtype=object,
                )
                refined_batch.non_tensor_batch["refined_response"] = np.array(
                    refined_responses,
                    dtype=object,
                )
                refined_batch.non_tensor_batch["original_parsed_action"] = np.array(
                    [item["parsed_action"] for item in original_action_quality],
                    dtype=object,
                )
                refined_batch.non_tensor_batch["original_action_tag_found"] = np.array(
                    [item["action_tag_found"] for item in original_action_quality],
                    dtype=bool,
                )
                refined_batch.non_tensor_batch["original_action_in_admissible"] = np.array(
                    [item["action_in_admissible"] for item in original_action_quality],
                    dtype=bool,
                )
                refined_batch.non_tensor_batch["original_action_no_think_valid"] = np.array(
                    [item["action_no_think_valid"] for item in original_action_quality],
                    dtype=bool,
                )
                refined_batch.non_tensor_batch["teacher_mentioned_action_found"] = np.array(
                    [item["teacher_mentioned_action_found"] for item in teacher_action_quality],
                    dtype=bool,
                )
                refined_batch.non_tensor_batch["teacher_mentioned_action_in_admissible"] = np.array(
                    [item["teacher_mentioned_action_in_admissible"] for item in teacher_action_quality],
                    dtype=bool,
                )
                refined_batch.non_tensor_batch["teacher_mentioned_action_all_in_admissible"] = np.array(
                    [item["teacher_mentioned_action_all_in_admissible"] for item in teacher_action_quality],
                    dtype=bool,
                )
                refined_batch.non_tensor_batch["parsed_action"] = np.array(
                    [item["parsed_action"] for item in action_quality],
                    dtype=object,
                )
                refined_batch.non_tensor_batch["action_tag_found"] = np.array(
                    [item["action_tag_found"] for item in action_quality],
                    dtype=bool,
                )
                refined_batch.non_tensor_batch["action_in_admissible"] = np.array(
                    [item["action_in_admissible"] for item in action_quality],
                    dtype=bool,
                )
                refined_batch.non_tensor_batch["action_no_think_valid"] = np.array(
                    [item["action_no_think_valid"] for item in action_quality],
                    dtype=bool,
                )
                if "is_action_valid" in infos[0]:
                    refined_batch.non_tensor_batch["is_action_valid"] = np.array(
                        [infos[i]["is_action_valid"] for i in active_indices],
                        dtype=bool,
                    )
                else:
                    refined_batch.non_tensor_batch["is_action_valid"] = np.ones(
                        active_indices.size,
                        dtype=bool,
                    )
                if "tool_calling" in infos[0]:
                    tool_callings[active_indices] += np.array(
                        [infos[i]["tool_calling"] for i in active_indices],
                        dtype=np.float32,
                    )

                episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
                episode_lengths[active_masks] += 1
                active_rewards = torch_to_numpy(rewards)[active_masks]
                refined_batch.non_tensor_batch["rewards"] = torch_to_numpy(
                    active_rewards,
                    is_object=True,
                )
                refined_batch.non_tensor_batch["active_masks"] = np.ones(
                    active_indices.size,
                    dtype=bool,
                )

                batch_list: list[dict] = to_list_of_dict(refined_batch)
                for local_idx, global_idx in enumerate(active_indices):
                    total_batch_list[global_idx].append(batch_list[local_idx])
                    total_infos[global_idx].append(infos[global_idx])

                is_done = np.logical_or(is_done, dones)
                obs = next_obs
                if is_done.all():
                    self._debug_log(f"turn={_step} all_done_break")
                    break
        finally:
            if reuse_weights:
                actor_rollout_wg.rollout_mode_exit()

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
        ref_policy_wg=None,
    ) -> DataProto:
        if not is_train:
            return super().multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
            )
        if ref_policy_wg is None:
            raise ValueError("StepwiseFeedbackTrajectoryCollector requires ref_policy_wg for teacher feedback.")

        gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        (
            total_batch_list,
            total_episode_rewards,
            total_episode_lengths,
            total_success,
            total_traj_uid,
            total_tool_callings,
        ) = self.stepwise_feedback_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            ref_policy_wg=ref_policy_wg,
            envs=envs,
        )

        return self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=total_tool_callings,
        )
