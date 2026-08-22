"""Compatibility shim for ATOD's stale rollout package import.

ATOD's rollout package imports ``NaiveRollout`` unconditionally, while the
checked-out repository does not contain the implementation. The experiments
in this branch use vLLM, so keep the import lightweight and fail explicitly
only if the unsupported naive backend is selected.
"""

from .base import BaseRollout

__all__ = ["NaiveRollout"]


class NaiveRollout(BaseRollout):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "ATOD's checked-out tree has no NaiveRollout implementation; "
            "use rollout.name=vllm for the OPRD-Bridge experiment."
        )

    def generate_sequences(self, prompts):
        raise NotImplementedError
