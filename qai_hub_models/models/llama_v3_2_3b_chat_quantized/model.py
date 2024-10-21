# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

from qai_hub_models.models._shared.llama3.model import (
    DEFAULT_CONTEXT_LENGTH,
    Llama3Base_Quantized,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_ENCODINGS = "llama32.encodings"
DEFAULT_ENCODINGS_ZIP = DEFAULT_ENCODINGS + ".zip"

NUM_LAYERS = 28
NUM_SPLITS = 3
NUM_LAYERS_PER_SPLIT = 14

# Hugging face repo name and url
HF_REPO_NAME = "meta-llama/Llama-3.2-3B-Instruct"
HF_REPO_URL = f"https://huggingface.co/meta-llama/{HF_REPO_NAME}"

# Minimum memory (RAM+swap) recommended for export.
# TODO: #10762 should reduce once AIMET export consumes less memory during export.   TODO!!! Not quite correct, since we are not using AIMET
MIN_MEMORY_RECOMMENDED = 40  # TODO: Does this work for Llama 3?


class Llama3_2_Quantized(Llama3Base_Quantized):
    def __init__(self, huggingface_model_name: str = HF_REPO_NAME, *args, **kwargs):
        super().__init__(
            huggingface_model_name=huggingface_model_name,
            min_memory_recommended=MIN_MEMORY_RECOMMENDED,
            *args,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        sequence_length: int,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        aimet_encodings: str | None = "DEFAULT",
        huggingface_model_name: str = HF_REPO_NAME,
    ) -> "Llama3_2_Quantized":
        """
        Load a pre-trained Llama 3.2 (3B) model from Meta via HuggingFace.

        sequence_length:
            Instantiate with this token sequence length input. A longer
            sequence length means the model is capable of processing more
            tokens at once. This can only be set to greater than one to process
            prompts, since responses are auto-regressive in nature and require
            this to be 1.
        context_length:
            Total context length of model. Longer context length means the
            model is more capable of making longer connections in the input
            prompt. However, it also hurts runtime performance (both time-to-
            first-token and tokens-per-second), so this is a tradeoff that may
            depend on the use case.
        aimet_encodings:
            Path to AIMET quantization encodings file.
        huggingface_model_name:
            Name or URL of the HuggingFace model. Change this if you want to
            change the weights.
        """
        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = os.path.join(
                    CachedWebModelAsset.from_asset_store(
                        MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS_ZIP
                    ).fetch(extract=True),
                    DEFAULT_ENCODINGS,
                )

        return cls(
            aimet_encodings=aimet_encodings,
            sequence_length=sequence_length,
            context_length=context_length,
            huggingface_model_name=huggingface_model_name,
        )

    @staticmethod
    def get_output_names(num_hidden_layers: int = NUM_LAYERS):
        return Llama3Base_Quantized.get_output_names(
            num_hidden_layers=num_hidden_layers
        )

    @staticmethod
    def get_input_spec(
        num_hidden_layers: int = NUM_LAYERS,
        input_seq_length: int = 128,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        hidden_size: int = 3072,
        num_key_value_heads: int = 8,
        num_attention_heads: int = 24,
    ) -> InputSpec:
        return Llama3Base_Quantized.get_input_spec(
            num_hidden_layers=NUM_LAYERS,
            input_seq_length=input_seq_length,
            context_length=context_length,
            hidden_size=hidden_size,
            num_key_value_heads=num_key_value_heads,
            num_attention_heads=num_attention_heads,
        )