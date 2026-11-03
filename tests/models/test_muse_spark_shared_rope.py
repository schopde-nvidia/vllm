# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.muse_spark.amd.model as amd_model


def test_target_layers_share_fp32_rope_cache(monkeypatch, default_vllm_config):
    monkeypatch.setattr(amd_model, "get_tensor_model_parallel_world_size", lambda: 1)
    for name in ("MuseSpark1xQKVOGateParallelLinear", "RowParallelLinear", "Attention"):
        monkeypatch.setattr(amd_model, name, lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(amd_model, "supports_fused_kv_insert", lambda attn: True)
    config = SimpleNamespace(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        layer_types=["sliding_attention"] * 2,
        layer_rope_theta=[5e5] * 2,
        sliding_window=2048,
        rms_norm_eps=1e-5,
        qk_scale_factor=1.0,
    )
    first, second = (
        amd_model.MuseSpark1xAttention(
            config=config,
            max_position_embeddings=4096,
            prefix=f"model.layers.{i}.self_attn",
        )
        for i in range(2)
    )

    assert first.rotary_emb is second.rotary_emb
    assert first.rotary_emb.cos_sin_cache.dtype == torch.float32


@pytest.mark.parametrize("backend", [None, "TRITON_ATTN"])
def test_nvidia_rope_cache_sharing_preserves_fp32_values(dist_init, backend):
    """Share read-only fused caches; keep dtype-mutating fallback caches private."""
    from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm.models.muse_spark.nvidia.model import MuseSpark1xAttention
    from vllm.transformers_utils.configs.muse_spark import MuseSparkTextConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    config = MuseSparkTextConfig(
        hidden_act="silu",
        attention_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 5e5},
        num_shared_experts=0,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        layer_types=["sliding_attention"] * 2,
        mlp_layer_types=["sparse"] * 2,
        layer_rope_theta=[5e5] * 2,
        sliding_window=2048,
        rms_norm_eps=1e-5,
        qk_scale_factor=1.0,
    )
    vllm_config = VllmConfig(attention_config=AttentionConfig(backend=backend))
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        first, second = (
            MuseSpark1xAttention(
                config,
                max_position_embeddings=4096,
                prefix=f"model.layers.{i}.self_attn",
            )
            for i in range(2)
        )
        reference = RotaryEmbedding(64, 64, 4096, 5e5, False, torch.float32)

    first_cache = first.rotary_emb.cos_sin_cache
    second_cache = second.rotary_emb.cos_sin_cache
    assert first_cache.dtype == second_cache.dtype == torch.float32
    assert torch.equal(first_cache, reference.cos_sin_cache)
    assert torch.equal(second_cache, reference.cos_sin_cache)
    shared = first_cache.data_ptr() == second_cache.data_ptr()
    assert shared == first.fused_qk_norm_rope_kv_insert
    if backend == "TRITON_ATTN":
        assert not shared
        positions = torch.tensor([0, 127, 4095], device="cuda")
        q = torch.ones(3, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.ones(3, 64, dtype=torch.bfloat16, device="cuda")
        first.rotary_emb.forward_cuda(positions, q, k)
        assert first.rotary_emb.cos_sin_cache.dtype == torch.bfloat16
        assert second.rotary_emb.cos_sin_cache.dtype == torch.float32
