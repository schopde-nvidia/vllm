# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Humming experts must normalize each Muse route before its weighted sum."""

import pytest
import torch

from tests.kernels.moe.muse_spark_utils import make_runner, run_moe, swiglu_moe_ref
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingIndexedExperts,
)
from vllm.model_executor.layers.quantization.humming import HummingConfig
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNormWithInputScale
from vllm.utils.humming import HummingWeightSchema


@pytest.mark.parametrize("down_scale", [1.0, 1e-4])
@torch.inference_mode()
def test_humming_applies_post_expert_norm_before_weighted_sum(
    dist_init, workspace_init, down_scale
):
    torch.manual_seed(29)
    experts, hidden, intermediate, topk = 8, 512, 256, 2
    schema = HummingWeightSchema(b_dtype="int3", weight_scale_group_size=128)
    checkpoint, decoded = {}, {}
    for prefix, shape in (
        ("w13", (experts, 2 * intermediate, hidden)),
        ("w2", (experts, hidden, intermediate)),
    ):
        weight = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 10
        if prefix == "w2":
            weight.mul_(down_scale)
        tensors = schema.quant_tensor(weight, schema, torch.bfloat16)
        decoded[prefix] = schema.dequant_tensors(tensors)
        checkpoint.update({f"{prefix}_{key}": value for key, value in tensors.items()})

    quant = HummingConfig.from_config(
        {
            "quant_method": "humming",
            "weight_dtype": "int3",
            "input_dtype": "bfloat16",
            "weight_scale_group_size": 128,
        }
    )
    norm = MuseSpark1xRMSNormWithInputScale(
        hidden,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    ).cuda()
    norm.weight.copy_(torch.linspace(-0.2, 0.2, hidden, device="cuda"))
    bias = torch.linspace(-0.5, 0.5, experts, device="cuda")
    torch.accelerator.synchronize()
    allocated_before = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    router, runner, config = make_runner(
        norm,
        bias,
        checkpoint,
        quant,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=topk,
    )
    torch.accelerator.synchronize()
    preparation_peak = torch.accelerator.max_memory_allocated() - allocated_before
    checkpoint_bytes = sum(tensor.nbytes for tensor in checkpoint.values())
    # Allow packed parameter copies and repacking, but not a full INT32 expansion.
    assert preparation_peak < 4 * checkpoint_bytes
    kernel = runner.routed_experts.quant_method.moe_kernel.fused_experts
    assert type(kernel) is HummingIndexedExperts
    x = torch.randn(17, hidden, dtype=torch.bfloat16, device="cuda") / 10
    logits = torch.randn(17, experts, device="cuda")
    actual = run_moe(runner, config, x, logits)
    weights, ids = router._compute_routing(x, logits, None)
    expected = swiglu_moe_ref(
        x,
        decoded["w13"],
        decoded["w2"],
        weights,
        ids,
        lambda rows: norm.forward_native(rows, out_dtype=torch.float32),
    )
    assert torch.isfinite(actual).all()
    assert (actual.float() - expected).norm() / expected.norm() < 0.02
