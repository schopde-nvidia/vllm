# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x Muse experts normalize individual routes before their weighted sum."""

import pytest
import torch

from tests.kernels.moe.muse_spark_utils import run_moe, swiglu_moe_ref
from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNormWithInputScale
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MuseMoERunner,
    MuseSpark1xRouter,
    MuseSparkRoutedExperts,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="The block-quantized Muse backend requires SM12x",
)


@pytest.mark.parametrize("codec", ["IQ2_XS", "BF16"])
@pytest.mark.parametrize("down_scale", [1 / 128, 1 / 65536])
@torch.inference_mode()
def test_muse_block_experts_normalize_before_weighted_sum(
    dist_init, workspace_init, codec, down_scale
):
    """The smaller down scale also exercises RMSNorm's non-negligible epsilon."""
    from b12x.testing.iq2_xs_reference import dequantize_blocks

    torch.manual_seed(29)
    experts, hidden, intermediate, topk = 4, 512, 1024, 2
    packed, decoded = {}, {}
    for name, rows, columns, scale in (
        ("w13_weight", 2 * intermediate, hidden, 1 / 128),
        ("w2_weight", hidden, intermediate, down_scale),
    ):
        raw = torch.randint(
            0, 256, (experts, rows, columns // 256, 74), dtype=torch.uint8
        )
        raw[..., :2] = torch.full(
            (experts, rows, columns // 256, 1), scale, dtype=torch.float16
        ).view(torch.uint8)
        packed[name] = raw.cuda()
        decoded[name] = dequantize_blocks(raw).bfloat16().cuda()
    quant = ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                ("moe.experts" if codec == "IQ2_XS" else "other.experts"): {
                    "quant_algo": "IQ2_XS",
                    "group_size": 256,
                    "block_payload_bytes": 74,
                    "packing": "ggml",
                }
            },
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
    config = VllmConfig(kernel_config=KernelConfig(moe_backend="auto"))
    with set_current_vllm_config(config):
        router = MuseSpark1xRouter(
            top_k=topk,
            num_experts=experts,
            expert_bias=bias,
            route_scale=1.0,
            router_score_normalization="sigmoid",
            router_score_division_eps=1e-15,
        )
        runner = FusedMoEFactory(
            num_experts=experts,
            top_k=topk,
            hidden_size=hidden,
            intermediate_size=intermediate,
            renormalize=True,
            router=router,
            quant_config=quant,
            prefix="moe.experts",
            params_dtype=torch.bfloat16,
            scoring_func="sigmoid",
            e_score_correction_bias=bias,
            routed_scaling_factor=1.0,
            routed_experts_cls=MuseSparkRoutedExperts,
            runner_cls=MuseMoERunner,
            runner_args={"post_expert_norm": norm},
        ).cuda()
        layer = runner.routed_experts
        checkpoint = packed if codec == "IQ2_XS" else decoded
        assert len(list(layer.load_weights(list(checkpoint.items())))) == 2
        layer.quant_method.process_weights_after_loading(layer)
        x = torch.randn(17, hidden, dtype=torch.bfloat16, device="cuda") / 10
        logits = torch.randn(17, experts, device="cuda")
        actual = run_moe(runner, config, x, logits)
        weights, ids = router._compute_routing(x, logits, None)
        expected = swiglu_moe_ref(
            x,
            decoded["w13_weight"],
            decoded["w2_weight"],
            weights,
            ids,
            lambda rows: norm.forward_native(rows, out_dtype=torch.float32),
        )
    assert torch.isfinite(actual).all()
    assert (actual.float() - expected).norm() / expected.norm() < 0.02
