# Muse Spark with Humming experts (experimental)

Muse Spark can load offline-converted Humming expert tensors with BF16 or FP16
activations. Select the path with `--quantization humming --moe-backend humming`.
The checkpoint must already contain Humming packed weights and scales; these
flags do not convert the original NVFP4 checkpoint to INT3.

This experimental integration supports tensor parallelism with DP1, PCP1, and
expert parallelism disabled. It requires symmetric Humming weights, without
zero points or online requantization. Other Muse layers can remain unquantized.

The indexed Humming experts pass their second GEMM outputs to Muse's existing
finalizer. Tensor-parallel partial outputs are summed before each expert's
RMSNorm; FP32 routing weights are then applied before summing selected experts.

## Checkpoint configuration

For INT3 group-128 expert weights, the checkpoint's `quantization_config` uses
`quant_method: humming`, `weight_dtype: int3`, `input_dtype: bfloat16`, and
`weight_scale_group_size: 128`. Preserve exclusions for unquantized projections,
including fused attention projections. The expert checkpoint uses consolidated
gate/up and down tensors with their corresponding group scales.

## Validation limits

Real Muse expert-layer weights have been exercised at TP1 and TP4 on GB10.
This verifies execution against the decoded INT3 reference. It does not measure
the quality loss caused by converting NVFP4 weights to INT3.

Full-model startup, generation quality, and serving performance remain under
investigation. Checkpoint byte size alone does not establish memory fit: runtime
parameters, replicated projections, repacking buffers, KV cache, and the operating
system also consume memory, especially on devices with shared CPU/GPU memory.
