# Muse Spark with Humming and IQ2 experts (experimental)

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

Muse keeps the checkpoint's integer representation during loading. It skips
Humming's optional integer re-encoding, which otherwise expands a complete
expert projection to INT32 temporarily. Kernel repacking still needs working
memory; avoiding that expansion does not establish full-model memory fit.

NVIDIA target layers using fused QK normalization, RoPE, and KV insertion share
identical FP32 RoPE tables. Draft and fallback attention paths keep private tables
because their rotary operation can change the cache dtype.

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

A converted 816B checkpoint has completed startup and short chat generations on
four GB10 nodes with TP4, BF16 activations, a 2,048-token context, one active
sequence, and 1 GiB of KV cache per GPU. vLLM reported 89.0 GiB of model memory per
rank; sharing target RoPE tables saved 11.25 GiB per rank before expert preparation.
Nsight captures were collected on all four ranks for a generation request.

The smoke checks answered basic arithmetic and geography prompts and completed
a short technical explanation. Runtime parameters, replicated projections,
repacking buffers, KV cache, and the operating system also consume memory,
especially on devices with shared CPU/GPU memory.

### Pilot evaluation

On 2026-10-29, the converted checkpoint completed 100 GSM8K test examples and
100 IFEval examples on four GB10 nodes. Each sample was drawn uniformly without
replacement with selection seed 42. The run used zero-shot prompts, temperature
0, generation seed 7, low reasoning effort, a 2,048-token context, and a
1,536-token generation budget including reasoning. Server and client concurrency
were both four; the other serving settings matched the smoke run above.

| Metric | Result |
| --- | --- |
| GSM8K exact match, strict and flexible extraction | 95/100 (95%) |
| IFEval prompt accuracy, strict and loose | 88/100 (88%) |
| IFEval instruction accuracy, strict and loose | 132/148 (89.2%) |
| Request errors | 0/200 |
| Token-limit truncations | 2/100 GSM8K; 11/100 IFEval |
| Empty final answers | 1/100 GSM8K; 4/100 IFEval |

Only final answer content was graded; the separate reasoning field was excluded.
GSM8K prompts requested a final `#### <number>` line, scored by decimal-normalized
numeric exact match. Flexible extraction used the last number in the final answer.
IFEval used the unmodified Google Research strict and loose scorers. All selected
cases, including truncations and empty final answers, remained in the denominator.
The 95% Wilson intervals are 88.8–97.8% for GSM8K and 80.2–93.0% for IFEval
prompt accuracy.

The data/scorer revisions were `3101c7d5072418e28b9008a6636bde82a006892c`
for OpenAI's GSM8K repository and `1ae1fded5c616deb6b4652818588577e8fca0f06`
for Google Research's IFEval implementation. All 189 rebuilt weight shards
matched the previously served conversion by SHA256. These are 100-example
pilots, not full benchmark runs. No original-NVFP4 baseline or controlled
performance comparison was evaluated, so quantization accuracy loss and serving
speedup remain unmeasured.

### IQ2_XS with ModelOpt and b12x

On DGX Spark (GB10, SM121), an offline-converted IQ2_XS checkpoint can select
the experimental SM12x path
with `--quantization modelopt_mixed --moe-backend auto --dtype bfloat16`.
This uses b12x kernels inside vLLM and requires an already converted checkpoint.
The Humming quantization flag does not select IQ2.

Install the optional [b12x dependencies](b12x.md) in the serving environment.
For Muse, keep `--moe-backend auto` so both IQ2 experts and retained BF16 experts
can select their respective backends. The generic b12x example's explicit
`--moe-backend b12x` is not the validated mixed Muse configuration.

For the validated four-node setup, put an identical checkpoint at `MODEL_PATH`
on every node and set `MASTER_ADDR` to the head node's reachable address.
This recipe and the reported checks cover text generation only;
`--language-model-only` disables multimodal inputs.
Run this on each node with `NODE_RANK` set to 0, 1, 2, or 3 respectively:

```bash
extra_args=(--host 0.0.0.0 --port 8000)
if [ "$NODE_RANK" != 0 ]; then extra_args=(--headless); fi
vllm serve "$MODEL_PATH" \
    --quantization modelopt_mixed --moe-backend auto --dtype bfloat16 \
    --distributed-executor-backend mp --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port 29541 --tensor-parallel-size 4 \
    --reasoning-parser muse --enforce-eager --disable-custom-all-reduce \
    --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 128 \
    --kv-cache-memory-bytes 4294967296 --language-model-only \
    --load-format safetensors --safetensors-load-strategy lazy \
    "${extra_args[@]}"
```

Configure the network interfaces for the actual cluster, following
[distributed serving guidance](../../serving/parallelism_scaling.md).

Use `quant_method: modelopt`, `quant_algo: MIXED_PRECISION`, and a
`quantized_layers` entry for each converted `model.language_model.layers.N.mlp.experts`
prefix. Each entry must specify `quant_algo: IQ2_XS`, `group_size: 256`,
`block_payload_bytes: 74`, and `packing: ggml`. Store consolidated gate/up and
down expert tensors as native IQ2_XS blocks. Unlisted experts use BF16.
IQ2_XS stores 256 weights in 74 bytes, including block metadata (2.3125 bits per
weight). This density does not include the checkpoint's retained BF16 tensors.

The adapter requires BF16 activations and tensor parallelism, with DP1, PCP1,
and expert parallelism disabled. On SM12x, retained BF16 experts use Triton.
The IQ2 adapter expands selected routes into separate b12x rows, sums TP partial
outputs before each expert's RMSNorm, and applies routing weights afterward.
Eager execution is the validated mode; this work does not establish CUDA graph
compatibility or a speedup over Humming.

Serving validation used b12x 1.5.1. Offline encoding used ModelOpt development
revision [`cb9ab4de0698bfdf701cad4924093867b790eb17`](https://github.com/NVIDIA/Model-Optimizer/tree/cb9ab4de0698bfdf701cad4924093867b790eb17);
ModelOpt 0.47 lacks that IQ2 encoder. Serving already packed expert tensors uses
vLLM's ModelOpt checkpoint loader and b12x kernels, without running the encoder.
Four complete real Muse experts passed TP4 checks on GB10 against a decoded IQ2
reference, with relative L2 errors of 0.00403, 0.00353, and 0.00306 at batch sizes
1, 4, and 17. Retained BF16 experts also passed TP4 checks. These tests establish
kernel correctness against the quantized weights, not model accuracy.

The experimental conversion uses unweighted IQ2_XS codebook search without
calibration data. It starts from the authorized original NVFP4 revision and
retains BF16 expert layers 0 and 61. The earlier INT3 checkpoint also quantized
those boundary experts, so any subsequent accuracy comparison must account for
the different retained precision as well as the codec.

The full conversion produced 189 weight shards with 299.53 GiB of tensor data.
All four node copies passed SHA256 verification, and vLLM reached serving
readiness with TP4 and a 32,768-token context. Model loading reported 80.38 GiB
per rank, excluding the subsequent 4 GiB KV cache and runtime workspaces.
Three arithmetic smoke requests returned the correct final answer under greedy
low, sampled low, and sampled xhigh reasoning. An earlier greedy request reached
its 256-token cap without a final answer; the three follow-ups used a 1,024-token
cap and ran concurrently. These small checks establish generation, not IQ2
accuracy or a controlled comparison with the earlier INT3 checkpoint.

#### IQ2 streaming and partial GPQA results

Eight streaming requests measured the TP4 deployment after warmup, with four
requests at each concurrency. Temperature was 1, seed 7, reasoning effort
`low`, and the output cap 512 tokens. These API measurements include reasoning
and control tokens; TTFT measures the first generated token.

| Measurement | Concurrency 1 | Concurrency 4 |
| --- | --- | --- |
| Aggregate output tokens/s | 11.59 | 30.37 |
| Median per-request decode tokens/s | 11.92 | 8.05 |
| Median TTFT | 0.946 s | 2.568 s |
| Mean observed inter-token latency | 83.94 ms | 124.75 ms |

There were zero request errors, six output-limit truncations, and four empty
final answers. This small performance sample does not establish a speedup
over INT3 or represent a production workload distribution.

A separate IQ2 GPQA Diamond run used the same 198-question manifest, scorer,
and generation settings as the INT3 run below, prioritizing its completed
question IDs. After cluster access recovered on 2026-10-31, all 118 completed
responses were retrieved, with 95 correct (80.5%). There were zero request
errors and 12 capped responses with empty, unparsed final answers; those 12
count as incorrect. The nominal 95% Wilson interval is 72.4–86.6%.
Across the same 75 question IDs completed by both deployments, IQ2 scored
65/75 (86.7%) and INT3 scored 68/75 (90.7%). Both used one pass; IQ2 also
retained BF16 boundary experts that the INT3 conversion quantized.

Slurm accounting confirms that the allocation timed out at 19:23:49 UTC on
2026-10-30. Four requests were interrupted without response records and 76
questions were never started. The 80 unfinished cases are missing, not scored
incorrect. All 66 responses saved before the connectivity outage remain
byte-identical within the recovered file, whose hash matches the remote copy.
Completion order can bias this partial subset, and the interval does not cover
that selection effect. This is not a complete GPQA score or a matched measurement
of loss against original NVFP4. Published NVFP4 scores remain external references.
Raw records, terminal logs, settings, and integrity hashes are preserved in the
private experiment archive.

### Interrupted INT3 GPQA Diamond evaluation

A separate run on 2026-10-29 attempted all 198 GPQA Diamond questions with
the Humming INT3 checkpoint. The eight-hour allocation expired after 75
responses completed; 68 of those 75 were correct (90.7%). Another four requests
were interrupted without response records, and 119 questions were not started.
The 123 unfinished cases are missing, rather than completed incorrect answers.
This is not a complete 198-question benchmark score.

The 75 completed requests had zero request errors. Four reached the output
limit with empty, unparsed final answers; all four remain in the denominator
as incorrect. The nominal 95% Wilson interval among completed cases is
82.0–95.4%. This incomplete single pass can favor requests that finish sooner;
that selection effect is not covered by the interval. There is no matched
original-NVFP4 run, so this result does not measure quantization accuracy loss.
Published NVFP4 scores are external references with different evaluation setups.

This run used TP4 on four GB10 nodes, BF16 activations, eager execution, a
32,768-token context, 4 GiB of KV cache per rank, concurrency four, `xhigh`
reasoning effort, temperature 1, `top_p=1`, disabled `top_k`, generation seed 7,
and 16,384 output tokens including reasoning. Settings stayed unchanged.
The prompt, choice permutation, and final-answer extraction followed
OpenAI simple-evals at `652c89d0ca9df547706735883097e9537d40dc47`, with
choice-permutation seed 0. The author dataset revision was
`56686c06f5e19865c153de0fdb11be3890014df7`. Only final answer content was scored.
All raw responses, missing-case IDs, settings, and termination logs were retained
in the private evaluation archive; the user stopped further INT3 evaluation.
Benchmark examples are omitted
from this documentation.
