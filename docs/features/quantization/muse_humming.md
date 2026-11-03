# Muse Spark with Humming and IQ2 experts (experimental)

Muse Spark supports offline-converted Humming INT3 and ModelOpt IQ2_XS expert
weights. Both paths preserve Muse's expert computation order: second projection,
TP all-reduce, per-expert RMSNorm, then routing weights and the expert sum.
The serving flags below require an already converted checkpoint.

## Humming INT3

Use `--quantization humming --moe-backend humming` with BF16 or FP16 activations.
The adapter requires symmetric Humming weights, DP1, PCP1, and no expert
parallelism. Online requantization and weights with zero points are unsupported.
Other Muse projections can remain unquantized.

For INT3 group-128 weights, set these fields in `quantization_config`:

```json
{
  "quant_method": "humming",
  "weight_dtype": "int3",
  "input_dtype": "bfloat16",
  "weight_scale_group_size": 128
}
```

Preserve checkpoint exclusions for unquantized projections, including fused
attention projections. Store experts as consolidated gate/up and down tensors
with their group scales.

Muse skips Humming's optional integer re-encoding to avoid expanding an entire
packed expert projection to INT32 during loading. Kernel repacking still needs
working memory. NVIDIA target layers using fused QK normalization, RoPE, and KV
insertion share FP32 RoPE tables. Draft and fallback paths keep private tables
because their rotary operation can change the cache dtype.

## IQ2_XS with ModelOpt and b12x

On DGX Spark (GB10, SM121), use
`--quantization modelopt_mixed --moe-backend auto --dtype bfloat16`.
Install the optional [b12x dependencies](b12x.md). Keep `--moe-backend auto`:
IQ2 experts use b12x, while retained BF16 experts use Triton on SM12x.

The adapter requires BF16 activations, DP1, PCP1, and no expert parallelism.
It expands each selected route into a separate b12x row so Muse can normalize
expert outputs before combining them. This adds intermediate storage and work.
The configuration below was tested with eager execution; CUDA graph support
has not been validated.

### Checkpoint configuration

Use `quant_method: modelopt`, `quant_algo: MIXED_PRECISION`, and a
`quantized_layers` entry for each converted
`model.language_model.layers.N.mlp.experts` prefix. Each entry needs:

```json
{
  "quant_algo": "IQ2_XS",
  "group_size": 256,
  "block_payload_bytes": 74,
  "packing": "ggml"
}
```

Store consolidated gate/up and down tensors as IQ2_XS blocks. Unlisted experts
remain BF16. Each block stores 256 weights in 74 bytes, or 2.3125 bits/weight
including metadata. Retained BF16 tensors increase the full checkpoint size.
IQ2_XXS and Q8_0 dispatch exists but has not been validated for Muse.

The tested conversion used unweighted codebook search in ModelOpt revision
[`cb9ab4de0698bfdf701cad4924093867b790eb17`](https://github.com/NVIDIA/Model-Optimizer/tree/cb9ab4de0698bfdf701cad4924093867b790eb17),
with no calibration or training data. ModelOpt 0.47 lacks this encoder.
Serving uses b12x 1.5.1 and does not run the encoder.

The source was the original NVFP4 checkpoint at revision
`4239e4db19376904f2e82c167e78f70abb5455fe`. Expert layers 0 and 61 stayed BF16;
the earlier INT3 conversion also quantized those layers. Comparisons between
these checkpoints therefore include a difference in retained precision.

### Four-node serving

Place identical checkpoint files at `MODEL_PATH` on every node. Set
`MASTER_ADDR` to the head node's reachable address and `NODE_RANK` to 0–3.
Run this on each node:

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

This recipe covers text generation; `--language-model-only` disables multimodal
inputs. Configure network interfaces using the
[distributed serving guide](../../serving/parallelism_scaling.md).

## DFlash2 speculative decoding with the IQ2 target

DFlash2 is off in the command above. To enable it, set
`VLLM_USE_V2_MODEL_RUNNER=1`, add `--gpu-memory-utilization 0.85`, and pass
`--speculative-config` with this JSON:

```json
{
  "method": "dflash",
  "model": "/path/to/draft",
  "num_speculative_tokens": 7,
  "draft_sample_method": "probabilistic",
  "draft_tensor_parallel_size": 4
}
```

Use the same draft path on every node. The tested draft was
[`nvidia/Muse-Spark-Open-816B-A42B-DFlash2`](https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2)
at revision `22e2669e4c60689f9bff0749a34d9d8e59ce7cd6`. This five-layer BF16
draft runs alongside the IQ2 target and shares its embedding and output-head
storage. Its weight file is 8.17 GiB, about 2.7% of the IQ2 target's tensor size.

### Tested runtime

The October 31 measurements used a vLLM wheel from
`5ea8301982648ac444ea01368ecb1b064f6de7bc`, with these files taken from tracking
branch commit `b1190e0f5780ef557cb2d3ee9f75d92a29c3dd95`:

- `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`
- `vllm/model_executor/layers/quantization/humming.py`
- `vllm/model_executor/layers/quantization/utils/humming/moe.py`
- `vllm/models/muse_spark/nvidia/model.py`
- `vllm/models/muse_spark/nvidia/muse_moe_runner.py`

The wheel alone lacks this Muse IQ2 adapter. Runs used b12x 1.5.1, BF16
activations, TP4, eager execution, a 32,768-token context, four sequences,
128 batched tokens, and 4 GiB KV cache per rank. Checkpoint hashes matched on
all four nodes. NCCL RoCE checks passed, native `DFlash2Speculator` execution
and accepted draft tokens were observed, and pointer checks confirmed shared
embedding/head storage on every rank.

## Validation

### Kernels and loading

INT3 real-expert checks passed at TP1 and TP4 against decoded INT3 weights.
IQ2 checks on four complete real experts passed at TP4, with relative L2 errors
of 0.00403, 0.00353, and 0.00306 at token counts 1, 4, and 17. Retained BF16
experts also passed. These checks compare kernels against the converted weights;
model accuracy is measured separately below.

| Deployment | Tensor storage | Model-loading memory/rank |
| --- | --- | --- |
| Humming INT3 | 334.03 GiB | 89.0 GiB |
| IQ2_XS | 299.53 GiB | 80.38 GiB |
| IQ2_XS + BF16 DFlash2 | 299.53 + 8.17 GiB | 83.46 GiB |

All three served on four GB10 nodes. The memory figures exclude later KV cache
allocations and runtime workspaces. INT3 startup used a 2,048-token context,
one sequence, and 1 GiB KV per rank; shared RoPE tables saved 11.25 GiB/rank
before expert preparation. INT3 Nsight captures were collected on all ranks.
No DFlash2 Nsight breakdown of draft, verification, and communication was collected.

### INT3 accuracy pilots

On October 29, 100 examples per benchmark were sampled uniformly without
replacement (selection seed 42). Settings: zero-shot prompts, temperature 0,
generation seed 7, low reasoning, 2,048-token context, 1,536-token output cap
including reasoning, TP4, four concurrent requests, and 1 GiB KV per rank.

| Metric | Result |
| --- | --- |
| GSM8K exact match, strict and flexible | 95/100 (95%) |
| IFEval prompt accuracy, strict and loose | 88/100 (88%) |
| IFEval instruction accuracy, strict and loose | 132/148 (89.2%) |
| Request errors | 0/200 |
| Truncations | 2 GSM8K; 11 IFEval |
| Empty final answers | 1 GSM8K; 4 IFEval |

Only final content was scored. GSM8K used decimal-normalized numeric exact
match with a requested `#### <number>` ending; flexible extraction used the
last number. IFEval used the unmodified Google Research scorers. Truncations
and empty answers stayed in the denominator. Wilson 95% intervals are
88.8–97.8% for GSM8K and 80.2–93.0% for IFEval prompt accuracy.

Data/scorer revisions: OpenAI GSM8K `3101c7d5072418e28b9008a6636bde82a006892c`;
Google Research IFEval `1ae1fded5c616deb6b4652818588577e8fca0f06`.
These are 100-case pilots. IQ2 and DFlash2 were not tested on these benchmarks.

### Partial GPQA Diamond evaluations

Each run attempted 198 questions and ended when its allocation expired.

| Deployment | Correct/completed | Missing | Truncated, empty and unparsed finals | Nominal 95% Wilson interval |
| --- | --- | --- | --- | --- |
| INT3, October 29 | 68/75 (90.7%) | 123 | 4 | 82.0–95.4% |
| IQ2_XS, October 30 | 95/118 (80.5%) | 80 | 12 | 72.4–86.6% |
| IQ2_XS + DFlash2, October 31 | 157/185 (84.9%) | 13 | 13 | 79.0–89.3% |

There were zero saved request errors. In each row, truncation, empty final, and
unparsed final refer to the same responses; they count as incorrect. Missing
questions are excluded. Four requests per run were interrupted without response
records; the other missing questions were never started.

All runs used the same manifest and final-only scorer: xhigh reasoning,
temperature 1, top_p 1, top_k disabled, seed 7, 16,384 output tokens including
reasoning, 32,768 context tokens, TP4, concurrency four, and 4 GiB KV per rank.
Prompt formatting, choice permutation (seed 0), and answer extraction followed
OpenAI simple-evals `652c89d0ca9df547706735883097e9537d40dc47`; the dataset
revision was `56686c06f5e19865c153de0fdb11be3890014df7`.
IQ2 prioritized the 75 completed INT3 IDs. DFlash2 prioritized the 118 completed
IQ2 IDs independently of correctness; all prompt token counts matched.

On shared IDs:

- INT3: 68/75; IQ2: 65/75. Both correct: 64; INT3 only: 4; IQ2 only: 1.
- IQ2: 95/118; DFlash2: 99/118. Both correct: 93; IQ2 only: 2; DFlash2 only: 6.

These are incomplete, single-pass results. Completion order can bias the subset,
and the intervals do not account for that bias. The retained expert precision
and historical runtime differ between runs. No matched original-NVFP4 baseline
was evaluated, so these scores do not measure quantization loss or establish
an accuracy gain from DFlash2. Raw responses and provenance are archived privately.

### Streaming performance

The October 31 comparison used four fixed 157–165-token prompts per concurrency
and per configuration, after warmup. Both configurations used the same nodes,
runtime and request settings: low reasoning, temperature 1, seed 7, and a
4,096-token output cap. Servers ran sequentially; unique cache salts prevented
prefix reuse. All 16 measured responses completed without errors, truncations,
or empty finals.

| Measurement | IQ2 C1 | DFlash2 C1 | IQ2 C4 | DFlash2 C4 |
| --- | --- | --- | --- | --- |
| Combined output tokens/s | 11.66 | 23.88 | 28.94 | 49.86 |
| Median per-request decode tokens/s | 11.93 | 25.78 | 8.25 | 15.25 |
| Median first-generated TTFT | 0.974 s | 0.973 s | 2.630 s | 2.912 s |
| Median effective time/output token | 83.84 ms | 38.81 ms | 121.20 ms | 65.60 ms |
| Median inter-chunk gap | 83.59 ms | 170.40 ms | 121.29 ms | 307.51 ms |
| Median first final-answer text | 43.19 s | 19.55 s | 68.97 s | 34.92 s |
| Mean tokens/streamed chunk | 1.00 | 4.36 | 1.00 | 4.75 |

C1/C4 mean one/four concurrent requests. Combined throughput is total generated
tokens divided by batch wall time, including queueing and prefill. Counts include
reasoning and control tokens. TTFT measures the first generated token; final
answer text arrives after reasoning. Effective time/output token averages the
decode span. Speculative bursts do not expose individual token times within a chunk.

With seven draft tokens per step, DFlash2 accepted 1,608/3,318 tokens at C1
(48.46%) and 1,733/3,206 at C4 (54.05%). Mean acceptance length, including the
bonus token, was 4.39/4.78 across 474/458 steps. The observed combined throughput
was 2.05x/1.72x the baseline, with different sampled output lengths in each run.
Four prompts in one pass are too few to characterize production performance.

An earlier IQ2-only run used a 512-token cap and four requests per concurrency.
Combined throughput was 11.59/30.37 tokens/s, median decode 11.92/8.05 tokens/s,
median TTFT 0.946/2.568 s, and mean ITL 83.94/124.75 ms. It had zero errors,
six truncations and four empty finals; it is separate from the comparison above.

#### DFlash2 with 8K uncached inputs

On November 1, synthetic background extended the same tasks to exactly 8,192
input tokens including the chat template. Tokenization, response usage and
server counters confirmed the input length and absence of prefix reuse.
The target, draft, runtime packages, server and request settings stayed the
same, including the 128-token scheduling budget. Physical nodes and drivers
differed from the short-input run. All eight responses completed without errors,
truncations or empty finals.

| Measurement | C1 | C4 |
| --- | --- | --- |
| Median first-generated TTFT | 37.18 s | 101.38 s |
| Median first final-answer text | 43.85 s | 119.98 s |
| Median per-request decode tokens/s | 18.84 | 6.10 |
| Combined output tokens/s | 5.06 | 7.83 |
| Median effective time/output token | 53.82 ms | 168.13 ms |
| Median inter-chunk gap | 172.27 ms | 581.27 ms |
| Mean acceptance length, including bonus | 3.41 | 3.81 |

Draft acceptance was 732/2,128 (34.40%) across 304 steps at C1 and
1,129/2,814 (40.12%) across 402 steps at C4. Output totaled 1,040/1,535 tokens,
including 486/975 reasoning tokens. Mean queue waits were below 0.001/58.69 s;
mean prefill spans were 37.02/41.86 s. These spans include scheduling interference
and are not isolated kernel timings. This is one pass with variable output
lengths on different nodes, so it does not isolate the effect of input length.
