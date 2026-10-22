---
license: other
license_name: nvidia-open-model-agreement
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
base_model:
  - DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - qwen3.8-flash-next
  - abliterated
  - trial117
  - nvfp4
  - mxfp8
  - dgx-spark
  - tensor-parallel
  - vllm
  - serving-recipe
  - speculative-decoding
---

# Qwen3.8-Flash-Next Abliterated · Dual DGX Spark TP=2

A reproducible **TP=2 serving recipe and evaluation bundle** for [DevelopingDad's Qwen3.8-Flash-Next-FP4-Local-Abliterated](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated/tree/51f24153ae9b27653a823074c631dfa941fdb536), tag **`trial117-mxfp8`**, running across two NVIDIA DGX Sparks.

The qualified v2 profile uses **16 active sequences, BF16 KV cache, MTP4, disk PLE, and 262,144-token context**. At 16 simultaneous short code/prose requests, the controlled comparison measured **313 aggregate output tokens/s**, up from 229, and **0.77 s p95 time to first token**, down from 11.89 s. Long-input gains were smaller, and serial prose decoding was about 8% slower.

This repository contributes the serving configuration, controlled comparisons, evaluation fixtures, and deployment guidance. **The checkpoint weights are unchanged and hosted by DevelopingDad on Hugging Face.** The abliteration and MXFP8 re-encode are DevelopingDad's work. No newly trained or re-encoded weights are distributed here.

## Model details

| Field | Value |
| --- | --- |
| Served model name | `Qwen3.8-Flash-Next` |
| Model family | Qwen3.8-Flash-Next; multimodal hybrid-attention MoE |
| Original model developer | [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Quantized base | [NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300) |
| Community derivative | [DevelopingDad / trial117-mxfp8](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated/tree/trial117-mxfp8) |
| Checkpoint revision | `51f24153ae9b27653a823074c631dfa941fdb536` |
| Weight format | Upstream ModelOpt mixed NVFP4 / MXFP8 / BF16; 21 safetensors shards |
| Serving recipe and evaluation | [XtraSaltyDev](https://github.com/XtraSaltyDev) |
| Qualified hardware | Two DGX Sparks, NVIDIA GB10, 128 GB unified memory each, direct CX7 RoCE |
| Parallelism | TP=2, PP=1, native multiprocessing, two nodes |
| Runtime | eugr B12X image, vLLM `0.1.dev21830+g8244d1c44.d20261015`, Torch `2.13.0+cu130` |
| Context / scheduler | 262144 tokens / 16 active sequences / 4096-token batch budget |
| Cache and speculation | BF16 KV, prefix caching, chunked prefill, MTP with four draft tokens |
| Qualification date | October 22, 2026 |

Pinned runtime image:

```text
eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595
```

## Model lineage and changes

```text
Qwen/Qwen3.8-Flash-Next
  → NVIDIA FP4-Local quantized checkpoint
  → DevelopingDad trial117 abliteration + MXFP8 re-encode
  → This repository: TP=2 serving v2, evaluation, and operations
```

DevelopingDad's [pinned model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated/blob/51f24153ae9b27653a823074c631dfa941fdb536/README.md) documents the baked refusal-direction intervention and direction-preserving MXFP8 re-encode. The frozen trial selection, weight transformation, and upstream refusal/KL measurements belong to that release; they are not new results from this serving study.

Our independent tensor-header review covered 375,487 entries. All 48 main attention output projections were MXFP8, the MTP output projection remained BF16, and the reviewed head/expert dimensions divided evenly across TP=2. See the [complete compressed review](evidence/checkpoint-review.json.gz) and [summary](evidence/checkpoint-summary.json).

Relative to our original eight-slot TP=2 reference, the selected profile raises active sequences to 16 and changes KV precision from FP8 to BF16. It retains disk PLE, fixed MTP4, the pinned checkpoint/image, and 262K context. BF16 KV retained the strict near-full-context JSON formatting gate that the tested 16-slot FP8 configurations failed.

## Intended uses

Local coding assistance, writing, tool-enabled agents, structured extraction, and image/text experimentation on a dual-Spark setup. Included fixtures exercise code execution, synthetic invoice OCR, tool calls, thinking controls, and planted-marker retrieval through an OpenAI-compatible chat-completions API.

Stock vLLM, generic Transformers loading, other accelerators, and other runtime versions were not qualified for this checkpoint here. Runtime/kernel compatibility is specific to the pinned B12X image.

## Quick start

Prepare two Sparks using the [eugr networking guide](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md). Docker needs NVIDIA CDI and RDMA access, working direct CX7 links, and sufficient free unified memory. Each container is capped at 104 GiB.

On **each Spark**, clone this repository and download the pinned checkpoint, preserving its directory layout:

```bash
git clone https://github.com/XtraSaltyDev/qwen38-flash-next-ablit.git
cd qwen38-flash-next-ablit
hf download DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated \
  --revision 51f24153ae9b27653a823074c631dfa941fdb536 \
  --local-dir "$HOME/models/qwen-trial117-mxfp8"
docker pull eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595
```

The example uses `10.100.225.2` for the head and `10.100.225.1` for the worker. Set addresses/interfaces for your CX7 network; see [operations](docs/OPERATIONS.md). `serve-tp2.sh` creates a named container and fails if that name already exists.

On the **worker**:

```bash
RANK=1 NODE_IP=10.100.225.1 MASTER_ADDR=10.100.225.2 \
  bash serve-tp2.sh "$HOME/models/qwen-trial117-mxfp8"
docker start qwen-flash-next-tp2-v2
```

Then on the **head**:

```bash
RANK=0 NODE_IP=10.100.225.2 MASTER_ADDR=10.100.225.2 \
  bash serve-tp2.sh "$HOME/models/qwen-trial117-mxfp8"
docker start qwen-flash-next-tp2-v2
docker logs --follow qwen-flash-next-tp2-v2
```

The head listens on `127.0.0.1:18383`; the worker is headless. Initial compilation/autotuning takes several minutes. When ready, on the head:

```bash
curl --fail http://127.0.0.1:18383/health
curl --fail http://127.0.0.1:18383/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Write a Python function that merges overlapping intervals."}],"temperature":0,"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}'
```

Thinking is controlled per request through `chat_template_kwargs.enable_thinking`. The recipe enables the `qwen3` reasoning parser, `qwen3_xml` tool parser, and automatic tool choice. Access loopback serving through your own authenticated proxy or SSH tunnel.

## Evaluation

Measured on an otherwise idle two-Spark cluster with fixed weights/runtime, temperature 0, seed 3741, thinking disabled, and fixed output lengths. Configurations received warmups. Stream-specific prefixes prevented prefix reuse between concurrent requests in the measured waves. The fresh baseline control used one repeat per code/prose fixture; v2 used two. Percentiles use nearest rank over a small sample.

| Workload / metric | Eight-slot FP8-KV reference | Sixteen-slot BF16-KV v2 | Change |
| --- | ---: | ---: | ---: |
| 16 clients, short prompts: aggregate output | 229.0 tok/s | **313.4 tok/s** | **+36.9%** |
| Same workload: p95 first token | 11.89 s | **0.77 s** | **−93.5%** |
| Same workload: p95 completion | 21.68 s | **15.84 s** | **−26.9%** |
| 16 clients, ~8K input: aggregate output | 38.64 tok/s | **40.98 tok/s** | **+6.0%** |

Short prompts were 62–64 input tokens with 256 forced output tokens. Long-input probes used about 8K input tokens and 128 output tokens; their p95 first-token latency improved about 9%. Aggregate output measures the whole concurrent wave, including queue/prefill time; it is not per-user decode speed. The shorter first-token latency reflects the change from eight to sixteen active slots. It does not imply subsecond prefill for long documents.

Single-stream coding decode stayed close to the initial reference. **Serial short-prose decode was approximately 8% slower.** C4/C8 aggregate throughput stayed within the predefined 5% regression floor. [Full results](docs/RESULTS.md), [machine-readable report](evidence/report.json), and [raw runs](evidence/runs) retain controls and tradeoffs.

### Quality and reliability

| Check | Result |
| --- | --- |
| Core deterministic/tool/retrieval fixtures | 13/13 |
| Baseline-passing extended fixtures | 7/7 retained; includes 3/3 invoice OCR |
| Strict JSON retrieval at 261623 input tokens | 10/10; schema control passed |
| Same retrieval after managed restart | 2/2 raw JSON and semantic passes; schema passed |
| Repeated word-count fixture | 100/100 vs reference 95/100; one prompt repeated |
| Returned target top-1 greedy audit | 384 tokens, zero mismatches on these fixtures |
| Qualified serving runs | No OOM, unexpected restart, or benchmark preemption |

Baseline and v2 both violated a no-import instruction in `sliding_max`; that failure remains recorded. The greedy audit is bounded and does not establish full distribution equivalence. A managed stop/start and configured client/API route checks passed on the original deployment; public evidence omits personal configuration and access details.

Final observed KV capacity was **3,281,415 tokens**, approximately **12.52 full 262K-context equivalents**. Sixteen active slots do not provide sixteen simultaneous maximum-length contexts. The original minimum of eight full-context equivalents was retained.

## Limitations and risks

These are small synthetic serving comparisons, not broad capability benchmarks, production SLO certification, or proof of a global optimum. DevelopingDad's refusal/KL results were not rerun in this study. No training dataset or weight transformation was introduced here.

The upstream checkpoint intentionally changes refusal behavior. Generated code and factual answers can still be incorrect; the tests cover only their stated fixtures. Video, broader vision tasks, general capability retention, larger clusters, and other platforms remain unqualified here. Long-prefill latency and the serial-prose tradeoff are the next performance targets.

Shared-PLE/FP8 candidates were rejected for a strict near-context JSON regression despite good speed. Adaptive MTP was rejected before scoring because the pinned B12X/CuTe graph path failed to become healthy. Historical `selected-*` filenames refer to a candidate later rejected. The final selection is `bf16kv-seq16`.

## Reproduce and inspect

- [Operations and rollback](docs/OPERATIONS.md)
- [Results and experimental decisions](docs/RESULTS.md)
- [Attribution and license scope](docs/ATTRIBUTION.md)
- [Profile](evidence/profile.json), [protocol](evidence/protocol.json), [qualification](evidence/qualification.json)
- [Evaluation scripts](scripts), [synthetic image fixtures](evidence/invoice-fixtures.json), and [experimental recipes](recipes)
- [Publication transformations and original hashes](evidence/PUBLICATION.json)

```bash
python3 scripts/evaluate.py --base http://127.0.0.1:18383 \
  --label v2-repeat --output /tmp/qwen-v2-repeat \
  --mode all --concurrency 1 4 8 16 --repeats 2 --tokens 256
shasum -a 256 -c SHA256SUMS
```

## Attribution and licensing

**Qwen** developed the original model. **NVIDIA** published the ModelOpt/QAD FP4-Local base. **DevelopingDad** created the trial117 community derivative and MXFP8 re-encode. **eugr / Eugene Rakhmatulin** provides the Spark Docker/runtime and cluster-recipe foundation, with inference work from **local-inference-lab**, **FlashInfer**, **vLLM**, and contributors. **XtraSaltyDev** contributed this TP=2 serving study, harnesses, measurements, and packaging. See [NOTICE](NOTICE) and [attribution](docs/ATTRIBUTION.md).

The checkpoint card identifies the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) and [Qwen Community License 1.0](third_party/licenses/QWEN-COMMUNITY-LICENSE-1.0.txt). **The YAML license metadata above refers to the checkpoint.** Original code and documentation contributed here use [MIT](LICENSE); that grant does not relicense model weights, upstream portions, or runtime dependencies. Upstream terms remain applicable as described in [license scope](docs/ATTRIBUTION.md#license-scope).

This community recipe is not endorsed or newly validated by Qwen, NVIDIA, DevelopingDad, or runtime maintainers.

## Citation

Use [CITATION.cff](CITATION.cff) for this serving/evaluation artifact. Cite upstream checkpoints separately when using their weights or reporting their model-level results.

```bibtex
@software{xtrasaltydev_qwen38_flash_next_tp2_2026,
  author = {XtraSaltyDev},
  title = {Qwen3.8-Flash-Next Abliterated: Dual DGX Spark TP=2 Serving Recipe},
  year = {2026},
  version = {tp2-v2},
  url = {https://github.com/XtraSaltyDev/qwen38-flash-next-ablit}
}
```
