# Qwen3.8-Flash-Next on 2× DGX Spark

_by [XtraSaltyDev](https://github.com/XtraSaltyDev)_

Serve [DevelopingDad's Qwen3.8-Flash-Next-FP4-Local-Abliterated](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated) across two DGX Sparks using vLLM and the B12X kernel stack.

**TP=2 · 262K context · 16 active sequences · BF16 KV · MTP4 · disk PLE**

The API model name is **`Qwen3.8-Flash-Next`**. This repo contains the launch recipe, evaluation scripts, and measured results. DevelopingDad's checkpoint is served unchanged.

## Performance

Measured on two GB10 DGX Sparks on **October 22, 2026**, against our original eight-slot FP8-KV profile. Both configurations use the same checkpoint, runtime image, disk PLE, and MTP4.

| Workload | Eight-slot reference | This recipe |
| --- | ---: | ---: |
| 16 concurrent short requests: aggregate output | 229.0 tok/s | **313.4 tok/s (+37%)** |
| Same workload: p95 first token | 11.89 s | **0.77 s** |
| Same workload: p95 completion | 21.68 s | **15.84 s** |
| 16 concurrent ~8K-input requests: aggregate output | 38.64 tok/s | **40.98 tok/s (+6%)** |

Short tests mix code and prose: 62–64 input tokens, 256 forced output tokens, greedy, thinking off, seed 3741. The fresh reference has one repeat per fixture; this recipe has two. Aggregate output includes queue and prefill time across the whole wave. These are small controlled probes; the p95 values are not production latency guarantees.

Single-stream coding stayed close to the reference; **serial prose decode was about 8% slower**. Long prompts still spend substantial time in prefill. See [full results and tradeoffs](docs/RESULTS.md), [report](evidence/report.json), and [raw runs](evidence/runs).

## Requirements

- Two DGX Sparks, each with GB10 and 128 GB unified memory.
- Working ConnectX RoCE links and Docker with NVIDIA CDI/RDMA access. See [eugr's networking guide](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md).
- The pinned image and complete checkpoint on **both** nodes; allow roughly 102 GB per checkpoint plus compilation/cache space.
- Enough free host memory for a 104 GiB container cap per node.

## Quick start

On **both Sparks**, clone the repo, download the weights pinned to the measured commit, and pull the runtime:

```bash
git clone https://github.com/XtraSaltyDev/qwen38-flash-next-ablit.git
cd qwen38-flash-next-ablit

hf download DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated \
  --revision 51f24153ae9b27653a823074c631dfa941fdb536 \
  --local-dir "$HOME/models/qwen38-flash-next-ablit"

docker pull eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595
```

The example uses head `10.100.225.2` and worker `10.100.225.1`. Adjust `MASTER_ADDR`, `NODE_IP`, `SOCKET_IFNAME`, and `IB_HCAS` for your CX7 network. The [configuration reference](docs/OPERATIONS.md#recipe-variables) lists the defaults.

**Worker first:**

```bash
RANK=1 NODE_IP=10.100.225.1 MASTER_ADDR=10.100.225.2 \
  bash serve-tp2.sh "$HOME/models/qwen38-flash-next-ablit"
docker start qwen-flash-next-tp2-v2
```

**Then the head:**

```bash
RANK=0 NODE_IP=10.100.225.2 MASTER_ADDR=10.100.225.2 \
  bash serve-tp2.sh "$HOME/models/qwen38-flash-next-ablit"
docker start qwen-flash-next-tp2-v2
docker logs --follow qwen-flash-next-tp2-v2
```

`serve-tp2.sh` creates the container; it does not start it or download weights. If the named containers already exist, use `docker start` rather than recreating them. First launch compiles/autotunes kernels and takes several minutes. Later starts reuse each node's cache.

Once the head is ready:

```bash
curl --fail http://127.0.0.1:18383/health
curl --fail http://127.0.0.1:18383/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Write a Python function that merges overlapping intervals."}],"temperature":0,"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}'
```

The OpenAI-compatible API listens on the head at **`http://127.0.0.1:18383/v1`**; the worker is headless. Use your own SSH tunnel or authenticated proxy for remote access. Stop the container on the head and then the worker with `docker stop -t 90 qwen-flash-next-tp2-v2`.

## Default profile

| Setting | Value |
| --- | --- |
| Parallelism | TP=2, PP=1, native multiprocessing across two nodes |
| Context / active sequences | 262144 tokens / 16 |
| Batch token budget | 4096 |
| KV / PLE | BF16 KV (`auto` with BF16 model dtype) / disk, io_uring |
| Speculation | MTP, four draft tokens |
| Memory | GPU utilization 0.8; container cap 104 GiB per node |
| Prefix caching / chunked prefill | Enabled |
| Thinking / tools | Per-request thinking control; `qwen3` reasoning and `qwen3_xml` tool parsers |

Final observed cache capacity was **3.28M tokens**, about **12.52 full 262K contexts**. Sixteen scheduler slots do not mean sixteen simultaneous maximum-length requests. The tested runtime was vLLM `0.1.dev21830+g8244d1c44.d20261015` with Torch `2.13.0+cu130` in the pinned image above.

See [operations](docs/OPERATIONS.md) for network variables, cache paths, startup checks, and rollback.

## Checks

The selected profile retained all baseline-passing quality gates: **13/13 core checks**, **7/7 baseline-passing extended checks**, and **10/10 strict JSON retrieval checks at 261623 input tokens**. A managed restart and actual client/API route tests passed on the original deployment. The baseline's no-import instruction failure in one code fixture remains recorded.

Shared-PLE/FP8 candidates failed the strict near-context formatting gate; adaptive MTP failed to start on the pinned graph/kernel path. Their results are retained alongside the winning `bf16kv-seq16` runs. These fixtures are bounded diagnostics, not a general capability benchmark. Details: [results](docs/RESULTS.md), [protocol](evidence/protocol.json), [qualification](evidence/qualification.json).

## Repository layout

| Path | Contents |
| --- | --- |
| [serve-tp2.sh](serve-tp2.sh) | Qualified serving defaults with configurable network/container names |
| [scripts/](scripts) | Performance, code/tool/OCR/retrieval, repeatability, and target-token checks |
| [evidence/](evidence) | Reports, raw runs, startup/failure logs, checkpoint review, and synthetic fixtures |
| [recipes/](recipes) | Reference and historical experimental configurations |
| [docs/](docs) | Operations, results, attribution, and license scope |
| [third_party/](third_party) | Pinned source references and retained upstream notices |

To repeat the short-prompt checks on the head:

```bash
python3 scripts/evaluate.py --base http://127.0.0.1:18383 \
  --label v2-repeat --output /tmp/qwen-v2-repeat \
  --mode all --concurrency 1 4 8 16 --repeats 2 --tokens 256
shasum -a 256 -c SHA256SUMS
```

## Credits and license

**[DevelopingDad](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated)** created the abliteration and MXFP8 re-encode, derived from **[NVIDIA's quantized checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)** and **[Qwen's original model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**. **[eugr / Eugene Rakhmatulin](https://github.com/eugr/spark-vllm-docker)** provides the Spark container and cluster-recipe foundation, with inference work from **[local-inference-lab](https://github.com/local-inference-lab)**, **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)**, and **[vLLM](https://github.com/vllm-project/vllm)**. This repo contributes the TP=2 serving study, harnesses, and deployment recipe.

README organization takes inspiration from [MiaAI-Lab's Spark recipes](docs/ATTRIBUTION.md#recipe-presentation-references). Full source credits and retained notices are in [attribution](docs/ATTRIBUTION.md) and [NOTICE](NOTICE).

Original contributions here use [MIT](LICENSE). Checkpoint-derived material retains its upstream [NVIDIA](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) and [Qwen](third_party/licenses/QWEN-COMMUNITY-LICENSE-1.0.txt) terms; the weights and external runtime are not relicensed by this repo. See [license scope](docs/ATTRIBUTION.md#license-scope). Citation metadata is available in [CITATION.cff](CITATION.cff).
