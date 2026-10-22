# Qwen3.8-Flash-Next on 2× DGX Spark

_by [XtraSaltyDev](https://github.com/XtraSaltyDev)_

Serve [DevelopingDad's Qwen3.8-Flash-Next-FP4-Local-Abliterated](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated) across two DGX Sparks using vLLM and the B12X kernel stack.

**TP=2 · 262K context · 16 active sequences · BF16 KV · MTP4 · disk PLE**

This companion repository provides the two-node launch recipe, evaluation scripts, and measured results. The checkpoint is served unchanged. For the model's method, model-level evaluation, and single-Spark instructions, see the [Hugging Face model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated).

[Performance](#performance) · [Quick start](#quick-start) · [API settings](#api-settings) · [Validation](#validation) · [Reproduce the results](#reproduce-the-results)

## Performance

This profile improves concurrent-request handling over our original eight-slot configuration while retaining the baseline-passing validation checks, including strict JSON retrieval near the 262K context limit.

Measured on two GB10 DGX Sparks on **October 22, 2026**:

| Workload | Eight-slot reference | This recipe |
| --- | ---: | ---: |
| 16 concurrent short requests: aggregate output | 229.0 tok/s | **313.4 tok/s (+37%)** |
| Same workload: p95 first token | 11.89 s | **0.77 s** |
| Same workload: p95 completion | 21.68 s | **15.84 s** |
| 16 concurrent ~8K-input requests: aggregate output | 38.64 tok/s | **40.98 tok/s (+6%)** |

Both profiles use the same abliterated checkpoint, runtime image, disk PLE, and MTP4. The reference uses eight active sequences and FP8 KV; this recipe uses sixteen active sequences and BF16 KV.

Short tests mix code and prose, with 62–64 input tokens, 256 forced output tokens, greedy decoding, and thinking off. Aggregate throughput includes queue and prefill time across the whole request wave. It is not a single user's decode rate, and these small controlled samples do not establish production latency guarantees.

**Tradeoffs:** single-stream coding stayed close to the reference, but serial prose decode was about 8% slower. Long prompts still spend substantial time in prefill.

See [full results and methodology](docs/RESULTS.md), the [report](evidence/report.json), and [raw runs](evidence/runs).

## Requirements

On each Spark:

- GB10 with 128 GB unified memory.
- Docker with NVIDIA CDI and RDMA device access.
- Working ConnectX RoCE networking between the nodes. Follow [eugr's networking guide](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md).
- The Hugging Face CLI, `hf`; see the [installation guide](https://huggingface.co/docs/huggingface_hub/guides/cli).
- Approximately 102 GB for a complete copy of the checkpoint, plus space for the Docker image and compilation caches.
- Enough free unified memory for a 104 GiB container cap. Avoid running another large model alongside this deployment.

Python 3 is required to run the evaluation scripts.

The recipe uses a pinned B12X runtime image compatible with NVIDIA's FP4-Local storage format. Keep the image digest and checkpoint revision below when reproducing the published results.

## Quick start

Run the preparation steps on **both Sparks**, then start the worker before the head.

`serve-tp2.sh` **creates a container**. It does not download weights or start the server. If the named containers already exist, use the [existing-container instructions](#start-and-stop-existing-containers).

### 1. Prepare both nodes

On each Spark:

```bash
git clone https://github.com/XtraSaltyDev/qwen38-flash-next-ablit.git
cd qwen38-flash-next-ablit

hf download DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated \
  --revision 51f24153ae9b27653a823074c631dfa941fdb536 \
  --local-dir "$HOME/models/qwen38-flash-next-ablit"

docker pull eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595
```

The checkpoint revision identifies the artifact used in the serving study. The current Hugging Face card describes the model and its latest documentation.

### 2. Configure networking

The examples use this topology:

| Node | Rank | Direct CX7 address | Role |
| --- | ---: | --- | --- |
| Head | `0` | `10.100.225.2` | Serves the API and participates in inference |
| Worker | `1` | `10.100.225.1` | Participates in inference; no API |

The qualified setup also used a second rail at `10.100.224.2` on the head and `10.100.224.1` on the worker. See [network prerequisites](docs/OPERATIONS.md#prerequisites) for the complete example.

In the terminal you will use on **each node**, set the head address, local network interface, and local RDMA device names:

```bash
export MASTER_ADDR=10.100.225.2
export SOCKET_IFNAME=enP2p1s0f0np0
export IB_HCAS=rocep1s0f0,roceP2p1s0f0
```

These are the tested setup's values. Replace the addresses and device names with those for your hosts. `MASTER_ADDR` must identify the same head on both nodes; interface and HCA names must match the local node.

The distributed rendezvous port defaults to `29503`. Full options are in the [configuration reference](docs/OPERATIONS.md#recipe-variables).

### 3. Start the worker

On the worker, from the repository root:

```bash
RANK=1 NODE_IP=10.100.225.1 \
  bash serve-tp2.sh "$HOME/models/qwen38-flash-next-ablit"

docker start qwen-flash-next-tp2-v2
```

Use the worker's actual CX7 address for `NODE_IP`.

### 4. Start the head

On the head, from the repository root:

```bash
RANK=0 NODE_IP=10.100.225.2 \
  bash serve-tp2.sh "$HOME/models/qwen38-flash-next-ablit"

docker start qwen-flash-next-tp2-v2
docker logs --follow qwen-flash-next-tp2-v2
```

Use the head's actual CX7 address for `NODE_IP`.

First launch compiles and autotunes kernels and can take several minutes. Later starts reuse each node's persistent cache.

### 5. Check readiness

In another terminal **on the head**:

```bash
curl --fail http://127.0.0.1:18383/health
curl --fail http://127.0.0.1:18383/v1/models
```

Wait for the health check to succeed. The model catalog should contain **`Qwen3.8-Flash-Next`**, with a maximum context of **262144**.

For startup problems, inspect the logs on both nodes. See [operations](docs/OPERATIONS.md#start-inspect-and-stop) for container status and memory checks.

### 6. Send a request

On the head:

```bash
curl --fail http://127.0.0.1:18383/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function that merges overlapping intervals."
      }
    ],
    "temperature": 0,
    "max_tokens": 512,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }'
```

The API binds to localhost on the head by default. For clients on another machine, use an SSH tunnel or authenticated proxy to the head.

### Start and stop existing containers

For later starts, run this command on the **worker first**, then on the **head**:

```bash
docker start qwen-flash-next-tp2-v2
```

Check head health before sending requests. Existing containers retain the configuration supplied when they were created.

To stop the deployment, run this command on the **head first**, then on the **worker**:

```bash
docker stop -t 90 qwen-flash-next-tp2-v2
```

## API settings

The two-Spark recipe has different API defaults from the model card's single-Spark launcher:

| Setting | HF single-Spark launcher | This two-Spark recipe |
| --- | --- | --- |
| Base URL | `http://127.0.0.1:8000/v1` | `http://127.0.0.1:18383/v1` on the head |
| API model ID | `qwen-trial117` | **`Qwen3.8-Flash-Next`** |
| KV cache | FP8 | BF16 |
| Active sequences | Up to 8 | Up to 16 |

Thinking is controlled per request with `chat_template_kwargs.enable_thinking`. The recipe enables the `qwen3` reasoning parser and `qwen3_xml` tool-call parser. See the [model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated#quick-start-dgx-spark) for additional request examples, using this recipe's base URL and model ID.

## Default profile

| Setting | Value |
| --- | --- |
| Parallelism | TP=2, PP=1, native multiprocessing across two nodes |
| Maximum context | 262,144 tokens |
| Active sequences | 16 |
| Batch token budget | 4096 |
| KV cache | BF16 (`auto` with BF16 model dtype) |
| PLE / n-gram embeddings | Disk-backed, io_uring |
| Speculative decoding | MTP, four draft tokens |
| Memory | GPU utilization 0.8; container cap 104 GiB per node |
| Prefix caching / chunked prefill | Enabled |

Final observed KV cache capacity was **3.28M tokens**, approximately **12.52 full 262K contexts**. Sixteen scheduler slots do not mean sixteen simultaneous maximum-length requests.

The tested runtime was vLLM `0.1.dev21830+g8244d1c44.d20261015` with Torch `2.13.0+cu130` in the pinned image.

See [operations](docs/OPERATIONS.md) for network variables, cache paths, startup checks, and rollback. Performance or topology changes need their own validation.

## Validation

The selected profile retained all baseline-passing checks:

| Check | Recorded result |
| --- | --- |
| Core deterministic, tool, and retrieval checks | **13/13 passed** |
| Baseline-passing extended checks, including synthetic invoice OCR | **7/7 retained** |
| Strict JSON retrieval at 261,623 input tokens | **10/10 passed** |
| Managed restart and client/API route tests | Passed on the original deployment |

Both the reference and selected profile violated the no-import instruction in one code fixture. That existing failure remains recorded.

These are bounded diagnostics and deployment checks, not a general capability benchmark. The near-context result belongs to this two-Spark BF16-KV profile; it does not extend the model card's single-Spark FP8-KV validation.

Full results, rejected experiments, and known limitations are in [RESULTS.md](docs/RESULTS.md), the [protocol](evidence/protocol.json), and [qualification record](evidence/qualification.json).

## Reproduce the results

Run from the repository root on the head against an otherwise idle deployment. The harness uses Python's standard library.

To repeat the short-prompt performance and core quality checks:

```bash
python3 scripts/evaluate.py \
  --base http://127.0.0.1:18383 \
  --label v2-repeat \
  --output /tmp/qwen-v2-repeat \
  --mode all \
  --concurrency 1 4 8 16 \
  --repeats 2 \
  --tokens 256
```

Results are written to `/tmp/qwen-v2-repeat/results.json`. Use a new output directory for each run.

See [repeat evaluations](docs/OPERATIONS.md#repeat-evaluations) for long-prefill, extended quality, near-context, repeatability, and target-token checks. Match the checkpoint, image, workload, sampling, and cache policy when comparing configurations.

### Verify repository files

From the repository root on Linux:

```bash
sha256sum -c SHA256SUMS
```

This checks the published repository files listed in the manifest. Model weights are downloaded separately and have their own release checksums.

## Repository layout

| Path | Contents |
| --- | --- |
| [serve-tp2.sh](serve-tp2.sh) | Published serving profile with configurable network and container names |
| [scripts/](scripts) | Performance, code/tool/OCR/retrieval, repeatability, and target-token checks |
| [evidence/](evidence) | Reports, raw runs, startup/failure logs, checkpoint review, and synthetic fixtures |
| [recipes/](recipes) | Historical reference and experimental configurations; use the root launcher for this profile |
| [docs/](docs) | Operations, results, attribution, and license scope |
| [third_party/](third_party) | Pinned source references and retained upstream notices |

## Credits and license

**[DevelopingDad](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated)** created the abliteration and MXFP8 re-encode, derived from **[NVIDIA's quantized checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)** and **[Qwen's original model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**. **[eugr / Eugene Rakhmatulin](https://github.com/eugr/spark-vllm-docker)** provides the Spark container and cluster-recipe foundation, with inference work from **[local-inference-lab](https://github.com/local-inference-lab)**, **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)**, and **[vLLM](https://github.com/vllm-project/vllm)**. This repo contributes the TP=2 serving study, harnesses, and deployment recipe.

README organization takes inspiration from [MiaAI-Lab's Spark recipes](docs/ATTRIBUTION.md#recipe-presentation-references). Full source credits and retained notices are in [attribution](docs/ATTRIBUTION.md) and [NOTICE](NOTICE).

Original contributions here use [MIT](LICENSE). Checkpoint-derived material retains its upstream [NVIDIA](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) and [Qwen](third_party/licenses/QWEN-COMMUNITY-LICENSE-1.0.txt) terms; the weights and external runtime are not relicensed by this repo. See [license scope](docs/ATTRIBUTION.md#license-scope). Citation metadata is available in [CITATION.cff](CITATION.cff).
