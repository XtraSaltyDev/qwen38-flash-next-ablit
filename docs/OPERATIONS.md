# Operations and reproduction

[README](../README.md) · [Results](RESULTS.md) · [Attribution](ATTRIBUTION.md) · [Recipes](../recipes/README.md) · [HF model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated)

Reference for configuring, inspecting, recovering, and evaluating the published two-Spark deployment.

Use the [README quick start](../README.md#quick-start) for first-time setup. The current published launcher is [`serve-tp2.sh`](../serve-tp2.sh), and the measured configuration is recorded in [`evidence/profile.json`](../evidence/profile.json).

## Prerequisites

Both nodes need:

- A DGX Spark with GB10 and 128 GB unified memory.
- Docker with NVIDIA CDI and RDMA device access.
- Working ConnectX RoCE networking between the nodes.
- The complete checkpoint at revision `51f24153ae9b27653a823074c631dfa941fdb536`.
- The pinned runtime image below.
- Enough free memory for a 104 GiB container cap, with no competing large-model deployment.

Follow [eugr's networking guide](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md) for link configuration and verification.

The qualified example uses:

| Node | Rank | Primary CX7 address | Second-rail address |
| --- | ---: | --- | --- |
| Head | `0` | `10.100.225.2` | `10.100.224.2` |
| Worker | `1` | `10.100.225.1` | `10.100.224.1` |

The tested socket interface was `enP2p1s0f0np0`; the RDMA HCAs were `rocep1s0f0,roceP2p1s0f0`. Replace these values with the addresses and device names on each host.

The rendezvous port defaults to `29503`. Distributed traffic must be able to pass between the paired nodes. The head API binds to localhost by default.

### Pinned inputs

| Input | Value |
| --- | --- |
| Checkpoint | `DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated` |
| Checkpoint revision | `51f24153ae9b27653a823074c631dfa941fdb536` |
| Runtime image | `eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595` |

These inputs identify the measured deployment. Current upstream documentation may have newer revisions.

## Recipe variables

Set these variables in the shell that creates the container:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RANK` | Required: `0` or `1` | Head or headless worker |
| `NODE_IP` | Required | This node's direct CX7 address |
| `MASTER_ADDR` | `10.100.225.2` | Head's direct address; same value on both nodes |
| `MASTER_PORT` | `29503` | Distributed rendezvous port |
| `SOCKET_IFNAME` | `enP2p1s0f0np0` | NCCL/Gloo/TP bootstrap interface |
| `IB_HCAS` | `rocep1s0f0,roceP2p1s0f0` | Local RDMA HCA list |
| `CONTAINER_NAME` | `qwen-flash-next-tp2-v2` | Docker container name on each host |
| `PORT` | `18383` | Head API port |
| `HOST` | `127.0.0.1` | Head API bind address |
| `CACHE_DIR` | `~/.cache/qwen-flash-next-tp2` | Node-local compilation and autotune caches |
| `MTP_TOKENS` | `4` | Speculative draft tokens |

The launcher creates a container; it does not start it. Weights are mounted read-only, and the restart policy is `unless-stopped`.

Environment changes apply when the container is created. Running `docker start` on an existing container retains its original settings.

The published profile uses TP=2, PP=1, BF16 KV cache, disk-backed PLE, MTP4, a 4096-token batch budget, 16 active sequences, and 262,144-token context. Performance or topology changes need their own validation.

Final observed KV cache capacity was **3,281,415 tokens**, approximately **12.52 full 262K contexts**. Sixteen active sequences do not imply sixteen simultaneous maximum-length requests.

## Start, inspect, and stop

Follow the README's [worker-first setup](../README.md#quick-start) when creating the deployment.

For existing containers, run this on the **worker first**, then on the **head**:

```bash
docker start qwen-flash-next-tp2-v2
```

First startup compiles and autotunes kernels and can take several minutes. Keep each node's cache for later starts.

### Inspect either node

```bash
docker inspect \
  -f 'status={{.State.Status}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}' \
  qwen-flash-next-tp2-v2

docker logs --tail 120 qwen-flash-next-tp2-v2
```

Use the container name chosen at creation if it differs from the default.

### Verify the head

```bash
curl --fail http://127.0.0.1:18383/health
curl --fail http://127.0.0.1:18383/v1/models
```

The catalog should contain `Qwen3.8-Flash-Next`, with maximum context `262144`.

Then run the README's [inference example](../README.md#6-send-a-request). A healthy catalog establishes discovery; a completed request establishes that the inference path works.

For clients on another machine, use an SSH tunnel or authenticated proxy to the head.

| Consumer | Base URL on the head |
| --- | --- |
| OpenAI-compatible client | `http://127.0.0.1:18383/v1` |
| Evaluation scripts' `--base` | `http://127.0.0.1:18383` |
| Health check | `http://127.0.0.1:18383/health` |
| Runtime metrics | `http://127.0.0.1:18383/metrics` |

The evaluation scripts append their own endpoint paths. Their `--base` value should not include `/v1`.

### Stop the pair

Run this on the **head first**, then on the **worker**:

```bash
docker stop -t 90 qwen-flash-next-tp2-v2
```

### Troubleshooting

| Symptom | First checks |
| --- | --- |
| Container name already exists | Start the existing container, or choose a separate name for a new experiment |
| Nodes do not connect | Check rank, head address, rendezvous port, local interface/HCA names, and logs on both nodes |
| Health check fails during startup | Check whether both containers are running and whether compilation or model loading is still in progress |
| A remote client cannot connect | Confirm head-local inference first, then inspect the client's tunnel or proxy, base URL, and model ID |
| Requests exceed capacity | Check total input/output length, active requests, KV usage, and preemptions |

`/metrics` exposes running and waiting requests, preemptions, speculative acceptance, and cache telemetry.

## Rollback

Before an experiment, retain the last passing containers and their caches. Give candidate containers and caches separate names so the passing deployment remains available.

The example below assumes:

- Preserved passing pair: `qwen-flash-next-tp2-v2`.
- Candidate pair: `qwen-flash-next-tp2-test`.
- The passing pair is stopped while the candidate runs.

Substitute the names used on your hosts.

### 1. Preserve candidate logs

On each node, save logs to a new file before changing the candidate:

```bash
docker logs --timestamps qwen-flash-next-tp2-test \
  > qwen-flash-next-tp2-test.log 2>&1
```

### 2. Stop the candidate

Run these commands on the **head first**, then on the **worker**:

```bash
docker update --restart=no qwen-flash-next-tp2-test
docker stop -t 90 qwen-flash-next-tp2-test
```

### 3. Restore the passing pair

Run these commands on the **worker first**, then on the **head**:

```bash
docker update --restart=unless-stopped qwen-flash-next-tp2-v2
docker start qwen-flash-next-tp2-v2
```

### 4. Verify recovery

Check health, model ID, context, actual inference, memory headroom, and any client routes used by the deployment.

If the preserved containers are unavailable, recreate the published profile using the pinned inputs and [README setup](../README.md#quick-start).

The historical eight-sequence FP8-KV reference is retained in [`recipes/baseline-tp2.sh`](../recipes/baseline-tp2.sh). Its machine-specific defaults are described in the [recipe index](../recipes/README.md).

Original dashboard/controller backups are installation-specific and are not dependencies of this recovery procedure.

## Repeat evaluations

Run from the repository root on the head against an otherwise idle endpoint. The harnesses use Python's standard library.

Each command writes `<output-directory>/results.json`. Use a new output directory for every run.

**Inspect the JSON results even when a command exits successfully.** Several scripts record failed checks without returning a failing process status. `complete: true` means the script finished; it does not mean every check passed.

### Short-prompt performance and core quality

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

Inspect `quality_summary`, individual quality cases, performance summaries, and request waves. The recorded selected profile passed 13/13 core checks.

Review external-completion flags and preemptions before comparing performance. Contaminated waves are retained in the output but excluded from the reported clean-wave summary.

### Long-prefill performance

```bash
python3 scripts/evaluate.py \
  --base http://127.0.0.1:18383 \
  --label v2-long-repeat \
  --output /tmp/qwen-v2-long-repeat \
  --mode performance \
  --concurrency 1 4 16 \
  --repeats 1 \
  --tokens 128 \
  --context-records 300
```

This adds a synthetic document of approximately 8K input tokens. Inspect actual token counts, first-token latency, completion latency, and aggregate throughput.

### Extended code, OCR, and retrieval

```bash
python3 scripts/quality_extended.py \
  --base http://127.0.0.1:18383 \
  --label v2-extended-repeat \
  --output /tmp/qwen-v2-extended-repeat \
  --fixtures evidence/invoice-fixtures.json
```

This executes generated pure-Python functions locally under the harness's AST checks and resource limits, then checks synthetic invoice OCR and near-context retrieval. Run it on Linux.

The recorded result was **7/8 total checks**, retaining all seven baseline passes. The remaining failure was the existing no-import instruction violation in `sliding_max`. Compare individual cases; a different set of seven passes is not equivalent.

Use the corrected `quality_extended.py`. The retained `quality_extended.initial.py` contains an inconsistent duration prompt and is historical.

### Strict near-context formatting

```bash
python3 scripts/near_context_control.py \
  --base http://127.0.0.1:18383 \
  --label v2-context-repeat \
  --output /tmp/qwen-v2-context-repeat \
  --samples 10 \
  --require-strict \
  --initial-cache-state unspecified
```

This checks strict JSON, semantic retrieval after removing a possible code fence, and a separate JSON-schema control at exactly 261,623 input tokens.

With `--require-strict`, strict-format or schema failure produces a failing exit status. Inspect `summary` for the pass counts and cache-state label.

Identical requests exercise prefix reuse. The cache-state flag records a label; it does not clear the cache. Label a run `cold` only when that state has been established.

### Word-count repeatability

```bash
python3 scripts/repeat_control.py \
  --base http://127.0.0.1:18383 \
  --label v2-word-count-repeat \
  --output /tmp/qwen-v2-word-count-repeat \
  --samples 100
```

Inspect `quality_summary` and the individual responses. The recorded selected result was 100/100 on one repeated prompt, compared with 95/100 for the reference. This is a repeatability check, not 100 independent questions.

### Returned target-token audit

```bash
python3 scripts/audit_target.py \
  --base http://127.0.0.1:18383 \
  --label v2-audit-repeat \
  --output /tmp/qwen-v2-audit-repeat
```

Inspect `greedy_target_audit` and the quality entries, including the thinking-on check. The recorded audit had zero mismatches over 384 returned tokens.

This compares emitted greedy tokens with returned target top-1 log probabilities. It is not a full-vocabulary distribution-equivalence test.

### Comparing runs

Match checkpoint, runtime image, prompts, output lengths, concurrency, sampling, warmups, and cache policy. Inspect per-fixture regressions, contaminated waves, startup reliability, preemptions, and host memory alongside aggregate speed.

[`protocol.json`](../evidence/protocol.json) preserves the evolving experiment. Its initial fixed FP8-KV setting was superseded by the final BF16 treatment. Use [`profile.json`](../evidence/profile.json) for the selected configuration and [RESULTS.md](RESULTS.md) for interpretation.
