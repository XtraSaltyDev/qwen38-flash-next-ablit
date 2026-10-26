# Qualification results

[README](../README.md) · [Operations](OPERATIONS.md) · [Attribution](ATTRIBUTION.md) · [Recipes](../recipes/README.md) · [HF model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated)

Controlled serving measurements from **October 22, 2026**, on two GB10 DGX Sparks.

These results compare serving configurations using the same abliterated checkpoint. They do not compare this derivative against NVIDIA's unmodified model or establish a general capability benchmark.

## Selected profile

| Setting | Eight-sequence reference | Selected profile |
| --- | --- | --- |
| Active sequences | 8 | 16 |
| KV cache | FP8 | BF16 |
| PLE / n-gram embeddings | Disk-backed | Disk-backed |
| Speculative decoding | MTP4 | MTP4 |
| Maximum context | 262,144 tokens | 262,144 tokens |
| Batch token budget | 4096 | 4096 |
| GPU memory utilization | 0.8 | 0.8 |

Both profiles used:

- Checkpoint `DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated`, revision `51f24153ae9b27653a823074c631dfa941fdb536`.
- Runtime image `eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595`.
- Tensor parallelism 2 across two DGX Sparks.
- A 104 GiB container memory cap on each node.

The selected treatment is **`bf16kv-seq16`**. It retained all baseline-passing quality checks and passed the recorded managed-restart and client-route checks.

The machine-readable summary is [`report.json`](../evidence/report.json), with configuration in [`profile.json`](../evidence/profile.json) and validation in [`qualification.json`](../evidence/qualification.json).

Historical `selected-*` run names refer to the shared-memory candidate before its rejection. The final published selection is `bf16kv-seq16`.

## Performance

| Workload | Eight-sequence reference | Selected profile |
| --- | ---: | ---: |
| 16 concurrent short requests: aggregate output | 229.0 tok/s | **313.4 tok/s (+37%)** |
| Same workload: p95 first token | 11.89 s | **0.77 s** |
| Same workload: p95 completion | 21.68 s | **15.84 s** |
| 16 concurrent ~8K-input requests: aggregate output | 38.64 tok/s | **40.98 tok/s (+6%)** |

The selected profile handled the sixteen-request wave without the reference's eight-sequence scheduling limit. Both sequence capacity and KV precision changed between the headline configurations; the experiments below provide the intermediate controls.

### Measurement method

| Workload | Input tokens per request | Output tokens per request | Repeats per fixture |
| --- | ---: | ---: | --- |
| Short code/prose | 62–64 | 256 | Reference: 1; selected: 2 |
| Synthetic long-prefill document | 8174–8176 | 128 | Reference: 1; selected: 1 |

Requests used temperature 0, seed 3741, thinking off, fixture warmups, and stream-specific prefixes. Output lengths were forced for the performance probes.

- **Aggregate throughput:** total output tokens divided by elapsed request-wave time, including queue and prefill time. The reported value is the median across clean waves.
- **Per-stream decode:** generation rate after the first content token.
- **TTFT:** time to first content token.
- **Completion latency:** time from sending a request to the end of its response.
- **p95:** nearest-rank percentile over the recorded stream samples.
- **C4, C8, C16:** waves of 4, 8, or 16 concurrent requests.

Waves with detected external completions remain in the evidence but are excluded from the clean-wave summaries.

These are small controlled samples, not production latency guarantees.

### Tradeoffs

Single-stream coding decode stayed close to the reference. Serial short-prose decode was approximately **8% slower**, while the combined single-stream decode summary remained within the original 5% floor. These are different aggregations.

Aggregate throughput at four and eight concurrent requests remained within the predefined 5% floor. The largest benefit appeared at sixteen concurrent requests.

Long-prefill throughput improved modestly. At sixteen concurrent ~8K-input requests, p95 first-token latency improved by about 9%; prefill remained the main latency bottleneck.

## Quality diagnostics

| Check | Recorded result | Scope |
| --- | --- | --- |
| Core checks | **13/13 passed** | Deterministic responses, tools, and retrieval |
| Extended suite | **7/8 passed** | Retained all seven baseline passes |
| Synthetic invoice OCR | **3/3 passed** | Included in the extended suite |
| Strict near-context JSON | **10/10 passed** | 261,623 input tokens |
| Structured JSON control | Passed | Separate schema-constrained request |
| Word-count repeatability | **100/100**, reference **95/100** | One repeated prompt |
| Returned target-token audit | **0 mismatches / 384 tokens** | Returned target top-1 comparison |
| Thinking-on arithmetic | Passed | Bounded functional probe |

Both the reference and selected profile violated the no-import instruction in `sliding_max`. The selected profile preserved the other seven extended checks; the result is not a perfect extended-suite pass.

The ten near-context requests reused an identical prompt and exercised prefix caching. The selected run's first request was already warm from the extended evaluation. This was not ten independent cold-prefill tests.

Strict JSON requires the expected object without a code fence. The semantic control permits removal of a surrounding fence, and the structured control requests a JSON schema. Those checks answer different questions.

The target-token audit does not establish full-vocabulary distribution equivalence. The repeatability and retrieval fixtures do not establish broad capability retention.

The [HF model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated#evaluation) describes separate single-Spark model-level tests. This two-Spark validation does not extend the single-Spark FP8-KV results.

## Deployment verification

On the original deployment:

- A managed stop/start and post-restart checks passed.
- Two further strict JSON and semantic retrieval requests passed, along with the structured JSON control. The restart record labels the initial cache state as cold; subsequent identical requests exercise prefix reuse.
- Both ranks remained healthy without OOM, benchmark preemption, or unexpected restart in the qualified runs.
- The canonical model catalog, existing API-route inference, and Goose generation passed.
- Codex Spark/mobile model, tool, and recovery checks passed.
- OpenCode/Herd evidence covered catalog, selection, and the shared route, rather than a full native OpenCode agent task.

These are recorded checks of the original installation. They do not establish that another installation or client configuration has passed.

## Experiments retained

| Candidate | Outcome |
| --- | --- |
| Shared PLE, eight sequences, manual 32 GiB KV per rank | Modest concurrent gain; investigated repeatability and the warm-start memory guard |
| Shared PLE, sixteen sequences, fixed MTP4 | Rejected: strict near-context JSON fell from reference 10/10 to 0/10; semantic and schema controls passed |
| Adaptive MTP counts | Rejected before scoring: B12X/CuTe CUDA-graph preparation did not reach a healthy server |
| Disk PLE, sixteen sequences, FP8 KV | Rejected: reproduced the strict-format regression without shared PLE |
| Disk PLE, sixteen sequences, BF16 KV | Selected: retained baseline-passing quality checks and concurrent-response gains |

The disk/FP8 intermediate isolated the formatting issue from shared-memory placement. Switching that treatment to BF16 KV restored the tested strict-format floor.

[`protocol.json`](../evidence/protocol.json) records the evolving experiment, including an initial fixed FP8-KV setting. The later BF16 treatment and final decision are documented in the [profile](../evidence/profile.json) and [report](../evidence/report.json).

Failed attempts, corrected fixture records, and rejected results remain available for review.

## Latency-v3 follow-up: archived

On **October 26, 2026**, the later latency-v3 study was assigned to the local
archive, with no candidate promoted. The published selection remains
`bf16kv-seq16`: BF16 KV, sixteen sequences, batch budget 4096, and MTP4.

The tested scheduler and batch-budget candidates failed performance gates.
The additive near-tail/MTP3 candidate improved warm document-extension latency,
but two matched cold 8K serial coding controls lost **5.85% and 7.37%** aggregate
output throughput, exceeding the study's 5% regression floor. Its full quality
checks were not run after that rejection. The adaptive successor failed startup;
CPU policy checks and GPU state-copy probes did not establish serving readiness.
No v3 candidate completed all performance, quality, capacity, restart, and route
gates. These findings do not extend or replace the v2 qualification above.

The results, experimental harnesses, patches, provenance, and license notices are
preserved with checksums in the maintainer's Git-ignored local archive at
`.local/archive/latency-v3/`, alongside the original installation bundle. The
staged v3 `PUBLICATION.json` is retained there as a historical file manifest; it
does not indicate that v3 was published. Archiving changed repository packaging
only and did not inspect or alter the live deployment.

## Capacity and checkpoint review

Final observed KV cache capacity was **3,281,415 tokens**, approximately **12.52 full 262K-context equivalents**, with 16 active-sequence slots. The original capacity floor of eight full-context equivalents was retained.

Capacity equivalents describe the observed cache allocation; they do not promise sixteen simultaneous maximum-length requests.

The checkpoint review covered **375,487 tensor entries**:

- All 48 main attention output projections were MXFP8.
- The MTP output projection remained BF16.
- PLE storage accounted for approximately 26.82 GiB total, or 13.41 GiB per TP rank.
- The reviewed attention, linear-head, and expert dimensions divided evenly across TP=2.

No checkpoint re-encoding or weight training was performed in this serving study.

## Evidence and reproduction

| Evidence | Location |
| --- | --- |
| Final report | [`evidence/report.json`](../evidence/report.json) |
| Selected configuration | [`evidence/profile.json`](../evidence/profile.json) |
| Validation summary | [`evidence/qualification.json`](../evidence/qualification.json) |
| Fresh short-prompt reference | [`baseline-control`](../evidence/runs/baseline-control/results.json) |
| Selected short-prompt/core run | [`bf16kv-seq16`](../evidence/runs/bf16kv-seq16/results.json) |
| Selected extended checks | [`bf16kv-seq16-extended`](../evidence/runs/bf16kv-seq16-extended/results.json) |
| Selected near-context controls | [`bf16kv-seq16-context-control`](../evidence/runs/bf16kv-seq16-context-control/results.json) |
| Post-restart context checks | [`managed-restart-context.json`](../evidence/managed-restart-context.json) |
| Other runs and experiments | [`evidence/runs/`](../evidence/runs/) |

Use the [reproduction commands](OPERATIONS.md#repeat-evaluations) to run new checks with new output directories.

Future work should prioritize long-prefill latency and the serial-prose tradeoff while retaining the strict-format, quality, restart, route, and memory checks. The rejected adaptive-MTP path needs runtime/kernel compatibility work before another performance comparison.
