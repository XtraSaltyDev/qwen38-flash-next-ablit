# Qualification results

These are controlled synthetic measurements from October 22, 2026, not a general model benchmark or production latency guarantee. The detailed authoritative evidence is [`evidence/report.json`](../evidence/report.json) and its referenced raw files.

## Selected profile

The winning treatment was `bf16kv-seq16`: original disk PLE placement and 0.8 memory utilization, 16 active slots, BF16 KV cache, fixed MTP4, 4096-token batch budget, and 262144-token context. Model weights and the runtime image remained pinned. All baseline-passing quality gates held; the selected configuration also passed a managed stop/start and post-restart checks.

| Workload | Original eight-slot baseline | Selected v2 | Interpretation |
| --- | --- | --- | --- |
| 16 simultaneous short code/prose prompts, aggregate output | 229.0 tok/s | 313.4 tok/s | About 37% higher throughput |
| Same short-prompt workload, p95 first token | 11.89 s | 0.77 s | Sixteen active slots remove the observed eight-slot queue |
| Same short-prompt workload, p95 completion | 21.68 s | 15.84 s | About 27% faster |
| 16 clients with approximately 8K input tokens, aggregate output | 38.64 tok/s | 40.98 tok/s | About 6% higher throughput; p95 first-token latency improved about 9% |

The short-prompt comparison used 62–64 input tokens and 256 forced output tokens, with temperature 0, seed 3741, thinking disabled, warmups, and stream-specific prefixes. The fresh baseline control had one repeat per fixture; v2 had two. Reported p95 values use nearest rank on a small sample. Long-prefill tests used 128 output tokens and one repeat per fixture. These workloads do not establish a universal optimum.

Single-stream coding decode stayed close to the reference; short-prose decode was approximately 8% slower. C4/C8 aggregate throughput remained within the predefined 5% floor. Long-prefill latency remains the main optimization target.

## Quality and reliability evidence

- Core deterministic, tool, and retrieval checks: 13/13.
- Extended baseline-passing checks retained: 7/7, including 3/3 synthetic invoice OCR fixtures. Both baseline and v2 violated the no-import instruction in `sliding_max`; that existing failure remains recorded.
- Strict raw JSON retrieval at a 261623-token prompt: 10/10, plus a passing JSON-schema control. Post managed restart: two further raw JSON and semantic retrieval passes plus schema success.
- One repeated word-count prompt: 100/100 versus baseline 95/100. This measures repeatability on one prompt, not 100 independent questions.
- Target greedy audit: 384 returned tokens, zero top-1 mismatches on the audited fixtures. This is not a full-vocabulary distribution comparison.
- Thinking-on arithmetic probe passed. Both ranks stayed healthy without OOM, benchmark preemption, or unexpected restart in the qualified runs.
- Managed dashboard restart, canonical model catalog, existing API route inference, Goose generation, and Codex Spark/mobile model/tool/recovery checks passed. OpenCode/Herd evidence covers catalog, selection, and the shared route, rather than a full native OpenCode agent task.

Final observed cache capacity was approximately 12.52 full 262K-context equivalents, with 16 scheduler slots. The original minimum of eight full contexts was retained.

## Experiments retained in this archive

| Candidate | Outcome |
| --- | --- |
| Shared PLE, eight slots, manual 32 GiB KV | Modest concurrent gain; investigated repeatability and warm-start memory guard |
| Shared PLE, sixteen slots, fixed MTP4 | Fast, but rejected after strict near-context JSON formatting fell from baseline 10/10 to 0/10; semantic/schema controls passed |
| Adaptive MTP counts | Rejected before scoring: B12X/CuTe CUDA-graph preparation failed to reach a healthy server |
| Disk PLE, sixteen slots, FP8 KV | Rejected after the same strict-format regression; isolated the behavior from shared placement |
| Disk PLE, sixteen slots, BF16 KV | Selected: retained baseline-passing quality gates and concurrent response gains |

Historical `selected-*` files refer to the shared-memory candidate before its later rejection. The final selection is `bf16kv-seq16`, documented by [report](../evidence/report.json), [profile](../evidence/profile.json), [qualification](../evidence/qualification.json), and [recipe](../serve-tp2.sh). Failed attempts, logs, and corrected harness records are preserved rather than overwritten.

The checkpoint review covered 375487 tensor entries. All 48 main attention output projections were MXFP8, while the MTP output projection remained BF16. PLE storage accounts for approximately 26.82 GiB total, or 13.41 GiB per TP rank. The reviewed attention, linear-head, and expert dimensions divide evenly across TP=2. No checkpoint re-encoding or weight training was performed.

Future work should prioritize long-prefill latency and the serial-prose tradeoff while retaining the same strict-format, quality, restart, routing, and memory gates. The rejected adaptive-MTP path needs runtime/kernel compatibility work before it is eligible for another performance comparison.
