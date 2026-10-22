# Historical reference and experimental recipes

[README](../README.md) · [Operations](../docs/OPERATIONS.md) · [Results](../docs/RESULTS.md) · [Attribution](../docs/ATTRIBUTION.md) · [HF model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated)

Use the root **[`serve-tp2.sh`](../serve-tp2.sh)** for the published BF16-KV/16-sequence profile.

The scripts in this directory preserve the reference configuration and experiments from the serving study. Their names and defaults reflect that history.

## Recipe index

| File | PLE placement | KV cache | Active sequences | Status |
| --- | --- | --- | --- | --- |
| [`baseline-tp2.sh`](baseline-tp2.sh) | Disk | FP8 | 8 | Original comparison reference |
| [`experiment-shared-ple.sh`](experiment-shared-ple.sh) | Shared by default | FP8, manual 32 GiB per rank by default | 8 by default; configurable | Experimental shared-PLE candidate builder |
| [`shared-ple-rejected.sh`](shared-ple-rejected.sh) | Shared by default | FP8, manual 32 GiB per rank by default | 16 by default | Rejected after strict near-context formatting regressed |
| [`experiment-disk16.sh`](experiment-disk16.sh) | Disk | FP8 | 16 | Rejected after reproducing the formatting regression |
| [`experiment-bf16kv16.sh`](experiment-bf16kv16.sh) | Disk | BF16 via `auto` | 16 | Winning treatment before final publication packaging |

The shared-PLE candidate builder also supported speculative-configuration experiments. Adaptive MTP failed to reach a healthy server on the pinned graph/kernel path.

The selected treatment is `bf16kv-seq16`. Historical `selected-*` evidence names identify an earlier shared-memory candidate, not the final published profile.

See [experiment outcomes](../docs/RESULTS.md#experiments-retained) for the measured decisions.

## Historical defaults

These files retain machine-specific network settings and container names. They do not expose every override available in the root launcher.

| Recipe | Default container name | Rendezvous port |
| --- | --- | ---: |
| `baseline-tp2.sh` | `qwen-flash-next-tp2` | `29501` |
| Shared-PLE scripts | `qwen-flash-next-tp2-v2`, configurable | `29503` |
| `experiment-disk16.sh` | `qwen-flash-next-tp2-v2-disk16` | `29503` |
| `experiment-bf16kv16.sh` | `qwen-flash-next-tp2-v2-bf16kv16` | `29503` |

The scripts default to head API port `18383`. Some historical usage comments retain earlier launcher names or settings; inspect the actual variable assignments and Docker arguments when reproducing a historical treatment.

Each script creates a container. Starting it is a separate step. The same two-node prerequisites and worker-first startup order apply.

Several historical recipes share a default container name or API port with the published deployment. Keep passing containers and caches available when preparing an experiment. See [rollback](../docs/OPERATIONS.md#rollback) for the preserved-container procedure.

## Reproduction and evidence

For a new deployment, follow the root [quick start](../README.md#quick-start).

For controlled experiments, use the [evaluation instructions](../docs/OPERATIONS.md#repeat-evaluations), match the pinned inputs, and keep outputs in new directories. Historical scripts and recorded results are references, not a guarantee that a changed environment will reproduce the same outcome.

## Attribution and license

These recipes adapt DevelopingDad's launcher and eugr's serving/cluster foundation.

See the root [NOTICE](../NOTICE), [attribution and license scope](../docs/ATTRIBUTION.md), and [retained third-party notices](../third_party/). Upstream portions retain their applicable terms.
