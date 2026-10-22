# Attribution and provenance

[README](../README.md) · [Operations](OPERATIONS.md) · [Results](RESULTS.md) · [Recipes](../recipes/README.md) · [HF model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated)

Model lineage, serving foundations, retained source records, and license scope for this repository.

## Model lineage

| Contributor | Contribution | Primary source |
| --- | --- | --- |
| Qwen | Original Qwen3.8-Flash-Next architecture and model | [Model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| NVIDIA | ModelOpt/QAD mixed-precision FP4-Local checkpoint and reference serving guidance | [Pinned quantized base](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300) |
| DevelopingDad | Refusal-direction intervention, baked checkpoint, MXFP8 re-encode, model validation, and single-Spark launcher | [HF model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated) |
| XtraSaltyDev | Two-Spark serving comparison, evaluation harnesses, BF16-KV/16-sequence validation, deployment verification, and packaging | [This repository](https://github.com/XtraSaltyDev/qwen38-flash-next-ablit) |

The serving study uses DevelopingDad's checkpoint unchanged at revision `51f24153ae9b27653a823074c631dfa941fdb536`.

The model's abliteration search, bake, and MXFP8 re-encode are upstream work. Their method and model-level measurements are described in the HF card.

## Current documentation and retained snapshots

| Record | Purpose |
| --- | --- |
| [Current HF model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated) | Current model description and usage guidance |
| [Current HF provenance](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated/blob/main/provenance.json) | Upstream build and validation summary |
| [Retained provenance snapshot](../third_party/model/provenance.json) | Historical provenance from the checkpoint revision used in this study |
| [Retained upstream notice](../third_party/model/NOTICE.txt) | Upstream attribution and notices |
| [`third_party/SOURCES.json`](../third_party/SOURCES.json) | Exact source URLs, revisions, and hashes for retained files |

The retained provenance records several stages of the model's history. Its earlier export fields describe BF16 attention output projections and a 4K/one-sequence smoke profile. Its later `attention_mxfp8_reencode` section records the October 22 MXFP8 conversion.

Those earlier fields are historical. The measured checkpoint's main attention output projections were MXFP8, and the published two-Spark profile uses BF16 KV, 16 active sequences, and 262K context.

Paths inside the copied upstream notice refer to the original HF repository. In particular, its `LICENSE` reference identifies the NVIDIA Open Model Agreement on HF. This GitHub repository's root [LICENSE](../LICENSE) covers its original MIT contributions.

Retained snapshots preserve provenance; current deployment settings are documented in the [README](../README.md) and [selected profile](../evidence/profile.json).

## Serving and kernel foundation

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker), Eugene Rakhmatulin and contributors: Docker images, Spark cluster/networking integration, and the [cluster recipe](../third_party/eugr-cluster-recipe.yaml) consulted during the study. Its [MIT notice](../third_party/licenses/EUGR-MIT.txt) is retained.
- [DevelopingDad's pinned serve-spark.sh](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated/blob/51f24153ae9b27653a823074c631dfa941fdb536/serve-spark.sh): the immediate single-node launcher foundation adapted into the two-node recipe.
- [vLLM](https://github.com/vllm-project/vllm): serving, scheduling, KV-cache management, speculative decoding, and OpenAI-compatible APIs.
- [local-inference-lab](https://github.com/local-inference-lab), including its [FlashInfer fork](https://github.com/local-inference-lab/flashinfer): B12X inference/kernel integration used by the eugr runtime.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [PyTorch](https://github.com/pytorch/pytorch), and NVIDIA CUDA/NCCL: runtime and kernel dependencies supplied through the separately downloaded image.

The runtime image is pinned by digest. The eugr source snapshot retained for attribution is not asserted to be that image's build commit.

## Changes made here

This repository contributes:

- Native two-node rank wiring and NVIDIA CDI device use.
- The selected BF16-KV/16-sequence serving profile.
- Matched serving comparisons, code/tool/OCR/retrieval fixtures, repeatability controls, and returned target-token audits.
- Recorded restart and client-route verification on the original deployment.
- Portable network/container settings and public documentation.

The study retained the checkpoint revision, runtime image, disk-backed PLE, MTP4, original memory profile, and 262K context. Rejected candidates and failures are documented in [RESULTS.md](RESULTS.md).

The published launcher exposes network and container names as environment variables and includes attribution comments. Its default Docker invocation was checked against the qualified recipe.

[`evidence/profile.json`](../evidence/profile.json) retains separate hashes for the original qualified recipe and the published launcher. Publication packaging did not change the live containers.

## License scope

The root [MIT license](../LICENSE) covers original code and documentation contributed by XtraSaltyDev. Third-party material retains its applicable terms.

| Material | Applicable scope |
| --- | --- |
| Checkpoint and checkpoint-derived material | The upstream card identifies the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) and [Qwen Community License 1.0](../third_party/licenses/QWEN-COMMUNITY-LICENSE-1.0.txt). Weights are downloaded separately. |
| Adapted upstream launcher portions | Retain their applicable upstream terms; the repository's MIT grant does not supersede them |
| eugr recipe portions and retained snapshot | [MIT notice](../third_party/licenses/EUGR-MIT.txt), copyright Eugene Rakhmatulin |
| Retained third-party notices and provenance | Original owners' notices and terms, identified by the source records |
| External runtime image and libraries | Their own distribution terms |
| Original harnesses and documentation | MIT, subject to the third-party scope above |

The model card and full license texts provide the governing model terms. This repository does not relicense the checkpoint or external runtime.

No upstream endorsement is implied.

## Public evidence and local archive

Public evidence includes measurements, synthetic fixtures, experimental failures, and tensor metadata.

Personal client backups, SSH/tunnel registrations, dashboard source snapshots, full Docker inspections, and original local copy manifests are excluded from Git. They remain part of the original installation's local archive and are not required for the public setup instructions.

Published logs replace local usernames, management endpoints, and host labels. The report omits personal connections and detailed runtime commands/environment; measured values remain unchanged.

- [`evidence/PUBLICATION.json`](../evidence/PUBLICATION.json) records original and published hashes and transformations for the archived evidence.
- [`SHA256SUMS`](../SHA256SUMS) verifies the listed publication files, excluding itself.
- [`CITATION.cff`](../CITATION.cff) provides citation metadata for the serving recipe and evaluation.

## Recipe presentation references

The README layout was informed by MiaAI-Lab's practical Spark recipes: concise descriptions, measured performance, requirements, launch commands, configuration, validation, and credits.

- [Qwen3.8-Flash-Next on two DGX Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
- [Qwen3.8-Flash-Next on one Spark with TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
- [DeepSeek V4 Flash DSpark on two DGX Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)

These are presentation references. No MiaAI-Lab launch code, patches, or benchmark numbers were copied.
