# Attribution and provenance

## Model lineage

| Contributor | Contribution | Primary source |
| --- | --- | --- |
| Qwen | Original Qwen3.8-Flash-Next architecture and model | [Model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| NVIDIA | ModelOpt/QAD mixed-precision FP4-Local checkpoint and reference serving guidance | [Pinned quantized base](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300) |
| DevelopingDad | Trial117 refusal-direction intervention, baked checkpoint, MXFP8 re-encode, validation, and Spark launcher | [Pinned community checkpoint](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated/tree/51f24153ae9b27653a823074c631dfa941fdb536) |
| XtraSaltyDev | TP=2 serving comparison, quality fixtures, BF16-KV/16-slot qualification, deployment verification, and packaging | [This repository](https://github.com/XtraSaltyDev/qwen38-flash-next-ablit) |

The upstream release uses a trial117/Heretic-style refusal-direction intervention. We neither repeated the abliteration search nor claim authorship of that intervention, the bake, or the re-encode. See the pinned [model card](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated/blob/51f24153ae9b27653a823074c631dfa941fdb536/README.md), [notice](../third_party/model/NOTICE.txt), and [provenance](../third_party/model/provenance.json).

## Serving and kernel foundation

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker), Eugene Rakhmatulin and contributors: Docker images, Spark cluster/networking integration, and the [cluster recipe](../third_party/eugr-cluster-recipe.yaml) consulted during this study. Its [MIT notice](../third_party/licenses/EUGR-MIT.txt) is retained.
- [DevelopingDad's pinned serve-spark.sh](https://huggingface.co/DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated/blob/51f24153ae9b27653a823074c631dfa941fdb536/serve-spark.sh): immediate single-node launcher foundation adapted into the two-node recipe.
- [vLLM](https://github.com/vllm-project/vllm): serving, scheduling, KV-cache management, speculative decoding, and OpenAI-compatible APIs.
- [local-inference-lab](https://github.com/local-inference-lab), including its [FlashInfer fork](https://github.com/local-inference-lab/flashinfer): B12X inference/kernel integration used by the eugr runtime.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [PyTorch](https://github.com/pytorch/pytorch), and NVIDIA CUDA/NCCL: runtime/kernel dependencies, downloaded in the external image rather than vendored here.

We do not claim to have built the pinned container or authored those runtime/kernel projects. `third_party/SOURCES.json` pins source snapshots retrieved for attribution. The eugr snapshot is not asserted to be the image build commit; the runtime image is pinned by digest.

## Changes made here

The qualified study preserved checkpoint revision and image digest. The final treatment uses 16 active sequences and BF16 KV while retaining disk PLE, MTP4, the original memory profile, and 262K context. We evaluated matched workloads, code/tool/OCR/retrieval fixtures, repeat controls, and returned target-token choices. Rejected candidates and failures remain documented.

The published `serve-tp2.sh` exposes network/container names as environment variables and adds attribution comments. Default inference arguments match the qualified profile. The original recipe hash remains in `evidence/profile.json`; the published recipe has a separate hash for these packaging changes. No live container was changed during publication.

## License scope

The root [MIT license](../LICENSE) grants rights in original code and documentation contributed by XtraSaltyDev. It does not grant additional rights in third-party material.

- **Checkpoint-derived material:** the pinned upstream card identifies the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) and [Qwen Community License 1.0](../third_party/licenses/QWEN-COMMUNITY-LICENSE-1.0.txt). Weights are not included. Upstream portions adapted from the checkpoint's launcher retain their applicable terms; MIT does not supersede them.
- **eugr recipe portions/snapshot:** retain the [MIT notice](../third_party/licenses/EUGR-MIT.txt), copyright Eugene Rakhmatulin.
- **`third_party/` notices/provenance:** retain original owners' copyright and terms, as identified by sources and notices.
- **External runtime:** libraries and the Docker image keep their own distribution terms.
- **Original harnesses and documentation:** MIT, subject to the exceptions above. Measurements and synthetic fixture outputs include provenance in `evidence/`.

The repository's MIT grant covers its original contributions; the checkpoint retains the upstream terms identified above. No endorsement by upstream authors is implied.

## Public evidence and local archive

Public evidence retains measurements, synthetic fixtures, experimental failures, and tensor metadata. Personal client backups, tunnel/SSH registrations, full Docker inspections, dashboard source snapshots, and original local copy manifests are excluded from Git. They remain in the complete local working bundle.

Published logs replace local usernames, management endpoints, and host labels. The report omits personal connections and detailed runtime commands/environment; measured values remain unchanged. `evidence/PUBLICATION.json` records original/published hashes and transformations. `SHA256SUMS` verifies tracked publication files, excluding itself.

## Recipe presentation references

The README layout was informed by MiaAI-Lab's practical Spark recipes: a short description, measured performance, requirements, launch commands, configuration, checks, and credits. Wording is original; no MiaAI-Lab launch code, patches, or benchmark numbers were copied.

- [Qwen3.8-Flash-Next on two DGX Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
- [Qwen3.8-Flash-Next on one Spark with TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
- [DeepSeek V4 Flash DSpark on two DGX Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)

These are presentation references, not dependencies or sources of this recipe's measured results.
