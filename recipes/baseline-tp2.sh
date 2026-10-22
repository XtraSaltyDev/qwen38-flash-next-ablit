#!/usr/bin/env bash
# Applicable attribution and license scope: ../NOTICE and ../docs/ATTRIBUTION.md.
# Historical DGX Spark TP=2 comparison recipe; see README.md and root NOTICE.
# Matches NVIDIA's published DGX Spark profile for the base checkpoint
# (https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local): 262,144-token context,
# 8 sequences, FP8 KV cache, prefix caching, MTP speculative decoding, B12X kernels.
# Speculative verification is exact; the baked weights are served unchanged.
#
# Usage: bash serve-spark.sh /absolute/path/to/downloaded/checkpoint
# Env:   PORT (default 8000), HOST (default 127.0.0.1), MTP_TOKENS (default 4; 0 disables),
#        CACHE_DIR (compile/autotune caches, default ~/.cache/qwen-flash-next-tp2)
# The container is capped at 104 GiB of the Spark's 128 GB unified memory; run one large model at a time.
set -euo pipefail
model_dir="${1:?Usage: bash serve-spark.sh /absolute/path/to/downloaded/checkpoint}"
model_dir="$(cd "$model_dir" && pwd)"
PORT="${PORT:-18383}"
RANK="${RANK:?RANK must be 0 or 1}"
NODE_IP="${NODE_IP:?NODE_IP must be the local CX7 address}"
extra=()
if [ "$RANK" != 0 ]; then extra=(--headless); fi
HOST="${HOST:-127.0.0.1}"
MTP_TOKENS="${MTP_TOKENS:-4}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/qwen-flash-next-tp2}"
IMAGE="eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595"

mkdir -p "$CACHE_DIR"/{cache,triton}
spec=()
if [ "$MTP_TOKENS" != "0" ]; then
  spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi

# First start compiles and autotunes kernels (several minutes); later starts reuse CACHE_DIR.
# Weight-plan compilation is serial and the allocator is bounded so startup stays within host memory.
docker create --name qwen-flash-next-tp2 --restart unless-stopped --network host --ipc host --device nvidia.com/gpu=all --device /dev/infiniband --cap-add IPC_LOCK \
  --shm-size 16g --ulimit memlock=-1 --ulimit stack=67108864 \
  --memory 104g --memory-swap 104g \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --user "$(id -u):$(id -g)" -e HOME=/cache -e XDG_CACHE_HOME=/cache/cache \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v "$model_dir:/model:ro" -v "$CACHE_DIR:/cache" \
  -e VLLM_HOST_IP="$NODE_IP" \
  -e NCCL_SOCKET_IFNAME=enP2p1s0f0np0 -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0 \
  -e TP_SOCKET_IFNAME=enP2p1s0f0np0 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_DISABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=INFO \
  -e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB \
  -e VLLM_PLE_TABLE_MEMORY=disk \
  -e B12X_DISK_BACKEND=io_uring \
  -e CUTE_DSL_ARCH=sm_121a \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_POLICY_MODE=auto \
  -e VLLM_MXFP8_LM_HEAD=1 \
  -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e B12X_W4A16_A4_PREFILL=1 \
  -e B12X_WEIGHTS_COMPILE_WORKERS=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6,per_process_memory_fraction:0.82,throw_on_cudamalloc_oom:True \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  --entrypoint vllm "$IMAGE" serve /model \
    --served-model-name Qwen3.8-Flash-Next \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 \
    --distributed-executor-backend mp --nnodes 2 --node-rank "$RANK" \
    --master-addr 10.100.225.2 --master-port 29501 --distributed-timeout-seconds 1800 \
    "${extra[@]}" \
    --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
    --dtype bfloat16 --kv-cache-dtype fp8 --quantization modelopt_mixed \
    --block-size 16 --load-format b12x \
    --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 \
    "${spec[@]}" \
    --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x \
    --no-enable-flashinfer-autotune --mm-encoder-tp-mode data \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
    --gpu-memory-utilization 0.8
