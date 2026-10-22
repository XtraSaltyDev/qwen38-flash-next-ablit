#!/usr/bin/env bash
# Qwen3.8-Flash-Next TP=2 profile v2, Spark 2BEF + C-3765.
# Defaults: 262K context, 16 active sequences, MTP4, BF16 KV (auto dtype), disk PLE.
# Model and image are pinned; parameter overrides require requalification.
# Create one container on each host using RANK=0/1 and NODE_IP=.225.2/.225.1.
# Start rank 1, then rank 0. See evidence/profile.json and docs/OPERATIONS.md.
# Adapted from DevelopingDad's pinned serve-spark.sh and eugr's cluster recipe.
# Original serving study and packaging: XtraSaltyDev, 2026.
# Attribution and applicable license scope: NOTICE and docs/ATTRIBUTION.md.
set -euo pipefail
model_dir="${1:?Usage: bash serve-tp2.sh /absolute/path/to/downloaded/checkpoint}"
model_dir="$(cd "$model_dir" && pwd)"
PORT="${PORT:-18383}"
RANK="${RANK:?RANK must be 0 or 1}"
NODE_IP="${NODE_IP:?NODE_IP must be the local CX7 address}"
case "$RANK" in 0|1) ;; *) printf 'RANK must be 0 or 1\n' >&2; exit 2 ;; esac
MASTER_ADDR="${MASTER_ADDR:-10.100.225.2}"
MASTER_PORT="${MASTER_PORT:-29503}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-flash-next-tp2-v2}"
SOCKET_IFNAME="${SOCKET_IFNAME:-enP2p1s0f0np0}"
IB_HCAS="${IB_HCAS:-rocep1s0f0,roceP2p1s0f0}"
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
docker create --name "$CONTAINER_NAME" --restart unless-stopped --network host --ipc host --device nvidia.com/gpu=all --device /dev/infiniband --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --memory 104g --memory-swap 104g \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --user "$(id -u):$(id -g)" -e HOME=/cache -e XDG_CACHE_HOME=/cache/cache \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v "$model_dir:/model:ro" -v "$CACHE_DIR:/cache" \
  -e VLLM_HOST_IP="$NODE_IP" \
  -e NCCL_SOCKET_IFNAME="$SOCKET_IFNAME" -e GLOO_SOCKET_IFNAME="$SOCKET_IFNAME" \
  -e TP_SOCKET_IFNAME="$SOCKET_IFNAME" \
  -e NCCL_IB_HCA="$IB_HCAS" -e NCCL_IB_DISABLE=0 \
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
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --distributed-timeout-seconds 1800 \
    "${extra[@]}" \
    --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
    --dtype bfloat16 --kv-cache-dtype auto --quantization modelopt_mixed \
    --block-size 16 --load-format b12x \
    --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 4096 \
    "${spec[@]}" \
    --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x \
    --no-enable-flashinfer-autotune --mm-encoder-tp-mode data \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
    --gpu-memory-utilization 0.8
