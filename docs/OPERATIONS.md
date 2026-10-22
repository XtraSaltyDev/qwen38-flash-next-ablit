# Operations and reproduction

The qualified configuration is in `../serve-tp2.sh`; measurements are in `../evidence/`. The model is `DevelopingDad/Qwen3.8-Flash-Next-FP4-Local-Abliterated`; commit hashes identify the measured artifact without relying on a release tag. Original defaults were qualified on two GB10 DGX Sparks on October 22, 2026. Repository publication made the network/container names configurable while preserving the default Docker invocation. Other topology or performance settings require their own qualification.

## Prerequisites

Both nodes need Docker access, NVIDIA CDI, RDMA devices, the pinned B12X image, and the complete checkpoint at revision `51f24153ae9b27653a823074c631dfa941fdb536`. Follow the [eugr networking guide](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md) for direct CX7 links and trusted SSH. Do not place two large model deployments in the same host memory budget.

The qualified example topology uses head `10.100.225.2`, worker `10.100.225.1`, and second-rail addresses `10.100.224.2` / `10.100.224.1`. These are private example addresses, not public API endpoints. The socket interface is `enP2p1s0f0np0`; the RDMA HCAs are `rocep1s0f0,roceP2p1s0f0`. Interface names and direct routes must match your actual hosts. The master port is 29503. Maintain the existing paired-node firewall scope.

## Recipe variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `RANK` | Required: `0` or `1` | Head or headless worker |
| `NODE_IP` | Required | This node's direct CX7 address |
| `MASTER_ADDR` | `10.100.225.2` | Head's direct address; same value on both nodes |
| `MASTER_PORT` | `29503` | Distributed rendezvous port |
| `SOCKET_IFNAME` | `enP2p1s0f0np0` | NCCL/Gloo/TP bootstrap interface |
| `IB_HCAS` | `rocep1s0f0,roceP2p1s0f0` | RDMA HCA list |
| `CONTAINER_NAME` | `qwen-flash-next-tp2-v2` | Docker name; use the same name on both hosts |
| `PORT` | `18383` | Head API port |
| `HOST` | `127.0.0.1` | Head API bind address |
| `CACHE_DIR` | `~/.cache/qwen-flash-next-tp2` | Per-node persistent compilation/autotune caches |
| `MTP_TOKENS` | `4` | Qualified speculation count; changing it requires new evaluation |

`serve-tp2.sh /absolute/checkpoint/path` creates a container but does not start it. It uses TP=2, PP=1, two native multiprocessing ranks, BF16 KV via auto dtype, disk PLE/io_uring, utilization 0.8, and a 104 GiB container memory cap. Weights are mounted read-only. Container restart policy is `unless-stopped`.

Final observed KV capacity was 3,281,415 tokens, about 12.52 full 262K-context equivalents. The 16-slot scheduler does not promise sixteen simultaneous full-length contexts.

## Start, inspect, and stop

Use the README's worker-first creation/start commands. For containers that already exist, run `docker start qwen-flash-next-tp2-v2` on the worker, then on the head. Do not rerun `docker create` against an existing name.

On either host:

```sh
docker inspect -f '{{.State.Status}} {{.State.OOMKilled}} {{.RestartCount}}' qwen-flash-next-tp2-v2
docker logs --tail 120 qwen-flash-next-tp2-v2
```

On the head after startup:

```sh
curl --fail http://127.0.0.1:18383/health
curl --fail http://127.0.0.1:18383/v1/models
```

The catalog should contain only `Qwen3.8-Flash-Next`, context 262144. First startup compiles/autotunes kernels and can take several minutes. Keep the node-local caches for later starts. `/metrics` provides running/waiting requests, preemptions, speculative acceptance, and cache telemetry.

To stop the pair, run `docker stop -t 90 qwen-flash-next-tp2-v2` on the head and then the worker. Clients should point to the head's API through their own SSH tunnel or authenticated proxy. A model catalog proves discovery; also run actual inference through every route used by your clients.

## Rollback

Before an experiment, keep the last passing containers, controller scripts, and cache directories. The original reference here used eight active sequences, FP8 KV, disk PLE, and MTP4; its creation recipe is [recipes/baseline-tp2.sh](../recipes/baseline-tp2.sh).

On both hosts, disable restart for the candidate and stop it. Restore the preserved controller scripts for your deployment. Set the preserved reference container's restart policy to `unless-stopped`, then start reference worker before reference head. Confirm health, canonical model name, context, actual route inference, and memory headroom. Preserve failed logs before changing a candidate.

The machine-specific rollback helpers and dashboard/client backups remain in the original local bundle and are intentionally excluded from the public repository. They depend on that installation's named containers, SSH configuration, and dashboard credentials; they are not clean-machine installers.

## Repeat evaluations

Harnesses use Python's standard library. Run on an otherwise idle endpoint and use a new output directory for every treatment. The extended code-execution harness uses Linux resource limits; run it on Linux. Execute these from the repository root on the head, or substitute your tunnel's base URL.

```sh
python3 scripts/evaluate.py --base http://127.0.0.1:18383 --label v2-repeat \
  --output /tmp/qwen-v2-repeat --mode all \
  --concurrency 1 4 8 16 --repeats 2 --tokens 256
python3 scripts/evaluate.py --base http://127.0.0.1:18383 --label v2-long-repeat \
  --output /tmp/qwen-v2-long-repeat --mode performance \
  --concurrency 1 4 16 --repeats 1 --tokens 128 --context-records 300
python3 scripts/quality_extended.py --base http://127.0.0.1:18383 \
  --label v2-extended-repeat --output /tmp/qwen-v2-extended-repeat \
  --fixtures evidence/invoice-fixtures.json
python3 scripts/near_context_control.py --base http://127.0.0.1:18383 \
  --label v2-context-repeat --output /tmp/qwen-v2-context-repeat \
  --samples 10 --require-strict
python3 scripts/repeat_control.py --base http://127.0.0.1:18383 \
  --label v2-word-count-repeat --output /tmp/qwen-v2-word-count-repeat --samples 100
python3 scripts/audit_target.py --base http://127.0.0.1:18383 \
  --label v2-audit-repeat --output /tmp/qwen-v2-audit-repeat
```

The near-context test must not label a cache state as cold unless that state has actually been established. The `quality_extended.initial.py` snapshot preserves an initial inconsistent duration prompt; use corrected `quality_extended.py` for new qualification.

Match weights, image, prompts, warmups, concurrency, output length, sampling, and cache policy when comparing configurations. Inspect contaminated waves, per-fixture regressions, preemptions, startup reliability, and host memory rather than relying only on aggregate speed. The [protocol](../evidence/protocol.json) preserves the evolving design: its initial fixed FP8 setting is superseded for the final BF16 treatment by the [selected profile](../evidence/profile.json).
