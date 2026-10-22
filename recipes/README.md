# Historical reference and experimental recipes

Use the top-level `serve-tp2.sh` for the qualified published profile. These files preserve comparison configurations and are not interchangeable recommendations:

- `baseline-tp2.sh`: original eight-slot FP8-KV reference, disk PLE.
- `experiment-shared-ple.sh`: shared-PLE/manual-KV candidate creation script.
- `shared-ple-rejected.sh`: earlier shared-PLE profile; rejected after strict near-context JSON formatting regressed.
- `experiment-disk16.sh`: sixteen-slot FP8-KV isolation; rejected for the same formatting regression.
- `experiment-bf16kv16.sh`: experimental container name used for the winning BF16 treatment before final packaging.

All adapt DevelopingDad's launcher and eugr's serving/cluster recipe. See the root NOTICE, docs/ATTRIBUTION.md, and retained third-party notices. Historical variable names and defaults remain as recorded; these scripts can create large containers and require the same two-node prerequisites.
