#!/usr/bin/env python3
"""Separate semantic retrieval from strict JSON formatting at near-full context."""
import argparse
import json
import re
from evaluate import Evaluation

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:18383")
p.add_argument("--label", required=True)
p.add_argument("--output", required=True)
p.add_argument("--samples", type=int, default=10)
p.add_argument("--require-strict", action="store_true")
p.add_argument("--initial-cache-state", choices=["cold", "warm", "unspecified"], default="unspecified")
args = p.parse_args()
args.concurrency = [1]; args.repeats = args.samples; args.tokens = 256
e = Evaluation(args)
unit = "Asset cedar is ready for inspection. "


def messages(n):
    section = unit * (n // 3)
    text = "Read the document and return the three qualified markers.\nEARLY_QUALIFIED=R7D4\n" + section + "\nMIDDLE_QUALIFIED=C2K9\n" + section + "\nLATE_QUALIFIED=P8M5\n" + section + '\nReturn only JSON with keys early, middle, late and their marker strings.'
    return [{"role": "user", "content": text}]


n = 32000
for _ in range(4):
    m = messages(n)
    with e.request("/tokenize", {"model": "Qwen3.8-Flash-Next", "messages": m, "chat_template_kwargs": {"enable_thinking": False}}) as r:
        count = json.load(r)["count"]
    if abs(count - 261600) <= 100:
        break
    n = max(3, int(n * (261600 - 128) / max(1, count - 128)))
assert count == 261623, count
expected = {"early": "R7D4", "middle": "C2K9", "late": "P8M5"}
rows = []
for i in range(args.samples):
    r = e.complete(m, max_tokens=256)
    text = r["choices"][0]["message"].get("content", "").strip()
    try:
        strict = json.loads(text) == expected
    except ValueError:
        strict = False
    semantic_text = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", text))
    try:
        semantic = json.loads(semantic_text) == expected
    except ValueError:
        semantic = False
    rows.append({"i": i, "strict_json": strict, "semantic_retrieval": semantic, "response": r})
    e.results["context_control"] = rows; e.save()
    print(json.dumps({"i": i, "strict_json": strict, "semantic_retrieval": semantic}), flush=True)
schema = {"type": "object", "properties": {k: {"type": "string"} for k in expected}, "required": list(expected), "additionalProperties": False}
r = e.complete(m, max_tokens=256, response_format={"type": "json_schema", "json_schema": {"name": "markers", "strict": True, "schema": schema}})
text = r["choices"][0]["message"].get("content", "").strip()
e.check("near_262k_structured_json", r, json.loads(text) == expected)
e.results["summary"] = {"samples": args.samples, "strict_json_passes": sum(x["strict_json"] for x in rows), "semantic_retrieval_passes": sum(x["semantic_retrieval"] for x in rows), "structured_json_passed": e.results["quality"][-1]["passed"], "prompt_tokens": count, "initial_cache_state": args.initial_cache_state, "cache_scope": "Identical prompt repeats exercise prefix reuse"}
e.results["complete"] = True
e.save()
if args.require_strict and (not all(x["strict_json"] for x in rows) or not e.results["summary"]["structured_json_passed"]):
    raise SystemExit("Strict near-context formatting gate failed; treatment is not qualified")
