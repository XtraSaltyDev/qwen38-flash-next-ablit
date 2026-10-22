#!/usr/bin/env python3
"""Audit emitted greedy tokens against returned target top-1 logprobs."""
import argparse
from evaluate import Evaluation

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:18383")
p.add_argument("--label", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
args.concurrency = [1]; args.repeats = 1; args.tokens = 128
e = Evaluation(args)
total = 0
mismatches = 0
for i, prompt in enumerate([
    "Write a Python function for binary search in a sorted list. Include a short explanation.",
    "Explain why write-ahead logs help a database recover after a power failure.",
    "Calculate 37 * 19 and explain a way to check the result.",
]):
    r = e.complete([{"role": "user", "content": prompt}], max_tokens=128, logprobs=True, top_logprobs=1)
    entries = r["choices"][0]["logprobs"]["content"]
    bad = []
    for j, token in enumerate(entries):
        top = token["top_logprobs"][0]
        same = token["token"] == top["token"] or token.get("bytes") == top.get("bytes") and token.get("bytes") is not None
        if not same and abs(token["logprob"] - top["logprob"]) > 1e-6:
            bad.append(j)
    total += len(entries); mismatches += len(bad)
    e.check("target_greedy_audit_" + str(i), r, not bad, audited_tokens=len(entries), mismatched_positions=bad)
r = e.complete([{"role": "user", "content": "What is 37 * 19? Give your final answer as only an integer."}], max_tokens=1024, chat_template_kwargs={"enable_thinking": True})
message = r["choices"][0]["message"]
e.check("thinking_on", r, message.get("content", "").strip() == "703" and bool(message.get("reasoning") or message.get("reasoning_content")))
e.results["greedy_target_audit"] = {"tokens": total, "mismatches": mismatches, "scope": "Returned target top-1 audit on these fixtures; not a full-vocabulary distribution equivalence test"}
e.results["complete"] = True
e.save()
