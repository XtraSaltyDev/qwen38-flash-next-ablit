#!/usr/bin/env python3
"""Pinned code execution, exact invoice OCR and near-262K retrieval gates."""
import argparse
import ast
import hashlib
import json
import pathlib
import re
import resource
import subprocess
import sys
from evaluate import Evaluation

CODE_CASES = [
    ("merge_intervals", "Implement merge_intervals(intervals). Input is a list of [start,end] integer pairs with start<=end. Return a new sorted list merging overlapping or touching intervals. Empty input returns []. Do not mutate the input. No imports. Return only the function definition, without markdown.", "assert merge_intervals([])==[]\nassert merge_intervals([[5,7],[1,3],[3,4],[9,9]])==[[1,4],[5,7],[9,9]]\na=[[2,4],[1,8]]; b=[x[:] for x in a]; assert merge_intervals(a)==[[1,8]] and a==b"),
    ("flatten_once", "Implement flatten_once(items). Replace each nested list with its elements, for exactly one level. Keep all other objects unchanged. Empty input returns []. No imports. Return only the function definition, without markdown.", "assert flatten_once([])==[]\nassert flatten_once([1,[2,3],[[4]],'ab',None])==[1,2,3,[4],'ab',None]\nassert flatten_once([[],[1],[]])==[1]"),
    ("parse_duration", "Implement parse_duration(text). Accept one or more non-negative integer duration components, each immediately followed by h, m, or s, e.g. '2h30m5s'. Return total seconds. Empty input, unknown units, missing numbers, and leftover characters must raise ValueError. Components can be in any order. No imports. Return only the function definition, without markdown.", "assert parse_duration('2h30m5s')==9005\nassert parse_duration('0s')==0\nassert parse_duration('45m')==2700\nfor s in ['', '12', '5d', 'h', '1s!']:\n try: parse_duration(s)\n except ValueError: pass\n else: raise AssertionError(s)"),
    ("sliding_max", "Implement sliding_max(nums,k), returning the maximum in every contiguous window of length k. Return [] for an empty nums, k<=0, or k>len(nums). No imports. Return only the function definition, without markdown.", "assert sliding_max([1,3,-1,-3,5,3,6,7],3)==[3,3,5,5,6,7]\nassert sliding_max([],2)==[]\nassert sliding_max([3,2],0)==[]\nassert sliding_max([3,2],3)==[]\nassert sliding_max([-2,-5,-1],1)==[-2,-5,-1]"),
]


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 2**20, 256 * 2**20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


def execute(source, assertions):
    text = source.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        tree = ast.parse(text)
        # These fixtures require only pure functions and standard builtins.
        if not tree.body or not all(isinstance(n, ast.FunctionDef) for n in tree.body):
            return False, "Expected function definitions only"
        forbidden = {"open", "eval", "exec", "compile", "input", "globals", "locals", "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Imports are outside fixture contract"
            if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith("__")):
                return False, "Unsafe builtin/name outside pure-function contract"
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return False, "Dunder attribute outside pure-function contract"
        r = subprocess.run([sys.executable, "-I", "-S", "-c", text + "\n" + assertions], capture_output=True, text=True, timeout=3, preexec_fn=limits, cwd="/tmp")
        return r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:18383")
    p.add_argument("--label", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fixtures", required=True)
    args = p.parse_args()
    args.concurrency = [1]; args.repeats = 1; args.tokens = 256
    e = Evaluation(args)
    for name, prompt, assertions in CODE_CASES:
        r = e.complete([{"role": "user", "content": prompt}], max_tokens=1024)
        passed, detail = execute(r["choices"][0]["message"].get("content", ""), assertions)
        e.check("executed_code_" + name, r, passed, test_detail=detail)
    fixtures = json.loads(pathlib.Path(args.fixtures).read_text())
    for invoice in fixtures:
        r = e.complete([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + invoice["png_base64"]}}, {"type": "text", "text": "Extract invoice number and total from this image. Return only JSON with exactly keys invoice_number (string), total (number), currency (three-letter code). No markdown."}]}])
        text = r["choices"][0]["message"].get("content", "").strip()
        try:
            valid = json.loads(text) == invoice["expected"]
        except ValueError:
            valid = False
        e.check("invoice_" + invoice["id"], r, valid, expected=invoice["expected"], image_sha256=invoice["sha256"])
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
    assert 260000 <= count <= 261850, count
    r = e.complete(m, max_tokens=256)
    text = r["choices"][0]["message"].get("content", "").strip()
    try:
        valid = json.loads(text) == {"early": "R7D4", "middle": "C2K9", "late": "P8M5"}
    except ValueError:
        valid = False
    e.check("near_262k_retrieval", r, valid, actual_prompt_tokens=r["usage"]["prompt_tokens"])
    e.results["quality_summary"] = {"passed": sum(x["passed"] for x in e.results["quality"]), "total": len(e.results["quality"]), "scope": "pure code fixtures, synthetic invoice OCR, and synthetic long-context retrieval; not a general capability benchmark"}
    e.results["complete"] = True
    e.save()


if __name__ == "__main__":
    main()
