"""Evaluate free-generated next actions on the fixed round-two eval states.

This is the middle layer between LLaMA-Factory reward accuracy and full
end-to-end task completion. It sends the held-out runtime-shaped state to a
served model, then scores the model's actual next tool call or final answer.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import check_completion, get_client, load_jsonl, save_jsonl, write_text


PLACEHOLDERS = ("某ID", "具体", "工具返回的", "转换后的")
ID_PATTERN = re.compile(r"\b(?:P\d{4}|[A-Z]{1,8}-[A-Za-z0-9-]+)\b")


def to_openai_messages(pair):
    messages = [{"role": "system", "content": pair["system"]}]
    last_call_id = None
    call_index = 0
    for message in pair["context_messages"]:
        role = message["from"]
        value = message["value"]
        if role == "human":
            messages.append({"role": "user", "content": value})
        elif role == "function_call":
            payload = json.loads(value)
            last_call_id = f"state-call-{call_index}"
            call_index += 1
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": last_call_id,
                    "type": "function",
                    "function": {
                        "name": payload["name"],
                        "arguments": json.dumps(
                            payload.get("arguments") or {}, ensure_ascii=False
                        ),
                    },
                }],
            })
        elif role == "observation":
            if last_call_id is None:
                raise ValueError("observation 前没有 function_call")
            messages.append({
                "role": "tool",
                "tool_call_id": last_call_id,
                "content": value,
            })
        else:
            raise ValueError(f"不支持的上下文角色: {role}")
    return messages


def prefix_tool_calls(pair):
    calls = []
    for message in pair["context_messages"]:
        if message["from"] != "function_call":
            continue
        payload = json.loads(message["value"])
        if payload["name"] != "finish_task":
            calls.append({
                "name": payload["name"],
                "args": payload.get("arguments") or {},
            })
    return calls


def tool_action_score(expected, actual):
    name_correct = actual.get("kind") == "tool" and actual.get("tool") == expected["tool"]
    if not name_correct:
        return False, False, False
    expected_args = expected.get("args") or {}
    actual_args = actual.get("args") or {}
    keys_correct = set(expected_args) == set(actual_args)
    if not keys_correct:
        return True, False, False
    values_correct = all(
        actual_args.get(key) == expected_value
        and not any(token in str(actual_args.get(key)) for token in PLACEHOLDERS)
        for key, expected_value in expected_args.items()
    )
    return True, True, values_correct


def score_action(pair, task, actual):
    expected = pair["chosen_action"]
    result = {
        "expected_kind": expected["kind"],
        "actual_kind": actual.get("kind"),
        "tool_name_correct": None,
        "argument_keys_correct": None,
        "argument_values_correct": None,
        "final_task_complete": None,
        "final_grounding_complete": None,
        "action_correct": False,
    }
    if expected["kind"] == "tool":
        name_ok, keys_ok, values_ok = tool_action_score(expected, actual)
        result.update({
            "tool_name_correct": name_ok,
            "argument_keys_correct": keys_ok,
            "argument_values_correct": values_ok,
            "action_correct": name_ok and keys_ok and values_ok,
        })
        return result

    if actual.get("kind") != "final":
        return result
    final = actual.get("content") or ""
    task_complete = check_completion(task, {
        "tool_calls": prefix_tool_calls(pair),
        "final_answer": final,
    })
    required_ids = {
        str(item["value"])
        for item in pair.get("chosen_grounding") or []
        if ID_PATTERN.fullmatch(str(item.get("value") or ""))
    }
    grounding_complete = all(identifier in final for identifier in required_ids)
    result.update({
        "final_task_complete": task_complete,
        "final_grounding_complete": grounding_complete,
        "action_correct": task_complete and grounding_complete,
    })
    return result


def response_action(message):
    calls = getattr(message, "tool_calls", None) or []
    if not calls:
        return {"kind": "final", "content": message.content or ""}
    call = calls[0]
    try:
        args = json.loads(call.function.arguments or "{}")
    except Exception:
        args = {"_raw": call.function.arguments}
    return {
        "kind": "tool", "tool": call.function.name, "args": args,
        "tool_call_count": len(calls),
    }


def evaluate_one(client, model, pair, task, temperature, seed):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=to_openai_messages(pair),
            tools=pair["tools"],
            # The model must choose between calling a tool and answering. Forcing
            # the gold action kind would turn this into an argument-filling test.
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=temperature,
            seed=seed,
        )
        actual = response_action(response.choices[0].message)
        score = score_action(pair, task, actual)
        usage = getattr(response, "usage", None)
        error = None
    except Exception as exc:
        actual = {"kind": "error"}
        score = score_action(pair, task, actual)
        usage = None
        error = str(exc)
    return {
        "pair_id": pair["pair_id"],
        "task_id": pair["task_id"],
        "stress": pair["stress"],
        "decision_index": pair["decision_index"],
        "model": model,
        "expected_action": pair["chosen_action"],
        "actual_action": actual,
        **score,
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "error": error,
    }


def summarize(rows):
    def rate(items, key):
        values = [
            row[key] for row in items
            if not row.get("error") and row.get(key) is not None
        ]
        return sum(bool(value) for value in values) / len(values) if values else None

    tools = [row for row in rows if row["expected_kind"] == "tool"]
    finals = [row for row in rows if row["expected_kind"] == "final"]
    return {
        "n": len(rows),
        "valid_n": sum(not row.get("error") for row in rows),
        "task_n": len({row["task_id"] for row in rows}),
        "errors": sum(bool(row.get("error")) for row in rows),
        "action_accuracy": rate(rows, "action_correct"),
        "tool_n": len(tools),
        "tool_name_accuracy": rate(tools, "tool_name_correct"),
        "tool_argument_accuracy": rate(tools, "argument_values_correct"),
        "final_n": len(finals),
        "final_task_accuracy": rate(finals, "final_task_complete"),
        "final_grounding_accuracy": rate(finals, "final_grounding_complete"),
        "by_stress": {
            stress: rate(
                [row for row in rows if row.get("stress") == stress],
                "action_correct",
            )
            for stress in sorted({row.get("stress") for row in rows})
        },
    }


def percentage(value):
    return "n/a" if value is None else f"{value:.1%}"


def align_rows(before, after):
    before_by_id = {row["pair_id"]: row for row in before}
    after_by_id = {row["pair_id"]: row for row in after}
    if len(before_by_id) != len(before) or len(after_by_id) != len(after):
        raise ValueError("state-action 结果包含重复 pair_id")
    if set(before_by_id) != set(after_by_id):
        raise ValueError("base/DPO state-action pair_id 不对齐")
    ids = sorted(before_by_id)
    return [(before_by_id[pair_id], after_by_id[pair_id]) for pair_id in ids]


def compare_report(before, after):
    aligned = align_rows(before, after)
    left, right = summarize(before), summarize(after)
    valid_pairs = [pair for pair in aligned if not pair[0].get("error") and not pair[1].get("error")]
    improved = sum(not left_row["action_correct"] and right_row["action_correct"] for left_row, right_row in valid_pairs)
    regressed = sum(left_row["action_correct"] and not right_row["action_correct"] for left_row, right_row in valid_pairs)
    lines = [
        "# Round-2 state-action evaluation",
        "",
        "| Metric | Base | DPO |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Next-action accuracy", "action_accuracy"),
        ("Tool-name accuracy", "tool_name_accuracy"),
        ("Tool-argument accuracy", "tool_argument_accuracy"),
        ("Final task accuracy", "final_task_accuracy"),
        ("Final grounding accuracy", "final_grounding_accuracy"),
    ):
        lines.append(f"| {label} | {percentage(left[key])} | {percentage(right[key])} |")
    lines.extend([
        "",
        f"Base/DPO pair rows: {left['n']}/{right['n']}; independent eval tasks: {left['task_n']}; API errors: {left['errors']}/{right['errors']}.",
        f"Aligned valid pairs improved/regressed: {improved}/{regressed}.",
        "",
        "| Stress | Base next-action | DPO next-action |",
        "|---|---:|---:|",
    ])
    for stress in sorted(set(left["by_stress"]) | set(right["by_stress"])):
        lines.append(
            f"| {stress} | {percentage(left['by_stress'].get(stress))} | "
            f"{percentage(right['by_stress'].get(stress))} |"
        )
    lines.extend([
        "",
        "API errors are excluded from accuracy denominators and reported separately. "
        "Pair rows are not independent samples when they share a task.",
        "",
        "This decision-level result is diagnostic. End-to-end holdout completion remains the primary product metric.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/round2/pref_pairs.jsonl")
    parser.add_argument("--tasks", default="tasks/tasks.jsonl")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--model", default="base")
    parser.add_argument("--base-url")
    parser.add_argument("--port", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--out", default="data/round2/state_eval_base.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare-before")
    parser.add_argument("--compare-after")
    parser.add_argument("--report", default="results/round2/state_action_compare.md")
    args = parser.parse_args()

    if args.compare_before or args.compare_after:
        if not (args.compare_before and args.compare_after):
            raise SystemExit("compare mode 必须同时提供 --compare-before 和 --compare-after")
        before = load_jsonl(args.compare_before)
        after = load_jsonl(args.compare_after)
        try:
            report = compare_report(before, after)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        write_text(report, args.report)
        print(report)
        return

    pairs = [
        row for row in load_jsonl(args.pairs)
        if args.split == "all" or row.get("dataset_split") == args.split
    ]
    tasks = {row["task_id"]: row for row in load_jsonl(args.tasks)}
    missing = sorted({row["task_id"] for row in pairs} - set(tasks))
    if missing:
        raise SystemExit(f"找不到 task: {missing}")
    rows = load_jsonl(args.out) if args.resume and os.path.exists(args.out) else []
    done = {row["pair_id"] for row in rows}
    base_url = args.base_url or (f"http://localhost:{args.port}/v1" if args.port else None)
    client = get_client(base_url=base_url)
    for index, pair in enumerate(pairs):
        if pair["pair_id"] in done:
            continue
        rows.append(evaluate_one(
            client, args.model, pair, tasks[pair["task_id"]],
            args.temperature, args.seed + index,
        ))
        rows.sort(key=lambda row: row["pair_id"])
        save_jsonl(rows, args.out)

    stats = summarize(rows)
    summary_path = os.path.splitext(args.out)[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n逐条结果 {args.out}\n汇总 {summary_path}")


if __name__ == "__main__":
    main()
