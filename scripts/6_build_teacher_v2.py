"""Build observation-grounded, single-decision DPO pairs from failed trajectories.

Unlike the v1 trajectory-level data, each row trains exactly one next action at
the first divergence from a teacher-authored workflow. The prompt contains the
correct prefix and deterministic tool observations, matching the runtime's
serial tool-call loop more closely.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import check_completion, load_jsonl, save_jsonl


TEACHER = "OpenAI Codex (GPT-5 family, interactive session)"
PLACEHOLDERS = (
    "工具返回的", "某ID", "具体patient", "具体record", "具体history",
    "具体measurements", "具体adherence", "具体resource", "具体slot",
)
ID_PATTERN = re.compile(r"\b(?:P\d{4}|[A-Z]{1,8}-[A-Za-z0-9-]+)\b")
CLOSURE_REJECTED = (
    "相关流程已处理完成，请在系统中查看结果。",
    "任务已经完成，后续无需操作。",
)
FLEXIBLE_TEXT_ARGS = {"reason", "message", "outcome", "instruction"}


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_tool_call(name, args):
    return "<tool_call>\n" + json.dumps(
        {"name": name, "arguments": args}, ensure_ascii=False
    ) + "\n</tool_call>"


def action_tool(name, args):
    return {"kind": "tool", "tool": name, "args": args}


def action_final(content):
    return {"kind": "final", "content": content or ""}


def render_action(action):
    if action["kind"] == "tool":
        return render_tool_call(action["tool"], action.get("args") or {})
    return action.get("content") or ""


def actions_equal(left, right):
    if left.get("kind") != right.get("kind"):
        return False
    if left["kind"] == "tool":
        if left.get("tool") != right.get("tool"):
            return False
        expected = left.get("args") or {}
        actual = right.get("args") or {}
        if set(expected) != set(actual):
            return False
        for key, value in expected.items():
            if key in FLEXIBLE_TEXT_ARGS:
                rendered = str(actual.get(key, "")).strip()
                if not rendered or any(token in rendered for token in PLACEHOLDERS):
                    return False
                continue
            if actual.get(key) != value:
                return False
        return True
    return (left.get("content") or "").strip() == (right.get("content") or "").strip()


def actual_actions(traj):
    actions = []
    saw_final = False
    for step in traj.get("steps") or []:
        if step.get("tool"):
            actions.append(action_tool(step["tool"], step.get("args") or {}))
        elif step.get("type") == "final":
            actions.append(action_final(step.get("content") or ""))
            saw_final = True
    if not saw_final:
        actions.append(action_final(traj.get("final_answer") or ""))
    return actions


def validate_spec(task, spec):
    errors = []
    tool_specs = {
        tool["name"]: set((tool.get("args") or {}).keys())
        for tool in task.get("tools") or []
    }
    steps = spec.get("gold_steps") or []
    finals = spec.get("final_variants") or []
    if not steps:
        errors.append("gold_steps 为空")
    if len(finals) < 3:
        errors.append("final_variants 少于 3 条")
    for index, step in enumerate(steps):
        name = step.get("tool")
        args = step.get("args") or {}
        if name not in tool_specs:
            errors.append(f"step {index}: 未声明工具 {name}")
            continue
        if set(args) != tool_specs[name]:
            errors.append(
                f"step {index} {name}: 参数键应为 {sorted(tool_specs[name])}，"
                f"实际 {sorted(args)}"
            )
        payload = compact(args)
        for placeholder in PLACEHOLDERS:
            if placeholder in payload:
                errors.append(f"step {index} {name}: 含占位文本 {placeholder}")

    pseudo_calls = [
        {"name": step.get("tool"), "args": step.get("args") or {}}
        for step in steps
    ]
    for final in finals:
        if not final.strip():
            errors.append("final variant 为空")
        if not check_completion(task, {
            "tool_calls": pseudo_calls,
            "final_answer": final,
        }):
            errors.append(f"final variant 未通过 checker: {final}")
    return errors


def gold_actions(spec, repeat):
    final_variants = spec["final_variants"]
    final = final_variants[int(repeat or 0) % len(final_variants)]
    actions = [
        action_tool(step["tool"], step.get("args") or {})
        for step in spec["gold_steps"]
    ]
    actions.append(action_tool("finish_task", {}))
    actions.append(action_final(final))
    return actions


def first_divergence(gold, actual):
    for index, chosen in enumerate(gold):
        if index >= len(actual):
            return index, chosen, action_final("")
        if not actions_equal(chosen, actual[index]):
            return index, chosen, actual[index]
    if len(actual) > len(gold):
        return len(gold) - 1, gold[-1], actual[len(gold)]
    return None


def observation_for(task, tool_name):
    if tool_name == "finish_task":
        return {"ready_for_final": True}
    return (task.get("mock_responses") or {}).get(
        tool_name,
        {"ok": True, "note": "deterministic mock has no task-specific payload"},
    )


def build_prompt(task, prefix):
    constraints = task.get("constraints") or []
    lines = [
        "你是慢病照护运营 Agent。根据任务、工具协议和已经发生的工具结果，只输出一个正确的下一动作。",
        "若任务未完成，只输出一个 <tool_call>；全部动作完成且 finish_task 已返回 ready_for_final 时，输出最终答复。",
        "不得改写枚举值、ID、时间、数值或单位；需要时必须逐字复用工具结果。",
        "",
        "[任务] " + task.get("goal", ""),
        "[约束] " + ("；".join(constraints) if constraints else "以工具观测中的约束为准"),
        "[可用工具] " + compact(task.get("tools") or []),
        "[控制工具] finish_task({})",
        "[已完成历史]",
    ]
    if not prefix:
        lines.append("(尚未调用工具)")
    for action in prefix:
        if action["kind"] != "tool":
            continue
        lines.append("ASSISTANT " + render_action(action).replace("\n", " "))
        lines.append(
            "TOOL " + action["tool"] + " -> "
            + compact(observation_for(task, action["tool"]))
        )
    lines.append("[现在输出一个下一动作]")
    return "\n".join(lines)


def grounding_audit(task, prefix, chosen):
    initial = "\n".join([
        task.get("goal", ""),
        *(task.get("constraints") or []),
    ])
    observations = []
    for action in prefix:
        if action["kind"] == "tool":
            observations.append((
                action["tool"],
                compact(observation_for(task, action["tool"])),
            ))

    grounding, errors = [], []
    if chosen["kind"] == "final":
        for identifier in ID_PATTERN.findall(chosen.get("content") or ""):
            source = None
            if identifier in initial:
                source = "initial_task"
            else:
                for tool_name, observation in observations:
                    if identifier in observation:
                        source = f"observation:{tool_name}"
                        break
            grounding.append({"value": identifier, "source": source})
            if source is None:
                errors.append(f"最终答复使用未在当前状态出现的 ID {identifier}")
        return grounding, errors

    for key, raw_value in (chosen.get("args") or {}).items():
        value = str(raw_value)
        source = None
        if value and value in initial:
            source = "initial_task"
        else:
            for tool_name, observation in observations:
                if value and value in observation:
                    source = f"observation:{tool_name}"
                    break
        if source is None:
            source = "teacher_policy"
        grounding.append({"arg": key, "value": raw_value, "source": source})

        for identifier in ID_PATTERN.findall(value):
            if identifier not in initial and not any(
                identifier in observation for _, observation in observations
            ):
                errors.append(
                    f"{chosen['tool']}.{key} 使用未在当前状态出现的 ID {identifier}"
                )
    return grounding, errors


def build_closure_pairs(task, spec, source_hash):
    prefix = [
        action_tool(step["tool"], step.get("args") or {})
        for step in spec["gold_steps"]
    ]
    prefix.append(action_tool("finish_task", {}))
    prompt = build_prompt(task, prefix)
    rows = []
    for variant, rejected in enumerate(CLOSURE_REJECTED):
        chosen_action = action_final(spec["final_variants"][variant])
        grounding, errors = grounding_audit(task, prefix, chosen_action)
        if errors:
            raise ValueError(f"{task['task_id']} closure#{variant}: {'；'.join(errors)}")
        pair_key = f"{task['task_id']}#closure#{variant}"
        rows.append({
            "pair_id": hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:16],
            "task_id": task["task_id"],
            "scenario_family": task.get("scenario_family"),
            "stress": task.get("stress"),
            "attr_label": "workflow_closure",
            "source_repeat": None,
            "decision_index": len(prefix),
            "decision_kind": "final",
            "pair_source": "teacher_generated_closure_hard_negative",
            "prompt": prompt,
            "chosen": render_action(chosen_action),
            "rejected": rejected,
            "chosen_action": chosen_action,
            "rejected_action": action_final(rejected),
            "chosen_grounding": grounding,
            "teacher": TEACHER,
            "generation_method": "teacher-authored outcome report versus generic closure",
            "source_badcase_sha256": source_hash,
        })
    return rows


def build_pair(task, spec, traj, source_hash):
    gold = gold_actions(spec, traj.get("repeat", 0))
    actual = actual_actions(traj)
    divergence = first_divergence(gold, actual)
    if divergence is None:
        return None, "失败轨迹与 teacher workflow 没有可见分歧"
    index, chosen_action, rejected_action = divergence
    chosen = render_action(chosen_action)
    rejected = render_action(rejected_action)
    if not rejected.strip():
        return None, "rejected 下一动作为空"
    if chosen == rejected:
        return None, "chosen/rejected 相同"

    prefix = gold[:index]
    grounding, grounding_errors = grounding_audit(task, prefix, chosen_action)
    if grounding_errors:
        return None, "；".join(grounding_errors)

    prompt = build_prompt(task, prefix)
    pair_key = f"{traj.get('task_id')}#{traj.get('repeat', 0)}"
    pair_id = hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:16]
    return {
        "pair_id": pair_id,
        "task_id": traj.get("task_id"),
        "scenario_family": traj.get("scenario_family"),
        "stress": traj.get("stress"),
        "attr_label": traj.get("attr_label"),
        "source_repeat": traj.get("repeat", 0),
        "decision_index": index,
        "decision_kind": chosen_action["kind"],
        "pair_source": "observed_badcase_first_divergence",
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_action": chosen_action,
        "rejected_action": rejected_action,
        "chosen_grounding": grounding,
        "teacher": TEACHER,
        "generation_method": "teacher-authored workflow compiled at first divergence",
        "source_badcase_sha256": source_hash,
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--badcase", default="data/train_badcases_labeled.jsonl")
    parser.add_argument("--tasks", default="tasks/tasks.jsonl")
    parser.add_argument("--specs", default="tasks/teacher_v2_specs.jsonl")
    parser.add_argument("--out", default="data/pref_pairs_teacher_v2.jsonl")
    args = parser.parse_args()

    task_rows = load_jsonl(args.tasks)
    tasks = {task["task_id"]: task for task in task_rows if task.get("split") == "train"}
    spec_rows = load_jsonl(args.specs)
    specs = {spec["task_id"]: spec for spec in spec_rows}
    badcases = load_jsonl(args.badcase)

    errors = []
    if set(tasks) != set(specs):
        errors.append(
            "teacher spec 与 train task 不一致: "
            f"missing={sorted(set(tasks) - set(specs))}, "
            f"extra={sorted(set(specs) - set(tasks))}"
        )
    for task_id in sorted(set(tasks) & set(specs)):
        for error in validate_spec(tasks[task_id], specs[task_id]):
            errors.append(f"{task_id}: {error}")
    if errors:
        raise SystemExit("teacher spec 校验失败:\n- " + "\n- ".join(errors))

    with open(args.badcase, "rb") as handle:
        source_hash = hashlib.sha256(handle.read()).hexdigest()

    pairs = []
    dropped = []
    for traj in badcases:
        task_id = traj.get("task_id")
        if task_id not in tasks:
            dropped.append((task_id, traj.get("repeat", 0), "非 train task"))
            continue
        pair, error = build_pair(tasks[task_id], specs[task_id], traj, source_hash)
        if error:
            dropped.append((task_id, traj.get("repeat", 0), error))
        else:
            pairs.append(pair)

    for task_id in sorted(tasks):
        pairs.extend(build_closure_pairs(tasks[task_id], specs[task_id], source_hash))

    pairs.sort(key=lambda row: (
        row["task_id"],
        row["pair_source"],
        -1 if row["source_repeat"] is None else row["source_repeat"],
        row["pair_id"],
    ))
    audit_path = os.path.splitext(args.out)[0] + ".audit.jsonl"
    save_jsonl(pairs, audit_path)

    unique_pairs = []
    seen_triplets = set()
    for row in pairs:
        key = (row["prompt"], row["chosen"], row["rejected"])
        if key in seen_triplets:
            continue
        seen_triplets.add(key)
        unique_pairs.append(row)
    save_jsonl(unique_pairs, args.out)

    duplicate_keys = Counter(
        (row["prompt"], row["chosen"], row["rejected"]) for row in pairs
    )
    stats = {
        "source_badcases": len(badcases),
        "candidate_pairs": len(pairs),
        "pairs": len(unique_pairs),
        "dropped": len(dropped),
        "unique_triplets": len(duplicate_keys),
        "duplicate_triplets": sum(count - 1 for count in duplicate_keys.values()),
        "by_stress": dict(Counter(row["stress"] for row in unique_pairs)),
        "by_attr_label": dict(Counter(row["attr_label"] for row in unique_pairs)),
        "by_decision_kind": dict(Counter(row["decision_kind"] for row in unique_pairs)),
        "by_pair_source": dict(Counter(row["pair_source"] for row in unique_pairs)),
        "by_decision_index": dict(Counter(str(row["decision_index"]) for row in unique_pairs)),
        "teacher": TEACHER,
        "source_badcase_sha256": source_hash,
    }
    stat_path = os.path.splitext(args.out)[0] + ".stat.json"
    with open(stat_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if dropped:
        print("\n丢弃明细:")
        for task_id, repeat, reason in dropped:
            print(f"- {task_id}#{repeat}: {reason}")
    print(f"\n训练集 {args.out}\n完整候选审计 {audit_path}\n统计 {stat_path}")
    if dropped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
