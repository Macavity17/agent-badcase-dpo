"""Validate the synthetic care-agent task set before running experiments."""

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_jsonl


CHECKER_TYPES = {
    "tool_call", "tool_not_called", "tool_call_sequence", "max_tool_calls",
    "final_contains", "final_contains_any", "final_contains_all",
    "final_not_contains", "all", "any",
}
STRESSES = {"tool_misuse", "context_forgetting", "planning_drift"}
SPLITS = {"train", "test"}


def walk_checks(check):
    yield check
    for child in check.get("checks") or []:
        yield from walk_checks(child)


def validate_task(task):
    errors = []
    required = ["task_id", "split", "scenario_family", "stress", "goal", "tools", "checker"]
    for key in required:
        if not task.get(key):
            errors.append(f"缺少字段 {key}")

    if task.get("split") not in SPLITS:
        errors.append(f"split 必须是 {sorted(SPLITS)}")
    if task.get("stress") not in STRESSES:
        errors.append(f"stress 必须是 {sorted(STRESSES)}")

    tools = task.get("tools") or []
    names = [t.get("name") for t in tools]
    if len(names) != len(set(names)):
        errors.append("工具名重复")
    if len(tools) < 3:
        errors.append("至少需要 3 个工具以形成真实选择空间")

    for tool in tools:
        if not tool.get("name") or not tool.get("desc") or not isinstance(tool.get("args"), dict):
            errors.append(f"工具定义不完整：{tool}")

    for check in walk_checks(task.get("checker") or {}):
        ctype = check.get("type")
        if ctype not in CHECKER_TYPES:
            errors.append(f"未知 checker 类型 {ctype}")
        if ctype == "tool_call" and check.get("expect_tool") not in names:
            errors.append(f"checker 引用了不存在的工具 {check.get('expect_tool')}")
        if ctype == "tool_call" and check.get("expect_tool") in names:
            tool = next(t for t in tools if t.get("name") == check.get("expect_tool"))
            declared = set((tool.get("args") or {}).keys())
            unexpected = set((check.get("expect_args_contains") or {}).keys()) - declared
            if unexpected:
                errors.append(f"checker 参数不在 {tool.get('name')} schema 中：{sorted(unexpected)}")
        if ctype in {"tool_not_called", "tool_call_sequence"}:
            missing = [name for name in (check.get("tools") or []) if name not in names]
            if missing:
                errors.append(f"checker 引用了不存在的工具 {missing}")

    expected = int(task.get("expected_steps", 0))
    maximum = int(task.get("max_steps", 0))
    if expected <= 0 or maximum < expected:
        errors.append("expected_steps/max_steps 配置不合法")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="tasks/tasks.jsonl")
    args = parser.parse_args()

    rows = load_jsonl(args.tasks)
    ids = Counter(row.get("task_id") for row in rows)
    errors = []
    for task in rows:
        for error in validate_task(task):
            errors.append(f"{task.get('task_id', '<unknown>')}: {error}")
    for task_id, count in ids.items():
        if count > 1:
            errors.append(f"{task_id}: task_id 重复 {count} 次")

    family_splits = defaultdict(set)
    for task in rows:
        family_splits[task.get("scenario_family")].add(task.get("split"))
    leaked = [family for family, splits in family_splits.items() if len(splits) > 1]
    if leaked:
        errors.append(f"scenario_family 跨训练/测试集泄漏：{leaked}")

    print(f"任务总数：{len(rows)}")
    print("按 split：", dict(Counter(row.get("split") for row in rows)))
    print("按失效模式：", dict(Counter(row.get("stress") for row in rows)))
    print("split x 失效模式：", dict(Counter((row.get("split"), row.get("stress")) for row in rows)))
    if errors:
        print("\n校验失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\n校验通过：字段、checker 引用与场景族隔离均有效。")


if __name__ == "__main__":
    main()
