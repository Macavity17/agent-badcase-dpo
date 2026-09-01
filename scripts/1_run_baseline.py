"""Step 1 — 跑基线：用 ReAct loop 采集完整轨迹，支持四种上下文组织策略（Phase 1 对照实验核心）。

用法：
    # Phase 1：四种策略各跑一遍（对照实验）
    python scripts/1_run_baseline.py --strategy full    --tasks tasks/tasks.jsonl --out data/p1_full.jsonl
    python scripts/1_run_baseline.py --strategy window  --tasks tasks/tasks.jsonl --out data/p1_window.jsonl
    python scripts/1_run_baseline.py --strategy summary --tasks tasks/tasks.jsonl --out data/p1_summary.jsonl
    python scripts/1_run_baseline.py --strategy layered --tasks tasks/tasks.jsonl --out data/p1_layered.jsonl

    # 难度校准（先跑这个！）
    python scripts/1_run_baseline.py --strategy full --tasks tasks/tasks.jsonl --limit 10 --out data/probe.jsonl

四种策略 = 同一模型，只变"上下文怎么组织"：
- full    全量历史（对照组，不做任何管理）
- window  滑动窗口：只保留最近 K 轮工具交互（模拟最朴素的截断）
- summary 摘要压缩：旧历史定期压成一段摘要（模拟多数"记忆"产品的做法）
- layered 分层结构化：目标/约束/已完成步骤/关键观测做成常驻状态块 + 最近 2 轮细节
          （模拟 event-memory 思路：该常驻的常驻，该滚动的滚动）

轨迹里记录每步实际发给模型的上下文长度（context_chars）——
这是"上下文预算"的一手数据，别浪费。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    build_tools_payload, check_completion, get_client, load_jsonl,
    mock_tool_call, save_jsonl, tool_names,
)

SYSTEM = (
    "你是一个严谨的办公助手，可以调用工具完成任务。\n"
    "要求：\n"
    "1. 每次只调用一个必要的工具，参数必须严格符合 schema；\n"
    "2. 仔细阅读任务中给出的所有约束（预算、禁忌、格式、单位），任何一步都不能违反；\n"
    "3. 达成目标后立即用自然语言给出最终答案，不要再调用多余工具；\n"
    "4. 如果信息已足够就直接作答，不要重复调用同一个工具。"
)

SUMMARIZE_PROMPT = (
    "把下面的智能体执行历史压缩成一段简洁的任务进展摘要（150字以内）。"
    "必须保留：任务目标、所有约束（预算/禁忌/单位/称呼等）、已完成步骤、"
    "已获得的关键数据（数字、名称、ID）、尚未完成的子目标。只输出摘要本身。"
)


# ---------------- 上下文组织策略（Phase 1 核心） ----------------

def _pair_up(messages):
    """把 [system, user, (assistant, tool...)*] 的中段拆成 (assistant+tools) 轮次对。"""
    rest = messages[2:]
    pairs, i = [], 0
    while i < len(rest):
        if rest[i].get("role") == "assistant":
            j = i + 1
            while j < len(rest) and rest[j].get("role") == "tool":
                j += 1
            pairs.append(rest[i:j])
            i = j
        else:
            i += 1
    return pairs


def apply_window(messages, keep_rounds=4):
    """滑动窗口：system+user 保留，中间只留最近 K 轮。模拟最朴素的截断做法。"""
    pairs = _pair_up(messages)
    kept = pairs[-keep_rounds:]
    flat = [m for p in kept for m in p]
    return messages[:2] + flat


def apply_summary(messages, client, model, keep_rounds=2, compress_threshold=3):
    """摘要压缩：比窗口更早的历史压成一段摘要，插入 system 后。模拟多数记忆产品。

    注意：用被测模型自己压缩（不用强模型）——模拟真实产品里小模型自压缩的
    场景，同时避免引入外部模型的额外 bias。这是个可以质疑的设计决策，
    详见实验手册思考题 T3.2。
    """
    pairs = _pair_up(messages)
    if len(pairs) <= compress_threshold + keep_rounds:
        return messages
    old, kept = pairs[:-keep_rounds], pairs[-keep_rounds:]
    history_text = "\n".join(
        f"[{i+1}] {json.dumps(m, ensure_ascii=False)[:300]}"
        for i, p in enumerate(old) for m in p
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SUMMARIZE_PROMPT + "\n\n" + history_text}],
            temperature=0.1,
        )
        summary_text = (resp.choices[0].message.content or "").strip()[:400]
    except Exception:
        summary_text = "（摘要失败，退化为窗口截断）" + history_text[-200:]
    summary_msg = {"role": "system", "content": f"[执行历史摘要]\n{summary_text}"}
    flat = [m for p in kept for m in p]
    return messages[:2] + [summary_msg] + flat


def build_state_block(task, steps_so_far):
    """分层结构化的常驻状态块：目标/约束/已完成/关键观测。

    这一层是纯规则维护（不花模型调用、零成本、确定性强）——
    "该常驻的常驻，该滚动的滚动"。
    """
    lines = ["=== 任务状态（常驻，每步可见）==="]
    lines.append(f"目标: {task['goal']}")
    cons = task.get("constraints") or []
    if cons:
        lines.append("约束（必须遵守）:")
        lines += [f"  - {c}" for c in cons]
    done = [s for s in steps_so_far if s.get("tool")]
    if done:
        lines.append("已完成步骤（不要重复）:")
        lines += [f"  {i+1}. {s['tool']}({json.dumps(s.get('args') or {}, ensure_ascii=False)})" for i, s in enumerate(done)]
    obs = [s.get("observation", "") for s in steps_so_far if s.get("observation")]
    if obs:
        lines.append("关键观测（最近3条）:")
        lines += [f"  - {o[:90]}" for o in obs[-3:]]
    lines.append("=== 继续执行任务 ===")
    return "\n".join(lines)


def apply_layered(messages, task, steps_so_far, keep_rounds=2):
    """分层：常驻状态块（规则维护）+ 最近 2 轮细节。event-memory 思路。"""
    pairs = _pair_up(messages)
    kept = pairs[-keep_rounds:]
    flat = [m for p in kept for m in p]
    state = {"role": "system", "content": build_state_block(task, steps_so_far)}
    return messages[:1] + [state] + messages[1:2] + flat


# ---------------- ReAct 主循环 ----------------

def build_user_message(task):
    msg = task["goal"]
    cons = task.get("constraints") or []
    if cons:
        msg += "\n\n约束条件（必须遵守）：\n" + "\n".join(f"- {c}" for c in cons)
    return msg


def run_one(client, model, task, strategy="full", verbose=False):
    tools = build_tools_payload(task.get("tools") or [])
    valid_names = tool_names(task)
    max_steps = int(task.get("max_steps", 8))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_message(task)},
    ]

    steps, tool_calls, final_answer = [], [], ""
    context_chars = []

    for i in range(max_steps):
        # ---- 上下文组织策略：每步调用模型前组装 ----
        if strategy == "full":
            eff = messages
        elif strategy == "window":
            eff = apply_window(messages)
        elif strategy == "summary":
            eff = apply_summary(messages, client, model)
        elif strategy == "layered":
            eff = apply_layered(messages, task, steps)
        else:
            raise ValueError(f"未知策略: {strategy}")

        try:
            resp = client.chat.completions.create(
                model=model, messages=eff, tools=tools, tool_choice="auto", temperature=0.2,
            )
        except Exception as e:
            steps.append({"step": i + 1, "type": "error", "error": str(e)})
            break

        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []

        if not calls:
            final_answer = msg.content or ""
            steps.append({"step": i + 1, "type": "final", "content": final_answer})
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{
                "id": c.id, "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            } for c in calls],
        })
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except Exception:
                args = {"_raw": c.function.arguments}
            obs = mock_tool_call(name, args, task)
            tool_calls.append({"name": name, "args": args, "valid_name": name in valid_names})
            steps.append({
                "step": i + 1, "type": "tool", "thought": msg.content or "",
                "tool": name, "args": args, "observation": obs[:2000],
            })
            messages.append({"role": "tool", "tool_call_id": c.id, "content": obs})

        context_chars.append(sum(len(str(m.get("content") or "")) for m in eff))
        if verbose:
            print(f"  step{i+1}: {name}({args}) ctx={context_chars[-1]}")

    traj_core = {"tool_calls": tool_calls, "final_answer": final_answer}
    return {
        "task_id": task.get("task_id"),
        "stress": task.get("stress"),
        "strategy": strategy,
        "goal": task.get("goal"),
        "constraints": task.get("constraints") or [],
        "model": model,
        "steps": steps,
        "tool_calls": tool_calls,
        "final_answer": final_answer,
        "n_steps": len(steps),
        "context_chars": context_chars,
        "max_context_chars": max(context_chars) if context_chars else 0,
        "success": check_completion(task, traj_core),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--out", default="data/p1_full.jsonl")
    ap.add_argument("--model", default="base")
    ap.add_argument("--strategy", default="full", choices=["full", "window", "summary", "layered"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tasks = load_jsonl(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    client = get_client(base_url=args.base_url)
    rows = []
    for t in tasks:
        if args.verbose:
            print(f"[{t.get('task_id')}] {t.get('goal')[:40]}...")
        rows.append(run_one(client, args.model, t, strategy=args.strategy, verbose=args.verbose))

    save_jsonl(rows, args.out)

    ok = sum(1 for r in rows if r["success"])
    avg_ctx = sum(r["max_context_chars"] for r in rows) / max(len(rows), 1)
    print(f"\n[{args.strategy}] 完成 {len(rows)} 条：成功 {ok}（完成率 {ok / max(len(rows), 1):.0%}），"
          f"平均上下文峰值 {avg_ctx:.0f} 字符")
    if args.limit:
        print("（难度校准模式）目标区间 30–50%。高于 70% 任务太简单，低于 20% 全是噪声——回去调 tasks.jsonl。")
    print(f"轨迹已写入：{args.out}")


if __name__ == "__main__":
    main()
