"""Step 1 — 用 ReAct loop 采集轨迹并对比上下文组织策略。

用法：
    python3 scripts/1_run_baseline.py --split test --strategy full --repeats 3 --out data/test_full.jsonl
    python3 scripts/1_run_baseline.py --split test --strategy window --repeats 3 --out data/test_window.jsonl
    python3 scripts/1_run_baseline.py --split test --strategy layered --repeats 3 --out data/test_layered.jsonl

    # 难度校准（先跑这个！）
    python scripts/1_run_baseline.py --strategy full --tasks tasks/tasks.jsonl --limit 10 --out data/probe.jsonl

主要策略 = 同一模型，只变"上下文怎么组织"：
- full    全量历史（对照组，不做任何管理）
- window  滑动窗口：只保留最近 K 轮工具交互（模拟最朴素的截断）
- layered 分层结构化：目标/约束/已完成步骤/关键观测做成常驻状态块 + 最近 2 轮细节
          （模拟 event-memory 思路：该常驻的常驻，该滚动的滚动）

`summary` 作为可选探索策略保留，但不进入两晚主实验。

轨迹同时记录 API usage 中的 prompt tokens 与字符数。
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    AGENT_SYSTEM, FINISH_TOOL, build_tools_payload, build_user_message,
    check_completion, get_client, load_jsonl, mock_tool_call, save_jsonl,
    tool_names,
)

SYSTEM = AGENT_SYSTEM

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


def apply_window(messages, keep_rounds=2):
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


def build_state_block(task, compressed_steps):
    """分层结构化的常驻状态块：目标/约束/早期步骤/关键观测。

    这一层是纯规则维护（不花模型调用、零成本、确定性强）——
    "该常驻的常驻，该滚动的滚动"。
    """
    lines = [f"[常驻目标] {task['goal']}"]
    cons = task.get("constraints") or []
    if cons:
        lines.append("[必须遵守] " + "；".join(cons))
    done = [s for s in compressed_steps if s.get("tool")]
    if done:
        lines.append("[已完成，不要重复]")
        lines += [
            f"- {s['tool']}({json.dumps(s.get('args') or {}, ensure_ascii=False)})"
            f" -> {str(s.get('observation') or '')[:160]}"
            for s in done
        ]
    lines.append("[控制] 根据最近原始结果继续未完成动作；全部完成后才调用 finish_task。")
    return "\n".join(lines)


def apply_layered(messages, task, steps_so_far, keep_rounds=2):
    """分层：稳定任务状态 + 压缩的早期记忆 + 最近两轮原始细节。"""
    pairs = _pair_up(messages)
    kept = pairs[-keep_rounds:]
    flat = [m for p in kept for m in p]
    compressed_count = max(0, len(steps_so_far) - keep_rounds)
    state = {
        "role": "user",
        "content": build_state_block(task, steps_so_far[:compressed_count]),
    }
    return messages[:1] + [state] + flat


# ---------------- ReAct 主循环 ----------------

def run_one(client, model, task, strategy="full", verbose=False,
            temperature=0.2, seed=None, repeat=0):
    tools = build_tools_payload(task.get("tools") or [])
    tools.append(FINISH_TOOL)
    valid_names = tool_names(task)
    declared_args = {t["name"]: set((t.get("args") or {}).keys()) for t in (task.get("tools") or [])}
    max_steps = int(task.get("max_steps", 8))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_message(task)},
    ]

    steps, tool_calls, final_answer = [], [], ""
    context_chars, prompt_tokens = [], []
    finish_requested = False

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

        context_chars.append(sum(len(str(m.get("content") or "")) for m in eff))
        try:
            resp = client.chat.completions.create(
                model=model, messages=eff, tools=tools,
                tool_choice="none" if finish_requested else "required",
                parallel_tool_calls=False, temperature=temperature, seed=seed,
            )
        except Exception as e:
            steps.append({"step": i + 1, "type": "error", "error": str(e)})
            break

        usage = getattr(resp, "usage", None)
        prompt_tokens.append(int(getattr(usage, "prompt_tokens", 0) or 0))

        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []

        if not calls:
            final_answer = msg.content or ""
            step_type = "final" if finish_requested else "protocol_error"
            steps.append({"step": i + 1, "type": step_type, "content": final_answer})
            break

        if finish_requested:
            steps.append({
                "step": i + 1, "type": "protocol_error",
                "error": "finish_task 后仍返回工具调用",
            })
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

            if name == "finish_task":
                steps.append({
                    "step": i + 1, "type": "control",
                    "tool": name, "args": args,
                    "observation": "{将进入最终答复阶段}",
                })
                messages.append({
                    "role": "tool", "tool_call_id": c.id,
                    "content": '{"ready_for_final": true}',
                })
                finish_requested = True
                continue

            obs = mock_tool_call(name, args, task)
            tool_calls.append({
                "name": name,
                "args": args,
                "valid_name": name in valid_names,
                "valid_args": name in declared_args and set(args) == declared_args[name],
            })
            steps.append({
                "step": i + 1, "type": "tool", "thought": msg.content or "",
                "tool": name, "args": args, "observation": obs[:2000],
            })
            messages.append({"role": "tool", "tool_call_id": c.id, "content": obs})

        if verbose:
            names = ",".join(c.function.name for c in calls)
            print(f"  step{i+1}: {names} ctx={context_chars[-1]} chars/{prompt_tokens[-1]} tokens")

    traj_core = {"tool_calls": tool_calls, "final_answer": final_answer}
    return {
        "task_id": task.get("task_id"),
        "stress": task.get("stress"),
        "strategy": strategy,
        "goal": task.get("goal"),
        "constraints": task.get("constraints") or [],
        "model": model,
        "split": task.get("split", "unspecified"),
        "scenario_family": task.get("scenario_family"),
        "repeat": repeat,
        "seed": seed,
        "steps": steps,
        "tool_calls": tool_calls,
        "final_answer": final_answer,
        "n_steps": len(steps),
        "context_chars": context_chars,
        "max_context_chars": max(context_chars) if context_chars else 0,
        "prompt_tokens": prompt_tokens,
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else 0,
        "total_prompt_tokens": sum(prompt_tokens),
        "success": check_completion(task, traj_core),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--out", default="data/p1_full.jsonl")
    ap.add_argument("--model", default="base")
    ap.add_argument("--strategy", default="full", choices=["full", "window", "summary", "layered"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--port", type=int, default=None, help="vLLM 端口；等价于 --base-url http://localhost:PORT/v1")
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="保留输出文件中已完成的 task/repeat")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tasks = load_jsonl(args.tasks)
    if args.split != "all":
        tasks = [t for t in tasks if t.get("split") == args.split]
    if args.limit:
        tasks = tasks[: args.limit]

    base_url = args.base_url or (f"http://localhost:{args.port}/v1" if args.port else None)
    client = get_client(base_url=base_url)
    rows = load_jsonl(args.out) if args.resume and os.path.exists(args.out) else []
    done = {(r.get("task_id"), r.get("repeat", 0), r.get("strategy")) for r in rows}
    jobs = []
    for task_index, task in enumerate(tasks):
        for repeat in range(args.repeats):
            key = (task.get("task_id"), repeat, args.strategy)
            if key not in done:
                jobs.append((task_index, task, repeat, args.seed + task_index * 1000 + repeat))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_one, client, args.model, task, args.strategy, args.verbose,
                args.temperature, seed, repeat,
            ): (task, repeat)
            for _, task, repeat, seed in jobs
        }
        for future in as_completed(futures):
            task, repeat = futures[future]
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: (r.get("task_id", ""), r.get("repeat", 0)))
            save_jsonl(rows, args.out)
            if args.verbose:
                print(f"[{task.get('task_id')}#{repeat}] success={row['success']}")

    save_jsonl(rows, args.out)

    ok = sum(1 for r in rows if r["success"])
    avg_ctx = sum(r["max_context_chars"] for r in rows) / max(len(rows), 1)
    avg_tokens = sum(r.get("max_prompt_tokens", 0) for r in rows) / max(len(rows), 1)
    print(f"\n[{args.strategy}] 完成 {len(rows)} 条：成功 {ok}（完成率 {ok / max(len(rows), 1):.0%}），"
          f"平均上下文峰值 {avg_tokens:.0f} tokens / {avg_ctx:.0f} 字符")
    if args.limit:
        print("（难度校准模式）目标区间 30–50%。高于 70% 任务太简单，低于 20% 全是噪声——回去调 tasks.jsonl。")
    print(f"轨迹已写入：{args.out}")


if __name__ == "__main__":
    main()
