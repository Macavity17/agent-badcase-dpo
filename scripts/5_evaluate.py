"""Step 5 — 评测与对比：完成率 + LLM-as-a-judge + 分类别统计，支持多策略对比。

用法：
    # 难度校准（只算完成率，不调 API，秒出）
    python scripts/5_evaluate.py --traj data/probe.jsonl --mode quick

    # 上下文策略对比（本实验的核心表）
    python scripts/5_evaluate.py \
        --files full=data/test_full.jsonl,window=data/test_window.jsonl,layered=data/test_layered.jsonl \
        --out results/context_compare.md

    # Phase 3：DPO 前后对比（含 judge，需配 OPENAI_API_KEY / OPENAI_BASE_URL）
    python scripts/5_evaluate.py --before data/p1_full.jsonl --after data/dpo_trajectories.jsonl --out results/phase3_compare.md

三块指标（口径写死在这里，前后必须一致）：
1. 任务完成率：规则 checker（客观主指标）
2. judge 质量分：工具选择 / 约束遵守 / 步骤效率，1–5 分（主观副指标）
3. 分类别改善：按三类失败模式拆开——本实验的核心视角
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import check_completion, format_trajectory, get_client, load_jsonl, write_text

RUBRIC = [
    ("tool_selection", "工具选择是否正确（选对工具、参数合理）"),
    ("constraint_adherence", "是否遵守任务中给出的约束（预算/禁忌/格式/单位）"),
    ("step_efficiency", "步骤是否高效（无冗余调用、无重复动作、无明显跑偏）"),
]

MODES = ["tool_misuse", "context_forgetting", "planning_drift"]

JUDGE_PROMPT = """你是 Agent 轨迹质量评审。请按三个维度给这条执行轨迹打 1–5 分。

## 任务
{goal}

## 约束
{constraints}

## 可用工具
{tools}

## 执行轨迹
{traj}

## 评分维度
{dims}

只输出 JSON，不要解释：{{"tool_selection": 1-5, "constraint_adherence": 1-5, "step_efficiency": 1-5}}"""


def load_tasks(path):
    return {t["task_id"]: t for t in load_jsonl(path)}


def stats_one(tasks, trajs):
    ok, steps, by_mode = 0, [], defaultdict(lambda: [0, 0])
    ctx, chars = [], []
    evaluated, calls, invalid_calls, trajectories_with_calls = 0, 0, 0, 0
    task_ids = set()
    for tr in trajs:
        task = tasks.get(tr.get("task_id"))
        if task is None:
            continue
        evaluated += 1
        task_ids.add(tr.get("task_id"))
        succ = bool(check_completion(task, tr))
        ok += succ
        steps.append(tr.get("n_steps") or 0)
        ctx.append(tr.get("max_prompt_tokens") or 0)
        chars.append(tr.get("max_context_chars") or 0)
        row_calls = tr.get("tool_calls") or []
        calls += len(row_calls)
        trajectories_with_calls += bool(row_calls)
        invalid_calls += sum(
            not c.get("valid_name", True) or not c.get("valid_args", True)
            for c in row_calls
        )
        mode = tr.get("stress") or "unknown"
        by_mode[mode][0] += succ
        by_mode[mode][1] += 1
    n = evaluated
    return {
        "n": n, "tasks": len(task_ids), "success": ok,
        "rate": ok / n if n else 0,
        "avg_steps": statistics.mean(steps) if steps else 0,
        "avg_ctx": statistics.mean(ctx) if ctx else 0,
        "avg_ctx_chars": statistics.mean(chars) if chars else 0,
        "tool_call_rate": trajectories_with_calls / n if n else 0,
        "invalid_call_rate": invalid_calls / calls if calls else 0,
        "by_mode": {k: (v[0] / v[1] if v[1] else 0, v[1]) for k, v in by_mode.items()},
    }


def judge_one(client, model, task, traj, samples=3, temperature=0.2):
    dims = "\n".join(f"- {k}：{d}" for k, d in RUBRIC)
    prompt = JUDGE_PROMPT.format(
        goal=task.get("goal"),
        constraints="\n".join(f"- {c}" for c in (task.get("constraints") or [])) or "(无)",
        tools=", ".join(t["name"] for t in (task.get("tools") or [])),
        traj=format_trajectory(traj)[:2500],
        dims=dims,
    )
    scores = defaultdict(list)
    for _ in range(samples):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature,
            )
            txt = resp.choices[0].message.content or ""
            s, e = txt.find("{"), txt.rfind("}")
            if s < 0:
                continue
            obj = json.loads(txt[s:e + 1])
            for k, _ in RUBRIC:
                if isinstance(obj.get(k), (int, float)):
                    scores[k].append(float(obj[k]))
        except Exception:
            continue
    return {k: (statistics.mean(v) if v else None) for k, _ in RUBRIC}


def judge_stats(client, model, tasks, trajs, samples=3):
    agg = defaultdict(list)
    for tr in trajs:
        task = tasks.get(tr.get("task_id"))
        if task is None:
            continue
        sc = judge_one(client, model, task, tr, samples=samples)
        for k, v in sc.items():
            if v is not None:
                agg[k].append(v)
    return {k: (statistics.mean(v) if v else None) for k, v in agg.items()}


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def build_multi_report(groups, ordered_labels):
    lines = ["# Phase 1：上下文策略 × 失效模式 对照结果", ""]
    lines.append(f"基线（full）覆盖 {groups[ordered_labels[0]]['tasks']} 个任务、"
                 f"{groups[ordered_labels[0]]['n']} 条轨迹/策略。"
                 "同一模型、同一任务集，只变上下文组织方式。")
    lines.append("")

    lines.append("## 1. 总体完成率")
    rows = []
    for lb in ordered_labels:
        st = groups[lb]
        rows.append([lb, st["n"], f"{st['rate']:.1%}", f"{st['avg_steps']:.2f}", f"{st['avg_ctx']:.0f}"])
    lines.append(md_table(["策略", "轨迹数", "完成率", "平均步数", "平均 prompt 峰值(tokens)"], rows))
    lines.append("")

    lines.append("## 2. 分类别完成率（核心表：策略 × 失效模式）")
    rows = []
    for lb in ordered_labels:
        st = groups[lb]
        row = [lb]
        for m in MODES:
            r, n = st["by_mode"].get(m, (None, 0))
            row.append(f"{r:.0%}(n={n})" if r is not None else "—")
        rows.append(row)
    lines.append(md_table(["策略"] + MODES, rows))
    lines.append("")

    lines.append("## 3. 上下文预算观察")
    rows = []
    for lb in ordered_labels:
        st = groups[lb]
        rows.append([lb, f"{st['avg_ctx']:.0f}", f"{st['rate']:.1%}"])
    lines.append(md_table(["策略", "平均 prompt 峰值(tokens)", "完成率"], rows))
    lines.append("")
    lines.append("看什么：layered 是否做到了「上下文峰值接近 window，但完成率显著更高」？"
                 "如果是，这就是「分层组织优于朴素压缩」的直接证据（JD 原文论断的复现）。")
    lines.append("")

    lines.append("## 4. 结论与局限")
    lines.append("- **结论**：（跑完再写。建议格式：对每一类失效，指明最优策略及差距幅度）")
    lines.append("- **局限**：任务集自建（设计偏差）、单模型、judge 为主观副指标")
    return "\n".join(lines)


def build_before_after(before, after, b_judge=None, a_judge=None):
    lines = ["# Phase 3：DPO 前后对比", ""]
    lines.append(f"样本：before {before['n']} / after {after['n']}")
    lines.append("")
    lines.append("## 1. 主指标")
    lines.append(md_table(
        ["指标", "Before", "After", "变化"],
        [
            ["任务完成率", f"{before['rate']:.1%}", f"{after['rate']:.1%}",
             f"{(after['rate'] - before['rate']) * 100:+.1f}pp"],
            ["平均步数", f"{before['avg_steps']:.2f}", f"{after['avg_steps']:.2f}",
             f"{after['avg_steps'] - before['avg_steps']:+.2f}"],
            ["工具调用率", f"{before['tool_call_rate']:.1%}", f"{after['tool_call_rate']:.1%}",
             f"{(after['tool_call_rate'] - before['tool_call_rate']) * 100:+.1f}pp"],
            ["工具协议错误占比", f"{before['invalid_call_rate']:.1%}", f"{after['invalid_call_rate']:.1%}",
             f"{(after['invalid_call_rate'] - before['invalid_call_rate']) * 100:+.1f}pp"],
        ],
    ))
    lines.append("")
    lines.append("## 2. 分类别（关键：DPO 到底修好了哪类失效）")
    rows = []
    for m in MODES:
        br, bn = before["by_mode"].get(m, (None, 0))
        ar, an = after["by_mode"].get(m, (None, 0))
        if not bn and not an:
            continue
        btxt = f"{br:.1%}" if br is not None else "—"
        atxt = f"{ar:.1%}" if ar is not None else "—"
        dtxt = f"{(ar - br) * 100:+.1f}pp" if (br is not None and ar is not None) else "—"
        rows.append([m, f"{bn}/{an}", btxt, atxt, dtxt])
    lines.append(md_table(["失效模式", "n(before/after)", "Before", "After", "变化"], rows))
    lines.append("")
    if b_judge and a_judge:
        lines.append("## 3. LLM-as-a-judge（1–5 分）")
        rows = []
        for k, d in RUBRIC:
            bv, av = b_judge.get(k), a_judge.get(k)
            if bv is None or av is None:
                continue
            rows.append([d, f"{bv:.2f}", f"{av:.2f}", f"{av - bv:+.2f}"])
        lines.append(md_table(["维度", "Before", "After", "变化"], rows))
        lines.append("")
    lines.append("## 4. 结论与局限")
    lines.append("- **结论**：（跑完再写。核心问题：DPO 修好了哪类、没修好哪类、有没有恶化哪类）")
    lines.append("- **局限**：偏好对规模小、单 seed（入职周补多 seed）、chosen 为合成数据")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", default=None)
    ap.add_argument("--files", default=None,
                    help="多策略对比：full=data/p1_full.jsonl,window=data/p1_window.jsonl,...")
    ap.add_argument("--traj", default=None, help="quick 模式：只算这一份的完成率")
    ap.add_argument("--mode", default="full", choices=["quick", "full"])
    ap.add_argument("--out", default="results/compare.md")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "deepseek-chat"))
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)

    if args.mode == "quick" or args.traj:
        trajs = load_jsonl(args.traj or args.before)
        st = stats_one(tasks, trajs)
        print(f"\n样本 {st['n']}：成功 {st['success']}（完成率 {st['rate']:.0%}），平均步数 {st['avg_steps']:.2f}")
        print("分类别完成率：", {k: f"{v[0]:.0%}(n={v[1]})" for k, v in st["by_mode"].items()})
        print("\n目标区间 30–50%：高于 70% 任务太简单，低于 20% 全是噪声。")
        return

    if args.files:
        groups, labels = {}, []
        for part in args.files.split(","):
            lb, path = part.split("=", 1)
            groups[lb] = stats_one(tasks, load_jsonl(path))
            labels.append(lb)
        report = build_multi_report(groups, labels)
        write_text(report, args.out)
        print(report)
        print(f"\n已写入：{args.out}")
        return

    if not (args.before and args.after):
        print("full 模式需要 --before 和 --after，或 --files 多策略对比")
        return

    b_tr, a_tr = load_jsonl(args.before), load_jsonl(args.after)
    before, after = stats_one(tasks, b_tr), stats_one(tasks, a_tr)

    b_judge = a_judge = None
    if args.judge:
        client = get_client()
        b_judge = judge_stats(client, args.judge_model, tasks, b_tr, args.samples)
        a_judge = judge_stats(client, args.judge_model, tasks, a_tr, args.samples)

    report = build_before_after(before, after, b_judge, a_judge)
    write_text(report, args.out)
    print(report)
    print(f"\n已写入：{args.out}")


if __name__ == "__main__":
    main()
