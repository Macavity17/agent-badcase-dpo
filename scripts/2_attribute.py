"""Step 2 — badcase 归因：把失败轨迹分到三类（对齐岗位三大课题）。

用法：
    python3 scripts/2_attribute.py --traj data/train_full.jsonl --out data/badcases_labeled.jsonl
    python3 scripts/2_attribute.py --traj data/train_full.jsonl --use-judge

三类定义（写 README 时可直接引用）：
- tool_misuse        工具误选：调用了不存在的工具名、参数缺失/多余、或选了与任务无关的相似工具
- context_forgetting 上下文遗忘：任务开头给了约束（预算/禁忌/单位/称呼），但最终答案未体现或违反
- planning_drift     规划发散：步数超预期 2 倍、重复同一动作 ≥3 次、或跑满步数仍未收敛

判定优先级：planning_drift > tool_misuse > context_forgetting
（发散往往同时伴随工具乱调，取最显著的那条作为主标签）

扩展方向（入职周晚上可做）：加上"上下文窗口内的信息分层观察"——
哪些信息该压缩、哪些该常驻，用这里的 context_chars 数据就能做出一手分析。
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import check_completion, format_trajectory, get_client, load_jsonl, save_jsonl, tool_names

LABELS = ["tool_misuse", "context_forgetting", "planning_drift"]


def rule_attribute(task, traj):
    """规则归因：返回 (label, evidence_list)。"""
    evidence = []
    steps = traj.get("steps") or []
    calls = traj.get("tool_calls") or []
    final = traj.get("final_answer") or ""
    valid = tool_names(task)
    expected = int(task.get("expected_steps", 3))
    max_steps = int(task.get("max_steps", 8))

    # ---- 1) planning_drift：未收敛 / 重复动作 / 步数爆炸 ----
    if not final:
        evidence.append("跑满步数或中途报错，未给出最终答案")
        return "planning_drift", evidence

    signatures = [(c.get("name"), json.dumps(c.get("args") or {}, sort_keys=True, ensure_ascii=False)) for c in calls]
    dup = Counter(signatures).most_common(1)
    if dup and dup[0][1] >= 3:
        evidence.append(f"重复调用同一工具 {dup[0][1]} 次：{dup[0][0][0]}")
        return "planning_drift", evidence

    if len(steps) >= expected * 2 and len(steps) >= max_steps:
        evidence.append(f"步数 {len(steps)} 达到上限（预期 {expected} 步），未收敛")
        return "planning_drift", evidence

    # ---- 2) tool_misuse：工具名不存在 / 参数缺失 / 选错相似工具 ----
    bad_names = [c.get("name") for c in calls if c.get("name") not in valid]
    if bad_names:
        evidence.append(f"调用了不存在的工具：{bad_names[:3]}")
        return "tool_misuse", evidence

    # 参数缺失：工具声明了 required 参数但调用时没给
    arg_errors = []
    for t in task.get("tools") or []:
        required = list((t.get("args") or {}).keys())
        for c in calls:
            if c.get("name") == t["name"] and required:
                missing = [k for k in required if k not in (c.get("args") or {})]
                if missing:
                    arg_errors.append(f"{t['name']} 缺少参数 {missing}")
                extra = [k for k in (c.get("args") or {}) if k not in required]
                if extra:
                    arg_errors.append(f"{t['name']} 包含未声明参数 {extra}")
    if arg_errors:
        evidence.extend(arg_errors[:3])
        return "tool_misuse", evidence

    if task.get("stress") == "tool_misuse" and not check_completion(task, traj):
        evidence.append("未完成任务要求的目标工具调用，可能选择了相似但错误的工具")
        return "tool_misuse", evidence

    # ---- 3) context_forgetting：有约束但最终答案没体现/违反了 ----
    cons = (task.get("constraints") or []) + (task.get("latent_constraints") or [])
    if cons:
        # 用 checker 里的 final_not_contains 反推：命中即说明违反了约束
        chk = task.get("checker") or {}
        checks = chk.get("checks") if chk.get("type") == "all" else [chk]
        for c in checks or []:
            if c.get("type") == "final_not_contains":
                hit = [v for v in (c.get("values") or []) if v in final]
                if hit:
                    evidence.append(f"最终答案违反约束，出现了禁用内容：{hit[:3]}")
                    return "context_forgetting", evidence

        # 约束里的数字/单位没出现在答案里（粗判：提取阿拉伯数字与单位词）
        import re
        nums = re.findall(r"\d+(?:\.\d+)?", " ".join(cons))
        missing_nums = [n for n in nums if n not in final]
        if nums and len(missing_nums) == len(nums):
            evidence.append(f"约束中的关键信息 {nums} 未出现在最终答案里")
            return "context_forgetting", evidence

        if task.get("stress") == "context_forgetting" and not check_completion(task, traj):
            evidence.append("轨迹未通过面向约束遵守设计的任务 checker")
            return "context_forgetting", evidence

    # ---- 兜底：按任务设计的 stress 类型标注（弱证据，README 里要说明）----
    evidence.append("规则未捕捉到明确信号，按任务设计类型标注（弱证据，人工复核后再用于训练）")
    return task.get("stress", "unknown"), evidence


def judge_attribute(task, traj, client, model):
    """可选：让强模型复核归因（更准，几十条的成本可忽略）。"""
    prompt = f"""你是 Agent 失败归因专家。下面是任务、可用工具、以及模型的执行轨迹。

任务：{task.get('goal')}
显式约束：{task.get('constraints') or []}
由环境观测暴露、仅供评审使用的约束：{task.get('latent_constraints') or []}
可用工具：{[t['name'] for t in (task.get('tools') or [])]}

执行轨迹：
{format_trajectory(traj)}

请判断这条失败轨迹的主因，只能从以下三类中选一个：
- tool_misuse：选错工具、参数错误、调用了不存在的工具
- context_forgetting：遗漏或违反了任务开头给出的约束
- planning_drift：规划发散、重复动作、步数超限未收敛

只输出 JSON：{{"label": "...", "reason": "一句话"}}"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        txt = resp.choices[0].message.content or ""
        s, e = txt.find("{"), txt.rfind("}")
        obj = json.loads(txt[s:e + 1]) if s >= 0 else {}
        return obj.get("label"), obj.get("reason", "")
    except Exception as ex:
        return None, f"judge 调用失败：{ex}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="data/train_full.jsonl")
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--out", default="data/badcases_labeled.jsonl")
    ap.add_argument("--use-judge", action="store_true", help="用强模型复核归因")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "deepseek-chat"))
    ap.add_argument("--include-errors", action="store_true", help="包含 API/服务错误轨迹（默认排除）")
    args = ap.parse_args()

    tasks = {t["task_id"]: t for t in load_jsonl(args.tasks)}
    trajs = load_jsonl(args.traj)
    client = get_client() if args.use_judge else None

    rows, dist = [], Counter()
    for tr in trajs:
        if check_completion(tasks.get(tr.get("task_id"), {}), tr):
            continue  # 只归因失败样本
        task = tasks.get(tr.get("task_id"))
        if task is None:
            continue
        if not args.include_errors and any(s.get("type") == "error" for s in (tr.get("steps") or [])):
            continue

        label, evidence = rule_attribute(task, tr)
        judge_label, judge_reason = (None, "")
        if args.use_judge and client:
            judge_label, judge_reason = judge_attribute(task, tr, client, args.judge_model)
            if judge_label in LABELS:
                label = judge_label  # 以 judge 为准

        dist[label] += 1
        rows.append({
            **tr,
            "attr_label": label,
            "attr_evidence": evidence,
            "judge_reason": judge_reason,
            "prompt": task["goal"] + (
                "\n\n约束条件（必须遵守）：\n" + "\n".join(f"- {c}" for c in (task.get("constraints") or []))
                if task.get("constraints") else ""
            ) + "\n\n可用工具定义：\n" + json.dumps(task.get("tools") or [], ensure_ascii=False),
        })

    save_jsonl(rows, args.out)

    print(f"\n失败样本 {len(rows)} 条，归因分布：")
    for k, v in dist.most_common():
        print(f"  {k:22s} {v:3d}  ({v / max(len(rows), 1):.0%})")
    if len(rows) < 15:
        print("\n⚠️ 失败样本不足 15 条，偏好数据会不够。回 tasks.jsonl 加量或调难度。")
    for k in LABELS:
        if dist[k] < 10:
            print(f"⚠️ {k} 只有 {dist[k]} 条，建议补到 10 条以上，否则分类别统计没意义。")
    print(f"\n已写入：{args.out}")


if __name__ == "__main__":
    main()
