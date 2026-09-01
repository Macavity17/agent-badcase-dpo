"""Step 3 — 偏好对构造：rejected=失败轨迹，chosen=强模型合成的正确轨迹。

用法：
    export OPENAI_API_KEY=xxx
    export OPENAI_BASE_URL=xxx      # DeepSeek / 通义 / OpenAI 兼容地址
    python scripts/3_build_preference.py --badcase data/badcases_labeled.jsonl --out data/pref_pairs.jsonl

成本：50 条 × 2–3 次调用 ≈ 几十元。

这是"合成数据生产"，也是岗位 JD 明确认可的动手项之一：
数据从哪来、怎么保证 chosen 质量、rejected 截断到哪一步，都要能讲清楚。

质量红线：
- chosen 必须真的满足 checker（脚本会验证，不通过的丢弃，别硬塞）；
- rejected 截断到失败点，不要把无关尾巴也算进去；
- 合成数据占比要在 README 里如实写，别包装成"人工标注数据"。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import check_completion, get_client, load_jsonl, save_jsonl

SYNTH_PROMPT = """你是 Agent 轨迹修复专家。下面是一个失败的执行案例，请给出**正确的**执行轨迹。

## 任务
{goal}

## 约束（必须全部遵守）
{constraints}

## 可用工具（只能从这些里选）
{tools}

## 失败轨迹（供参考，找出它错在哪）
{failed}

## 输出要求
严格输出 JSON，不要任何解释文字、不要 markdown 代码块：
{{
  "steps": [
    {{"tool": "工具名", "args": {{"参数名": "参数值"}}}}
  ],
  "final_answer": "给用户的最终答复（必须体现上述约束）"
}}

规则：
1. steps 里只放必要的工具调用，最多 {max_steps} 步；
2. 工具名和参数名必须严格来自上面的可用工具列表；
3. final_answer 必须体现所有约束（如预算上限、禁忌、单位、称呼）。"""


def render_steps(steps):
    lines = []
    for s in steps or []:
        if s.get("tool"):
            lines.append(f"[{s.get('step')}] CALL {s['tool']}({json.dumps(s.get('args') or {}, ensure_ascii=False)})")
        elif s.get("type") == "final":
            lines.append(f"FINAL: {s.get('content')}")
    return "\n".join(lines) or "(无有效步骤)"


def synth_chosen(client, model, task, traj):
    """让强模型生成正确轨迹，返回 (steps_text, ok)。"""
    tools_desc = "\n".join(
        f"- {t['name']}({', '.join((t.get('args') or {}).keys())})：{t.get('desc','')}"
        for t in (task.get("tools") or [])
    )
    cons = task.get("constraints") or []
    prompt = SYNTH_PROMPT.format(
        goal=task.get("goal"),
        constraints="\n".join(f"- {c}" for c in cons) or "(无)",
        tools=tools_desc,
        failed=render_steps(traj.get("steps"))[:2000],
        max_steps=int(task.get("max_steps", 8)),
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content or ""
        s, e = txt.find("{"), txt.rfind("}")
        if s < 0:
            return None, "输出中没有 JSON"
        obj = json.loads(txt[s:e + 1])
        steps = obj.get("steps") or []
        final = obj.get("final_answer") or ""
        # 校验：chosen 必须真的能过 checker，否则丢弃（宁缺毋滥）
        pseudo = {"tool_calls": [{"name": x.get("tool"), "args": x.get("args") or {}} for x in steps],
                  "final_answer": final}
        ok = check_completion(task, pseudo)
        text = render_steps_from_synth(steps) + f"\nFINAL: {final}"
        return text, (ok, "checker 通过" if ok else "checker 未通过，已丢弃")
    except Exception as ex:
        return None, f"调用/解析失败：{ex}"


def render_steps_from_synth(steps):
    return "\n".join(
        f"[{i+1}] CALL {x.get('tool')}({json.dumps(x.get('args') or {}, ensure_ascii=False)})"
        for i, x in enumerate(steps)
    )


def truncate_rejected(traj):
    """rejected：截断到失败点（最后一个工具调用），避免把发散的尾巴也算进去。"""
    return render_steps(traj.get("steps"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--badcase", default="data/badcases_labeled.jsonl")
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--out", default="data/pref_pairs.jsonl")
    ap.add_argument("--model", default=os.environ.get("SYNTH_MODEL", "deepseek-chat"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tasks = {t["task_id"]: t for t in load_jsonl(args.tasks)}
    bad = load_jsonl(args.badcase)
    if args.limit:
        bad = bad[: args.limit]

    client = get_client()
    pairs, dropped = [], 0

    for b in bad:
        task = tasks.get(b.get("task_id"))
        if task is None:
            continue
        chosen, info = synth_chosen(client, args.model, task, b)
        if chosen is None or info is True or (isinstance(info, tuple) and not info[0]):
            dropped += 1
            print(f"[{b.get('task_id')}] 丢弃：{info}")
            continue
        pairs.append({
            "task_id": b.get("task_id"),
            "stress": b.get("stress"),
            "attr_label": b.get("attr_label"),
            "prompt": b.get("prompt"),
            "chosen": chosen,
            "rejected": truncate_rejected(b),
        })
        print(f"[{b.get('task_id')}] ✓ {b.get('attr_label')}")

    save_jsonl(pairs, args.out)
    print(f"\n生成偏好对 {len(pairs)} 条，丢弃 {dropped} 条（chosen 未通过 checker）")
    if len(pairs) < 100:
        print("⚠️ 少于 100 条，DPO 效果会很弱。建议：加任务量 / 每条任务多采样几个失败轨迹 / 人工补几条 chosen。")
    print(f"已写入：{args.out}")


if __name__ == "__main__":
    main()
