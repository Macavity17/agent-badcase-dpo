"""Step 3 — 偏好对构造：rejected=失败轨迹，chosen=强模型合成的正确轨迹。

用法：
    export OPENAI_API_KEY=xxx
    export OPENAI_BASE_URL=xxx      # DeepSeek / 通义 / OpenAI 兼容地址
    python scripts/3_build_preference.py --badcase data/badcases_labeled.jsonl --out data/pref_pairs.jsonl

成本：50 条 × 2–3 次调用 ≈ 几十元。

这是"合成数据生产"，也是岗位 JD 明确认可的动手项之一：
数据从哪来、怎么保证 chosen 质量、rejected 保留哪些失败证据，都要能讲清楚。

质量红线：
- chosen 必须同时通过工具协议校验与任务 checker（不通过即丢弃）；
- rejected 保留能够呈现主失败原因的完整响应轨迹；
- 合成数据占比要在 README 里如实写，别包装成"人工标注数据"。
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

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

## 模拟环境的确定性工具返回
{mock_responses}

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
3. 关键约束可能出现在工具返回中，final_answer 必须保留这些约束。"""


def render_steps(steps):
    lines = []
    for s in steps or []:
        if s.get("tool"):
            lines.append(render_tool_call(s["tool"], s.get("args") or {}))
        elif s.get("type") == "final":
            lines.append(s.get("content") or "")
    return "\n".join(lines) or "(无有效步骤)"


def render_tool_call(name, args):
    payload = {"name": name, "arguments": args}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def validate_synth_steps(task, steps):
    tools = {t["name"]: set((t.get("args") or {}).keys()) for t in (task.get("tools") or [])}
    if not steps:
        return False, "steps 为空"
    for step in steps:
        name = step.get("tool")
        args = step.get("args") or {}
        if name not in tools:
            return False, f"调用了未声明工具 {name}"
        if set(args) != tools[name]:
            return False, f"{name} 参数键不匹配：期望 {sorted(tools[name])}，实际 {sorted(args)}"
    return True, "工具协议通过"


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
        mock_responses=json.dumps(task.get("mock_responses") or {}, ensure_ascii=False),
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
        protocol_ok, protocol_info = validate_synth_steps(task, steps)
        if not protocol_ok:
            return None, protocol_info
        # 校验：chosen 必须真的能过 checker，否则丢弃（宁缺毋滥）
        pseudo = {"tool_calls": [{"name": x.get("tool"), "args": x.get("args") or {}} for x in steps],
                  "final_answer": final}
        ok = check_completion(task, pseudo)
        text = render_steps_from_synth(steps) + f"\n{final}"
        return {
            "text": text,
            "steps": steps,
            "final_answer": final,
        }, (ok, "checker 通过" if ok else "checker 未通过，已丢弃")
    except Exception as ex:
        return None, f"调用/解析失败：{ex}"


def render_steps_from_synth(steps):
    return "\n".join(render_tool_call(x.get("tool"), x.get("args") or {}) for x in steps)


def walk_checks(check):
    yield check
    for child in check.get("checks") or []:
        yield from walk_checks(child)


def find_mock_values(value, key):
    found = []
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if child_key == key and isinstance(child_value, (str, int, float)):
                found.append(str(child_value))
            found.extend(find_mock_values(child_value, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_mock_values(child, key))
    return found


def canonical_arg(task, tool_name, arg_name, expected):
    """从任务、checker 和确定性 mock 中构造可追溯的具体参数。"""
    if arg_name in expected.get(tool_name, {}):
        value = str(expected[tool_name][arg_name])
        return value + "30" if arg_name == "send_at" and value.endswith(":") else value

    goal = task.get("goal") or ""
    mocks = task.get("mock_responses") or {}
    dataflow_sources = {
        ("draft_followup_message", "context"): ("get_last_followup", "summary"),
        ("convert_measurements", "records"): ("get_measurement_history", None),
        ("generate_summary", "records"): ("convert_measurements", None),
        ("generate_plan_summary", "history"): ("get_execution_history", None),
        ("generate_care_report", "measurements"): ("get_weekly_measurements", None),
        ("generate_care_report", "adherence"): ("get_task_adherence", None),
    }
    source = dataflow_sources.get((tool_name, arg_name))
    if source:
        source_value = mocks.get(source[0])
        if source[1] and isinstance(source_value, dict):
            source_value = source_value.get(source[1])
        if isinstance(source_value, str):
            return source_value
        return json.dumps(source_value, ensure_ascii=False, separators=(",", ":"))
    if arg_name == "slot":
        slots = find_mock_values(mocks, "slots")
        if slots:
            return slots[0]
        for value in mocks.values():
            if isinstance(value, dict) and value.get("slots"):
                return str(value["slots"][0])
    if arg_name == "record_ids":
        ids = find_mock_values(mocks, "id")
        if ids:
            return ",".join(ids)
    if arg_name in {"task_id_a", "task_id_b"}:
        task_ids = re.findall(r"T-[A-Z0-9-]+", goal)
        offset = 0 if arg_name == "task_id_a" else 1
        if len(task_ids) > offset:
            return task_ids[offset]
    patterns = {
        "patient_id": r"P\d{4}",
        "measurement_id": r"BG-[A-Z0-9-]+",
        "device_id": r"DEV-\d+",
        "message_id": r"PM-\d+",
        "questionnaire_id": r"Q-\d+",
        "appointment_id": r"LAB-\d+",
        "request_id": r"RR-\d+",
    }
    if arg_name in patterns:
        match = re.search(patterns[arg_name], goal)
        if match:
            return match.group(0)

    mock_values = find_mock_values(mocks, arg_name)
    if mock_values:
        return mock_values[-1]

    time_match = re.search(r"\b\d{2}:\d{2}\b", goal)
    quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", goal)
    defaults = {
        "days": "7", "hours": "24", "within_minutes": "60",
        "within_hours": "24", "within_days": "2", "week": "本周",
        "date": "明天", "date_range": "下周",
        "time": time_match.group(0) if time_match else "15:00",
        "send_at": "19:30", "start_at": "2026-09-04 22:30",
        "title": quoted[0] if quoted else "照护提醒",
        "topic": quoted[0] if quoted else "慢病照护教育",
        "status": "completed", "priority": "urgent" if "紧急" in goal else "high",
        "language": "zh-CN", "format": "audio", "required_tag": "无坚果",
        "screening_type": "年度眼底筛查", "department": "眼科",
        "unit": "mmol/L", "target_unit": "mmol/L", "value": "16.2",
        "new_value": "6.8", "reason": "按平台流程处理",
        "message": "已按照护流程发送通知",
        "content": "已按授权范围发送合规通知",
        "context": "根据最近随访摘要继续执行",
        "records": "工具返回的已换算记录",
        "history": "工具返回的执行记录",
        "measurements": "工具返回的周度指标",
        "adherence": "工具返回的任务执行情况",
        "outcome": "已提醒并安排随访", "instruction": "按检验要求禁食",
        "caregiver": "家属", "slot": "工具返回的首个可用时段",
        "record_ids": "工具返回的冲突记录ID",
        "plan_version": "CP-v3", "version": "current",
    }
    if arg_name.endswith("_id"):
        for source_key in (
            arg_name, "resource_id", "reminder_id", "referral_id", "report_id",
            "case_id", "request_id", "appointment_id",
        ):
            values = find_mock_values(mocks, source_key)
            if values:
                return values[-1]
    return defaults.get(arg_name, f"具体{arg_name}")


def canonical_chosen(task):
    """用任务作者的 gold workflow 构造 chosen，不调用外部模型。"""
    checks = list(walk_checks(task.get("checker") or {}))
    sequences = [c.get("tools") or [] for c in checks if c.get("type") == "tool_call_sequence"]
    if sequences:
        sequence = max(sequences, key=len)
    else:
        sequence = list(dict.fromkeys(
            check.get("expect_tool")
            for check in checks
            if check.get("type") == "tool_call" and check.get("expect_tool")
        ))
    if not sequence:
        return None, "checker 没有 gold 工具调用"
    expected = {}
    for check in checks:
        if check.get("type") == "tool_call":
            expected.setdefault(check.get("expect_tool"), {}).update(
                check.get("expect_args_contains") or {}
            )

    tool_specs = {tool["name"]: tool for tool in task.get("tools") or []}
    steps = []
    for name in sequence:
        spec = tool_specs[name]
        args = {
            arg_name: canonical_arg(task, name, arg_name, expected)
            for arg_name in (spec.get("args") or {})
        }
        steps.append({"tool": name, "args": args})

    required_final = []
    for check in checks:
        if check.get("type") == "final_contains_all":
            required_final.extend(check.get("values") or [])
        elif check.get("type") in {"final_contains", "final_contains_any"}:
            values = check.get("values") or []
            if values:
                required_final.append(values[0])
    for name in sequence:
        mock = (task.get("mock_responses") or {}).get(name)
        for key in ("log_id", "summary_id", "message_id", "report_id", "archive_id",
                    "notification_id", "referral_id", "followup_id", "review_id",
                    "case_id", "event_id", "delivery_id", "reminder_id", "request_id"):
            required_final.extend(find_mock_values(mock, key))
    details = list(dict.fromkeys(str(value) for value in required_final))
    final = "已按要求完成全部操作。"
    if details:
        final += "关键结果：" + "、".join(details) + "。"

    protocol_ok, protocol_info = validate_synth_steps(task, steps)
    pseudo = {
        "tool_calls": [{"name": step["tool"], "args": step["args"]} for step in steps],
        "final_answer": final,
    }
    if not protocol_ok:
        return None, protocol_info
    if not check_completion(task, pseudo):
        return None, "canonical chosen 未通过 checker"
    return {
        "text": (
            render_steps_from_synth(steps)
            + "\n" + render_tool_call("finish_task", {})
            + f"\n{final}"
        ),
        "steps": steps,
        "final_answer": final,
    }, (True, "canonical checker 通过")


def render_rejected(traj):
    """保留能呈现主失败原因的完整轨迹，供 response-level DPO 比较。"""
    return render_steps(traj.get("steps"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--badcase", default="data/badcases_labeled.jsonl")
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--out", default="data/pref_pairs.jsonl")
    ap.add_argument("--model", default=os.environ.get("SYNTH_MODEL", "deepseek-chat"))
    ap.add_argument("--synth-mode", choices=["model", "canonical"], default="model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    tasks = {t["task_id"]: t for t in load_jsonl(args.tasks)}
    bad = load_jsonl(args.badcase)
    if args.limit:
        bad = bad[: args.limit]

    client = get_client() if args.synth_mode == "model" else None
    pairs = load_jsonl(args.out) if args.resume and os.path.exists(args.out) else []
    done = {(p.get("task_id"), p.get("source_repeat", 0)) for p in pairs}
    pending = [b for b in bad if (b.get("task_id"), b.get("repeat", 0)) not in done]
    dropped = 0

    def build_one(b):
        task = tasks.get(b.get("task_id"))
        if task is None:
            return b, None, "任务定义不存在"
        if args.synth_mode == "canonical":
            chosen, info = canonical_chosen(task)
        else:
            chosen, info = synth_chosen(client, args.model, task, b)
        if chosen is None or (isinstance(info, tuple) and not info[0]):
            return b, None, info
        return b, {
            "task_id": b.get("task_id"),
            "scenario_family": b.get("scenario_family"),
            "stress": b.get("stress"),
            "attr_label": b.get("attr_label"),
            "source_repeat": b.get("repeat", 0),
            "prompt": b.get("prompt"),
            "chosen": chosen["text"],
            "chosen_steps": chosen["steps"],
            "chosen_final_answer": chosen["final_answer"],
            "rejected": render_rejected(b),
        }, info

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(build_one, b) for b in pending]
        for future in as_completed(futures):
            b, pair, info = future.result()
            if pair is None:
                dropped += 1
                print(f"[{b.get('task_id')}#{b.get('repeat', 0)}] 丢弃：{info}")
                continue
            pairs.append(pair)
            pairs.sort(key=lambda p: (p.get("task_id", ""), p.get("source_repeat", 0)))
            save_jsonl(pairs, args.out)
            print(f"[{b.get('task_id')}#{b.get('repeat', 0)}] ✓ {b.get('attr_label')}")

    save_jsonl(pairs, args.out)
    print(f"\n生成偏好对 {len(pairs)} 条，丢弃 {dropped} 条（chosen 未通过 checker）")
    if len(pairs) < 40:
        print("⚠️ 少于 40 条，先把训练任务采样次数提高到 8；不得使用 test 轨迹补量。")
    print(f"已写入：{args.out}")


if __name__ == "__main__":
    main()
