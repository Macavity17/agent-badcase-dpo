"""公共工具：数据读写、工具 schema、mock 执行、完成判定。

设计要点：
- 工具执行全部走 mock（不接真实 API），保证实验可复现、零外部依赖成本；
- checker 用规则判定，可自动化跑大批量，不依赖主观判断；
- 单独抽出来是为了让脚本 1 和 5 共用同一套判定口径（前后评测必须口径一致）。
"""

import json
import os
import sys


AGENT_SYSTEM = (
    "你是一个严谨的慢病照护运营助手，可以调用平台工具完成信息读取、记录、提醒和人工升级。\n"
    "要求：\n"
    "1. 这是串行 ReAct 执行：每次响应只调用一个必要工具，参数必须严格符合 schema；\n"
    "2. 保留任务和工具观测中的约束、授权、单位、时区与最新计划状态；\n"
    "3. 不得诊断、开药或自行修改处方；遇到超出权限或紧急情况必须升级人工；\n"
    "4. 必须完整执行用户目标中明示的工作流；当目标还要求记录、发送、预约或升级时，读取和检索信息不等于完成；\n"
    "5. 收到工具结果后，若目标未完成，下一次响应必须继续调用工具。不得用‘接下来’、‘将要’、‘请稍等’等自然语言代替工具调用，也不得把 <tool_call> 标记当普通文本输出；\n"
    "6. 参数必须复用任务或工具结果中的具体值，不得填写‘某ID’、‘转换后的记录’、‘当前单位’、‘高值’等占位词；\n"
    "7. 只有所有必要动作都完成后才调用 finish_task；不要重复调用同一个工具；\n"
    "8. finish_task 成功后，用自然语言报告关键结果和工具返回的 ID。"
)

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_task",
        "description": "仅当用户目标中的所有必要动作已完成时调用，用于进入最终答复阶段。",
        "parameters": {
            "type": "object", "properties": {},
            "required": [], "additionalProperties": False,
        },
    },
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows, path):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_text(text, path):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def build_tools_payload(tools):
    """转成 OpenAI function calling 的 tools 格式。"""
    out = []
    for t in tools:
        props = {k: {"type": "string", "description": v} for k, v in (t.get("args") or {}).items()}
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("desc", ""),
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(props.keys()),
                    "additionalProperties": False,
                },
            },
        })
    return out


def build_user_message(task):
    """按运行时口径组装初始用户消息。"""
    message = task["goal"]
    constraints = task.get("constraints") or []
    if constraints:
        message += "\n\n约束条件（必须遵守）：\n" + "\n".join(
            f"- {constraint}" for constraint in constraints
        )
    return message


def tool_names(task):
    return {t["name"] for t in task.get("tools", [])}


def mock_tool_call(tool_name, args, task):
    """假工具：优先用任务预设的 mock_responses，否则返回通用成功响应。

    mock 数据质量直接决定实验成败 —— 假数据太敷衍，模型只能产出噪声轨迹。
    """
    mocks = task.get("mock_responses") or {}
    if tool_name in mocks:
        return json.dumps(mocks[tool_name], ensure_ascii=False)
    return f"[mock] {tool_name} executed. args={args}. Result: 2 records found (sample data)."


def _check_one(check, traj):
    calls = traj.get("tool_calls") or []
    final = traj.get("final_answer") or ""
    ctype = check.get("type")

    if ctype == "tool_call":
        expect = check.get("expect_tool")
        contains = check.get("expect_args_contains") or {}
        for c in calls:
            if c.get("name") == expect:
                args = c.get("args") or {}
                if all(str(args.get(k, "")).find(str(v)) >= 0 for k, v in contains.items()):
                    return True
        return False

    if ctype == "tool_arg_not_contains":
        expect = check.get("expect_tool")
        arg_name = check.get("arg")
        forbidden = [str(value) for value in (check.get("values") or [])]
        matching = [call for call in calls if call.get("name") == expect]
        if not matching:
            return False
        return all(
            not any(value in str((call.get("args") or {}).get(arg_name, "")) for value in forbidden)
            for call in matching
        )

    if ctype in {"final_contains", "final_contains_any"}:
        return any(v in final for v in (check.get("values") or []))

    if ctype == "final_contains_all":
        return all(v in final for v in (check.get("values") or []))

    if ctype == "final_not_contains":
        return not any(v in final for v in (check.get("values") or []))

    if ctype == "tool_not_called":
        forbidden = set(check.get("tools") or [])
        return all(c.get("name") not in forbidden for c in calls)

    if ctype == "tool_call_sequence":
        expected = check.get("tools") or []
        actual = [c.get("name") for c in calls]
        pos = 0
        for name in actual:
            if pos < len(expected) and name == expected[pos]:
                pos += 1
        return pos == len(expected)

    if ctype == "max_tool_calls":
        return len(calls) <= int(check.get("value", 0))

    if ctype == "all":
        return all(_check_one(c, traj) for c in (check.get("checks") or []))

    if ctype == "any":
        return any(_check_one(c, traj) for c in (check.get("checks") or []))

    return False


def check_completion(task, traj):
    """判定一条轨迹是否完成任务（前后评测共用此口径）。"""
    placeholder_tokens = ("具体", "某ID", "工具返回的", "待填写")
    for call in traj.get("tool_calls") or []:
        rendered_args = json.dumps(call.get("args") or {}, ensure_ascii=False)
        if any(token in rendered_args for token in placeholder_tokens):
            return False
    chk = task.get("checker")
    if not chk:
        return None
    return _check_one(chk, traj)


def format_trajectory(traj):
    """把轨迹序列化成文本，供 DPO 的 chosen/rejected 与 judge 评分使用。"""
    lines = []
    for s in traj.get("steps") or []:
        if s.get("tool"):
            lines.append(f"[{s.get('step')}] CALL {s['tool']}({json.dumps(s.get('args') or {}, ensure_ascii=False)})")
            lines.append(f"      OBS  {str(s.get('observation'))[:300]}")
    lines.append(f"FINAL: {traj.get('final_answer') or '(未给出最终答案)'}")
    return "\n".join(lines)


def get_client(base_url=None, api_key=None):
    """统一的 OpenAI 兼容客户端（vLLM 本地服务 / 强模型 API 都走它）。"""
    from openai import OpenAI
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=api_key)
