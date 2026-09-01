"""公共工具：数据读写、工具 schema、mock 执行、完成判定。

设计要点：
- 工具执行全部走 mock（不接真实 API），保证实验可复现、零外部依赖成本；
- checker 用规则判定，可自动化跑大批量，不依赖主观判断；
- 单独抽出来是为了让脚本 1 和 5 共用同一套判定口径（前后评测必须口径一致）。
"""

import json
import os
import sys


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
                },
            },
        })
    return out


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

    if ctype == "final_contains":
        return any(v in final for v in (check.get("values") or []))

    if ctype == "final_not_contains":
        return not any(v in final for v in (check.get("values") or []))

    if ctype == "all":
        return all(_check_one(c, traj) for c in (check.get("checks") or []))

    return False


def check_completion(task, traj):
    """判定一条轨迹是否完成任务（前后评测共用此口径）。"""
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
