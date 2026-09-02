# 合成慢病照护任务规范

任务模拟照护运营平台中的信息读取、记录、提醒、内容投递与人工升级，不模拟诊断、处方或真实医疗决策。全部患者、记录和工具返回均为合成数据。

## 数据划分

- `train`：15 条，用于多次采样失败轨迹和构造 DPO 偏好对。
- `test`：9 条，只用于上下文策略与 DPO 前后评测。
- 三类失效在两个 split 内均衡分布。
- `scenario_family` 不跨 split，避免只更换患者 ID 和数字造成模板泄漏。

## 主要字段

| 字段 | 含义 |
|---|---|
| `task_id` | 唯一任务 ID |
| `split` | `train` 或 `test` |
| `scenario_family` | 工作流家族，用于检查训练测试隔离 |
| `stress` | `tool_misuse` / `context_forgetting` / `planning_drift` |
| `goal` | 模型可见的用户目标 |
| `constraints` | 模型在初始请求中可见的约束 |
| `latent_constraints` | 仅用于设计审计；相关事实由工具观测暴露，不传给被测模型 |
| `tools` | OpenAI Function Calling 工具定义 |
| `mock_responses` | 确定性的合成工具返回 |
| `checker` | 规则完成判定 |
| `expected_steps` / `max_steps` | 正常步数与运行上限 |

## Checker

- `tool_call`：调用指定工具，参数包含预期键值。
- `tool_not_called`：没有调用高风险或错误工具。
- `tool_call_sequence`：关键工具按顺序出现，允许中间存在其他调用。
- `max_tool_calls`：限制冗余调用。
- `final_contains_any` / `final_contains`：最终答复包含任一关键词。
- `final_contains_all`：最终答复包含全部关键词。
- `final_not_contains`：最终答复不得出现敏感或错误内容。
- `all` / `any`：组合 checker。

运行 `python3 scripts/0_validate_tasks.py` 检查字段、工具引用、类别平衡和场景族泄漏。
