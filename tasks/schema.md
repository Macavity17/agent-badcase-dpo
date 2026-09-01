# 任务集设计说明

自建 30–50 个轻量 tool-use 任务（当前 `tasks.jsonl` 有 12 个样例，照格式扩到 30–50）。

## 设计原则：场景贴岗位，难度可校准

**场景全部选办公 / 差旅 / 数据 / 金融类**——正对目标岗位描述的用户群（"办公、金融、法律、医疗等场景中的专业工作者"）。这不是凑巧，是让面试官一眼看出我理解他们的产品场景。

**每类失败模式定向设计**，对应岗位 JD 的三大课题：

| 前缀 | 失败模式 | 对应 JD 课题 | 设计手法 |
|---|---|---|---|
| `tm_` | **tool_misuse** 工具误选 | 工具体系设计、Function Calling 协议 | 放 3–4 个名字/参数相近的工具，只有一个正确 |
| `cf_` | **context_forgetting** 上下文遗忘 | 上下文分层与预算分配、记忆管理 | 任务开头给一个约束（预算/禁忌/格式），中后期必须用到 |
| `pd_` | **planning_drift** 规划发散 | 任务规划与 Subagent 协作、反思收敛 | 3–5 步串联 + 条件分支，诱导模型循环或跑偏 |

## JSON 字段

```jsonc
{
  "task_id": "tm_001",
  "stress": "tool_misuse",                 // 三类之一，用于后续分类别统计
  "goal": "帮我订 9 月 10 日北京到上海最早的航班",
  "constraints": ["只要上午起飞的"],         // 早期约束，cf_ 类必填
  "tools": [
    {"name": "search_flight", "desc": "查询航班", "args": {"from": "出发城市", "to": "到达城市", "date": "日期"}},
    {"name": "search_train",  "desc": "查询火车", "args": {...}}
  ],
  "mock_responses": {                      // 工具的假返回，让模型能"跑通"流程
    "search_flight": {"flights": [{"no": "CA1501", "dep": "08:00", "price": 1180}]}
  },
  "checker": {                             // 完成判定（规则，可自动跑）
    "type": "tool_call",
    "expect_tool": "search_flight",
    "expect_args_contains": {"date": "2026-09-10"}
  },
  "expected_steps": 2,                     // 用于判定 planning_drift（超 2 倍即发散）
  "max_steps": 8
}
```

### checker 类型

- `tool_call`：最终调用了 `expect_tool`，且参数包含 `expect_args_contains` 里的键值
- `final_contains`：最终答案文本必须包含 `values` 中任一项（检验约束是否被遵守，如"避开海鲜"）
- `final_not_contains`：最终答案**不得**包含（如给过敏用户推荐了海鲜）
- `all`：`checks` 数组里的多个条件全部满足

## 扩充时的三条纪律

1. **每类至少 12–15 条**，否则分类别统计没意义
2. **mock_responses 要写具体**（航班号、价格、日期），模型才有可能"正确"完成——敷衍的假数据只会让轨迹全是噪声
3. **先跑 10 条校准难度**（见 QUICKSTART 第 6 步），完成率目标 30–50%，再全量扩
