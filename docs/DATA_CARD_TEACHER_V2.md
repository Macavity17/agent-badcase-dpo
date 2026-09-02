# Teacher-v2 偏好数据卡

## 状态

数据已完成构造与本地校验，尚未用于训练或评测。任何第二轮效果数字都必须在实际运行后填写，不能从数据质量推断。

## 目标

第一轮将一整条多工具轨迹序列化为单个 assistant response，而线上协议要求每轮只调用一个工具、接收 observation 后再决定下一步。teacher-v2 将训练单元改为“状态 -> 一个下一动作”：

- prompt 包含任务、工具定义、正确历史前缀和确定性工具 observation；
- chosen 是首次分歧点的一个正确工具调用，或 `finish_task` 后的最终答复；
- rejected 是真实失败轨迹在同一决策点的错误动作；
- 额外构造 outcome-closure hard negative，训练最终答复复用结果 ID，而不是只说“已完成”。

## 来源与真实性边界

- 源 badcase：第一轮 Qwen2.5-1.5B-Instruct 在 15 个 train 任务上的 68 条真实失败轨迹。
- teacher 规范：由 OpenAI Codex（GPT-5 family，交互式会话）辅助逐任务编写，固化在 `tasks/teacher_v2_specs.jsonl`。
- 编译与过滤：`scripts/6_build_teacher_v2.py` 确定性执行。
- 数据不属于人工专家标注，也不是公司生产数据；不得宣传为医生标注或独立 teacher API 蒸馏。
- 交互式模型调用本身无法按 API 参数完全复演；可复现对象是已经提交的 teacher 规范、编译器、源 badcase 哈希和最终数据哈希。

全部患者、工具、观察和结果均为合成数据。

## 规模

| 项目 | 数量 |
|---|---:|
| 源 badcase | 68 |
| 编译候选 pair | 98 |
| 完全重复候选 | 24 |
| 最终唯一训练 pair | 74 |
| 真实 badcase 首次分歧 | 44 |
| teacher outcome-closure hard negative | 30 |
| 工具动作 chosen | 38 |
| 最终答复 chosen | 36 |

按任务 stress 统计：`tool_misuse=17`、`context_forgetting=29`、`planning_drift=28`。按原 badcase 归因统计的唯一 pair 为 `tool_misuse=5`、`context_forgetting=19`、`planning_drift=20`，另有 30 条 `workflow_closure`。

## 质量规则

1. 所有 15 个 train 任务都有 teacher workflow 和 3 条最终答复变体。
2. teacher workflow 的工具名与参数键必须严格匹配任务 schema。
3. 完整 teacher workflow 加任一最终答复必须通过任务 checker。
4. ID 必须来自初始任务或此前工具 observation；未出现的 ID 会使构造失败。
5. `reason`、`message`、`outcome`、`instruction` 等自由文本允许合理措辞差异，避免学习无意义的文案偏好。
6. ID、枚举、时间、数值、单位和结构化 observation 值保持严格比较。
7. 空 chosen/rejected、相同 pair、占位文本和不完整 grounding 均不得进入训练集。
8. 98 条候选完整保存在 audit 文件，训练集对相同 prompt/chosen/rejected 去重为 74 条。

## 第二份盲测集

`tasks/holdout_v2.jsonl` 包含 9 个新任务，三类 stress 各 3 条。患者 ID 为 `P9601-P9803`，9 个 `scenario_family` 均不与 train、dev 或第一份 holdout 重合。

第二份 holdout 只完成字段、checker 引用、类别平衡、ID/family 隔离等结构校验；没有用 base 或 DPO 模型运行，也没有根据模型表现修改。第二轮训练前应先固定其 SHA-256，之后 base 与 DPO 使用完全相同的 seed 运行。由于任务由项目作者构造，它是模型未见的评测集，不是外部公开 benchmark。

## 构造与导出

```bash
python3 scripts/6_build_teacher_v2.py \
  --badcase data/train_badcases_labeled.jsonl \
  --tasks tasks/tasks.jsonl \
  --specs tasks/teacher_v2_specs.jsonl \
  --out data/pref_pairs_teacher_v2.jsonl

python3 scripts/4_to_llamafactory.py \
  --pref data/pref_pairs_teacher_v2.jsonl \
  --outdir data/lf_data_teacher_v2 \
  --train-config config/dpo_teacher_v2.yaml

python3 scripts/0_validate_tasks.py --tasks tasks/holdout_v2.jsonl
```

## 当前文件哈希

```text
source badcases                         3f90aff256e085ee418a9d2af0accd625cf3d107580632d8bc373d25fc148c7d
pref_pairs_teacher_v2.jsonl             8407d7d2211b41608be53b142f9c6ceea32f270294c35299c250eaeca007d6bf
pref_pairs_teacher_v2.audit.jsonl       8b1ca977f97c2836e000f26390487d9f14250f115f7af1d632183c728ad0138f
LLaMA-Factory agent_pref.json           4db6cb98713e2776b575f09e3e768d3b1002cb4a3b464cec82e14222b31dc638
teacher_v2_specs.jsonl                  475a96c6bf7b473238ddd7ba825410e419b8e1d2e93f6895a02b4ede708e623e
holdout_v2.jsonl                        f2f251e43df49d0a7f999513b31b9078a2f48372aa8564d7236e817467b7eee2
```

`data/` 被 Git 忽略，训练数据和 audit 文件需随实验 evidence archive 备份；teacher 规范、编译器、holdout 和本数据卡进入 Git。

## 已知局限

- 74 条仍是小规模、单领域合成偏好数据。
- 30 条 closure rejected 是 teacher 构造的通用弱答复，不是线上采样。
- 不同任务的唯一 pair 数量不完全均衡，保留了原 badcase 分布。
- state-action prompt 是显式文本化的工具历史，仍不是运行时 OpenAI 消息数组的完全同构表示。
- 只有实际训练并在固定 holdout-v2 上比较后，才能判断它是否修复第一轮退化。
