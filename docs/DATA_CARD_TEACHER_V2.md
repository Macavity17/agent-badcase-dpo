# Teacher-v2 偏好数据卡

## 状态

数据已完成构造、逐条人工复核与本地校验，尚未用于训练或评测。任何第二轮效果数字都必须在实际运行后填写，不能从数据质量推断。

## 目标

第一轮将一整条多工具轨迹序列化为单个 assistant response，但运行时要求每轮只调用一个工具、接收 observation 后再决定下一步。teacher-v2 将训练单元改为“状态 -> 一个下一动作”：

- LLaMA-Factory 输入使用 `human -> function_call -> observation` 多轮角色，并通过 `tools` 列携带与运行时一致的工具 schema；
- chosen 是首次有意义分歧点的一个正确工具调用，或 `finish_task` 后的最终答复；
- rejected 优先使用第一轮真实失败轨迹在同一决策点的错误动作；
- 额外构造 outcome-closure hard negative，训练最终答复复用工具返回 ID，而不是只说“已完成”。

## 来源与作者边界

- 源 badcase：第一轮 Qwen2.5-1.5B-Instruct 在 15 个 train 任务上的 68 条真实失败轨迹。
- teacher 规范：由 OpenAI Codex（GPT-5 family，交互式会话）逐任务编写，固化在 `tasks/teacher_v2_specs.jsonl`。
- 2026-09-03 已将 15 条 workflow、45 条最终答复变体、9 条 holdout 和最终 66 条 pair 全量逐条复核；不再仅是抽样检查。
- 这是强模型辅助的合成标注，不是独立人类专家标注，也不是公司生产数据或 teacher API 批量蒸馏。
- 可复现对象是已提交的 teacher 规范、人工审阅清单、编译器、源 badcase 哈希和最终数据哈希。

全部患者、工具、观察和结果均为合成数据。

## 规模与切分

| 项目 | 数量 |
|---|---:|
| 源 badcase | 68 |
| 编译候选 pair | 98 |
| 人工规则排除的候选行 | 22 |
| 审阅与去重后的唯一 pair | 66 |
| 真实 badcase 首次分歧 | 36 |
| teacher outcome-closure hard negative | 30 |
| 工具动作 chosen | 27 |
| 最终答复 chosen | 39 |
| 任务级 train / eval | 52 / 14 |

按任务 stress 统计：`tool_misuse=20`、`context_forgetting=24`、`planning_drift=22`。人工审阅排除近义文案、等价序列化、未声明时间窗口、未定义枚举等不能证明 rejected 更差的偏好，原始候选和理由仍保留在 audit 文件。

`experiments/round2/split.json` 固定 3 个 eval task，每类 stress 各 1 个。同一 `task_id` 不会同时出现在 train/eval，避免同任务不同 repeat 随机泄漏导致 reward accuracy 虚高。

## 质量规则

1. 所有 15 个 train 任务都有 teacher workflow 和 3 条最终答复变体。
2. teacher workflow 的工具名与参数键必须严格匹配任务 schema。
3. 完整 teacher workflow 加任一最终答复必须通过任务 checker。
4. ID 必须来自初始任务或此前工具 observation；未出现的 ID 会使构造失败。
5. 只有能归因到工具选择、工作流闭环、明确值约束或观察丢失的差异才进入训练。
6. 空 chosen/rejected、相同 pair、占位文本和不完整 grounding 均不得进入训练集。
7. 所有候选完整保存在 audit 文件，审阅决定固化在 `experiments/round2/pair_review.jsonl`。

## 第二份盲测集

`tasks/holdout_v2.jsonl` 包含 9 个新任务，三类 stress 各 3 条。患者 ID 为 `P9601-P9803`，9 个 `scenario_family` 均不与 train、dev 或第一份 holdout 重合。

除字段、checker 引用、平衡与隔离外，每个 holdout 还生成了一条无占位符的 canonical trace 并通过 checker。但它从未用 base 或 DPO 模型运行，也未根据模型表现修改。它是模型未见的作者合成评测集，不是外部公开 benchmark。

## 预注册评测协议

第二轮按三个层级判断，低层指标不能替代高层结果：

| 层级 | 数据与隔离 | 指标 | 能回答的问题 |
|---|---|---|---|
| 训练偏好 | 14 条 task-grouped eval pair，来自 3 个未进入 train 的任务 | reward accuracy、chosen/rejected reward、reward margin | 模型是否学会区分本数据集的 chosen/rejected |
| 状态决策 | 同一批 eval pair 的运行时消息、工具 schema；base/DPO pair ID 对齐 | 自由生成 next-action accuracy、工具名、精确参数、final checker、grounding ID | 偏好分离是否迁移为未见任务上的局部动作选择 |
| 端到端 | 9 个全新 `holdout_v2` 任务，每任务 3 个对齐 seed | 规则完成率（主指标）、分类别完成率、协议错误、工具调用与轨迹审计 | 局部变化是否真正改善完整 Agent 工作流 |

状态决策评测必须使用 `tool_choice=auto`；如果评测器按 gold 强制 tool/none，就泄漏了正确动作类型，只能测试参数填充。工具参数默认精确匹配 chosen，目的是捕获第一轮出现的枚举、数值、时间和 ID 语义改写；可能合理的自由文本改写进入人工误差审阅，不悄悄算作正确。API/service error 从准确率分母中排除并单独计数。

14 条 eval pair 只有 3 个独立 `task_id`，且可能共享相同状态或 chosen 变体。因此报告必须同时给出 pair 数和独立任务数，不能把 pair 行数当样本独立性。状态层是定位指标，最终产品结论仍以未触碰的 9-task holdout 完成率为准。

预注册解释规则：

- reward 指标改善、状态动作不改善：偏好 shortcut、记忆数据或小样本过拟合更可能，不能声称 Agent 行为改善；
- 状态动作改善、端到端不改善：局部决策学习未沿长轨迹传播，需要检查状态分布偏移、错误累积和后续恢复；
- 端到端完成率改善且无协议、安全或分类别明显退化：才构成 DPO 有用效果的探索性证据；
- 任何 stress 或关键能力退化都单独报告，不用总平均掩盖；API 错误也不当模型失败混入；
- 9 个独立 holdout 任务和单 seed 不支持显著性或广泛外推。

## 轮次隔离

第一轮路径保持不变；第二轮的派生数据、日志、模型和结果分别只写入 `data/round2/`、`runs/round2/`、`outputs/round2/` 和 `results/round2/`。两轮完整路径清单见 `experiments/round1/README.md` 和 `experiments/round2/README.md`。

## 构造与导出

```bash
python3 scripts/6_build_teacher_v2.py \
  --badcase data/train_badcases_labeled.jsonl \
  --tasks tasks/tasks.jsonl \
  --specs tasks/teacher_v2_specs.jsonl \
  --review experiments/round2/pair_review.jsonl \
  --split experiments/round2/split.json \
  --out data/round2/pref_pairs.jsonl

python3 scripts/4_to_llamafactory.py \
  --pref data/round2/pref_pairs.jsonl \
  --outdir data/round2/lf_data \
  --train-config config/dpo_teacher_v2.yaml

python3 scripts/0_validate_tasks.py --tasks tasks/holdout_v2.jsonl
```

## 当前文件哈希

```text
source badcases                         3f90aff256e085ee418a9d2af0accd625cf3d107580632d8bc373d25fc148c7d
data/round2/pref_pairs.jsonl            f25a9f90acb9c689bdd0a016d7a8766e197414e4927cd6c9a7d0540b824055f1
data/round2/pref_pairs.audit.jsonl      2a7ee3151f896c57f8250d17601e5708830684204c0092b1f154be753de887cf
LLaMA-Factory train                     c5ef211d5cde49c08783774640b55b982c63c153d288f03c763239ff11d0ac33
LLaMA-Factory eval                      7cc74cf0458c0722f54d8af65048317cea1b2663cbc983c1aaf03f1ff850af34
teacher_v2_specs.jsonl                  9fc9f65175e7fadcf96b0be63cc1ad226a1741d8055a44ba8227b0a045c59a58
holdout_v2.jsonl                        78d6cc68b8954c6bc9c3ded1daf6a71a129ee3daf8397bfa48bcc963b21769b9
pair_review.jsonl                       862b459fe5874fd57cf4b09178f0ac0f2c8a34fe9614bf0233d2739f7143f827
split.json                              1fb2d39c939f3b5b437f37fac51c4a0f47406187a4568a349ed1109558680cc7
```

`data/` 被 Git 忽略，训练数据和 audit 文件需随实验 evidence archive 备份；teacher 规范、审阅清单、切分、holdout 和本数据卡进入 Git。

## 已知局限

- 66 条仍是小规模、单领域合成偏好数据。
- 30 条 closure rejected 是 teacher 构造的通用弱答复，不是线上采样。
- 不同任务的 pair 数不均衡，eval 仅 3 个独立任务；reward accuracy 只是训练诊断，不是效果结论。
- 状态决策仍来自合成 train-domain 任务的 task-isolated eval，能诊断迁移但不是独立公开 benchmark，也不能替代完整轨迹。
- 多轮 ShareGPT 角色与工具 schema 已对齐运行时语义，但 LLaMA-Factory/Qwen 模板与 vLLM/OpenAI 协议的底层 token 仍不保证逐 token 完全同构。
- 只有实际训练并在固定 holdout-v2 上比较后，才能判断它是否修复第一轮退化。
