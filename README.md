# Care Agent Failure Lab

> 一项关于长程 Agent 失效、上下文组织与 LoRA-DPO 干预边界的独立受控实验

## 摘要

长程 Agent 的失败往往发生在模型之外：早期约束没有进入当前决策、相似工具被混淆、工具返回值没有在后续步骤中被准确复用，或规划循环在看似合法的 Function Calling 中逐步偏离目标。这些问题有时被简化为“模型不够强”，但不同根因应当对应不同干预。

本项目以合成慢病照护运营工作流为受控环境，在 Qwen2.5-1.5B-Instruct 上研究三类定向失效：`tool_misuse`、`context_forgetting` 和 `planning_drift`。实验先比较 `full`、`window` 和 `layered` 三种上下文策略，再从训练任务的真实失败轨迹出发，完成 badcase 归因、偏好数据构造、LoRA-DPO 训练和独立 holdout 评测。第一轮 DPO 退化后，项目进一步修复训练/运行协议错位、弱偏好对和任务级泄漏，并以新盲测集完成第二轮评测。

两轮都没有得到可宣称的效果提升。上下文实验中，`window` 节省 13.5% 的 prompt 峰值，但完成率从 14.8% 降为 0；`layered` 仅节省 2.4%，完成率为 11.1%，没有超过 `full`。第一轮 DPO 的训练内 eval reward accuracy 达到 100%，但冻结 holdout 从 4/27 降至 2/27。第二轮在任务隔离 eval 上达到 92.9% reward accuracy，但 Base/DPO 的自由 next-action accuracy 均为 14.3%，新 `holdout_v2` 上均为 0/27。

结果支持一个有限但可验证的结论：**小规模偏好训练可以学会区分当前数据中的 chosen/rejected，却不必然学会在自由生成和长轨迹中稳定复用相同决策规则。** 因此，reward、局部状态动作和端到端任务完成必须分层评测；Function Calling 格式正确也不能代替参数语义、观察引用和工作流闭环。

**关键词：** Agent evaluation；context management；tool use；badcase attribution；DPO；LoRA；long-horizon workflow

## 1. 问题背景

本项目受既往慢病照护 Agent 实践启发。在相关实践中，我参与过模型选型、event-memory、badcase 数据分析和自动化评测。一次未完成的照护任务，表面上都可以概括为“Agent 做错了”，但底层介入点并不相同：

- 模型没有在当前上下文中看到过敏史、授权范围或时区，更像上下文组织问题。
- 模型看到了信息，却选错记录/更正、通知/升级工具，更像工具选择偏好问题。
- 每一步都符合 schema，但模型重复查询、提前结束或漏掉最终回传，则是规划和闭环问题。

如果不先区分根因，就容易把所有 badcase 都变成 prompt 修改，或让一次微调同时承担上下文、记忆、工具语义和规划问题。本项目建立了一条可审计路径：

```text
长程失败轨迹
  -> 失效模式归因
  -> 选择干预层（上下文 / 后训练）
  -> 构造受控数据与反例
  -> 分层评测与轨迹审计
  -> 确认效果边界或推翻原假设
```

### 1.1 真实性边界

本项目不是九安医疗或腾讯的内部项目，DPO 训练也没有在实习或生产系统中部署。仓库不包含公司代码、真实患者数据、内部提示词或内部指标。所有患者、工具、照护计划、观察和执行结果均为合成数据。

准确表述是：**问题受既往慢病照护 Agent 实践启发，实验是离岗后独立完成的合成受控 pilot。**

## 2. 研究问题与假设

**RQ1：上下文策略能否在有限窗口中同时降低 token 成本和保持任务完成率？**

- H1a：朴素滑动窗口可以降低 prompt 峰值，但会因丢失早期事实而伤害完成率。
- H1b：将目标、约束和较早事件压缩为结构化状态，同时保留最新原始轮次，应比朴素窗口更好地保留任务能力。

**RQ2：对训练任务的失败轨迹进行 LoRA-DPO，能否改善未见任务的端到端完成率？**

- H2：如果 chosen/rejected 真正捕捉工具选择、约束复用和工作流闭环差异，DPO 应在同条件 holdout 上改善完成率，且不以工具调用抑制或协议错误换取表面提升。

**RQ3：训练偏好分离能否预测 Agent 的真实行为改善？**

- H3：reward accuracy 或 margin 提升不足以证明 Agent 改善；还需检查自由下一动作，以及局部决策是否传播为完整工作流成功。

## 3. 实验系统

### 3.1 应用环境

受控环境模拟慢病照护运营平台中的完整工作流。Agent 可以读取记录、修正错误事件、建立提醒、投递教育内容、协调随访、通知家属或升级人工。Agent 无权诊断、开药或自行修改处方；紧急情况和权限外决策必须升级人工。

每个任务由用户目标、工具 JSON schema、确定性 mock observation 和显式 checker 构成。Checker 验证工具顺序、必要/禁止调用、参数来源、调用次数和最终答复内容。因此，“任务完成”意味着一组工作流条件同时成立，而不是模型仅说出“已完成”。

### 3.2 失效模式

| 失效模式 | 操作性定义 | 典型风险 | train / test |
|---|---|---|---:|
| `tool_misuse` | 已知当前状态，但选择错误工具或顺序 | 记录与更正混淆，普通消息代替紧急升级 | 5 / 3 |
| `context_forgetting` | 关键事实出现在早期 observation，后续未正确使用 | 忘记过敏、授权、单位、时区或最新计划 | 5 / 3 |
| `planning_drift` | 工具调用合法，但任务路径重复、遗漏或越权 | 重复查询、漏掉闭环、越权高风险动作 | 5 / 3 |

主任务集共 24 条，三类失效各 8 条。`context_forgetting` 的关键约束由早期工具 observation 暴露，不直接泄漏在用户请求中。

### 3.3 模型与运行环境

| 项目 | 设置 |
|---|---|
| 基座模型 | Qwen2.5-1.5B-Instruct |
| 推理服务 | vLLM 0.28.0，OpenAI-compatible API，Hermes tool parser |
| 最大上下文 | 8,192 tokens |
| 后训练 | LLaMA-Factory 0.9.6.dev0，LoRA-DPO |
| 训练硬件 | AutoDL，NVIDIA GeForce RTX 4090 D 24 GB |
| 环境隔离 | Python 3.11 `care-infer` / `care-train` |
| DPO 参数 | rank 16，alpha 32，beta 0.1，3 epochs，bf16 |

运行时要求串行工具调用，每次接收 observation 后再决定下一步，并通过显式 `finish_task` 进入最终答复。

## 4. 数据隔离与评测口径

主任务集使用 15 条 train 和 9 条 test，并以 `scenario_family` 隔离。早期调试过的 9 条 test 任务在被人工检视后归档为 `tasks/dev_tasks.jsonl`；第一轮最终结果使用重新冻结的 9 条 holdout，不再根据模型表现修改 checker。

第二轮另建 `tasks/holdout_v2.jsonl`，包含 9 个新患者/event ID 和 9 个不重叠 `scenario_family`。每个任务在模型评测前都由无占位符 canonical trace 验证 checker 可达，之后才对 Base/DPO 解冻评测。

端到端评测对每个 holdout 任务重复采样 3 次，即每组 27 条轨迹。重复采样可以观察生成波动，但不会把 9 个独立任务变成 27 个独立样本。API/服务错误从模型准确率分母中排除并单独报告。

主指标是基于确定性 checker 的端到端任务完成率。辅助指标包括 prompt token 峰值、平均步骤、工具调用率、协议错误率、精确参数、最终 checker 与 grounding ID。

## 5. 实验一：上下文组织

### 5.1 策略与实现校正

- `full`：保留全部用户、assistant 和 tool 消息。
- `window`：只保留最近 2 轮工具交互。
- `layered`：常驻任务目标与显式约束，将较早动作和 observation 压缩为结构化状态，保留最新原始轮次。

三组共用同一模型、任务、工具、温度和 seed 规则，只改变上下文装配。早期 `layered v1` 同时放入原始请求、包含全部观察的状态块和最近原始轮次，导致信息重复，prompt 峰值反而升至 1,107 tokens。修正后，状态块替代而不是叠加原始历史：旧轮次进入压缩状态，最新轮次保留原文。下表只使用修正后的冻结 holdout 结果。

### 5.2 结果

| 策略 | 完成率 | `tool_misuse` | `context_forgetting` | `planning_drift` | 平均步数 | prompt 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| `full` | 14.8% (4/27) | 0/9 | 1/9 | 3/9 | 5.81 | 1,234 |
| `window` | 0.0% (0/27) | 0/9 | 0/9 | 0/9 | 7.33 | 1,068 |
| `layered` | 11.1% (3/27) | 1/9 | 2/9 | 0/9 | 6.00 | 1,205 |

### 5.3 解读

H1a 得到定向支持：`window` 相对 `full` 节省约 13.5% prompt 峰值，但完成率降为 0，平均步数还从 5.81 上升到 7.33。“历史更少”没有自动带来更短路径，反而可能让模型重复查询或失去闭环信息。

H1b 没有得到支持。`layered` 优于 `window`，但只节省约 2.4% prompt 峰值，完成率仍低于 `full`。分类别上，它在 `tool_misuse` 和 `context_forgetting` 中产生少量成功，却将 `planning_drift` 从 3/9 降为 0/9。当前结果只能说明分层效果依赖失效类型、压缩内容和模型如何解释状态表示。

该轮确定的干预边界是：**上下文管理的目标不能只是 token 更少，还必须证明关键事实可召回、最新 observation 被准确复用，且规划不会因状态重写而提前结束。**

## 6. 实验二：第一轮整轨迹 DPO

### 6.1 Badcase 与偏好数据

在 15 个 train 任务上，基座模型以 `full` 策略每任务采样 6 次，生成 90 条轨迹。其中 22 条通过 checker，68 条失败，服务错误为 0。

| 归因 | 数量 | 占比 |
|---|---:|---:|
| `planning_drift` | 32 | 47.1% |
| `context_forgetting` | 23 | 33.8% |
| `tool_misuse` | 13 | 19.1% |

第一轮为每条失败构造一条“规则约束 canonical chosen”：步骤来自任务 gold workflow，参数来自初始任务或确定性 observation，并同时通过工具协议和任务 checker。68/68 chosen 都包含 `finish_task`，没有使用 test 失败补量。这些 chosen 不是公司数据、独立人类专家标注或强模型批量标注。

### 6.2 训练与结果

| 训练项 | 实际值 |
|---|---:|
| 偏好对 | 68 |
| train / eval | 61 / 7 |
| epochs / steps | 3 / 24 |
| 运行时间 | 44.46 s |
| train loss / eval loss | 0.6239 / 0.5857 |
| eval reward accuracy / margin | 1.000 / 0.217 |

| 指标 | Base | Round-1 DPO | 变化 |
|---|---:|---:|---:|
| 任务完成率 | 14.8% (4/27) | 7.4% (2/27) | -7.4 pp |
| 平均步数 | 5.81 | 5.78 | -0.04 |
| 工具调用率 | 100% | 100% | 0 pp |
| 工具协议错误 | 0% | 0% | 0 pp |
| `tool_misuse` | 0/9 | 0/9 | 0 |
| `context_forgetting` | 1/9 | 1/9 | 0 |
| `planning_drift` | 3/9 | 1/9 | -2 |

在同 seed 的 90 条 train-task 轨迹上，Base 和 DPO 均为 22/90，没有净训练任务增益。

### 6.3 轨迹归因

训练内 reward accuracy 达到 100%，却没有传播为任务成功。对齐轨迹暴露两类关键退化：

1. **精确参数复用被语义改写。** schema 所需的 `2` 被写成“连续两天”，`next_week` 被改成 `next`。表面语义相近，但不再满足值域或下游协议。
2. **闭环和结果回传退化。** 工具调用成功后，最终答复漏掉返回 ID，使轨迹在业务上不可追踪。

也有一个局部改善：DPO 在某条设备轨迹中避免了 Base 的禁止同步调用，但其他闭环条件仍未满足。工具调用率 100%、协议错误 0%，仍然可以在参数语义和结果回传上失败。

## 7. 实验三：teacher-v2 状态动作 DPO

### 7.1 为什么需要第二轮

第一轮存在四个明显混杂因素：

1. 多工具完整轨迹被放入一个 assistant response，与运行时多轮工具循环不同。
2. 部分 chosen/rejected 只是近义措辞或等价序列化。
3. 随机行级 eval 可能把同一任务的不同 repeat 分到 train 和 eval。
4. 只看 reward 和端到端完成率，无法定位偏好向行为迁移在哪一层中断。

第二轮的目的不是为第一轮“翻盘”，而是排除这些可见问题，让“reward 是否迁移到 Agent 行为”可以被单独检验。

### 7.2 数据重构

训练单元改为“当前运行时状态 -> 一个下一动作”：

- 历史使用 LLaMA-Factory 原生 `human -> function_call -> observation` 多轮角色；
- 工具 schema、system prompt、初始用户消息和 `finish_task` 与运行时共用；
- chosen 是首个有意义分歧点的正确工具调用，或 `finish_task` 后的最终答复；
- rejected 优先取自真实 badcase 的同状态错误动作；
- 额外构造 outcome-closure hard negative，要求最终答复复用工具返回 ID。

teacher 规范由强模型辅助编写，并对 15 条 workflow、45 条答复变体、9 条新 holdout 和最终 66 条 pair 逐条复核。这是强模型辅助合成标注，不是独立人类专家标注或公司数据。

| 数据项 | 数量 |
|---|---:|
| 源 badcase | 68 |
| 编译候选 pair | 98 |
| 审阅排除 | 22 |
| 审阅去重后 pair | 66 |
| 真实 badcase 首次分歧 | 36 |
| outcome-closure hard negative | 30 |
| 工具动作 / 最终答复 chosen | 27 / 39 |
| task-grouped train / eval | 52 / 14 |

22 条排除涵盖近义文案、等价记录序列化、未声明时间窗口和未定义枚举。所有候选和排除理由都保留在 audit 中。固定 eval 含 3 个任务，每类 stress 各 1 个；12 个 train task 与 3 个 eval task 零重叠。

### 7.3 预注册三级评测

| 证据层 | 数据与隔离 | 指标 | 能回答的问题 |
|---|---|---|---|
| 训练偏好 | 14 条 task-grouped eval pair，3 个未进入 train 的任务 | reward accuracy、loss、margin | 模型是否区分这批 chosen/rejected |
| 状态决策 | 同一批 eval state，Base/DPO pair ID 对齐 | 自由 next-action、工具名、精确参数、final checker、grounding | 偏好是否迁移为局部决策 |
| 端到端 | 9 个全新 `holdout_v2`，每任务 3 个对齐 seed | 任务完成率、分 stress、协议错误、轨迹审计 | 局部变化是否改善完整工作流 |

状态评测使用 `tool_choice=auto`，不根据 gold 强制工具/最终回答类型，避免泄漏正确动作类型。预注册解释为：reward 改善而状态决策不变，优先考虑偏好 shortcut 或小样本过拟合；状态改善而端到端不变，说明局部学习未沿长轨迹传播；只有 holdout 完成率改善且无安全或分类别退化，才算有用效果。

### 7.4 训练结果

首次 micro-batch 2、`cutoff_len=4096` 在 step 0 发生 CUDA OOM。失败日志、PID 和空输出目录均被保留。最终使用 micro-batch 1、gradient accumulation 8 保持有效 batch 8，并设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

| 训练项 | 实际值 |
|---|---:|
| train / task-grouped eval | 52 / 14 |
| epochs / steps | 3 / 21 |
| 运行时间 | 82.897 s |
| train loss / eval loss | 0.672958 / 0.666803 |
| eval reward accuracy / margin | 0.928571 / 0.054155 |
| best checkpoint | step 20 |

### 7.5 状态决策结果

| 指标 | Base | Round-2 DPO |
|---|---:|---:|
| Next-action accuracy | 14.3% | 14.3% |
| Tool-name accuracy | 50.0% | 50.0% |
| Exact tool-argument accuracy | 0.0% | 0.0% |
| Final task accuracy | 33.3% | 33.3% |
| Final grounding accuracy | 16.7% | 16.7% |

Base/DPO 均评估 14 条 pair row，但只来自 3 个独立 eval 任务；API error 为 0，paired improved/regressed 为 0/0。Reward 分离没有迁移为自由下一动作改善。

### 7.6 新盲测端到端结果

| `holdout_v2` 指标 | Base | Round-2 DPO | 变化 |
|---|---:|---:|---:|
| 任务完成率 | 0/27 | 0/27 | 0 |
| 平均步数 | 6.67 | 6.67 | 0 |
| 工具调用率 | 100% | 100% | 0 pp |
| 工具协议错误 | 0% | 0% | 0 pp |
| 三类 stress | 各 0/9 | 各 0/9 | 0 |

27 组对齐轨迹中，15 组最终输出改变，4 组工具轨迹改变，但每个 checker 子条件的 Base/DPO 通过数完全相同。这不是“输出完全没变”，而是变化没有稳定跨过任务成功阈值。

代表性变化包括：`pd_h2_001` 的授权跟进 `within_days` 从错误 `"1"` 变为正确 `"3"`；`pd_h2_002` 的通知增加未来 24 小时指引；但 `pd_h2_003` 的出院摘要内容反而变为空字符串。正向变化和新回归同时存在。

部分失败也暴露 checker 的字面限制，例如最终答复写“语音”，checker 要求字面 `voice`。这会影响绝对完成率的解释，但本轮每个子条件的总通过数均未变，没有支持隐藏 uplift 的证据。

## 8. 综合讨论

### 8.1 证据支持什么

| 命题 | 状态 | 证据 |
|---|---|---|
| 朴素截断会损害长程任务 | 支持 | `window` 节省 13.5% token，完成率 14.8% -> 0% |
| 当前 `layered` 优于完整历史 | 不支持 | 仅节省 2.4% token，完成率 11.1% < 14.8% |
| 第一轮 DPO 改善端到端任务 | 否定 | holdout 4/27 -> 2/27 |
| 第二轮 DPO 改善局部决策 | 不支持 | Base/DPO next-action 均 14.3% |
| 第二轮 DPO 改善新盲测 | 不支持 | Base/DPO 均 0/27 |
| reward 可以代替 Agent 评测 | 否定 | 两轮 reward 可分，均未带来端到端改善 |
| schema-valid Function Calling 等于成功 | 否定 | 工具调用率 100%、协议错误 0%，完成率仍可为 0 |

### 8.2 Reward 未迁移的候选解释

实验不能唯一确认某个机制原因，但有四个与证据一致的候选：

1. **偏好 shortcut。** chosen/rejected 的局部形式线索足以改善 reward 排序，却不足以形成可泛化决策规则。
2. **训练信号密度不足。** 66 条 pair 要同时覆盖工具选择、精确值、observation 引用、闭环和 ID 回传，对 1.5B 模型可能过于稀疏。
3. **局部目标与轨迹目标不等价。** 正确 next action 可能在后续被新错误抵消，长程任务对连续决策和错误恢复的要求高于独立偏好对。
4. **分布偏移与容量限制。** task-grouped eval 仍来自合成 train domain，新 holdout 需要组合多个未见细节；基座模型 0/27 也形成地板效应。

地板效应使完成率无法表示成功阈值以下的局部改善。但状态决策和 checker 子条件也没有净变化，所以不能将 0/27 辩护为“其实已经提升”。

### 8.3 第二轮的边际价值

第二轮没有带来新的正向业务结果，但收窄了负结果的解释空间：

- 对齐多轮工具协议，排除第一轮训练/运行序列化不同的明显问题；
- 逐条排除 22 条模糊候选，减少弱偏好对；
- 改为任务级固定切分，避免同任务 repeat 泄漏；
- 将迁移失败进一步定位到 reward -> 自由 next-action 这一层。

因此，第二轮是一次归因加固实验：结论从“第一轮数据可能做坏了”推进到“修复这些可见问题后，当前规模的状态动作 DPO 仍只形成偏好分离，未产生决策迁移”。

## 9. 对 Agent 产品与评测的启示

### 9.1 三道评测门

1. **偏好门：** 模型是否对未进入 train 的 chosen/rejected 形成分离？
2. **决策门：** 不泄漏正确动作类型时，模型是否选对下一动作、工具和精确参数？
3. **工作流门：** 局部决策是否在完整轨迹中累积为任务完成，且没有安全、协议或分类别退化？

低层指标只能解释低层问题。偏好门通过而决策门失败，问题在偏好向决策的迁移；决策门通过而工作流门失败，问题更可能在轨迹误差累积、状态分布偏移或缺乏恢复策略。

### 9.2 从“好答案”到“可执行决策标准”

后训练数据应分别覆盖：

- 工具选择与禁止动作；
- schema 值域和精确参数；
- observation 中 ID、时间、单位和授权范围的复用；
- 必要工具顺序和最大调用次数；
- `finish_task` 与最终结果 ID 回传；
- 失败后的恢复动作，而不只是理想路径。

如果一对 chosen/rejected 只在文风、语气或等价表达上不同，就不应进入决策型 DPO 数据，否则模型可能学习表面风格而非工作流规则。

### 9.3 上下文需要信息保全评测

上下文策略不应只报告 token 节省率。对长程工作流，还应评估关键约束是否写入、是否在正确步骤召回、更新后旧值是否失效，以及新旧事件冲突如何消解。只有 token 减少且信息保全通过，才应继续比较端到端效果。

## 10. 局限与有效性威胁

1. **数据规模小。** 主任务只有 24 条，第二轮偏好数据只有 66 对，不支持生产级统计结论。
2. **任务全部合成。** 优点是无隐私风险且可精确控制；缺点是作者可能在任务和 checker 中引入同源偏差。
3. **单模型、单训练 seed、单组超参。** 负结果不能外推为“DPO 在 Agent 上无效”。
4. **重复轨迹不等于独立任务。** 27 条 holdout 轨迹来自 9 个任务，14 条状态 pair 来自 3 个 eval 任务。
5. **第二轮存在地板效应。** Base 已是 0/27，主指标无法表示成功阈值以下的局部进步。
6. **checker 不等于完整语义理解。** 精确参数适合验证业务约束，字面 grounding 会将部分语义等价自然语言判为错误。
7. **canonical chosen 风格单一。** 它们可验证，但可能诱导模型匹配表面特征，而非更广泛的工作流原则。
8. **未使用公开 benchmark 或真实线上分布。** 结论只适用于当前合成受控环境。

这些限制规定了结论能够外推到哪里，并直接导向下一步实验设计。

## 11. 后续实验设计

在不针对当前冻结 holdout 搜索超参的前提下，后续应优先：

1. 建立新的 development 任务和另一份 untouched holdout，先解决 0/27 地板效应；
2. 将复合 checker 拆为可训练的原子能力标签，如 exact-value reuse、observation grounding、closure 和 recovery；
3. 为每类原子能力构造更多充分区分且多样化的 preference pair；
4. 在独立 dev 上比较更强基座、更多 seed 或其他训练目标，再一次性解冻新 holdout；
5. 对上下文策略单独评估事实保全、召回和失效，而不只看 token 与最终完成率。

项目截止时没有对冻结 holdout 进行超参搜索，也没有为获得正结果而放宽 checker。

## 12. 项目意义

这不是一个证明“我用 DPO 把模型做得更强”的项目。如果只把训练 reward 写成效果，第一轮可以被包装成 100% reward accuracy，第二轮也可以被写成 92.9%。但端到端证据表明，这种表述是错误的。

项目的实际贡献在于：

- 将“Agent 做错了”拆成上下文遗忘、工具误用和规划偏移三类可操作失效；
- 构建从轨迹采集、错误归因、偏好数据、LoRA-DPO 到冻结 holdout 的端到端实验链路；
- 用负结果推动第二轮协议、数据切分和评测层级重设计；
- 证明 reward、Function Calling 合法性和任务完成率不能直接画等号；
- 保留失败 run、完整命令账本、产物哈希和两轮独立路径，使负结果也可审计、可复现。

在长程 Agent 策略工作中，从 badcase 出发，将模糊目标转成可训练规则和可验证指标，然后在证据不支持假设时收窄结论，比得到一个好看但无法传播的训练数字更重要。

## 13. 复现与审计

### 13.1 仓库导航

| 内容 | 路径 |
|---|---|
| AutoDL 端到端复现指南 | [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md) |
| 全量命令、失败和输出日志 | [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) |
| teacher-v2 数据卡 | [`docs/DATA_CARD_TEACHER_V2.md`](docs/DATA_CARD_TEACHER_V2.md) |
| 任务字段和 checker 协议 | [`tasks/schema.md`](tasks/schema.md) |
| 第一轮产物清单 | [`experiments/round1/README.md`](experiments/round1/README.md) |
| 第二轮产物清单 | [`experiments/round2/README.md`](experiments/round2/README.md) |
| teacher-v2 审阅决定 | [`experiments/round2/pair_review.jsonl`](experiments/round2/pair_review.jsonl) |
| 固定任务级切分 | [`experiments/round2/split.json`](experiments/round2/split.json) |
| 状态动作评测器 | [`scripts/7_evaluate_state_actions.py`](scripts/7_evaluate_state_actions.py) |

### 13.2 本地校验

```bash
python3 scripts/0_validate_tasks.py
python3 scripts/0_validate_tasks.py --tasks tasks/holdout_v2.jsonl
PYTHONPYCACHEPREFIX=/tmp/agent-badcase-pycache \
  python3 -m unittest discover -s tests -v
```

当前验证结果：18 个单元测试通过；24 条主任务与 9 条 `holdout_v2` 均通过字段、checker 引用和场景隔离校验。

### 13.3 证据保全

两轮轻量证据包已下载并通过 SHA-256 校验。它们包含派生数据、配置、训练日志、评测轨迹、结果报告、环境元数据和模型哈希，不包含体积较大的 adapter 或 merged model 本体。

| 证据包 | SHA-256 |
|---|---|
| Round 1 | `e9ad0708ce2b2764053aa0a37e598fd4c5ccc0d08d8a9bf2dae109a6dfc88347` |
| Round 2 | `590974ca8bad782eb957b10c06e40f44429c96b02750a4016ab34f534408483c` |

## 14. 结论

本研究没有找到一个能在当前受控环境中稳定提升长程 Agent 完成率的上下文策略或 DPO 配方。朴素窗口用任务成功换取 token 节省；当前分层上下文没有超过完整历史；第一轮 DPO 出现端到端退化；第二轮在修复数据和协议问题后，仍只学到偏好分离，未改善自由状态决策或新盲测完成率。

最有价值的结论并不是“DPO 无效”，而是：**在长程 Agent 中，上下文组织、局部动作偏好和端到端工作流是三个相关但不等价的优化层。** 训练指标只有通过自由决策和冻结工作流评测后，才能成为产品效果证据。
