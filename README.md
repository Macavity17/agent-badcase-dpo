# Care Agent Failure Lab

> 一个慢病照护 Agent 的独立受控实验：上下文分层与 DPO 分别能修复哪类长程失效？

## 项目背景

在既往慢病照护 Agent 实践中，我参与过模型选型、event-memory、badcase 数据分析与自动化评测。一个反复出现的问题是：相似的失败表象，根因可能完全不同。有些来自早期约束在长链路中丢失，有些来自模型对相似工具的选择偏差，还有些来自规划循环不收敛。它们不应该默认用同一种手段修复。

因此，我在实习结束后独立构建了这个 controlled pilot，研究三个问题：

1. 滑动窗口丢失早期关键观测时，结构化事件记忆能否恢复完成率？
2. 上下文组织难以修复的工具选择偏差，能否通过小规模 LoRA-DPO 改善？
3. 两种干预对 `tool_misuse`、`context_forgetting`、`planning_drift` 的作用边界是否不同？

### 真实性边界

本项目不是九安医疗或腾讯的内部项目，DPO 训练也未在任何生产系统部署。仓库不包含公司代码、真实用户数据、内部提示词或内部指标。患者、工具、照护计划和执行结果均为合成数据。应用问题来自既往实践启发，实验设计、代码、数据与结论由个人独立完成。

## 受控环境

环境模拟慢病照护运营平台中的信息读取、记录、提醒、教育内容投递、随访协调与人工升级。Agent 无权诊断、开药或自行修改处方；紧急情况与权限外决策必须升级人工。

任务集中三类定向失效各 8 条，共 24 条：

| 失效模式 | 典型风险 | train / test |
|---|---|---:|
| `tool_misuse` | 记录与更正混淆、普通消息代替紧急升级 | 5 / 3 |
| `context_forgetting` | 遗忘过敏、授权、单位、时区或最新计划 | 5 / 3 |
| `planning_drift` | 重复查询、遗漏闭环、越权采取高风险动作 | 5 / 3 |

训练集和测试集按 `scenario_family` 隔离。早期调试过的 9 条 test 任务已归档为 `tasks/dev_tasks.jsonl`，最终结果使用全新患者和事件 ID 的 9 条冻结 holdout；冻结后不再根据 holdout 表现修改任务或 checker。`context_forgetting` 的关键事实由早期工具观测暴露，不直接写进用户提示。

## 实验设计

```text
合成照护任务 + 确定性 mock 工具
                |
        Qwen2.5-1.5B ReAct 轨迹
                |
      +---------+----------+
      |                    |
上下文策略对照         训练集失败轨迹归因
full/window/layered         |
      |            canonical chosen + 双重校验
      |                    |
      |               LoRA-DPO
      +---------+----------+
                |
       独立 test 集统一评测
```

上下文策略：

- `full`：保留完整消息历史。
- `window`：只保留最近 2 轮工具交互。
- `layered`：常驻目标与显式约束，将较早动作和事件观测压缩为结构化状态，同时保留最近 2 轮原始交互。

DPO 使用 Qwen2.5-1.5B-Instruct、LoRA rank 16、`beta=0.1`。偏好对仅来自 `train` 失败轨迹；`chosen` 是由任务 gold workflow、checker 和确定性 mock 构造的“规则约束 canonical 合成轨迹”，不是人工标注或强模型标注。每条 chosen 必须同时通过工具协议校验和任务 checker。`test` 在数据构造和训练期间保持不可见。

训练前将 chosen/rejected 统一序列化为 Qwen 兼容的 `<tool_call>` 标记，避免用自定义 `CALL(...)` 文本训练、再用原生 Function Calling 评测造成协议错位。

主指标是规则完成率，并按失效模式拆分。辅助指标包括平均步骤数、无效工具名占比、工具调用率和 API 返回的 prompt token 峰值。LLM-as-a-judge 仅用于人工难以覆盖的轨迹质量复核，不替代规则主指标。

## 两晚执行路径

安装依赖并启动启用了 Qwen 工具解析器的 OpenAI-compatible vLLM 服务：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen2.5-1.5B-Instruct --served-model-name base \
  --port 8000 --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

然后执行：

```bash
python3 scripts/0_validate_tasks.py

for strategy in full window layered; do
  python3 scripts/1_run_baseline.py \
    --split test --strategy "$strategy" --repeats 3 --temperature 0.2 \
    --seed 20260902 \
    --base-url http://localhost:8000/v1 \
    --out "data/holdout_base_${strategy}.jsonl" --resume
done

python3 scripts/5_evaluate.py \
  --files full=data/holdout_base_full.jsonl,window=data/holdout_base_window.jsonl,layered=data/holdout_base_layered.jsonl \
  --out results/holdout_context_compare.md
```

采集训练失败并构造 canonical 偏好对：

```bash
python3 scripts/1_run_baseline.py \
  --split train --strategy full --repeats 6 --temperature 0.7 \
  --base-url http://localhost:8000/v1 \
  --seed 4242 --out data/train_base_full_r6.jsonl --resume

python3 scripts/2_attribute.py \
  --traj data/train_base_full_r6.jsonl --out data/train_badcases_labeled.jsonl

python3 scripts/3_build_preference.py \
  --synth-mode canonical \
  --badcase data/train_badcases_labeled.jsonl \
  --tasks tasks/tasks.jsonl \
  --out data/pref_pairs_canonical_v2.jsonl --workers 4

python3 scripts/4_to_llamafactory.py \
  --pref data/pref_pairs_canonical_v2.jsonl --outdir data/lf_data
```

安装 LLaMA-Factory 后，从仓库根目录运行；训练配置会直接读取 `data/lf_data/`：

```bash
llamafactory-cli train config/dpo_qwen15b.yaml
llamafactory-cli export config/merge_lora.yaml
```

用合并模型在 `8001` 端口启动 vLLM，然后在同一测试集上评测：

```bash
python3 scripts/1_run_baseline.py \
  --split test --strategy full --repeats 3 --temperature 0.2 \
  --seed 20260902 --port 8001 --model dpo \
  --out data/holdout_dpo_full_seed20260902.jsonl --resume

python3 scripts/5_evaluate.py \
  --before data/holdout_base_full.jsonl \
  --after data/holdout_dpo_full_seed20260902.jsonl \
  --out results/dpo_compare_seed20260902.md
```

完整操作与止损条件见 [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md)，实际执行命令、输出、失败 run 和实验决策见 [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)，任务字段见 [`tasks/schema.md`](tasks/schema.md)。

## 结果

最终冻结 holdout 含 9 个独立任务，每任务重复 3 次，即每组 27 条轨迹。上下文策略只改变消息组装；DPO 前后使用相同 `full` 策略、任务、温度与 27/27 对齐的 seed。

| 干预 | 总完成率 | tool_misuse | context_forgetting | planning_drift | 平均 prompt 峰值 |
|---|---:|---:|---:|---:|---:|
| full 基线 | 14.8% (4/27) | 0/9 | 1/9 | 3/9 | 1,234 |
| window | 0.0% (0/27) | 0/9 | 0/9 | 0/9 | 1,068 |
| layered | 11.1% (3/27) | 1/9 | 2/9 | 0/9 | 1,205 |
| DPO（full 上下文） | 7.4% (2/27) | 0/9 | 1/9 | 1/9 | 1,230 |

`window` 节省约 13.5% 的 prompt 峰值，但完成率降为 0；`layered` 仅节省约 2.4%，且没有超过 full。因此这轮不支持“分层上下文带来净改善”，它更明确地显示了朴素截断的风险以及小模型对压缩状态的敏感性。

DPO 使用 68 对 canonical 偏好数据训练 3 epoch，44.46 秒完成。训练内 eval reward accuracy 为 1.0、reward margin 为 0.217，但这种偏好分离没有转化为任务收益：训练任务上 base/DPO 都为 22/90，冻结 holdout 则从 4/27 降至 2/27。对齐轨迹显示两类退化：精确参数复用变成语义改写（如 `2` -> `连续两天`、`next_week` -> `next`），以及最终答复漏掉工具返回 ID。也观察到一个局部改善：DPO 在一条设备轨迹中避免了 base 的禁止同步调用，但仍未使该轨迹通过 checker。这说明“工具协议 0 错误”并不等于“工作流完成”，也为下一轮数据标准提供了明确方向：对 schema 值域、观测引用和结果回传分别设置评测与训练信号。

### 第二轮数据修复（已准备，尚未训练）

针对第一轮暴露的训练/运行协议错位，第二轮不再把整条多工具轨迹放进单个 assistant response，而是把 68 条 train badcase 编译成首次分歧点的“状态 -> 一个下一动作”偏好。去除自由文本措辞伪差异和 24 条完全重复候选后，得到 44 条真实 badcase 决策 pair；另加入 30 条结果回传 hard negative，共 74 条唯一训练 pair（工具动作 38、最终答复 36）。

同时新建了 9 条未运行的 `holdout_v2`，三类失效各 3 条，患者 ID 和 scenario family 均不与 train/dev/旧 holdout 重合。当前只完成数据构造与结构校验，没有第二轮训练结果。来源、过滤规则、哈希和局限见 [`docs/DATA_CARD_TEACHER_V2.md`](docs/DATA_CARD_TEACHER_V2.md)。

## 局限

- 24 条任务仍是小规模合成 pilot，结论只能解释当前受控环境。
- 单基座、单次训练 seed，不支持生产级稳定性声明。
- 测试集每类只有 3 个独立任务；重复采样增加的是轨迹数，不等同于增加独立任务数。
- checker 能验证关键行为和安全边界，但不能覆盖自然语言质量的全部维度。
- canonical chosen 由任务规则构造，可验证但风格单一，可能导致对规则文本而非未见场景的过拟合。
- DPO 只跑了单 seed 和一组超参；本轮负结果不能外推为“DPO 无效”。
