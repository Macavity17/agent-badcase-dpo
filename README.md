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

训练集和测试集按 `scenario_family` 隔离，不通过替换患者 ID 或数字构造同模板测试题。`context_forgetting` 的关键事实由早期工具观测暴露，不直接写进用户提示，以真实测试窗口截断和事件记忆的差异。

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
      |                合成 chosen + 双重校验
      |                    |
      |               LoRA-DPO
      +---------+----------+
                |
       独立 test 集统一评测
```

上下文策略：

- `full`：保留完整消息历史。
- `window`：只保留最近 2 轮工具交互。
- `layered`：常驻目标与显式约束，将较早动作和事件观测压缩为结构化状态，同时保留最近 1 轮原始交互。

DPO 使用 Qwen2.5-1.5B-Instruct、LoRA rank 16、`beta=0.1`。偏好对仅来自 `train`；`chosen` 必须同时通过工具协议校验和任务 checker。`test` 在数据构造和训练期间保持不可见。

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
    --base-url http://localhost:8000/v1 \
    --out "data/test_${strategy}.jsonl" --resume
done

python3 scripts/5_evaluate.py \
  --files full=data/test_full.jsonl,window=data/test_window.jsonl,layered=data/test_layered.jsonl \
  --out results/context_compare.md
```

采集训练失败并构造偏好对：

```bash
python3 scripts/1_run_baseline.py \
  --split train --strategy full --repeats 6 --temperature 0.7 \
  --base-url http://localhost:8000/v1 \
  --out data/train_full.jsonl --resume

python3 scripts/2_attribute.py \
  --traj data/train_full.jsonl --out data/badcases_labeled.jsonl

export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
python3 scripts/3_build_preference.py \
  --badcase data/badcases_labeled.jsonl --out data/pref_pairs.jsonl \
  --workers 4 --resume

python3 scripts/4_to_llamafactory.py \
  --pref data/pref_pairs.jsonl --outdir data/lf_data
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
  --port 8001 --model dpo --out data/test_dpo.jsonl --resume

python3 scripts/5_evaluate.py \
  --before data/test_full.jsonl --after data/test_dpo.jsonl \
  --out results/dpo_compare.md
```

完整操作与止损条件见 [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md)，实际执行命令、输出、失败 run 和实验决策见 [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)，任务字段见 [`tasks/schema.md`](tasks/schema.md)。

## 结果

仓库当前完成了实验设计、受控环境和运行链路重构。首轮上下文对照已在实验日志中保留为地板效应与实现缺陷的诊断 run，不作为正向结论。完成修复后重跑和 DPO 评测后，只报告独立测试集的实际数字：

| 干预 | tool_misuse | context_forgetting | planning_drift |
|---|---:|---:|---:|
| full 基线 | 待运行 | 待运行 | 待运行 |
| window | 待运行 | 待运行 | 待运行 |
| layered | 待运行 | 待运行 | 待运行 |
| DPO（full 上下文） | 待运行 | 待运行 | 待运行 |

## 局限

- 24 条任务仍是小规模合成 pilot，结论只能解释当前受控环境。
- 单基座、单次训练 seed，不支持生产级稳定性声明。
- 测试集每类只有 3 个独立任务；重复采样增加的是轨迹数，不等同于增加独立任务数。
- checker 能验证关键行为和安全边界，但不能覆盖自然语言质量的全部维度。
- 强模型生成的 chosen 可能引入风格偏差，因此保留协议校验、规则校验和人工抽检三道质量控制。
