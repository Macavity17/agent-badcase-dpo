# 两晚实验执行手册

目标是在 2026-09-04 前得到一组可以诚实写进简历的独立测试结果。优先级依次是：训练测试隔离、真实完成 DPO、统一评测、结果可复现。图表、公开 benchmark、多模型和多训练 seed 均不在本轮范围内。

## 第一晚：环境、基线与偏好数据

### 1. 环境检查

```bash
python3 scripts/0_validate_tasks.py
python3 -m compileall scripts
```

启动 Qwen2.5-1.5B-Instruct 的 vLLM 服务，并确认 `curl http://localhost:8000/v1/models` 可用。先用一条任务冒烟：

```bash
python3 scripts/1_run_baseline.py --split test --limit 1 --strategy full \
  --base-url http://localhost:8000/v1 --out data/smoke.jsonl --verbose
```

### 2. 上下文策略对照

按 README 的命令运行 `full/window/layered`，每个测试任务重复 3 次。三组必须使用相同任务、温度和 seed 规则。

检查点：

- 输出中不存在 `type=error` 的轨迹；服务错误不能算模型失败。
- 每组应有 27 条轨迹、9 个独立任务。
- `max_prompt_tokens` 非零；否则当前服务没有返回 usage，需要在报告中退回字符口径。
- 若全部完成率高于 80% 或低于 10%，先检查 checker 和任务难度，不直接解释策略效果。

### 3. 训练轨迹与偏好对

在 15 条训练任务上以 `temperature=0.7` 重复采样 6 次，共 90 条候选轨迹。只从规则失败且无服务错误的轨迹构造偏好数据。

目标是 40–70 条有效偏好对。这里不追求虚假的“至少 100 条”门槛；小模型受控 pilot 的关键是独立测试与数据质量。

生成完成后人工抽查至少 5 对：

- chosen 使用的工具名和参数来自 schema。
- chosen 经过必要的读取与核对，没有利用 checker 走捷径。
- chosen 没有诊断、开药或修改处方。
- rejected 的主要失败确实对应归因标签。
- chosen 与 rejected 的差异不只是措辞和长度。

第一晚停止条件：上下文对照报告生成，且至少 40 条偏好对通过自动校验。若偏好对不足，优先把训练采样提高到 8 次；不要动测试集，也不要把测试失败加入训练。

## 第二晚：训练、评测与材料更新

### 1. DPO

使用 `config/dpo_qwen15b.yaml` 完成单 seed LoRA-DPO。记录：

- 实际偏好对数量和三类分布；
- GPU、训练时长和显存峰值；
- `rewards/accuracies`、`rewards/margins` 与 loss；
- 最终采用的 checkpoint。

如果训练失败，最多解决一次明确的格式或环境错误。不要在截止日前临时改做 GRPO、多模型或大规模超参搜索。

### 2. 独立测试

DPO 模型只在 9 条 `test` 任务上评测，条件与 base/full 完全一致。除分类别完成率外，必须检查：

- 工具调用率是否突然下降；
- 无效工具名占比是否上升；
- 平均步骤是否明显增加；
- 是否出现重复输出或拒绝调用工具；
- 至少人工对比 3 条 base/DPO 轨迹。

只有完成率上升且这些退化检查没有明显异常，才能表述为“改善”。

### 3. 最终交付

把真实数字填入 README，并保留失败结果。简历只写以下可验证事实：

- 基座模型；
- 24 条任务、训练/测试划分和实际轨迹数；
- 实际有效偏好对数量；
- LoRA-DPO 配置；
- 独立测试的分类别变化；
- 合成数据、单 seed 和未部署生产的边界。

禁止把该项目写成九安内部训练项目。推荐表述为“受慢病照护 Agent 实践启发，离岗后独立完成的合成受控实验”。

## 截止日前可砍项

可以砍：LLM judge、summary 策略、图表、多 seed、公开 benchmark、额外超参实验。

不能砍：测试隔离、chosen 校验、实际训练、DPO 后同条件评测、真实性声明。
