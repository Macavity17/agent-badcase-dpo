# AutoDL 两晚实验 Runbook

本手册覆盖从空白 AutoDL 实例到最终 DPO 对比报告的完整命令。默认使用单卡 RTX 4090/4090D、Ubuntu、CUDA 12.x，工作目录为 `/root/autodl-tmp/agent-badcase-dpo`。

实验截止日期为 2026-09-04。主线只做 Qwen2.5-1.5B-Instruct、`full/window/layered`、单 seed LoRA-DPO 和独立测试集评测。

## 实验日志协议

`docs/EXPERIMENT_LOG.md` 是实际执行记录。每完成一次实质性操作（环境变更、数据校验、采样、训练、评测、错误归因或实验决策）后立即追加：当时 commit、实际命令和参数、关键输出、产物路径、失败与下一步。服务器上的每一条 shell 命令都要进入文末的命令账本，包括诊断、等待、日志查看、失败命令和重试。保留失败 run，修复后使用新文件名，不覆盖旧结果。

日志不得包含 SSH 密码、API key、Token 或其他凭据。

## 0. 开机与代码准备

在 AutoDL 创建带 PyTorch/CUDA 的单卡实例，SSH 登录后执行：

```bash
cd /root/autodl-tmp
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi

git clone https://github.com/Macavity17/agent-badcase-dpo.git
cd /root/autodl-tmp/agent-badcase-dpo

nvidia-smi
python3 --version
mkdir -p data results runs models outputs vendor
```

如果仓库已经存在：

```bash
cd /root/autodl-tmp/agent-badcase-dpo
git pull --ff-only origin main
```

所有生成目录均被 Git 忽略。不要把 API key 写进仓库或日志。

## 1. 创建 Python 3.11 推理环境并下载模型

基础镜像可以使用 Python 3.12；实验通过 Miniconda 显式创建 Python 3.11 环境。推理与训练使用不同 Conda 环境，避免 vLLM 和训练依赖发生版本冲突。

```bash
cd /root/autodl-tmp/agent-badcase-dpo
eval "$(conda shell.bash hook)"
conda create -n care-infer python=3.11 -y
conda activate care-infer

python -m pip install --upgrade pip
python -m pip install "openai>=1.40" "vllm>=0.6.0" "huggingface_hub[cli]>=0.24"

hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir ./models/Qwen2.5-1.5B-Instruct
```

验证代码和数据：

```bash
python3 scripts/0_validate_tasks.py
PYTHONPYCACHEPREFIX=/tmp/care-agent-pycache python3 -m compileall -q scripts tests
python3 -m unittest discover -s tests -v
```

预期任务统计：24 条；train 15、test 9；三类失效各 8 条。

## 2. 启动基座 vLLM

Qwen2.5 的训练响应和部署评测都使用兼容的 `<tool_call>` / Hermes 工具协议。

```bash
cd /root/autodl-tmp/agent-badcase-dpo
eval "$(conda shell.bash hook)"
conda activate care-infer

nohup python3 -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen2.5-1.5B-Instruct \
  --served-model-name base \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > runs/vllm_base.log 2>&1 &

echo $! > runs/vllm_base.pid
tail -f runs/vllm_base.log
```

看到服务启动完成后按 `Ctrl+C` 退出日志查看，不会停止后台服务。验证接口：

```bash
curl -s http://localhost:8000/v1/models
```

## 3. 冒烟测试

```bash
python3 scripts/1_run_baseline.py \
  --split test --limit 1 --strategy full --repeats 1 \
  --base-url http://localhost:8000/v1 \
  --out data/smoke.jsonl --verbose

sed -n '1p' data/smoke.jsonl
grep -n '"type": "error"' data/smoke.jsonl || true
```

继续前确认：

- 轨迹包含结构化 `tool_calls`，而不是把工具调用写成普通文本。
- `prompt_tokens` 中至少有一个非零值。
- 不存在 `type=error`。

如果工具调用只出现在文本中，先检查 vLLM 是否带了 `--enable-auto-tool-choice --tool-call-parser hermes`，不要直接跑全量。

## 4. 上下文策略对照

三种策略必须保持相同模型、测试集、重复数、温度和 seed 规则：

```bash
for strategy in full window layered; do
  python3 scripts/1_run_baseline.py \
    --split test \
    --strategy "$strategy" \
    --repeats 3 \
    --temperature 0.2 \
    --seed 42 \
    --workers 4 \
    --base-url http://localhost:8000/v1 \
    --out "data/test_${strategy}.jsonl" \
    --resume
done
```

生成报告：

```bash
python3 scripts/5_evaluate.py \
  --files full=data/test_full.jsonl,window=data/test_window.jsonl,layered=data/test_layered.jsonl \
  --out results/context_compare.md

sed -n '1,220p' results/context_compare.md
wc -l data/test_full.jsonl data/test_window.jsonl data/test_layered.jsonl
grep -n '"type": "error"' data/test_*.jsonl || true
```

每组应有 27 条轨迹。服务错误不能算模型失败。如果三组均高于 80% 或低于 10%，先检查任务难度与 checker，不解释策略差异。

## 5. 采集训练失败轨迹

只在 train split 采样，test 绝不进入偏好数据：

```bash
python3 scripts/1_run_baseline.py \
  --split train \
  --strategy full \
  --repeats 6 \
  --temperature 0.7 \
  --seed 2026 \
  --workers 4 \
  --base-url http://localhost:8000/v1 \
  --out data/train_full.jsonl \
  --resume

wc -l data/train_full.jsonl
grep -n '"type": "error"' data/train_full.jsonl || true
```

应有 90 条候选轨迹。然后归因真实失败，默认排除 API/服务错误：

```bash
python3 scripts/2_attribute.py \
  --traj data/train_full.jsonl \
  --tasks tasks/tasks.jsonl \
  --out data/badcases_labeled.jsonl

wc -l data/badcases_labeled.jsonl
```

## 6. 用强模型合成 chosen

以下以 OpenAI-compatible API 为例。将占位符替换为实际服务信息，不要把真实 key 写入 Markdown 或 Git：

```bash
export OPENAI_API_KEY='<YOUR_API_KEY>'
export OPENAI_BASE_URL='<YOUR_OPENAI_COMPATIBLE_BASE_URL>'
export SYNTH_MODEL='<YOUR_SYNTH_MODEL_NAME>'

python3 scripts/3_build_preference.py \
  --badcase data/badcases_labeled.jsonl \
  --tasks tasks/tasks.jsonl \
  --out data/pref_pairs.jsonl \
  --model "$SYNTH_MODEL" \
  --workers 4 \
  --resume

wc -l data/pref_pairs.jsonl
sed -n '1,5p' data/pref_pairs.jsonl
```

人工检查前 5 对：chosen 是否完成必要读取、是否使用真实 schema、是否越权诊断/开药、rejected 是否确实更差。目标是 40–70 条有效偏好对。

如果少于 40 条，只增加 train 的重复采样：

```bash
python3 scripts/1_run_baseline.py \
  --split train --strategy full --repeats 8 --temperature 0.7 \
  --seed 2026 --workers 4 \
  --base-url http://localhost:8000/v1 \
  --out data/train_full.jsonl --resume

python3 scripts/2_attribute.py \
  --traj data/train_full.jsonl --out data/badcases_labeled.jsonl

python3 scripts/3_build_preference.py \
  --badcase data/badcases_labeled.jsonl \
  --out data/pref_pairs.jsonl \
  --model "$SYNTH_MODEL" --workers 4 --resume
```

不要降低 checker 标准，也不要使用 test 轨迹补量。

## 7. 导出 LLaMA-Factory 数据

```bash
python3 scripts/4_to_llamafactory.py \
  --pref data/pref_pairs.jsonl \
  --outdir data/lf_data

python3 -m json.tool data/lf_data/agent_pref.json > /dev/null
python3 -m json.tool data/lf_data/dataset_info.json > /dev/null
sed -n '1,160p' data/lf_data/stat.json
```

`config/dpo_qwen15b.yaml` 会直接读取 `./data/lf_data`，不需要覆盖 LLaMA-Factory 自带的 `dataset_info.json`。

## 8. 停止 vLLM 并创建训练环境

DPO 训练前必须释放基座 vLLM 占用的显存：

```bash
kill "$(cat runs/vllm_base.pid)"
sleep 5
nvidia-smi
```

创建独立训练环境并安装 LLaMA-Factory：

```bash
conda deactivate
cd /root/autodl-tmp/agent-badcase-dpo

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git vendor/LLaMA-Factory

conda create -n care-train python=3.11 -y
conda activate care-train
python -m pip install --upgrade pip
python -m pip install -e "./vendor/LLaMA-Factory[torch,metrics]"

git -C vendor/LLaMA-Factory rev-parse HEAD > runs/llamafactory_commit.txt
llamafactory-cli --help
```

如果 `vendor/LLaMA-Factory` 已存在：

```bash
git -C vendor/LLaMA-Factory pull --ff-only
python -m pip install -e "./vendor/LLaMA-Factory[torch,metrics]"
git -C vendor/LLaMA-Factory rev-parse HEAD > runs/llamafactory_commit.txt
```

## 9. 运行 LoRA-DPO

从仓库根目录启动训练。`nohup` 确保 SSH 断开后训练继续：

```bash
cd /root/autodl-tmp/agent-badcase-dpo
eval "$(conda shell.bash hook)"
conda activate care-train

nohup llamafactory-cli train config/dpo_qwen15b.yaml \
  > runs/dpo_train.log 2>&1 &

echo $! > runs/dpo_train.pid
tail -f runs/dpo_train.log
```

训练完成后确认进程退出、adapter 存在，并查看训练记录：

```bash
ps -p "$(cat runs/dpo_train.pid)" || true
find outputs/dpo-qwen15b -maxdepth 2 -type f | sort | sed -n '1,120p'
tail -n 80 runs/dpo_train.log
nvidia-smi
```

记录实际偏好对数量、三类分布、GPU、训练时长、显存、loss、`rewards/accuracies`、`rewards/margins` 和采用的 checkpoint。快速冲到 100% reward accuracy 只能说明区分训练偏好容易，不等于 Agent 行为已经改善。

## 10. 合并 LoRA

```bash
cd /root/autodl-tmp/agent-badcase-dpo
eval "$(conda shell.bash hook)"
conda activate care-train

llamafactory-cli export config/merge_lora.yaml \
  > runs/merge_lora.log 2>&1

find outputs/dpo_merged -maxdepth 1 -type f | sort
```

## 11. 启动 DPO 模型

退出训练环境并回到推理环境：

```bash
conda deactivate
conda activate care-infer

nohup python3 -m vllm.entrypoints.openai.api_server \
  --model ./outputs/dpo_merged \
  --served-model-name dpo \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > runs/vllm_dpo.log 2>&1 &

echo $! > runs/vllm_dpo.pid
tail -f runs/vllm_dpo.log
```

看到启动完成后按 `Ctrl+C` 退出日志查看并验证：

```bash
curl -s http://localhost:8001/v1/models
```

## 12. 独立测试与前后对比

DPO 只在 untouched test split 上以与 base/full 相同的条件评测：

```bash
python3 scripts/1_run_baseline.py \
  --split test \
  --strategy full \
  --repeats 3 \
  --temperature 0.2 \
  --seed 42 \
  --workers 4 \
  --port 8001 \
  --model dpo \
  --out data/test_dpo.jsonl \
  --resume

grep -n '"type": "error"' data/test_dpo.jsonl || true
wc -l data/test_dpo.jsonl

python3 scripts/5_evaluate.py \
  --before data/test_full.jsonl \
  --after data/test_dpo.jsonl \
  --out results/dpo_compare.md

sed -n '1,240p' results/dpo_compare.md
```

必须同时检查完成率、工具调用率、工具协议错误、平均步骤和重复行为。至少人工对比 3 个相同 task/repeat 的 base 与 DPO 轨迹。只有目标指标改善且没有明显退化，才能写“提升”。

## 13. 备份证据

`data/`、`results/`、`runs/` 和 `outputs/` 均被 Git 忽略。先在 AutoDL 数据盘生成证据包：

```bash
cd /root/autodl-tmp/agent-badcase-dpo
tar -czf /root/autodl-tmp/care-agent-evidence-20260903.tar.gz \
  data results runs config tasks README.md

ls -lh /root/autodl-tmp/care-agent-evidence-20260903.tar.gz
```

通过 AutoDL Jupyter 文件管理器下载，或在本地终端使用平台提供的 SSH 主机和端口：

```bash
scp -P <SSH_PORT> \
  root@<SSH_HOST>:/root/autodl-tmp/care-agent-evidence-20260903.tar.gz .
```

不要只把结果留在实例盘。确认本地备份后再关机停止计费。

## 14. 最终材料

把真实数字填入 README，保留失败结果和局限。简历只写实际完成的：基座模型、24 条任务、训练/测试划分、轨迹数、有效偏好对数、LoRA-DPO 参数、独立测试分类别变化。

项目必须描述为“受慢病照护 Agent 实践启发、离岗后独立完成的合成受控实验”，不能写成九安内部训练或生产部署。

## 截止日前止损顺序

可以砍：LLM judge、图表、summary 策略、额外 seed、公开 benchmark、超参搜索。

不能砍：train/test 隔离、chosen 校验、真实 DPO、同条件后测、退化检查、真实性声明和结果备份。
