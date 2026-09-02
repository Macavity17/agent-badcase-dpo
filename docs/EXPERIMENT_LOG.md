# 实验日志

Last updated: 2026-09-03 (Asia/Shanghai)

本文档记录实际执行的实验，与 `docs/EXPERIMENT_GUIDE.md` 的预期操作分开。未执行的命令、未观测的结果不得写成完成项。失败的 run 保留原始结果，修复后使用新的记录和输出文件。

全部患者、工具、观测和照护计划均为合成数据。本项目是受既往慢病照护 Agent 实践启发的独立受控实验，不是九安或腾讯的内部项目，也没有使用任何公司代码、真实用户数据或生产指标。

## 记录规则

每次实质性操作后更新本文档，至少记录：

- 日期、执行环境、当时的 Git commit；
- 实际命令和关键参数；
- 输入数据分割、样本数、seed 和重复数；
- 命令的关键输出、产物路径和错误；
- 对结果的归因、决策与下一步；
- 如果失败，明确标记 `FAILED` 或 `INVALID FOR CONCLUSION`，不覆盖原结果。

对服务器的每一条 shell 命令都要进入文末的完整命令账本，包括诊断、等待、查看日志、失败命令和重试；不只记录最终成功的主流程命令。

不记录 SSH 密码、API key、Token 或其他凭据。

## 2026-09-02 / Run 0：本地代码与数据准备

### 目标

将原型收敛为两个晚上可完成的受控实验：在同一 Qwen2.5-1.5B-Instruct 上对比 `full/window/layered` 上下文策略，然后用 train split 失败轨迹构造偏好对并进行 LoRA-DPO，最后在隔离的 test split 上对比训练前后。

### 固定实验设计

- 基座模型：Qwen2.5-1.5B-Instruct。
- 数据：24 个合成慢病照护任务。
- 划分：train 15，test 9；每个失效类别在 train/test 中为 5/3。
- 失效类别：`tool_misuse`、`context_forgetting`、`planning_drift`，各 8 个任务。
- 泄漏控制：`scenario_family` 不跨 train/test。
- 上下文对照：9 个 test 任务 x 3 次重复 = 27 条轨迹/策略。
- 初始训练采样：15 个 train 任务 x 6 次 = 90 条候选轨迹。
- DPO 目标：40-70 对合格偏好数据，LoRA rank 16，beta 0.1，单 seed。

### 已完成实现

- 重构 24 个合成照护任务，并加入 train/test 与场景族隔离。
- 加入复合 checker：工具顺序、禁止工具、最大调用数、最终答案关键字和敏感信息排除。
- runner 加入 split、repeats、seed、workers、resume、port 和 prompt token 记录。
- 偏好数据链路加入协议校验、checker 校验、并发和续传，并输出 Qwen/Hermes `<tool_call>` 格式。
- 补全 LLaMA-Factory DPO 与 LoRA merge 配置。
- 将 AutoDL 命令整理为 `docs/EXPERIMENT_GUIDE.md`。

### 本地验证

执行：

```bash
python3 scripts/0_validate_tasks.py
PYTHONPYCACHEPREFIX=/tmp/care-agent-pycache python3 -m compileall -q scripts tests
python3 -m unittest discover -s tests -v
git diff --check
```

观测：

```text
任务总数：24
按 split： {'train': 15, 'test': 9}
按失效模式： {'tool_misuse': 8, 'context_forgetting': 8, 'planning_drift': 8}
split x 失效模式：
  train: 5 / 5 / 5
  test: 3 / 3 / 3
校验通过：字段、checker 引用与场景族隔离均有效。
```

当时的服务器执行基线 commit：`1f87b2a` (`docs: add end-to-end AutoDL runbook`)。

## 2026-09-02 / Run 1：AutoDL 环境建立

### 服务器与仓库

- SSH 实际登录验证用户为平台提供的 `root`。凭据未写入文件、shell 命令或日志。
- 服务器仓库路径：`/root/autodl-tmp/agent-badcase-dpo`。
- 服务器 checkout：`1f87b2a62183806e2ec180be54e1bccb7cf2aefb`。
- OS：Ubuntu 22.04 基础镜像，Linux `5.15.0-94-generic x86_64`。
- GPU：NVIDIA GeForce RTX 4090 D，24,564 MiB，Driver `580.76.05`。
- 数据盘：`/root/autodl-tmp` 170 GiB；补采时已用 2.9 GiB，可用 168 GiB。
- 服务器报告内存 1.0 TiB；这是容器可见值，未用它推断专属物理内存配额。

### Conda 推理环境

实际命令：

```bash
eval "$(conda shell.bash hook)"
conda create -n care-infer python=3.11 -y
conda activate care-infer
python -m pip install --upgrade pip
python -m pip install "openai>=1.40" "vllm>=0.6.0" "huggingface_hub[cli]>=0.24"
```

Conda 创建阶段在 `Collecting package metadata` 停留较久，但未中断，最终完成。实际安装版本：

```text
Python             3.11.16
PyTorch            2.13.0+cu130
PyTorch CUDA       13.0
CUDA available     True
vLLM               0.28.0
OpenAI Python      3.6.0
Hugging Face Hub   1.29.0
GPU detected       NVIDIA GeForce RTX 4090 D
```

`pip` 提示 Hugging Face Hub 1.29.0 不再提供 `cli` extra，但 `hf` 命令已安装且后续可正常下载，因此未额外变更环境。

### 服务器代码验证

实际命令：

```bash
python scripts/0_validate_tasks.py
python -m unittest discover -s tests -v
```

实际输出：

```text
任务总数：24
按 split： {'train': 15, 'test': 9}
按失效模式： {'tool_misuse': 8, 'context_forgetting': 8, 'planning_drift': 8}
校验通过：字段、checker 引用与场景族隔离均有效。

Ran 2 tests in 0.001s
OK
```

## 2026-09-02 / Run 2：基础模型下载

### 命令

```bash
mkdir -p models runs
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir ./models/Qwen2.5-1.5B-Instruct
```

### 第一次尝试：`FAILED, RESUMABLE`

- 未登录 Hugging Face Hub；命令明确提示会有较低限额，但无需 Token 才能下载该公开模型。
- 主权重建进度到达 `3.10GB / 3.10GB`。
- 在获取其余文件元数据时，远端代理断开，报错：

```text
httpx.RemoteProtocolError: Server disconnected without sending a response.
```

- Hugging Face 本地缓存和已重建权重均被保留，没有删除或重新下载。

### 第二次尝试：`SUCCESS`

执行相同命令续传，只补齐约 2.78 MB 缺失内容。关键输出：

```text
Fetching 10 files: 100%|...| 10/10
Downloaded
path: /root/autodl-tmp/agent-badcase-dpo/models/Qwen2.5-1.5B-Instruct
```

产物验证：

```text
目录大小: 2.9G
权重文件: model.safetensors, 2.9G
SHA-256: dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
```

## 2026-09-02 / Run 3：vLLM 基础模型服务

### 启动命令

```bash
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
```

### 启动观测

最初两次 `curl http://localhost:8000/v1/models` 返回 connection refused。这不是服务崩溃；进程仍在，vLLM 正在首次编译和捕获 CUDA Graph。日志关键行：

```text
21:15:45 vLLM version 0.28.0
21:15:52 Resolved architecture: Qwen2ForCausalLM
21:15:52 Using max model len 8192
21:16:06 Loading weights took 1.03 seconds
21:16:07 Model loading took 2.98 GiB memory and 3.285363 seconds
21:16:24 Available KV cache memory: 15.85 GiB
21:16:24 GPU KV cache size: 593,424 tokens
         Maximum concurrency for 8,192 tokens per request: 72.44x
21:16:28 Graph capturing finished in 3 secs, took 0.48 GiB
Application startup complete.
```

健康检查：

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

返回模型 `base`、root `./models/Qwen2.5-1.5B-Instruct`、`max_model_len=8192`。补采时 GPU 占用 21,187 MiB，GPU utilization 0%，与模型已加载但空闲的状态一致。

## 2026-09-02 / Run 4：Function Calling 冒烟测试

### 命令与参数

```bash
python3 scripts/1_run_baseline.py \
  --split test \
  --limit 1 \
  --strategy full \
  --repeats 1 \
  --base-url http://127.0.0.1:8000/v1 \
  --out data/smoke.jsonl \
  --verbose
```

- 任务：`tm_test_001`。
- 模型：`base`。
- 策略：`full`。
- repeat: 0，seed: 42。

### 实际轨迹摘要

```text
step 1: pair_device(patient_id=P9001, device_id=DEV-901)
observation: {"ok": true, "pairing_id": "PAIR-901"}
step 2 final: 设备已配对，配对 ID 为 PAIR-901。
```

运行统计：

```text
structured tool_calls: 有
prompt_tokens: [610, 680]
max_prompt_tokens: 680
context_chars: [283, 321]
service/API error: 0
success: false
```

### 结果判定

`SUCCESS AS A SMOKE TEST`，但任务本身失败。模型使用了原生结构化 `tool_calls`，而不是把调用写成普通文本；usage 中也有非零 token 数。任务 checker 要求先查设备状态、再配对，模型直接配对，所以 `success=false` 是真实 badcase，不是 API 或 checker 故障。

## 2026-09-02 / Run 5：首轮上下文策略对照

### 命令与参数

```bash
for strategy in full window layered; do
  python3 scripts/1_run_baseline.py \
    --split test \
    --strategy "$strategy" \
    --repeats 3 \
    --temperature 0.2 \
    --seed 42 \
    --workers 4 \
    --base-url http://127.0.0.1:8000/v1 \
    --out "data/test_${strategy}.jsonl" \
    --resume
done

python3 scripts/5_evaluate.py \
  --files full=data/test_full.jsonl,window=data/test_window.jsonl,layered=data/test_layered.jsonl \
  --out results/context_compare.md
```

样本口径：9 个独立 test 任务，每任务每策略 3 次，27 条/策略，81 条总轨迹。三策略使用同一模型、任务、temperature 与 seed 派生规则，只改变上下文组装。

文件检查：

```text
data/test_full.jsonl      27
data/test_window.jsonl    27
data/test_layered.jsonl   27
total                     81
type=error                0 / 0 / 0
```

### 总体结果

| 策略 | 轨迹数 | 成功数 | 完成率 | 平均步数 | 平均 prompt 峰值 |
|---|---:|---:|---:|---:|---:|
| full | 27 | 3 | 11.1% | 3.67 | 864 tokens |
| window | 27 | 0 | 0.0% | 3.74 | 868 tokens |
| layered v1 | 27 | 1 | 3.7% | 3.74 | 1,107 tokens |

### 分失效模式结果

| 策略 | tool_misuse | context_forgetting | planning_drift |
|---|---:|---:|---:|
| full | 11% (1/9) | 0% (0/9) | 22% (2/9) |
| window | 0% (0/9) | 0% (0/9) | 0% (0/9) |
| layered v1 | 11% (1/9) | 0% (0/9) | 0% (0/9) |

full 按任务的成功数：

| 任务 | 类别 | 成功/重复 |
|---|---|---:|
| tm_test_001 | tool_misuse | 0/3 |
| tm_test_002 | tool_misuse | 0/3 |
| tm_test_003 | tool_misuse | 1/3 |
| cf_test_001 | context_forgetting | 0/3 |
| cf_test_002 | context_forgetting | 0/3 |
| cf_test_003 | context_forgetting | 0/3 |
| pd_test_001 | planning_drift | 1/3 |
| pd_test_002 | planning_drift | 0/3 |
| pd_test_003 | planning_drift | 1/3 |

### 结论有效性：`INVALID FOR POSITIVE CONTEXT CLAIM`

本轮运行是有效的工程执行（81 条、0 服务错误），但不能支持“layered 提高完成率且降低上下文开销”的正向结论，原因有两个：

1. 总体成功率过低，`context_forgetting` 三策略全为 0%，存在明显地板效应。
2. layered v1 的 token 峰值 1,107，高于 full/window，与预期压缩目标相反。

因此不得将该结果写成成功实验，也未将 README 结果占位符替换为正向描述。

### badcase 定性检查

对 full 的 27 条轨迹按任务检查工具序列、参数与最终回答，观察到：

- `tm_test_001`：3/3 直接 `pair_device`，遗漏前置状态读取。
- `tm_test_002`：3/3 直接升级，遗漏留言读取；priority 用 `high` 而 checker 要求 `urgent`。
- `cf_test_001`：3/3 调用了全部五个工具，但部分参数是占位文本；有轨迹在最终回答中泄露了家属未获授权接收的测量值，另有轨迹未返回预约 ID。
- `cf_test_002`：3/3 在搜索审核材料后停止，没有实际调用投递工具；部分轨迹搜索格式错用为文本。
- `cf_test_003`：3/3 调用了四个工具，但 appointment ID 和时间参数为幻觉/占位值，未将 `09:00 - 8小时` 稳定计算成 checker 要求的 `01:00`。
- `pd_test_002`：3/3 直接升级，没有读问卷和基础档案，也没有记录安全事件。
- `pd_test_003`：3/3 调用了预期的五个工具，但只有 1/3 在最终回答中同时带回升级和跟进 ID。

这些轨迹说明失败不是单一原因：同时存在前置读取遗漏、工具参数未从观测中绑定、长流程提前终止、最终答案遗漏 ID 和授权信息忘却。

## 2026-09-02 / Run 6：layered v1 实现归因与修复

### 代码归因

layered v1 的有效上下文由以下部分组成：

```text
system
+ 包含目标/约束/全部步骤/全部观测的 state block
+ 原始 user 目标与约束
+ 最近 2 轮原始 assistant/tool 细节
```

这导致目标与约束重复，最近观测也同时存在于 state block 和原始 tool message。因此 layered v1 不是压缩策略，而是上下文叠加策略；1,107 token 峰值不是模型偶然噪声，而是实现缺陷的直接结果。

### 修复设计

layered v2 改为：

```text
system
+ 替代原始 user 的稳定状态块
  - 目标
  - 约束
  - 只压缩较早步骤与观测
+ 最近 1 轮原始 assistant/tool 细节
```

变更文件：

- `scripts/1_run_baseline.py`：修正 `build_state_block` 和 `apply_layered`。
- `tests/test_core.py`：加入 `test_layered_context_compresses_old_rounds`。

本地验证：

```bash
PYTHONPYCACHEPREFIX=/tmp/care-agent-pycache python3 -m compileall -q scripts tests
python3 -m unittest discover -s tests -v
git diff --check
```

实际输出：

```text
test_composite_checker ... ok
test_layered_context_compresses_old_rounds ... ok
test_task_set_is_balanced_and_disjoint ... ok
Ran 3 tests in 0.001s
OK
```

本地提交：

```text
7de4762 fix: make layered context actually compress history
2 files changed, 51 insertions(+), 10 deletions(-)
```

### 推送状态：`BLOCKED BY LOCAL NETWORK`

- 第一次推送返回：`Could not resolve host: github.com`。
- 后续推送未完成；本地 `main` 比 `origin/main` 领先 4 个待推送提交：layered v2 代码修复、实验日志与记录协议、推送状态纠正，以及服务器完整命令账本。
- 服务器仍在 `1f87b2a`，尚未获得 layered v2 修复。
- 等用户切换本地网络后，再推送并让服务器 `git pull --ff-only origin main`。

## 当前状态与下一步

### 已验证

- AutoDL、CUDA、PyTorch、vLLM 和 Qwen2.5-1.5B-Instruct 服务可用。
- OpenAI-compatible Function Calling 可返回结构化工具调用。
- 首轮 81 条轨迹已保留，且没有服务错误。
- layered v1 结果已忠实记录为无法支持正向结论。
- layered v2 修复在本地通过编译和 3 个单测。

### 未完成

1. 推送 `7de4762` 及本日志相关更新。
2. 服务器拉取修复，将旧 `data/test_layered.jsonl` 保留为 v1 证据，使用新文件运行 layered v2。
3. 对比 layered v2 的完成率与 token 峰值；如果仍为地板效应，先校准任务，不直接进入偏好合成。
4. 运行 train split 90 条基座轨迹并归因真实失败。
5. 配置强模型 API，生成并人工检查至少 40 对 preference pairs。
6. 创建 `care-train` 环境，运行 LLaMA-Factory LoRA-DPO、merge 和独立 test 评测。

### 下一次应记录的证据

- 推送后的远端 commit。
- 服务器 pull 前后 commit。
- layered v2 完整命令、输出文件、27 条行数、0 服务错误检查、完成率、分类完成率和 token 峰值。
- 是保留任务集还是进行难度校准的明确决策与依据。

## 附录 A：2026-09-02 服务器完整命令账本

以下清单根据本次仍在线的 SSH shell 中 `history` 的 69 条记录还原，并与 Agent 工具调用输出交叉核对。编号代表实际执行顺序。

登录命令在本地终端执行，不属于服务器 shell history：

```bash
ssh -p <AUTODL_PORT> root@<AUTODL_HOST>
```

AutoDL 端口和主机名是实例级动态连接信息，不写入公开仓库。SSH 密码从未作为 shell 命令执行，也不记录。

### A.1 初始环境检查与仓库克隆（history 1-11）

```bash
nvidia-smi
python3 --version
conda --version
git --version
df -h / /root/autodl-tmp
cd /root/autodl-tmp
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
ls -la
git clone https://github.com/Macavity17/agent-badcase-dpo.git
cd agent-badcase-dpo
git log -1 --oneline --decorate
```

### A.2 Conda 环境、依赖与服务器校验（history 12-21）

```bash
eval "$(conda shell.bash hook)"
conda create -n care-infer python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate care-infer
python --version
python -m pip install --upgrade pip
python -m pip install "openai>=1.40" "vllm>=0.6.0" "huggingface_hub[cli]>=0.24"
python -c 'import torch, vllm, openai; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0)); print("vllm", vllm.__version__, "openai", openai.__version__)'
python scripts/0_validate_tasks.py
python -m unittest discover -s tests -v
```

### A.3 模型下载与产物检查（history 22-25）

```bash
mkdir -p models runs
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/Qwen2.5-1.5B-Instruct
du -sh models/Qwen2.5-1.5B-Instruct
ls -lh models/Qwen2.5-1.5B-Instruct/*.safetensors
```

`hf download` 实际执行了两次：第一次在主权重下载后因 `RemoteProtocolError` 退出，第二次使用完全相同的命令续传成功。当前 shell 对连续重复命令进行了历史去重，所以 `history` 只显示一条；Agent 执行记录明确包含两次调用。

### A.4 vLLM 启动与第一轮就绪诊断（history 26-32）

```bash
nohup python3 -m vllm.entrypoints.openai.api_server --model ./models/Qwen2.5-1.5B-Instruct --served-model-name base --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes > runs/vllm_base.log 2>&1 &
echo $! > runs/vllm_base.pid
cat runs/vllm_base.pid
sleep 8
ps -p "$(cat runs/vllm_base.pid)" -o pid,stat,cmd
curl -sS http://localhost:8000/v1/models
tail -n 120 runs/vllm_base.log
```

此处 `curl` 返回 connection refused，后续 `tail` 证明服务正在加载模型和编译，不是进程退出。

### A.5 vLLM 第二轮就绪诊断（history 33-41）

```bash
sleep 20
curl -sS http://localhost:8000/v1/models
ps -p "$(cat runs/vllm_base.pid)" -o pid,stat,cmd
tail -n 120 runs/vllm_base.log
date
ss -ltnp | grep 8000 || true
ps -ef | grep -E 'vllm|EngineCore' | grep -v grep
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
curl -sS http://localhost:8000/v1/models
```

该轮关键输出：

- 两次 `curl localhost` 仍在服务就绪前返回 connection refused。
- `ss` 命令失败：`ss: command not found`。
- `ps` 显示 API server 和 `VLLM::EngineCore` 进程均存活。
- `nvidia-smi` 显示 EngineCore 已占用约 20,874 MiB GPU 内存。

### A.6 vLLM 第三轮诊断与成功健康检查（history 42-46）

```bash
sleep 20
tail -n 30 runs/vllm_base.log
ps -o pid,stat,pcpu,pmem,etime,wchan:30 -p 2617,2704
cat /proc/2617/net/tcp
curl -sS http://127.0.0.1:8000/v1/models
```

`/proc/2617/net/tcp` 中出现 `00000000:1F40`，`1F40` 为十六进制端口 8000。最后一条 `curl 127.0.0.1` 成功返回 `base` 模型。PID 是本次 run 的动态值，复现时应从 `runs/vllm_base.pid` 和实际子进程获取，不应复制固定 PID。

### A.7 冒烟测试与轨迹展开（history 47-48）

```bash
python3 scripts/1_run_baseline.py --split test --limit 1 --strategy full --repeats 1 --base-url http://127.0.0.1:8000/v1 --out data/smoke.jsonl --verbose
python -m json.tool data/smoke.jsonl
```

### A.8 81 条上下文对照、报告与完整性检查（history 49-53）

```bash
for strategy in full window layered; do
  python3 scripts/1_run_baseline.py --split test --strategy "$strategy" --repeats 3 --temperature 0.2 --seed 42 --workers 4 --base-url http://127.0.0.1:8000/v1 --out "data/test_${strategy}.jsonl" --resume
done
python3 scripts/5_evaluate.py --files full=data/test_full.jsonl,window=data/test_window.jsonl,layered=data/test_layered.jsonl --out results/context_compare.md
sed -n '1,220p' results/context_compare.md
wc -l data/test_full.jsonl data/test_window.jsonl data/test_layered.jsonl
grep -n '"type": "error"' data/test_*.jsonl || true
```

### A.9 badcase 结构化诊断（history 54-56）

```bash
command -v jq || true
python - <<'PY'
import json
from collections import Counter, defaultdict
for strategy in ('full','window','layered'):
    path=f'data/test_{strategy}.jsonl'
    rows=[json.loads(line) for line in open(path)]
    print('\n', strategy)
    by=defaultdict(list)
    for r in rows: by[r['task_id']].append(r)
    for task_id, group in sorted(by.items()):
        seqs=Counter('>'.join(c['name'] for c in r['tool_calls']) or '<none>' for r in group)
        print(task_id, f"ok={sum(r['success'] for r in group)}/3", dict(seqs))
PY
python - <<'PY'
import json
rows=[json.loads(line) for line in open('data/test_full.jsonl')]
for r in rows:
    print('\n', r['task_id'], r['repeat'], 'success=', r['success'])
    for c in r['tool_calls']:
        print(' ', c['name'], c['args'])
    print(' final:', r['final_answer'])
PY
```

`command -v jq` 没有输出，因此使用 Python 的 JSON 解析器展开 JSONL，没有安装额外系统包。

### A.10 环境、产物和服务证据补采（history 57-68）

```bash
hostname
uname -srmo
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
free -h
df -h /root/autodl-tmp
git rev-parse HEAD
python -c 'import sys,torch,vllm,openai,huggingface_hub; print(sys.version.split()[0]); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(vllm.__version__, openai.__version__, huggingface_hub.__version__)'
sha256sum models/Qwen2.5-1.5B-Instruct/model.safetensors
curl -sS http://127.0.0.1:8000/v1/models
wc -l data/test_full.jsonl data/test_window.jsonl data/test_layered.jsonl
grep -h -c '"type": "error"' data/test_full.jsonl data/test_window.jsonl data/test_layered.jsonl
grep -E 'version 0.28.0|Resolved architecture|Using max model len|Loading weights took|Model loading took|Available KV cache|GPU KV cache size|Graph capturing finished|Application startup complete|Uvicorn running' runs/vllm_base.log
```

### A.11 shell 历史完整性检查（history 69）

```bash
history
```

此命令的输出是附录 A 的原始依据。自此开始，后续每次服务器操作都必须在同一次实验日志更新中追加精确命令、参数、关键输出和状态。

## 2026-09-02 / Run 7：网络切换后重连、服务器同步与校验

### 时间与风险预评估

开始前将后续工作分为两类：

- 本次允许执行：重连、pull、代码/数据校验、vLLM 健康检查和 27 条 layered v2 采样，预计单项不超过 10 分钟。
- 本次不启动：强模型合成、LLaMA-Factory 安装、DPO 训练与 merge；这些步骤合计可能需要 30-90 分钟，并受网络下载影响。

断链预期：SSH 断开不会关闭 AutoDL 实例，实例仍保持运行和计费。`nohup ... &` 启动的 vLLM 应继续运行；普通前台采样或训练可能随终端断开而终止。

### 实际连接事件

切换本地网络后，旧 SSH 会话返回：

```text
Read from remote host: Connection reset by peer
client_loop: send disconnect: Broken pipe
```

旧会话中刚发送的 pull/校验命令没有到达服务器 shell，因此不计入服务器命令账本。重新建立 SSH 后，AutoDL 欢迎信息报告 15 CPU cores、80 GB RAM、RTX 4090 D x1 和 170 GB 数据盘。

### 服务器同步与校验命令

```bash
cd /root/autodl-tmp/agent-badcase-dpo
eval "$(conda shell.bash hook)"
conda activate care-infer
git status --short
git rev-parse HEAD
git pull --ff-only origin main
git rev-parse HEAD
python scripts/0_validate_tasks.py
python -m unittest discover -s tests -v
curl -sS http://127.0.0.1:8000/v1/models
```

### 关键输出

```text
git status --short: 无输出，tracked worktree clean
pull 前 HEAD: 1f87b2a62183806e2ec180be54e1bccb7cf2aefb
git pull: Fast-forward 1f87b2a..0454f39
pull 后 HEAD: 0454f39796d296ac05263a1c0221dc5ad4fcde31

任务总数: 24
train/test: 15/9
三类失效: 8/8/8
任务字段、checker 引用与场景族隔离: 通过

Ran 3 tests in 0.002s
OK

vLLM /v1/models: 成功返回 base
max_model_len: 8192
```

### 状态判断

- `SUCCESS`：服务器已部署 layered v2 修复和完整日志协议。
- `SUCCESS`：3 个单测在服务器通过。
- `SUCCESS`：vLLM 经历 SSH 断链后仍正常响应，后台服务不会因本地网络切换自动结束。
- 下一步仍限定为短任务：使用新文件名执行 layered v2，保留 layered v1 原始结果。

### A.12 重连后的服务器同步与校验

```bash
cd /root/autodl-tmp/agent-badcase-dpo
eval "$(conda shell.bash hook)"
conda activate care-infer
git status --short
git rev-parse HEAD
git pull --ff-only origin main
git rev-parse HEAD
python scripts/0_validate_tasks.py
python -m unittest discover -s tests -v
curl -sS http://127.0.0.1:8000/v1/models
```

## 2026-09-02 / Run 8：layered v2 重跑与暂停决策

### 运行前时间评估

- 27 条 layered v2 采样：预计 2-5 分钟，最坏预留 10 分钟。
- 报告、行数和服务错误检查：预计 1 分钟。
- 只读轨迹归因：预计 5 分钟以内。
- 如果仍需修改上下文协议、校准任务并重跑三策略：预计 30-60 分钟，达到本次暂停阈值，不在本轮启动。

### 采样与报告命令

旧 `data/test_layered.jsonl` 作为 v1 失败证据保留。v2 使用新文件名：

```bash
test ! -e data/test_layered_v2.jsonl && echo "new output path confirmed"
python3 scripts/1_run_baseline.py --split test --strategy layered --repeats 3 --temperature 0.2 --seed 42 --workers 4 --base-url http://127.0.0.1:8000/v1 --out data/test_layered_v2.jsonl
python3 scripts/5_evaluate.py --files full=data/test_full.jsonl,window=data/test_window.jsonl,layered_v2=data/test_layered_v2.jsonl --out results/context_compare_v2.md
wc -l data/test_full.jsonl data/test_window.jsonl data/test_layered_v2.jsonl
grep -n '"type": "error"' data/test_layered_v2.jsonl || true
sed -n '1,220p' results/context_compare_v2.md
```

### 完整性与总体结果

```text
new output path confirmed
full:       27 trajectories
window:     27 trajectories
layered_v2: 27 trajectories
layered_v2 service/API errors: 0
```

| 策略 | 完成率 | 平均步数 | 平均 prompt 峰值 |
|---|---:|---:|---:|
| full | 11.1% (3/27) | 3.67 | 864 tokens |
| window | 0.0% (0/27) | 3.74 | 868 tokens |
| layered v1 | 3.7% (1/27) | 3.74 | 1,107 tokens |
| layered v2 | 0.0% (0/27) | 2.96 | 894 tokens |

layered v2 分类别完成率：

```text
tool_misuse:       0/9
context_forgetting: 0/9
planning_drift:    0/9
```

### 结果判断：`INVALID FOR POSITIVE CONTEXT CLAIM`

修复达成了部分工程目标：平均 prompt 峰值从 v1 的 1,107 降到 894，接近 window 的 868，证明 v1 的重复上下文缺陷已经消除。但完成率下降为 0%，平均步骤降至 2.96，说明模型更频繁地提前终止。因此 v2 仍不能支持 structured/layered context 改善完成率的结论。

### 只读 badcase 归因命令

```bash
python - <<'PY'
import json
from collections import Counter, defaultdict
rows=[json.loads(line) for line in open('data/test_layered_v2.jsonl')]
by=defaultdict(list)
for row in rows:
    by[row['task_id']].append(row)
for task_id, group in sorted(by.items()):
    sequences=Counter('>'.join(call['name'] for call in row['tool_calls']) or '<none>' for row in group)
    print(task_id, f"ok={sum(row['success'] for row in group)}/3", f"avg_steps={sum(row['n_steps'] for row in group)/3:.2f}", dict(sequences))
PY
python - <<'PY'
import json
rows=[json.loads(line) for line in open('data/test_layered_v2.jsonl')]
for row in rows:
    print('\n', row['task_id'], row['repeat'], 'steps=', row['n_steps'], 'tokens=', row['max_prompt_tokens'])
    print(' tools:', [(call['name'], call['args']) for call in row['tool_calls']])
    print(' final:', row['final_answer'][:500].replace('\n', ' '))
PY
```

### 按任务诊断

| 任务 | v2 的稳定行为 | 主要失败 |
|---|---|---|
| `cf_test_001` | 3/3 只调用 `get_caregiver_permissions` | 用文本描述“接下来读取记录/时段”，但不继续调用工具 |
| `cf_test_002` | 3/3 读偏好、学习历史并搜索材料 | 搜索后用文本描述将要发送，不调用 `send_voice_resource` |
| `cf_test_003` | 3/3 完成三个读取 | 不创建提醒；appointment ID/时间仍含占位或错误值 |
| `pd_test_001` | 2/3 只读来源，1/3 做到人工复测 | 未闭环到 `create_care_review` |
| `pd_test_002` | 3/3 直接升级 | 未读问卷/档案，未调用 `log_safety_event` |
| `pd_test_003` | 3/3 调用五个预期工具 | 工具参数使用虚构 case ID，最终答案未带真实 `ESC-924`/`SFU-924` |
| `tm_test_001` | 3/3 直接配对 | 遗漏前置设备状态读取 |
| `tm_test_002` | 3/3 直接升级 | 遗漏留言读取，priority 为 `high` 而非 `urgent` |
| `tm_test_003` | 3/3 只查回电时段 | 用文本描述下一步，不调用 `create_callback_request` |

核心失效不是 token 开销，而是模型在 tool observation 后把计划写成 final answer。layered v2 的 `system + state-as-user + latest assistant/tool` 组装虽然压缩了历史，但没有让 1.5B 模型稳定继续执行；当前 SYSTEM 中已有“达成目标后才最终作答”，仍不足以约束该模型。

### 暂停决策

下一步不能只改一个词后立即宣称成功，至少需要：

1. 审计 layered 的消息角色和“未完成子目标”表达；
2. 识别初始输入缺失必要 ID、工具 schema 要求与 checker 不一致的任务；
3. 先用小 probe 校准 full 到约 30-50%；
4. 所有策略使用同一校准任务集重新运行，不能只挑 layered 重跑；
5. 在获得非地板基线前，不进入 preference synthesis 或 DPO。

预计该链路需要 30-60 分钟，因此按用户的断链约束在此停止服务器实验。vLLM 继续以 nohup 后台运行；AutoDL 实例不会因 SSH 断开自动关闭，仍会运行和计费。生成的 v1/v2 JSONL 与报告保存在 `/root/autodl-tmp/agent-badcase-dpo/data` 和 `results` 中。

### A.13 layered v2 采样、评测与完整性检查

```bash
test ! -e data/test_layered_v2.jsonl && echo "new output path confirmed"
python3 scripts/1_run_baseline.py --split test --strategy layered --repeats 3 --temperature 0.2 --seed 42 --workers 4 --base-url http://127.0.0.1:8000/v1 --out data/test_layered_v2.jsonl
python3 scripts/5_evaluate.py --files full=data/test_full.jsonl,window=data/test_window.jsonl,layered_v2=data/test_layered_v2.jsonl --out results/context_compare_v2.md
wc -l data/test_full.jsonl data/test_window.jsonl data/test_layered_v2.jsonl
grep -n '"type": "error"' data/test_layered_v2.jsonl || true
sed -n '1,220p' results/context_compare_v2.md
```

### A.14 layered v2 轨迹诊断

```bash
python - <<'PY'
import json
from collections import Counter, defaultdict
rows=[json.loads(line) for line in open('data/test_layered_v2.jsonl')]
by=defaultdict(list)
for row in rows:
    by[row['task_id']].append(row)
for task_id, group in sorted(by.items()):
    sequences=Counter('>'.join(call['name'] for call in row['tool_calls']) or '<none>' for row in group)
    print(task_id, f"ok={sum(row['success'] for row in group)}/3", f"avg_steps={sum(row['n_steps'] for row in group)/3:.2f}", dict(sequences))
PY
python - <<'PY'
import json
rows=[json.loads(line) for line in open('data/test_layered_v2.jsonl')]
for row in rows:
    print('\n', row['task_id'], row['repeat'], 'steps=', row['n_steps'], 'tokens=', row['max_prompt_tokens'])
    print(' tools:', [(call['name'], call['args']) for call in row['tool_calls']])
    print(' final:', row['final_answer'][:500].replace('\n', ' '))
PY
```

### A.15 主动结束本次 SSH 会话

```bash
exit
```

实际输出：

```text
logout
Connection closed.
```

该命令只关闭交互式 SSH。AutoDL 实例仍运行和计费，使用 `nohup ... &` 启动的 vLLM 不随该会话退出。

## 2026-09-02 / Run 9：协议修复、train-only 校准与冻结 holdout

### 修复决策

前两轮数据显示的主要问题是“模型在工具观测后直接输出文本，没有继续工具循环”，且多个任务的 ID 不充分。本轮实施：

- 加入共享 `finish_task` 控制工具；完成前 `tool_choice=required`，完成后 `tool_choice=none` 产生最终答复。
- 强制串行工具调用，`parallel_tool_calls=False`。
- `finish_task` 不计入业务工具数。
- layered 保留最近两轮原始 assistant/tool，早期历史压缩为状态。
- 为欠规定任务补充具体事件 ID 和可追溯 mock 值。

先前已人工查看的 9 个 test 任务移入 `tasks/dev_tasks.jsonl`。新建 9 个患者/事件 ID 不重复的 test 任务作为最终 holdout；此后没有根据 holdout 表现修改任务或 checker。

### 验证与校准结果

关键命令（代码变更通过 `git apply - <<'PATCH' ... PATCH` 完成）：

```bash
python3 scripts/0_validate_tasks.py
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py
git diff --check
```

结果：24 条任务、15 train / 9 test、三类失效各 8 条；dev/holdout 患者 ID 无交集；5 个单测全部通过。

train-only 小样本校准每任务跑 1 次，失败 run 使用不同文件名保留。最终同一版任务的 v3 结果：

| 策略 | 成功 | 平均 prompt 峰值 |
|---|---:|---:|
| full | 4/15 (26.7%) | 1,201 |
| window | 4/15 (26.7%) | 1,048 |
| layered | 5/15 (33.3%) | 1,182 |

冻结 holdout 随后每策略跑 9 任务 x 3 次，无服务错误：

| 策略 | 成功 | 平均 prompt 峰值 |
|---|---:|---:|
| full | 4/27 (14.8%) | 1,234 |
| window | 0/27 (0.0%) | 1,068 |
| layered | 3/27 (11.1%) | 1,205 |

分类为：full `0/9, 1/9, 3/9`，window `0/9, 0/9, 0/9`，layered `1/9, 2/9, 0/9`（顺序均为 tool misuse / context forgetting / planning drift）。window 节省约 13.5% 峰值 token 但完成率归零；layered 只节省约 2.4% 且不超过 full。结论标记为 `VALID NEGATIVE RESULT`：不支持正向上下文收益，但显示了截断与状态压缩的干预边界。

一次评测命令错把多文件参数写成 `--inputs`，脚本拒绝该参数；正确参数为 `--files`。错误未影响 JSONL 轨迹。

## 2026-09-02 / Run 10：train-only 归因、canonical 偏好数据与导出

### 训练轨迹与归因

`full` 在 15 个 train 任务上每任务采样 6 次，得到 90 条轨迹：22 成功、68 失败、0 服务错误。

```bash
python3 scripts/2_attribute.py --traj data/train_base_full_r6.jsonl --tasks tasks/tasks.jsonl --out data/train_badcases_labeled.jsonl
```

归因分布：

```text
planning_drift:     32
context_forgetting: 23
tool_misuse:        13
```

### canonical chosen 构造

首个 `data/pref_pairs_canonical.jsonl` 虽通过基础校验，但人工查看发现部分参数存在“工具返回的...”式占位文本，因此拒绝作为训练数据并保留文件。修复参数从 goal/checker/mock 取值、显式加入 `finish_task` 后，生成 v2：

```bash
python3 scripts/3_build_preference.py --synth-mode canonical --badcase data/train_badcases_labeled.jsonl --tasks tasks/tasks.jsonl --out data/pref_pairs_canonical.jsonl --workers 4
python3 scripts/3_build_preference.py --synth-mode canonical --badcase data/train_badcases_labeled.jsonl --tasks tasks/tasks.jsonl --out data/pref_pairs_canonical_v2.jsonl --workers 4
python3 scripts/4_to_llamafactory.py --pref data/pref_pairs_canonical_v2.jsonl --outdir data/lf_data
cat data/lf_data/stat.json
```

v2 输出 68 对，0 丢弃，0 chosen/rejected 相同。68/68 chosen 包含 `finish_task`，且每条同时通过协议和任务 checker。LLaMA-Factory 导出为 `data/lf_data/agent_pref.json`、`dataset_info.json`和 `stat.json`。正确定性描述是“规则约束 canonical 合成 chosen”，不是强模型或人工标注。

## 2026-09-02 / Run 11：LLaMA-Factory 环境修复与 LoRA-DPO

### 第一次安装：`FAILED / INTERRUPTED`

初始使用系统盘 named env 和 `[torch,metrics]` extra，pip 开始以约 270 KB/s 下载 526.6 MB Torch wheel。这不符合两晚时间约束，也解释了 AutoDL 数据盘曲线基本不动：当时写入的是 `/root/miniconda3/envs` 和 `/root/.cache/pip`，而非 `/root/autodl-tmp`。下载被中断，88 MB 失败环境后续删除。

相关命令：

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git vendor/LLaMA-Factory
conda create -n care-train python=3.11 -y
conda run -n care-train python -m pip install --upgrade pip
conda run -n care-train python -m pip install -e './vendor/LLaMA-Factory[torch,metrics]'
pgrep -af 'pip install|conda run|LLaMA-Factory'
ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd --sort=-%cpu | head -25
du -sh /root/miniconda3/envs/care-train /root/.cache/pip 2>/dev/null
df -h / /root/autodl-tmp
ss -tpn | grep -E 'python|pip|conda' || true
conda remove -n care-train --all -y
```

### 数据盘环境：`SUCCESS`

将已验证 CUDA 可用的 `care-infer` 克隆到数据盘，再只安装 `[metrics]`：

```bash
conda create --prefix /root/autodl-tmp/envs/care-train --clone /root/miniconda3/envs/care-infer -y
du -sh /root/autodl-tmp/envs/care-train
conda run --no-capture-output -p /root/autodl-tmp/envs/care-train python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
grep -n -A25 '^\[project.optional-dependencies\]' vendor/LLaMA-Factory/pyproject.toml
conda run --no-capture-output -p /root/autodl-tmp/envs/care-train python -m pip install -e './vendor/LLaMA-Factory[metrics]'
du -sh /root/autodl-tmp/envs/care-train
conda run --no-capture-output -p /root/autodl-tmp/envs/care-train llamafactory-cli --help
conda run --no-capture-output -p /root/autodl-tmp/envs/care-train llamafactory-cli version
conda run --no-capture-output -p /root/autodl-tmp/envs/care-train python -c 'import torch, transformers, accelerate, peft, trl; print("torch", torch.__version__); print("cuda", torch.cuda.is_available(), torch.version.cuda); print("transformers", transformers.__version__); print("accelerate", accelerate.__version__); print("peft", peft.__version__); print("trl", trl.__version__)'
git -C vendor/LLaMA-Factory rev-parse HEAD
```

克隆后 8.2 GB，安装后 9.1 GB，路径均为 `/root/autodl-tmp/envs/care-train`。版本：Torch `2.13.0+cu130`、CUDA available `True`、Transformers `5.8.0`、Accelerate `1.11.0`、PEFT `0.18.1`、TRL `0.24.0`、LLaMA-Factory `0.9.6.dev0`，Git commit `4451765a6b04ff08a6c5650f5953513608ae9e64`。`llamafactory-cli --help` 输出 `Unknown command: --help` 但同时显示了完整用法；正确版本命令是 `llamafactory-cli version`。

pip 将 `starlette` 降为 0.52.1，与克隆进来的 vLLM 0.28.0 需求冲突。该环境被限定为训练用，推理仍由原 `care-infer` 承担，因此不混用两种角色。

### 训练与 merge

```bash
ls -la runs
cat runs/vllm_base.pid
ps -fp 2617
kill 2617
ps -fp 2617
nvidia-smi
nohup conda run --no-capture-output -p /root/autodl-tmp/envs/care-train llamafactory-cli train config/dpo_qwen15b.yaml > runs/dpo_train.log 2>&1 &
tail -80 runs/dpo_train.log
ps -fp 7831
tail -120 runs/dpo_train.log
tail -80 runs/dpo_train.log
tail -120 runs/dpo_train.log
conda run --no-capture-output -p /root/autodl-tmp/envs/care-train llamafactory-cli export config/merge_lora.yaml
```

基座 vLLM 停止后显存为 0 MiB。训练使用 61 train / 7 eval、3 epoch、24 optimization steps、18,464,768 可训参数（1.1820%）。总训练时间 44.46 秒，train loss 0.6239。最佳 checkpoint 为 step 20，最终 eval loss 0.5857、reward accuracy 1.0、chosen reward 0.1611、rejected reward -0.0558、margin 0.217。merge 成功输出 `outputs/dpo_merged/model.safetensors`。

## 2026-09-03 / Run 12：冻结 holdout DPO 评测、负结果归因与备份

### 服务启动中的两个失败

第一次使用不存在的数据盘 `care-infer` 前缀：

```bash
nohup conda run --no-capture-output -p /root/autodl-tmp/envs/care-infer python -m vllm.entrypoints.openai.api_server --model ./outputs/dpo_merged --served-model-name dpo --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes > runs/vllm_dpo.log 2>&1 &
echo 8191 > runs/vllm_dpo.pid
tail -120 runs/vllm_dpo.log
conda env list
```

失败为 `EnvironmentLocationNotFound: /root/autodl-tmp/envs/care-infer`，模型未加载。正确路径是 `/root/miniconda3/envs/care-infer`：

```bash
nohup conda run --no-capture-output -p /root/miniconda3/envs/care-infer python -m vllm.entrypoints.openai.api_server --model ./outputs/dpo_merged --served-model-name dpo --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes > runs/vllm_dpo.log 2>&1 &
echo 8220 > runs/vllm_dpo.pid
tail -80 runs/vllm_dpo.log
tail -60 runs/vllm_dpo.log
curl -s http://localhost:8001/v1/models
```

vLLM 0.28.0 在 8001 成功提供 `dpo`，`max_model_len=8192`。第一次评测直接用 shell 的 `python3`，失败为 `ModuleNotFoundError: No module named 'openai'`：

```bash
python3 scripts/1_run_baseline.py --split test --strategy full --repeats 3 --temperature 0.2 --seed 42 --workers 4 --port 8001 --model dpo --out data/holdout_dpo_full.jsonl --resume
```

改用推理环境 Python 后完成 27 条，但随后发现它与 base holdout 的 seed 不同，因此 3/27 只作为 `INVALID FOR BEFORE/AFTER CONCLUSION`，文件保留：

```bash
/root/miniconda3/envs/care-infer/bin/python scripts/1_run_baseline.py --split test --strategy full --repeats 3 --temperature 0.2 --seed 42 --workers 4 --port 8001 --model dpo --out data/holdout_dpo_full.jsonl --resume
/root/miniconda3/envs/care-infer/bin/python scripts/5_evaluate.py -h
/root/miniconda3/envs/care-infer/bin/python scripts/5_evaluate.py --before data/holdout_base_full.jsonl --after data/holdout_dpo_full.jsonl --out results/dpo_compare.md
```

### 同 seed 最终结果

```bash
/root/miniconda3/envs/care-infer/bin/python scripts/1_run_baseline.py --split test --strategy full --repeats 3 --temperature 0.2 --seed 20260902 --workers 4 --port 8001 --model dpo --out data/holdout_dpo_full_seed20260902.jsonl --resume
/root/miniconda3/envs/care-infer/bin/python scripts/5_evaluate.py --before data/holdout_base_full.jsonl --after data/holdout_dpo_full_seed20260902.jsonl --out results/dpo_compare_seed20260902.md
```

| 指标 | Base | DPO | 变化 |
|---|---:|---:|---:|
| 任务完成率 | 14.8% (4/27) | 7.4% (2/27) | -7.4pp |
| 平均步数 | 5.81 | 5.78 | -0.04 |
| 工具调用率 | 100% | 100% | 0pp |
| 工具协议错误 | 0% | 0% | 0pp |
| 平均 prompt 峰值 | 1,234 | 1,230 | -4 |

分类为tool misuse `0/9 -> 0/9`，context forgetting `1/9 -> 1/9`，planning drift `3/9 -> 1/9`。27/27 seed 对齐；21/27 轨迹的完整 tool call 相同；base/DPO 均 27/27 调用 `finish_task`，均没有重复工具名轨迹。

差异轨迹显示 DPO 产生了语义参数退化：`days="2"` 改为 `连续两天/连续两次`，`week="next_week"` 改为 `next`。同时也有一个局部改善：设备任务中 DPO 没有像 base 那样误调被禁止的 `sync_measurements`，但该轨迹仍因结果回传不完整而失败。两条原本成功的 planning drift 轨迹工具序列未变，但 DPO 最终答复漏掉 `ESC-974`/`SFU-974`，因此 checker 失败。

为区分训练不足与泛化失败，在 train 任务上以原 90 条的 seed/温度又评测 DPO：

```bash
/root/miniconda3/envs/care-infer/bin/python scripts/1_run_baseline.py --split train --strategy full --repeats 6 --temperature 0.7 --seed 4242 --workers 4 --port 8001 --model dpo --out data/train_dpo_full_r6_seed4242.jsonl --resume
/root/miniconda3/envs/care-infer/bin/python scripts/5_evaluate.py --before data/train_base_full_r6.jsonl --after data/train_dpo_full_r6_seed4242.jsonl --out results/dpo_train_compare_seed4242.md
```

base/DPO 均为 22/90（24.4%）；context forgetting `23.3% -> 26.7%`，planning drift `3.3% -> 0%`，tool misuse 维持 46.7%。因此最终结论是：训练内 reward 成功分离，但没有转化为任务完成率，holdout 还出现退化。这不支持 DPO 正向提升声明。

### 停止服务与证据包

`nohup conda run ...` 的 PID 8220 是 wrapper，只 `kill 8220` 后子进程仍占用显存；随后定位 API PID 8229 并停止，显存才归零。

```bash
ps -fp 8220
kill 8220
nvidia-smi
ps -fp 8274
ps -fp 8229
kill 8229
nvidia-smi
du -sh data results runs outputs/dpo-qwen15b outputs/dpo_merged
find outputs/dpo-qwen15b -maxdepth 1 -type f -printf '%f %s bytes\n'
ls -lh /root/autodl-tmp/care-agent-evidence-20260903.tar.gz
tar -czf /root/autodl-tmp/care-agent-evidence-20260903.tar.gz data results runs config tasks README.md
ls -lh /root/autodl-tmp/care-agent-evidence-20260903.tar.gz
sha256sum /root/autodl-tmp/care-agent-evidence-20260903.tar.gz
```

证据包：`/root/autodl-tmp/care-agent-evidence-20260903.tar.gz`，215 KB，SHA-256 `e9ad0708ce2b2764053aa0a37e598fd4c5ccc0d08d8a9bf2dae109a6dfc88347`。`outputs/dpo-qwen15b` 为 527 MB，`outputs/dpo_merged` 为 2.9 GB，因可由配置重现而未放入这个轻量证据包。

### A.16 补充 shell 诊断与历史检查命令

除上述主流程外，本轮还执行了以下只读诊断/检查命令，均未修改实验数据：

```bash
sed -n '1,130p' scripts/4_to_llamafactory.py
sed -n '1,240p' config/dpo_qwen15b.yaml
sed -n '1,180p' config/merge_lora.yaml
conda env list
command -v llamafactory-cli || true
conda run -n care-train python --version
conda run -n care-train sh -lc 'command -v llamafactory-cli || true'
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
grep -n -A35 -B10 'care-train\|LLaMA-Factory\|llamafactory' docs/EXPERIMENT_GUIDE.md | head -260
find vendor -maxdepth 2 -type d 2>/dev/null | head -20
conda run --no-capture-output -n care-train python -m pip show torch llamafactory transformers accelerate peft
conda run --no-capture-output -n care-train python -c 'import sys; print(sys.version)'
conda run --no-capture-output -n care-train python -m pip cache info
conda run --no-capture-output -n care-train python -m pip cache list | grep -E 'torch|nvidia|transformers' | tail -40
du -sh /root/miniconda3/envs/care-infer /root/miniconda3/envs/care-train
conda run --no-capture-output -n care-infer python -m pip show torch transformers accelerate peft | grep -E '^(Name|Version):'
conda run --no-capture-output -n care-infer python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
find vendor/LLaMA-Factory/examples -path '*dpo*' -name '*.yaml' | head -20
sed -n '1,220p' vendor/LLaMA-Factory/examples/train_lora/qwen3_lora_dpo.yaml
ls -lh data
head -1 data/holdout_dpo_full.jsonl
head -1 data/train_base_full_r6.jsonl
sed -n '1,240p' results/holdout_context_compare.md
history
fc -l 1700 1918
fc -l 1300 1600
fc -l 850 1040
history | grep -E 'calibration_|holdout_base_|train_base_full_r6|2_attribute.py|pref_pairs_canonical|holdout_context_compare' | tail -120
```

`history` 输出因 heredoc patch 被展开成大量 diff 行，终端显示有截断；本日志依据未截断的命令尾部、实际产物和日志输出记录命令与结果，没有重写失败历史。

## 2026-09-03 / Run 13：本地结果文档与评测报告收口

更新 `README.md`、`docs/EXPERIMENT_GUIDE.md`和本地 `PROJECT_MEMORY.md`，用观测结果替换所有“待运行”占位，并明确 canonical chosen 的真实来源。同时修复 `scripts/5_evaluate.py` 仍生成“跑完再写”占位结论的问题：报告现在根据完成率、分失效模式和协议错误自动生成保守结论，并显式区分协议正确与任务完成。

本地验证命令：

```bash
python3 scripts/0_validate_tasks.py
PYTHONPYCACHEPREFIX=/tmp/care-agent-pycache python3 -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/care-agent-pycache python3 -m py_compile scripts/*.py tests/*.py
git diff --check
```

实际结果：24 条任务校验通过；6 个单测全部通过；Python 编译通过；`git diff --check` 无输出。本轮未 commit、未 push，也没有重启 AutoDL GPU 服务。

最后执行 `exit`，输出 `logout` 和 `Connection ... closed`，主动关闭空闲 SSH 会话。该操作不会关闭 AutoDL 实例；实例仍需在 AutoDL 控制台手动关机才停止计费。

## 2026-09-03 / Run 14：本地提交与 GitHub 同步

提交前重新执行任务校验、6 个单测、Python 编译和 `git diff --check`，全部通过。Git 作者邮箱确认为 `paxonw@163.com`。提交：

```text
a3eb5a5 feat: complete agent failure pilot
10 files changed, 756 insertions(+), 145 deletions(-)
```

首次 `git push origin main` 没有更新远程跟踪分支；使用 `git push --verbose origin main` 重试后得到明确失败：

```text
Failed to connect to github.com port 443 after 75013 ms
```

本地 Git 和系统均无代理配置。`curl -I --connect-timeout 10 https://github.com` 随后返回 HTTP 200，说明是短时网络波动。再次执行 verbose push 成功：

```text
To https://github.com/Macavity17/agent-badcase-dpo.git
   0454f39..a3eb5a5  main -> main
```

本次 GitHub 同步包含先前未推送的 2 个日志提交和本轮完整实验提交。`AGENTS.md`、`PROJECT_MEMORY.md`、本地凭据、AutoDL 原始 JSONL、模型和大型运行产物未进入 Git。

## 2026-09-03 / Run 15：服务器工作区核验与三端代码对齐

用户要求明确服务器实验代码、Mac 本地仓库和 GitHub 的关系，并要求直接完成安全的服务器对齐。该轮不使用真实凭据记录；SSH 连接参数为实例动态信息，未写入本日志。

### 核验结论

服务器原先的 Git `HEAD` 为 `0454f39`，但确实保留了用于最终实验的未提交工作区改动，而非用旧代码完成训练。逐文件内容哈希证明，服务器工作区中的以下实验核心文件与 GitHub 最终版完全一致：

- `scripts/1_run_baseline.py`: `c046e9c613b21dda05e9456dd4dec35f97052029`
- `scripts/3_build_preference.py`: `4957134d86ae262b76ee90ab34e9ea609a492a74`
- `scripts/4_to_llamafactory.py`: `94c6bedaeffb425eba275141a94284a30d5d0ca4`
- `tasks/tasks.jsonl`: `d4440c433ad447a4c2a5831e7ea704386f9db61a`
- `tasks/dev_tasks.jsonl`: `4e01396b6bb24abaf75194126b7f34908db61591`

服务器的 `tests/test_core.py` 仍是较早的 `84f28b3899cdf7db758699c1e045f86f949084c2`，最终版为 `9307ae672371af132f135280a85cdad99e069501`；README、实验指南、完整实验日志和 `scripts/5_evaluate.py` 也尚未同步到服务器。因此结论是：服务器直接运行了最终的核心实验代码，随后 Mac 将完整代码、补充测试和结果文档提交为 `a3eb5a5`，并继续提交同步记录为 `956780e`。

### 服务器完整命令账本

以下为本轮在服务器执行的全部 shell 命令，按执行顺序记录。第一组仅检查状态：

```bash
cd /root/autodl-tmp/agent-badcase-dpo
git status --short
git rev-parse HEAD
git log --oneline -5
git diff --stat
git diff --name-status
git diff --check
```

结果：tracked 改动为 `scripts/1_run_baseline.py`、`scripts/3_build_preference.py`、`scripts/4_to_llamafactory.py`、`tasks/tasks.jsonl`、`tests/test_core.py`，未跟踪项为 `tasks/dev_tasks.jsonl` 和 `vendor/`；HEAD 为 `0454f39`。`git diff --check` 无输出。

随后拉取远程引用并比较对象，不修改工作区：

```bash
git fetch origin main
git rev-parse origin/main
git hash-object scripts/1_run_baseline.py scripts/3_build_preference.py scripts/4_to_llamafactory.py tasks/tasks.jsonl tests/test_core.py tasks/dev_tasks.jsonl
git ls-tree origin/main -- scripts/1_run_baseline.py scripts/3_build_preference.py scripts/4_to_llamafactory.py tasks/tasks.jsonl tests/test_core.py tasks/dev_tasks.jsonl scripts/5_evaluate.py README.md docs/EXPERIMENT_GUIDE.md
git diff --stat origin/main
```

`origin/main` 更新到 `956780eaa433c3710c83f2aaaf05d029db58588a`。上述哈希比较形成了前述结论；相对于最终版的其余差异为 README、指南、实验日志、`scripts/5_evaluate.py` 和补充测试。

在覆盖受 Git 管理的工作区前，先将服务器 tracked diff 导出到数据盘。该补丁不含模型、JSONL、训练输出、证据包或未跟踪 `vendor/`：

```bash
test ! -e /root/autodl-tmp/server-worktree-before-sync-20260903.patch
git diff --binary > /root/autodl-tmp/server-worktree-before-sync-20260903.patch
sha256sum /root/autodl-tmp/server-worktree-before-sync-20260903.patch
git reset --mixed origin/main
git restore --source=origin/main --worktree .
git rev-parse HEAD
git status --short
git diff --check
python3 scripts/0_validate_tasks.py
python3 -m unittest discover -s tests -v
```

补丁 SHA-256 为 `8337a826c7b3c7fc3619b18afc90ef3eba1a2e644cbe9897330878ff365e2bad`。`reset` 和 `restore` 后服务器 HEAD 成功成为 `956780e`，tracked 工作区干净，仅 `vendor/` 仍未跟踪。最后两条验证命令未运行成功，原因是默认服务器 shell 的 `python3` 不存在（`bash: python3: command not found`）；这是环境路径问题，不是代码或测试失败。

改用已验证的推理 Conda 环境完成验证：

```bash
/root/miniconda3/envs/care-infer/bin/python --version
/root/miniconda3/envs/care-infer/bin/python scripts/0_validate_tasks.py
/root/miniconda3/envs/care-infer/bin/python -m unittest discover -s tests -v
git diff --check
git status --short
git rev-parse HEAD
```

输出：Python `3.11.16`；24 个任务通过字段、checker 引用和场景族隔离校验；6 个单测全部通过；`git diff --check` 无输出；状态仅为 `?? vendor/`；HEAD 为 `956780eaa433c3710c83f2aaaf05d029db58588a`。至此，Mac、GitHub 与服务器的受 Git 管理代码对齐；服务器独有的模型、训练输出、原始 JSONL、证据包和 LLaMA-Factory `vendor/` 目录仍保留在服务器。
