# 实验日志

Last updated: 2026-09-02 (Asia/Shanghai)

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
