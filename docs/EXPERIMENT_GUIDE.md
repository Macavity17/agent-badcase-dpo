# 实验手册：上下文策略 × 训练矫正——谁能修好 Agent 的哪类失效

> 这不是一份"照着敲就能出结果"的操作文档。操作步骤我会写清楚，但**每一个关键决策点我都埋了思考题（标 T\*.\*）**——先自己想、写下你的答案，再去看 `ANSWERS.md` 对参考答案。做完实验你收获的应该不只是几组数字，而是"我能独立设计并执行一个严谨对照实验"的手感。
>
> **实验正式名称**：Context Strategy vs. Training-based Correction: An Empirical Study on Agent Failure Modes

---

## 0. 实验总览：先想清楚你要回答什么

### 0.1 核心问题

**同一个基座模型下，Agent 的三类典型失效（工具误选 / 上下文遗忘 / 规划发散），分别该用"改上下文组织"还是"改训练"来修？边界在哪？**

这个问题不是凭空的——目标岗位 JD 原文写着：

> 同一个模型，上下文组织方式不同，长任务的完成率会有数倍差距……核心工作是智能体运行策略，包括：在有限的上下文窗口内完成信息的分层、筛选与压缩；在长程任务中控制记忆的写入、召回与失效……

你要做的，就是把这句话**从论断变成你自己的实验证据**，并且比它多走一步——量化"上下文策略治不了的部分"。

### 0.2 三个假设（写下来，实验就是去验证/推翻它们）

- **H1**：上下文组织策略（尤其分层结构化）对**上下文遗忘类**失效有大幅改善，对工具误选类收效甚微。
- **H2**：DPO 训练对**工具误选类**失效有效（行为矫正），但对上下文遗忘类可能无效甚至恶化。
- **H3**：两类干预手段近似正交——最优解是组合使用，而不是二选一。

**✏️ T0.1（实验前必做，5 分钟）**：在跑任何东西之前，写下你自己的预测矩阵：

| | tool_misuse | context_forgetting | planning_drift |
|---|---|---|---|
| full（基线） | ?% | ?% | ?% |
| window | ?% | ?% | ?% |
| summary | ?% | ?% | ?% |
| layered | ?% | ?% | ?% |
| DPO | ?% | ?% | ?% |

**先猜再验证**——猜错了比猜对了收获更大，因为错误的预测暴露的是你心智模型里的 bug。把这张表存好，实验结束后逐格对答案。

### 0.3 实验全景图

```
Phase 0  任务集与难度校准        ── 确保地基没问题（半天内）
Phase 1  上下文策略对照实验      ── 你的主场，四种策略 × 同一任务集（周末白天）
Phase 2  归因 + 偏好对构造      ── 失败轨迹变成训练数据（周日晚）
Phase 3  DPO 训练与验证         ── 防守目标：跑通 + 看懂训练内部信号（周日晚-入职周）
Phase 4  综合分析与写作         ── 把数字变成结论（入职周晚上，纯分析活）
```

---

## 0.5 开跑前必须搞清的六个前提

> 这一节是后来补的——你问的六个问题，答案都在这里。

### Q1 AutoDL 国内能直接访问吗？

**能。** autodl.com 是国内 GPU 算力平台（中文界面、学生认证、可开发票），不需要任何特殊网络工具。注册 → 实名 → 充值即可用。

### Q2 关掉浏览器 / 断网后，任务还会继续跑吗？

分两层，别混淆：

- **实例本身**：它是独立的云主机，跟你本地浏览器、SSH 连接完全无关——你关机睡觉，实例照跑照计费。
- **你在 SSH 终端里前台跑的命令**：SSH 一断，进程就被挂断信号杀掉。所以长任务必须守护：
  ```bash
  # 方式 A：nohup + 日志重定向（手册里 vLLM / 训练都用这个）
  nohup python xxx.py > run.log 2>&1 &
  tail -f run.log                     # 随时回来看进度

  # 方式 B：tmux（推荐，可随时回来交互）
  tmux new -s exp
  python xxx.py                       # Ctrl+B 然后 D 脱离；tmux attach -t exp 回来
  ```
  另外：JupyterLab 里跑的程序**不受关闭浏览器影响**（官方 FAQ 明确），但日志必须重定向到文件才留得住。

- **关机与数据**：用完在控制台关机即停计费；**关机后数据保留 15 天不收费**，之后自动释放且**不可找回**。
- ⚠️ 实例本地盘是物理盘、无冗余备份，有丢失可能——**结果文件务必及时拉回本地或推 git**（见 §1 第 6 条）。

### Q3 手册里是全部代码吗？

**不是**，手册是操作 + 思考指南；代码在 `scripts/` 下（6 个文件，五步流水线骨架，已通过编译与单元测试）。三处需要你动手：

1. `tasks/tasks.jsonl`：现在只有 12 条样例，**你要扩到 36–48 条**（规范见 `tasks/schema.md`）
2. 强模型 API：合成 chosen / judge 用的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`
3. `config/merge_lora.yaml`：合并 LoRA 权重用的配置，模板在 `config/dpo_qwen15b.yaml` 末尾注释里，复制出来另存即可

### Q4 数据从哪来？（本实验最大的软肋，务必看清楚）

分四层，性质完全不同：

| 数据 | 来源 | 性质 |
|---|---|---|
| 任务与工具 | 我写的 12 条样例 + 你扩写 | **自建**，存在设计偏差 |
| 工具返回值 | `tasks.jsonl` 里的 `mock_responses` | **假数据**（我编的航班/酒店/价格） |
| 失败轨迹 | 模型真实跑出来的 | **真实行为数据**（实验的实证基础） |
| chosen 修正轨迹 | 强模型合成 | 合成数据（须过 checker 才保留） |

**诚实的定位**：任务环境是模拟的，但**模型行为是真的**。这与很多 agent 研究一致（用受控环境诱发并观察真实行为），但局限必须写进 README——我已经在"局限"里标了"任务集自建（设计偏差）"。

想提高外部效度，入职周可以补一件事：在 τ-bench / AgentBench 这类公开 benchmark 上跑一遍同样的四策略对比，作为"自建任务的结论能否外推"的佐证。**周末两天别碰**——环境配置成本高，且难以定向控制失效类别。

### Q5 用什么 agent 框架？

**不用框架。** `1_run_baseline.py` 里是手写的 ReAct loop（OpenAI 兼容 function calling + vLLM 推理，约 100 行）。

为什么不用 LangGraph / AutoGen 这类框架：**因为本实验的自变量就是"上下文怎么组织"**。框架会自己管理记忆和上下文，你无法精确控制每一步 messages 的构成——等于把要研究的变量交给了黑箱。手写 loop 才能让每步输入完全由 `apply_window / apply_summary / apply_layered` 决定。

这是实验设计的需要，不是偷懒。面试被问"为什么不用框架"，这就是答案。

### Q6 `<你的仓库>` 里放什么？

就是**这个项目文件夹本身**（`agent-badcase-dpo/`）。三条路任选：

1. **git（推荐，便于迭代）**：GitHub 建**私有**仓 → 本地 `git init && git push` → AutoDL 上 `git clone`（本地改代码 → push → 服务器 pull，改起来最快）
2. **直接上传（最快）**：AutoDL 控制台 / JupyterLab 支持上传文件夹，整个项目传上去即可
3. **网盘挂载**：适合同步大文件（模型权重）

⚠️ 代码先保持私有，README 打磨好再公开——仓库链接是要进简历的。

---

## 1. 准备（30 分钟）

1. **AutoDL**：注册 autodl.com（国内直连），充值 **100 元**。租 **RTX 4090D**（约 ¥1.88/时，以实例页实际报价为准；重庆 A 区通常最充足；选"剩余 2 卡以上"的机器）。镜像选 PyTorch / CUDA 12.x 基础镜像。
2. **SSH 连上后第一件事**：
   ```bash
   source /etc/network_turbo        # 不开这个 HuggingFace 下载会慢到几 KB/s
   ```
3. **代码放到数据盘**（系统盘在重置镜像时会丢）：
   ```bash
   cd /root/autodl-tmp          # AutoDL 数据盘，关机后保留
   git clone <你的仓库> && cd agent-badcase-dpo
   pip install -r requirements.txt
   ```
4. **下载模型 + 起 vLLM 服务**（后台运行，断 SSH 也不中断）：
   ```bash
   pip install -U huggingface_hub
   hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/Qwen2.5-1.5B-Instruct
   nohup python -m vllm.entrypoints.openai.api_server \
     --model ./models/Qwen2.5-1.5B-Instruct --served-model-name base \
     --port 8000 --max-model-len 8192 > vllm.log 2>&1 &
   sleep 30 && curl http://localhost:8000/v1/models    # 能看到 "base" 即成功
   ```
5. **强模型 API**（合成 chosen 用）：准备好 DeepSeek / 通义 / GPT 任一的 `API_KEY` 和 `BASE_URL`。
6. **备份纪律**：每跑完一个 Phase，把 `data/` 和 `results/` 拉回本地或 push 到 git。实例盘无冗余备份，丢了不可找回。

> 💡 长任务统一用 `nohup ... > log 2>&1 &` 或 `tmux`（见 §0.5 Q2）。Phase 1 只推理不训练，最便宜的卡都够；Phase 3 训练再上 4090D。

---

## 2. Phase 0：任务集与难度校准（周六上午，1–2 小时）

### 2.1 为什么任务集是地基

你的所有结论都建立在"任务集能稳定诱发三类失效"上。任务集有偏，后面全歪。

现有 `tasks/tasks.jsonl` 有 12 条样例（三类各 4 条）。**先把样例跑通，再扩充到 36–48 条**（每类 12–16）。扩写规范见 `tasks/schema.md`。

### 2.2 操作

```bash
# ① 先跑 10 条探测难度（用 full 策略）
python scripts/1_run_baseline.py --strategy full --tasks tasks/tasks.jsonl --limit 10 --out data/probe.jsonl --verbose

# ② 秒出完成率
python scripts/5_evaluate.py --traj data/probe.jsonl --mode quick
```

### 2.3 验收标准

- 总完成率落在 **30–50%**
- 三类任务各自都有成功有失败（不是某一类全挂或全过）

**✏️ T2.1**：为什么目标区间是 30–50%？高于 70% 和低于 20% 分别会导致什么问题？想两个层面：数据层面（偏好对的数量和质量）、结论层面（天花板和地板效应）。

**✏️ T2.2**：如果你拿到 `tool_misuse: 90%`、`context_forgetting: 10%` 这种极端偏斜的分类别完成率，这说明任务设计有哪两种可能的偏差？分别怎么修？

### 2.4 调难度的旋钮（按此顺序尝试）

| 想调什么 | 动哪里 |
|---|---|
| 整体太难 | 缩短 `goal`、减少 `tools` 数量、工具 `desc` 写得更明确 |
| 整体太简单 | 加干扰工具（名字相近的）、加 `constraints`、加步数（`expected_steps`） |
| tool_misuse 诱发不足 | 让相似工具的差异只体现在一个参数上（如 `search_flight` vs `search_train` 的 `date` 格式不同） |
| context_forgetting 诱发不足 | 把约束写成**自然语言埋进 goal**（"我预算有限"而不是列表式"预算≤800"），且要求在**第 3 步之后**才用到 |
| planning_drift 诱发不足 | 加条件分支（"如果缺货就查替代"）、多阶段产出（查→筛→写文件） |

改完重跑 probe，直到落在区间内。**这一步不许跳过**——地基歪了后面全部白跑。

---

## 3. Phase 1：上下文策略对照实验（周六下午 + 晚上）

### 3.1 四种策略是什么、各自模拟什么真实做法

| 策略 | 实现 | 模拟真实产品的什么 |
|---|---|---|
| **full** | 全量历史塞进上下文 | 对照组：不做任何上下文管理（模型原生行为） |
| **window** | 只保留最近 K 轮工具交互 | 最朴素的截断：上下文超预算时砍掉最老的 |
| **summary** | 旧历史定期用模型压缩成摘要 | 多数"记忆"产品的做法：滑窗 + 摘要 |
| **layered** | 目标/约束/已完成/关键观测做成**常驻状态块** + 最近 2 轮细节 | event-memory 思路：**该常驻的常驻，该滚动的滚动** |

实现都在 `scripts/1_run_baseline.py` 的 `apply_*` 函数里（每个不到 30 行，**跑之前通读一遍**，你要能在面试里讲清楚每个策略的实现细节）。

**✏️ T3.1**：window 的 `keep_rounds` 默认是 4。这个数字怎么定才算"公平"？定太大和定太小分别对哪种策略不公平？（提示：想想它和 layered 的 `keep_rounds=2` 是不是该用同一个值、为什么。）

**✏️ T3.2**：summary 策略的压缩用的是**被测模型自己**（1.5B），不是强模型。这是一个可以质疑的设计——用强模型压缩会引入什么 bias？用被测模型自己压缩又有什么问题？各说一条，然后判断哪种 bias 对本实验更致命。

**✏️ T3.3**：预测：layered 在 context_forgetting 类上应该显著占优——机制是什么（一句话说清楚信息流向）？再想深一层：它**治得好 planning_drift 吗**？为什么？

### 3.2 操作（四组各跑一遍，全量任务）

```bash
for s in full window summary layered; do
  python scripts/1_run_baseline.py --strategy $s --tasks tasks/tasks.jsonl --out data/p1_${s}.jsonl --verbose
done

# 出核心对比表
python scripts/5_evaluate.py \
  --files full=data/p1_full.jsonl,window=data/p1_window.jsonl,summary=data/p1_summary.jsonl,layered=data/p1_layered.jsonl \
  --out results/phase1_compare.md
```

### 3.3 验收标准 + 要盯的三个点

1. **总体完成率**：layered vs full 的差距——这就是 JD 说的"数倍差距"在你实验里的版本（大概率不到数倍，诚实报告实际数字）
2. **分类别表**（核心）：layered 应该在 context_forgetting 上拉开差距；四策略在 tool_misuse 上应该都差不多平——**如果这个"平"出现了，它就是 Phase 3 的理由**（上下文治不了 → 试试训练）
3. **上下文峰值**：layered 的 `avg_ctx` 应该显著低于 full。如果 layered 完成率高但上下文峰值也最高，你的故事就变成"堆上下文也能解决"——那就换叙事，别硬编

**⚠️ 中间检查点**：如果 full 的完成率 > 70%（任务太简单）或 < 20%，回 Phase 0。别硬跑完四组再发现地基歪了。

### 3.4 额外赚的一手数据（下班前 10 分钟）

```bash
python scripts/5_evaluate.py --traj data/p1_layered.jsonl --mode quick
```

看轨迹里的 `context_chars` 序列（每步实际发给模型的上下文长度）。这是"上下文预算"的一手观察：layered 的曲线应该明显平缓（状态块是常数级），full 是线性膨胀。README 里放这两条曲线的对比，胜过一千字解释。

---

## 4. Phase 2：归因 + 偏好对构造（周六深夜 / 周日上午）

### 4.1 归因

用 **full 策略**的失败轨迹做归因（理由见 T4.0 思考）：

```bash
python scripts/2_attribute.py --traj data/p1_full.jsonl --out data/badcases_labeled.jsonl --use-judge
```

看输出分布，确认三类各 ≥10 条。分布严重偏斜就补任务，别硬凑。

**✏️ T4.0**：为什么用 full（无策略）的失败轨迹做训练数据，而不是用 layered 的？提示：想想 DPO 学到的行为在部署时会遇到什么环境。

### 4.2 偏好对构造

```bash
export OPENAI_API_KEY=xxx
export OPENAI_BASE_URL=xxx
python scripts/3_build_preference.py --badcase data/badcases_labeled.jsonl --out data/pref_pairs.jsonl
```

脚本会自动丢弃 chosen 不过 checker 的对（宁缺毋滥）。

**✏️ T4.1**：chosen 过了 checker 就算高质量吗？checker 是规则匹配，它抓不到 chosen 的哪些质量问题？至少想出两条，并说明这些质量问题会以什么方式反噬 DPO。

**✏️ T4.2**：rejected 是失败轨迹截断到失败点。为什么截断？不截断（把整条发散轨迹全给）会怎样？提示：DPO 的 loss 同时压低 rejected 的概率——想想"压低整条轨迹"和"压低失败那一步"学到的东西有什么区别。

**✏️ T4.3**（本 Phase 最难的一题）：你的偏好对**全部来自失败任务**——DPO 同时也在学"这批任务长什么样"。怎么设计一个对照，把"学到正确行为"和"过拟合任务分布"区分开？

### 4.3 验收标准

- 偏好对 ≥100 条（少于这个数，DPO 效果会很弱，如实报告）
- 随机抽 5 条人工读：chosen 确实比 rejected 好，且 chosen 没有明显胡编（比如调用了不存在的工具）

---

## 5. Phase 3：DPO 训练与验证（周日下午 + 晚上）

### 5.1 训练操作

```bash
# ① 导出训练数据
python scripts/4_to_llamafactory.py --pref data/pref_pairs.jsonl --outdir data/lf_data

# ② 装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
pip install -e ".[torch,metrics]"
cp data/lf_data/agent_pref.json data/lf_data/dataset_info.json LLaMA-Factory/data/

# ③ 训练（配置在 config/dpo_qwen15b.yaml，先通读一遍再跑）
cd LLaMA-Factory && llamafactory-cli train ../config/dpo_qwen15b.yaml

# ④ 合并 LoRA
llamafactory-cli export ../config/merge_lora.yaml    # 配置见 dpo_qwen15b.yaml 末尾注释

# ⑤ 用训练后的模型起第二个 vLLM 服务（端口 8001）
nohup python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/dpo_merged --served-model-name dpo \
  --port 8001 --max-model-len 8192 > vllm_dpo.log 2>&1 &
```

### 5.2 训练时盯什么（这是"不玩具"的关键证据链）

打开 `outputs/dpo-qwen15b/trainer_log.jsonl`（或 TensorBoard：`tensorboard --logdir outputs/dpo-qwen15b`）：

1. **rewards/accuracies**：chosen 比 rejected 打分高的比例——健康走势是升到 0.7–0.9 并稳定；**快速冲到 1.0 要警惕**（可能过拟合）
2. **rewards/margins**：chosen 与 rejected 的 reward 差——应该温和扩大；**陡增 = 可能 reward hacking**（模型学会区分好坏但行为没真变好）
3. **train_loss**：DPO 的 loss 下降很慢是正常的（它优化的是相对偏好不是绝对似然）

**✏️ T5.1**：训练完成后、下任何结论之前，必须先跑什么？（提示：你不只想知道"变好的部分"，你还想知道"没变坏的部分"。至少给出两个检查。）

**✏️ T5.2**（本实验最重要的思考题）：如果 DPO 后 context_forgetting 类**反而变差**了，给出至少三种可能原因，并说明各自的验证方法。

**✏️ T5.3**：怎么区分"DPO 真学到了工具选择行为"和"模型整体退化（比如变得不敢调工具、疯狂复读），但恰好在这批任务上 checker 判过更多"？

### 5.3 评测操作

```bash
# 训练后模型跑同一任务集（full 策略，和基线同条件）
python scripts/1_run_baseline.py --strategy full --port 8001 --model dpo \
  --tasks tasks/tasks.jsonl --out data/dpo_trajectories.jsonl

# 前后对比
python scripts/5_evaluate.py \
  --before data/p1_full.jsonl --after data/dpo_trajectories.jsonl \
  --out results/phase3_compare.md
```

### 5.4 验收标准

- 分类别表出来了：**DPO 在哪类失效上有效、哪类无效、哪类恶化**——三格都有答案
- 对照 T5.1 的检查至少做了一项（比如：拿 5 条不在任务集里的开放问题问两个模型，确认没退化成复读机）

---

## 6. Phase 4：综合分析与结论（入职周晚上，纯分析活）

### 6.1 交叉分析：把三张表拼成一张

| | tool_misuse | context_forgetting | planning_drift |
|---|---|---|---|
| full 基线 | a% | b% | c% |
| 最佳上下文策略（+幅度） | ? | ? | ? |
| DPO（+幅度） | ? | ? | ? |

每一格填"该干预手段对该失效模式的净效果"。**这张表就是整个实验的产出**——它回答"资源该往哪投"：每一类失效，你都有了证据支持的最优干预。

**✏️ T6.1**：什么情况下"上下文策略 + DPO"叠加会**互相打架**？（提示：想想 DPO 学到的行为里有没有可能包含"依赖某种上下文组织方式"的隐性假设。）

### 6.2 结论写作的三条纪律

1. **只报分类别结论，不吹总体数字**——"整体提升 X%" 是玩具项目的写法；"DPO 对工具误选 +18pp、对上下文遗忘 −3pp（不显著）" 是实验的写法
2. **失败也写**——v1 哪里失败了、归因是什么、v2 怎么修的。这正是 JD 的工作方式原文
3. **如实标注局限**——任务集自建、单 seed、合成 chosen、1.5B 规模。局限写得越诚实，可信度越高

---

## 7. README 与简历落点

### 7.1 README 重写（结构照抄这个骨架）

```
1. 一句话：从 badcase 归因出发，量化"上下文组织 vs 训练矫正"对三类 Agent 失效的修复边界
2. 实验设计：四策略 × DPO × 三类失效，任务集 N 条，基座 Qwen2.5-1.5B
3. 核心结果：那张 3×2 交叉表（只放分类别数字）
4. 关键发现：2-3 条非显然结论（从 T3.3/T5.2/T6.1 的答案里提炼）
5. 失败与迭代：v1 的坑和修复
6. 局限与下一步
```

### 7.2 简历落点（3–4 行，放在项目经历栏）

写法重心：**量化结论 + 交叉实验设计**，训练只是其中一格。参考措辞方向（数字跑完再填）：

> **Agent 失效模式与干预边界实证研究（独立项目）**：设计"上下文组织策略 × DPO 矫正"交叉对照实验（4 策略 × 3 类失效 × N 任务，Qwen2.5-1.5B）；发现分层上下文组织使约束遗忘类任务完成率 +X%，但工具误选类不敏感；针对性构造偏好数据 DPO 训练使工具误选 +X pp，验证两类干预近似正交——为"线上失效该改上下文还是提训需"提供决策依据。

**✏️ T7.1**：上面这段措辞里，为什么把"发现分层使 +X%"放在"训练"前面？调整这个顺序会削弱什么？

### 7.3 面试叙事准备

见 `ANSWERS.md` 末尾的"面试防守手册"——30 秒版、3 分钟版、以及被追问训练细节时的接法。**做完实验再读那部分，否则没有素材**。

---

## 8. 时间盒与止损线（保命用）

| 阶段 | 时间盒 | 超时了砍什么 |
|---|---|---|
| Phase 0 | 周六上午 2h | 任务集 36 条→24 条也够（每类 8） |
| Phase 1 | 周六下午+晚上 5h | 砍 summary 策略（保留 full/window/layered 三组也成立——它是最"标准"的一组） |
| Phase 2 | 周日晚 2h | 砍 --use-judge（纯规则归因也够） |
| Phase 3 | 周日晚 3h | 单 seed 直接跑，多 seed 留给入职周 |
| Phase 4 | 入职周 3×1.5h | 只写 README + 简历段，不做曲线图 |

**两条红线**：
1. 8/31 入职九安第一周**不许熬夜**——丛明刚定调，第一印象比实验重要。宁可 Phase 3 只有单 seed。
2. 任何时刻发现"我在为了出好看数字而调实验"——停。诚实的负结果比编的正结果值钱，而且是 JD 工作方式的原话。

---

## 附：思考题索引

| 编号 | 主题 | 在哪一步 |
|---|---|---|
| T0.1 | 实验前预测矩阵 | §0.2 |
| T2.1 | 难度校准的区间依据 | §2.3 |
| T2.2 | 偏斜诊断 | §2.3 |
| T3.1 | 窗口大小公平性 | §3.1 |
| T3.2 | 压缩器的 bias | §3.1 |
| T3.3 | layered 的机制与边界 | §3.1 |
| T4.0 | 训练数据用哪个策略的轨迹 | §4.1 |
| T4.1 | chosen 的隐性质量问题 | §4.2 |
| T4.2 | rejected 截断的理由 | §4.2 |
| T4.3 | 行为学习 vs 分布过拟合 | §4.2 |
| T5.1 | 训后先检查什么 | §5.2 |
| T5.2 | DPO 恶化的归因 | §5.2 |
| T5.3 | 真学 vs 退化的区分 | §5.2 |
| T6.1 | 干预叠加的冲突 | §6.1 |
| T7.1 | 简历措辞的顺序设计 | §7.2 |

答案见 `ANSWERS.md`。规则：**先写你自己的答案（哪怕一行），再看参考**。
