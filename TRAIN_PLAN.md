# SFT / RL dataset + LoRA fine-tuning plan (v2.1, 2026-10-28)

_Plan for the training lane of this repo. Everything that quotes task text (trajectories, rendered prompts, adapters) stays under `runs/` and `adapters/`, which are gitignored; only the scripts under `train/` are public. Run results are logged in `RESULTS.md`. Written in Chinese; the "精简版" section at the end is the one-page execution checklist._


## v2.1 的针对性修订（6 处，不改范围）

1. **Schema 改为 call 级**：训练的最小单位是一次模型调用，每次调用保存它真实看到的全部输入（之前的 assistant 输出、tool_call、tool_response、compaction 摘要、nudge），每次调用有自己的目标和 loss mask。segment 只是 nudge/compaction 的边界标记；同一段内把多次调用打包成一条序列是**优化**，且只在逐调用验证了"前缀性质"（call k 的输入 = call k−1 的输入 + k−1 的输出 + 工具返回）后才允许。
2. **最新 scorer 一致性**：424 条只证明与 wave 7 时的环境一致。Phase 0 加版本/哈希核对（wheelhouse、swegemma、adk-submission、google-adk、vLLM、tokenizer、chat template）并在最新环境采一批新 prompt 做第二份对照；在带 adapter 的 vLLM 上实测可用上下文、KV 容量、超时。
3. **Canary 前移到正式训练之前**：真实底座训 1–5 step 的小 adapter → PEFT 导出 → tensor key/shape 与 `-ct` 模型的模块名核对 → 在 patched vLLM 0.19.1 + `-ct` 上加载 → logprob 变化 → 跑 1 道真实任务看 tool call 能被 parser 解析。任何一步失败，正式训练不开始。
4. **thinking 重标注改成独立实验**：原生 Gemma thinking（D2）是默认基准；D1 的 Qwen reasoning、D3 的 Opus 无 thought 轨迹、事后生成的 synthetic rationale 三种来源**分别打标**（`thought_source` 字段）并各自消融；先用 20–50 个回合量重标注的成本和质量，再决定是否批量。**D1（Qwen 轨迹）的 thought 重放用 GLM**（MindForge 论文的 teacher 就是 GLM-5.2，用它做 reasoning rewrite；外部 LLM 按许可证使用是允许的，#742807），`thought_source = synthetic_glm`；D3（Opus）的候选仍是 Gemma 31B（`synthetic_31b`）。两者都和原生对比。
5. **混合与评测**：E4B on-policy 与 31B teacher 分开标记；不默认复制 D2；混合比例由小规模消融定。alpha 和混合比例只在**开发集**（rest96 中划出的 dev26 + dev33）扫，OOD-45 只做最终验收。所有训练来源与评测任务按 repo / commit / PR / patch 去重（D1 的 814 个 repo 要对照 OOD-45 的 17 个 repo）。
6. **先实测成本再排期**：31B 先采 10–20 次、E4B 先跑 20 个训练 step，记录 tokens/s、显存、费用、数据有效率，再更新 Phase 1 预算。基线和训练结果都必须导出完整 JSONL，所以**基线从 box 挪到 RunPod**（box 只能拍照）。

## 对 v1 的修改（仔细过了一遍后改的 9 处）

1. **算力分工有矛盾，改掉**：v1 说"训练在 box"，但 box 只能拉不能推（结果靠拍照），训出的 adapter 和采样出的轨迹都出不来。v2：box 只做**不需要带走文件**的事（基线评测、gold 复核）；**所有产出文件的步骤（31B 采样、thought 重标注、训练）都在 RunPod**，Kaggle 4×L4 的每周 30 h 免费额度做 31B 采样的补充。
2. **先量基线**：v1 没有 E4B / 31B 在 OOD-45 上的 base 成绩，没有基线就没法判断 adapter。Phase 0 加基线（E4B ×2 seed，31B ×1）。
3. **渲染单位从"整条轨迹"改成"段"**：推理时 nudge 会丢掉更早的 thought，compaction 会把历史换成模型自己写的摘要。整条轨迹一次渲染会和模型实际看到的 prompt 不一致。v2 按 nudge / compaction 切段，每段的 prompt 用当时真实的 prompt（traj 日志里有），loss 只算该段的 model 回合。无事故的轨迹就是一段。
4. **采样温度**：拒绝采样用 temp 0.7（Ayush：1.0 ≈ 0.4 > 0.2，解题数不降而多样性高）；评测仍用 latest 的 0.2。
5. **thought 重标注**：Opus 轨迹没 thought、Qwen 的 reasoning 零散，v1 打算用 Opus 补写。v2 改为一个工具 `relabel_thoughts.py --model glm|gemma31b`，在当时的 context 下给定下一步动作生成 thought；（v2.1 修订）D1 用 GLM、D3 用 Gemma 31B，各自与原生 thinking 消融。
6. **thought 长度上限**：训练目标里的 thought ≤ 3,000 token（推理 budget 4,096），超过的回合截掉或整条剔除。
7. **第二轮默认是 expert iteration，不是 GRPO**：用 v1 adapter 再采样 → 只留通过的 → SFT v2。零新基建、就是 RL 的保守版。GRPO 降为 stretch，只在自建题 ≥ 60 且 expert iteration 有增益时做。
8. **数据中转 + 备份**：Opus 86 条通过轨迹和 E4B/31B traj 现在只在笔记本上。建一个**私有** Kaggle dataset `gemma4-sft-private` 作为中转站（笔记本 → Kaggle → RunPod），同时就是备份。私有数据集可以含任务文本；公开仓库仍然不含。
9. **自建题缩成 pilot**：先 1 个 repo × 10 题走通 goldcheck，再决定扩到 60。它主要服务第二轮采样和 GRPO，不挡 SFT v1。

## Context

比赛允许提交 PEFT LoRA adapter（`adapters/<name>/adapter_model.safetensors`，rank ≤ 128 可加载、≤ 64 已出分，zip < 3 GiB，adapter 名 ≠ base 别名），小模型奖同样允许。外部数据允许：Gemma 自己的轨迹、其他开源模型的轨迹、公开 HF SWE 数据集、外部 LLM 按许可证做 teacher（#742807 #743964 #746046）；课程要求用到的外部数据集在论坛声明。

论坛至今**没人做出 SFT 增益**：icfoer（31B，697 条 teacher 轨迹）训练数据没 thinking 而推理有 → 错配，alpha 过大 → 重复循环；TTX（E4B，170 条）数据太少。本计划的差异点：**thinking 与推理格式逐 token 一致**、**on-policy 拒绝采样为主**、**数据上千**、**alpha 推理期扫描**、**在未见 repo（OOD-45）上评测**。

用户决定：E4B 先、31B 后；四种数据来源都允许；先 SFT（拒绝采样）再看 RL。参考 arXiv 2607.27146（MindForge）：只留正常结束且产物可用的 teacher 轨迹 → 修复坏步骤（回滚重放、重写引用坏调用的 reasoning）→ loss 只算 assistant 的 reasoning + tool call，mask system/user/tool 输出。

截止：论文 11-12（15 天），最终 12-02（35 天）。

## 硬事实（已核实）

- 比赛 vLLM 0.19.1：`max_model_len` 32,768，`reasoning_parser`/`tool_call_parser` = gemma4，`enable_lora, max_loras 8, max_lora_rank 128`（HARNESS_README 的配置值；adk_submission 0.2.13 启动时实际用 `--max-loras` = adapter 个数、`--max-lora-rank` = 秩向上取档，`serve_vllm.sh` 自 2026-10-30 起照此计算）；带一个 r16 adapter 时 KV ≈ 27.7K token，我们的 agent 峰值 16–24K。
- PEFT 挂不上 `-ct` 权重。训练用 bf16：(a) Kaggle `google/gemma-4` transformers 变体，或 (b) `-ct` 反量化副本（icfoer 的做法，#750713）。Phase 0 比对两者权重差异后二选一。
- scorer 加载过 rank ≤ 64 的 adapter（Ryan Holbrook，#749191）；adapter 在前向里生效（#749528）；**没人展示 adapter 超过 base**。
- 推理时的 prompt 规则：同一回合内所有 thought 都在 prompt 里；harness 的 nudge（user 消息）之后更早的 thought 消失；compaction（interval 15 · threshold 14,336 · retain 5）用**模型自己**写的摘要替换历史（`scripts/compaction_shim.py`）。
- 本地资源：129 题（gold 112 过）+ docker 评测；Opus 通过轨迹 86/129（ATIF，无 thinking，`runs/passing_traces.json`）；E4B wave 7 轨迹 99 条（11 通过，含 thinking，`runs/pod_w7/*/traj/`）；**424 条真实渲染 prompt**（`runs/pod_w7/w7_example/rendered_base/`）；`scripts/traj_log.py` `walkthrough.py` `passing_traces.py` `eval_dev.sh`（TRAJ=1）`serve_vllm.sh`（adapter 加载）`make_submission.py`。
- 公开数据 adesolanke/gemma4-swe-agent-trajectories（CC BY 4.0）：3,586 条 SWE-rebench Qwen3-Coder-480B 轨迹已转成我们的工具格式，curated 1,427；**没有 resolved 字段**；源数据集 nebius/SWE-rebench-openhands-trajectories（67,074 行）有 `resolved` `model_patch` `exit_status`。
- OOD-45：45 题 17 个未见 repo，swegemma 格式，gold 44/45。
- 仓库里没有任何训练代码；`.venv-eval` 不能训练（vllm/torch 锁死），笔记本没有 vLLM。

## 关键设计决定

1. **中间格式（call 级）**：`runs/sft/<source>/<id>.json` =
   `{instance_id, repo, base_commit, pr, patch_sha, source (e4b_onpolicy | 31b_teacher | opus | d1_qwen | mined), resolved, policy_model, sampling, env_fingerprint, calls[]}`；
   每个 call = `{call, segment, input_messages[], target: {reasoning, content, tool_calls}, thought_source (native | qwen | synthetic_glm | synthetic_31b | none), usage_prompt_tokens, events_before: [compaction | nudge | reset]}`。`input_messages` 是这次调用真实看到的完整消息列表（从 traj 日志的 `new`/`reset`/`compaction` 逐步重放得到）。
2. **渲染** = 用 `-ct` tokenizer 的 chat template 把每个 call 渲成 `{input_ids, labels}`，labels 只在该 call 的目标（thought + 文字 + tool_call + 结束 token），其余 -100。同一 segment 内若逐调用验证了前缀性质，`render.py --pack` 把它们打包成一条序列、多段 labels；验证不过的段退回逐 call 样本。打包前后两种样本的 labels 总数必须相等（断言）。
3. **系统提示** = `scripts/arms/latest/prompts/system.md`，任务 prompt = `build_agent_prompt`（`swegemma/harness/agent_runner.py:317`）格式；外部数据也重新套这套 prompt。
4. **长度**：渲染后 ≤ 26,624 token，调用 ≤ 28，thought ≤ 3,000 token。
5. **采样 vs 评测**：拒绝采样 temp 0.7 · top_p 0.95 · thinking 4096 · 28 calls · 7 min（其余同 latest）；评测用 latest（0.2）。
6. **评测协议**：开发期每个 adapter 跑 dev26 ×2 seed（+ dev33 ×1 作 31B 回归）；只有开发期选出的**一个** adapter 跑 OOD-45 ×2 seed 做最终验收；`serve_vllm.sh` 带 adapter 启动时 KV ≥ 27K。**决策规则：OOD-45 两 seed 均值 ≥ base + 4 题才上 LB**（LB 单次噪声 ±3 题）。所有评测产出 `task_results.jsonl` + `traj/`，训练产出 `trainer_log.jsonl`（每 step loss、lr、tokens/s、显存）；每次都写 `notes/05_run_log.md` + `RESULTS.md`。
7. **数据分割**：`rest96` 去掉 env_defect 10 题 ≈ 86 → 按 repo 分层切成 **train60**（采样池）和 **dev26**（开发集，alpha / 混合比例 / epoch 扫描）；dev33 = 回归；**OOD-45 = 最终验收，整个开发过程不看**。去重：训练来源与 dev26 / dev33 / OOD-45 按 repo、base_commit、PR 号、patch 哈希互斥；D1 先剔除 OOD-45 的 17 个 repo 和我们 4 个 repo 的所有轨迹（打印剔除数）。
8. **算力**：
   - box：E4B/31B 在 OOD-45 的基线、gold 复核、不出文件的消融（免费）。
   - GLM API（付费，需确认；key 放 `~/.zshrc`，不进仓库）：D1 的 thought 重放。
   - RunPod（每次新建 pod，付费需确认）：31B 采样（1× 96 GB 卡），D3 的 thought 重标注，E4B 训练（≥ 48 GB 卡，32K 序列 + grad ckpt），31B QLoRA（96 GB 卡），adapter 评测（RTX PRO 4500 跑 E4B 即可）。
   - Kaggle 4×L4：31B 采样补充（`build.py --traj`，~1 h/10 题，每周 30 h）；最终提交。
   - 中转：私有 Kaggle dataset `gemma4-sft-private`。
9. **训练**：HF Trainer + PEFT（不用 TRL，自己按 span 生成 labels）；E4B bf16 LoRA r=32 α=32 all-linear（语言层），dropout 0，LR 1e-4 cosine warmup 3%，2 epoch，micro-batch 1 + grad-accum 16，无 packing，clip 1.0；31B QLoRA nf4 + bf16 adapter，同超参。导出 PEFT safetensors，`base_model_name_or_path` 写比赛别名。alpha 在推理期扫 0.5/1/2 × r（只改 adapter_config.json）。

## 阶段

### Phase 0 · 环境核对 + 基线 + 格式 + canary + 成本实测（10-29 → 11-02）
- **环境核对**（`train/env_fingerprint.py`，输出 JSON 并写进每条样本的 `env_fingerprint`）：wheelhouse 版本与 sha256、swegemma / adk-submission / google-adk / vllm / transformers 版本、`-ct` tokenizer 与 chat template 的 sha256、`compaction_shim` 的四个值；与 Kaggle 最新 Getting Started notebook 和当前 wheelhouse 对照，差异逐项列出。
- **新 prompt 对照组**：在最新环境（RunPod，最新 wheelhouse）用 E4B 跑 dev33 的 5 题、TRAJ=1、/tokenize 渲染 → `rendered_latest/`（预计 30–60 次调用）。渲染器要同时过 424 旧 + 新对照组；两组模板若有差异，以新的为准并记录差异。
- **LoRA 下的实际容量**：同一 pod 上挂一个 r32 全层 canary adapter 启动 vLLM，读日志里的 KV token 数、`max_model_len` 实际值；跑 5 题记超时率和每调用耗时，与无 adapter 对比。
- **基线**（RunPod，不是 box——结果要有 JSONL）：OOD-45 上 E4B ×2 seed、31B ×1（31B 用 96 GB 卡）；dev26 上 E4B ×2、31B ×1；dev33 已有。产物 `task_results.jsonl` + `traj/`。
- **Canary 前移**（`train/canary.py`，正式训练前必过）：真实底座训 1–5 step 的 r32 全层 adapter → PEFT 导出 → 核对 `adapter_model.safetensors` 的 key 名、数量、shape 与 `-ct` 模型模块名一一对应（E4B 的 per-layer 模块名要特别核对）→ 在 patched vLLM 0.19.1 + `-ct` 上加载 → 固定 prompt 的 top-1 logprob 与无 adapter 相比有变化 → 跑 1 道真实任务，tool call 能被 gemma4 parser 解析。
- **成本实测**：31B 在 train60 上采 10–20 次（temp 0.7，TRAJ=1），E4B 用 20 条样本跑 20 个训练 step；记录 tokens/s、峰值显存、每小时费用、采样有效率（resolved / 总）。用实测数更新 Phase 1 的预算行。
- **训练环境**：RunPod 上新建 `.venv-train`（torch + transformers + peft + accelerate，`train/requirements-train.txt` 锁版本），与 `.venv-eval` 隔离。
- **中转 + 备份**：`train/pack_private.py` 把 `runs/passing/`、`runs/dev33_base_ceil*`、`runs/pod_w7/*/traj`、`rendered_base/` 打成 tar → 私有 Kaggle dataset（需用户确认创建；本轮不上传）。
- `train/schema.md`、`convert_traj.py`（traj → 段）、`convert_atif.py`（Opus ATIF → 段，reasoning 空）、`convert_adesolanke.py`、`render.py`、`filter.py`（每条样本打印：段数、token、thought 长度、剔除原因）。
- **权重比对**：下载 transformers 变体和 `-ct`，反量化后比各层权重差（RunPod 或笔记本 CPU），选训练底座。
- 验收（`render.py --check`，四条全过才进 Phase 1）：
  1. 逐调用渲染：wave 7 base arm 的 424 次调用 + 新对照组全部逐 token 相同（有 nudge / compaction 的调用也必须一致）；打包模式下每段的前缀性质逐调用断言，打包前后 labels 总数相等。
  2. 每个 model 回合拆 thought / 文字 / tool call 三类 token 区间，边界落在控制 token 上；抽 50 个回合解码回文本，与 `reasoning` / `content` / `tool_calls` 原文一致。
  3. system、user（含任务 prompt、nudge、compaction 摘要）、tool_response 全部 `labels = -100`；model 回合（含结束 token）无一被 mask；打印每条样本有效 label 占比。
  4. 目标文本喂给 vLLM 0.19.1 的 gemma4 reasoning / tool_call parser（在 RunPod 的 `.venv-eval` 跑）：工具名在 9 个里、参数合法 JSON 且符合 `setup.tool_schemas`、与轨迹 `tool_calls` 等价；失败的剔除并打印剔除率。

### Phase 1 · SFT v1 数据 + 第一个 E4B adapter（11-01 → 11-07）
数据（数量以 Phase 0 实测的有效率为准，下面是估计）：
- **D2a E4B on-policy**：E4B 在 train60 上 k=4，temp 0.7，TRAJ=1（RunPod 4500）。`source = e4b_onpolicy`，`thought_source = native`。
- **D2b 31B teacher**：31B 在 train60 上 k=4（RunPod 96 GB 卡 + Kaggle 补充），`source = 31b_teacher`，`thought_source = native`。
  两者都 strict `reverify.py` 只留 resolved；同题近重复（相同 patch + 相同调用序列）去重。预计 D2b 40–80 条、D2a ~10 条。
- **D3 Opus 86 条**：`thought_source = none`（无 thought）先原样入库；是否重标注见下面的 thinking 实验。
- **Thinking 重标注实验**（先做，再决定是否批量）：`relabel_thoughts.py`，给定 context 和下一步动作写 thought，禁止提到"给定的动作"，≤ 3,000 token。
  - D1：取 20 个回合，`--model glm`（GLM-5.2 API，按 MindForge 的 reasoning-rewrite 做法）→ `thought_source = synthetic_glm`。
  - D3：取 30 个回合，`--model gemma31b`（RunPod）→ `thought_source = synthetic_31b`。
  记录每回合成本（token、秒、费用），人工看各 10 条质量，并用 E4B 对这 50 个回合算目标 NLL（原生 D2 vs qwen vs synthetic_glm vs synthetic_31b vs 无 thought）。结论写进 run log；只有合成明显更接近原生分布才批量重标注。
- **D1 公开 1,427 条**：先过 `audit_d1.py` 四道检查，每项打印数字，任一不过就不进 v1（v1 直接 D2 + D3）：
  1. 结果真实性：按 `instance_id` + 最终 patch 回联 nebius 源数据集，只留 `resolved = true`；抽 20 条用 SWE-rebench 的 `docker_image` 复验。
  2. 工具返回保真：抽 20 条在镜像里用我们的 `swegemma.tools` 逐步重放，比对 tool_response；一致率 ≥ 95%。
  3. 行为冲突：改测试文件（21）、根目录/构建文件残留（**626/1,427**）、pip install / 联网、复现脚本没删、无 submit_patch 结尾、调用 > 28 → 剔除或修复（补 `rm` 回合）。预计剩 500–700。
  4. Qwen reasoning 零散：统计有 reasoning 的回合占比和长度；保留为 `thought_source = qwen`，是否换成 synthetic 由重标注实验决定。
  5. 去重：剔除 OOD-45 的 17 个 repo 和我们 4 个 repo；与 dev26/dev33 按 PR/patch 哈希互斥。
- **最小 SFT 实验矩阵**（E4B，每个 adapter 在 dev26 ×2 seed 上评；见文末）：先 M0–M3 决定混合，再 alpha 扫描，最后一个 adapter 去 OOD-45。
- 论坛声明帖：列出 D1 和 OOD-45（用户发）。
训练 + 评测：`train/sft_lora.py`（RunPod ≥ 48 GB，时长按 Phase 0 实测）→ `adapters/e4b_sft_<M>/` → 每次导出都跑 `canary.py` 的 key/shape/加载检查 → dev26 评测 → 选出的一个做 alpha 扫 0.5/1/2（仍在 dev26）→ **最终一个 adapter 跑 OOD-45 ×2** → 通过决策规则则提交 LB（需用户确认）。
打包改动：`make_submission.py` NOLORA 默认保留 adapter；`kaggle_notebook_cell.py:23`、`build_notebook.py:62` 的"LoRA trap"断言改成：adapter 名 ≠ base 别名、rank ≤ 128、zip < 3 GiB。
产出：v1 的 OOD-45 / dev33 结果行 + alpha 扫描表。

### Phase 2 · 轨迹修复 + expert iteration + 自建题 pilot（11-07 → 11-12，与论文并行）
- `train/refine.py`（MindForge 式）：删失败的 `edit_file`（not found / occurs N times）、未声明工具调用、空文字回合；引用了被删步骤的 thought 用 31B 重写一句；统计删前/删后。
- **Expert iteration**：用 v1 adapter 在采样池上 k=4（temp 0.7）→ 只留 resolved → 与 v1 数据合并（新轨迹权重 2）→ SFT v2 → 评测协议。
- **自建题 pilot**：1 个 pytest Python 库（不与 OOD-45、公开题、D1 的 814 个 repo 重叠，PR 在 2025-06 以后）× 10 题 → 8 字段 `tasks.jsonl` + snapshot → `goldcheck.py` gold 过 / 无 patch 不过。走通才扩到 60（`scripts/mine_prs.py`，产物 `runs/mined/`）。
- 论文（11-12）可写：数据构建流水线、渲染一致性四条验收、D1 审计数字、alpha 扫描、OOD-45 结果（无论增益与否）。

### Phase 3 · 31B（11-13 → 11-22）
- 数据 = v2 全集（31B 自采样权重 2）；QLoRA nf4 r=32，RunPod 96 GB 卡，≈ 1 天。
- 评测 dev33 ×2 + OOD-45 ×1；确认带 adapter 不超时（解码慢 ~11%，7 min 预算内）。通过规则后上 LB（Kaggle GPU 只用于有把握的改动）。

### Phase 4 · GRPO（stretch，11-22 →，go/no-go 在 Phase 2 末）
- 条件：expert iteration 有增益 ∧ 自建题 ≥ 60 ∧ RunPod 预算允许。
- E4B，reward = `verify_task` 通过（+ 有 patch、没改测试文件的小项），环境 = 采样池 + 自建题，rollout = 本地 vLLM + 当前 adapter，8/题，docker 评测并发 4。
- 不满足条件则做第三轮 expert iteration。

## 文件清单
- 新：`code-repo/train/{schema.md, env_fingerprint.py, pack_private.py, convert_traj.py, convert_atif.py, convert_adesolanke.py, audit_d1.py, relabel_thoughts.py, render.py, filter.py, refine.py, sft_lora.py, canary.py, requirements-train.txt, README.md}`（公开仓库，不含任务文本）；`splits/train60.txt`、`splits/dev26.txt`（task id 可公开）
- 新：`scripts/mine_prs.py`；产物 `runs/sft/`、`runs/mined/`、`adapters/`（gitignored）
- 改：`scripts/make_submission.py`、`scripts/kaggle_notebook_cell.py:23`、`scripts/build_notebook.py:62`、`scripts/eval_dev.sh`（`ADAPTER=` 和 `TEMP=` 选项，打印有效值）、`.gitignore`（加 `runs/sft/`）
- 笔记：`notes/12_finetune_plan.md`（本计划）+ `05_run_log.md` 每次采样/训练/评测一行 + `01_weekly_plan.md` 的 LoRA 决策门

## 验证
1. Phase 0：`python3 train/render.py --check runs/pod_w7/w7_example/rendered_base/` → 四条验收全过。
2. 每个 adapter：`serve_vllm.sh` 带 adapter 启动 → KV ≥ 27K；canary 输出变化；OOD-45 ×2 + dev33 ×1 结果行；alpha 扫描表。
3. 提交前：`make_submission.py` 8 项检查 + `validate_directory`；zip < 3 GiB；adapter 名 ≠ base 别名；本地 `eval_sample.sh` 用提交包跑 3 题。
4. 决策规则：OOD-45 两 seed 均值 ≥ base + 4 题才上 LB。

## 风险
- SFT 可能仍无增益（论坛零成功）：Phase 1 结束即知；差异点是 on-policy + thinking 一致。无增益时论文仍有数据流水线和审计结果可写。
- D1 四道检查任一不过 → v1 只用 D2 + D3（~150–200 条）；数据少则 epoch 提到 3，并把 expert iteration 提前。
- 31B 采样成本：RunPod ≈ 8 h × 96 GB 卡；若预算紧，改用 Kaggle 免费额度分 2–3 周跑。
- 段切分若和 424 条对不上：先修渲染器，不绕过。
- E4B QAT 的 `per_layer_model_proj` 拷贝缺陷（#753277）：在 OOD-45 上看 adapter 前后 edit_file 命中率变化，作为副指标。
- 带 adapter 解码慢 ~11%：7 min 预算内看超时率，必要时 28 calls → 24。

---

## 精简版（执行时看这一页）

### Phase 0 硬性验收（全部通过才进 Phase 1）
| # | 项 | 通过标准 |
|---|---|---|
| A1 | 环境指纹 | `env_fingerprint.json` 生成；与最新 Getting Started / wheelhouse 的差异逐项列出，无未解释差异 |
| A2 | 渲染一致 | 旧 424 + 新对照组逐 token 100% 一致；打包前缀性质逐调用断言通过；打包前后 labels 总数相等 |
| A3 | 区间标注 | thought / 文字 / tool_call 边界落在控制 token；抽 50 回合解码回文与原文一致 |
| A4 | mask | system / user / nudge / compaction 摘要 / tool_response 全 -100；model 目标（含结束 token）无一被 mask |
| A5 | parser 回路 | 目标文本经 gemma4 reasoning + tool parser 解析：工具名 ∈ 9、参数合法且符合 schema、与原 tool_calls 等价；剔除率打印 |
| A6 | canary | 1–5 step adapter：key/shape 与 `-ct` 模块一一对应 → vLLM 加载 → logprob 变化 → 1 题真实 tool call 可解析 |
| A7 | LoRA 容量 | 带 r32 全层 adapter：KV ≥ 27K，5 题无超时，每调用耗时增幅 ≤ 15% |
| A8 | 基线 | OOD-45（E4B ×2、31B ×1）、dev26（E4B ×2、31B ×1）有 `task_results.jsonl` + `traj/` |
| A9 | 成本实测 | 31B 10–20 次采样、E4B 20 step 的 tokens/s、显存、费用、有效率记入 run log；Phase 1 预算据此改写 |

### 最小 SFT 实验矩阵（E4B，r32 α32，2 epoch，dev26 ×2 seed）
| ID | 数据 | 目的 |
|---|---|---|
| M0 | base，无 adapter | 基线 |
| M1 | D2b（31B native thinking） | teacher 轨迹单独 |
| M2 | D2a + D2b（全 native） | on-policy 加入是否有用 |
| M3 | M2 + D1（qwen thought） | 外部数据是否有用 |
| M4 | M2 + D1（synthetic_glm） | 仅当 GLM 重放实验通过才跑 |
| M5 | M2 + D3（opus，none / synthetic 二选一） | Opus 轨迹是否有用 |
| α | 最优 M 的 adapter，alpha 0.5r / 1r / 2r | 推理期扫描 |
| F | 最优 (M, α) → OOD-45 ×2 | 最终验收，只跑一次 |
每行一个 adapter，同一 seed、同一超参；dev26 结果差 < 2 题视为平手，取数据更简单的那个。

### 数据 Schema（call 级）
```
{instance_id, repo, base_commit, pr, patch_sha,
 source: e4b_onpolicy | 31b_teacher | opus | d1_qwen | mined,
 resolved: bool, policy_model, sampling: {temperature, top_p, thinking_budget, max_calls, time_min},
 env_fingerprint: sha256,
 calls: [{call, segment, events_before: [nudge|compaction|reset],
          input_messages: [...全部真实输入...],
          target: {reasoning, content, tool_calls},
          thought_source: native | qwen | synthetic_glm | synthetic_31b | none,
          usage_prompt_tokens}]}
```
渲染产物：每 call 一条 `{input_ids, labels, meta}`；`--pack` 时每段一条，labels 多段。

### 成本测量方法
- 采样：`eval_dev.sh` 末尾的 `run_costs.py` 已打印 token/时间；加一行 `$ = 小时 × pod 单价`，有效率 = resolved / runs。
- 训练：`sft_lora.py` 每 step 写 `trainer_log.jsonl`（loss、lr、tokens/s、`torch.cuda.max_memory_allocated`），结束打印总 token、总时长、费用。
- 所有数字进 `notes/05_run_log.md` 一行；Phase 1 的时间预算 = 实测 tokens/s × 数据总 token × epoch ÷ 0.8。

### 回滚条件
- A1–A9 任一不过 → 不开正式训练，先修。
- 重标注实验（GLM 对 D1、31B 对 D3 分别判）：合成 thought 的 E4B 目标 NLL 不比无 thought 低，或人工 10 条里 ≥ 3 条不合理 → 该来源不批量重标注，D1 退回 qwen、D3 退回 none。GLM 的每回合费用 × D1 回合数超出预算时也不批量。
- M1/M2 在 dev26 ≤ base → 停止加数据，先查渲染和 alpha（扫 0.5r），再查 D2 的去重与长度分布；仍 ≤ base 则本轮只交论文，不交 adapter。
- OOD-45 F < base + 4 → 不上 LB；当前 LB 入口维持无 adapter 的 latest。
- 带 adapter 超时率 > 10% → 28 calls → 24，重评；仍超 → 不上 LB。
- 任何 Kaggle / RunPod 付费动作前都要用户确认；本轮只改 plan。


## Decision (2026-10-28): D1, D2a and D2b dropped; training data = D3 + self-mined tasks

D2's sampling pool is the same local task bank as the Opus trajectories (D3), and D1 leaves 13 trajectories under the
harness limits. Changes: data = D3 (Opus passing trajectories on the 96 non-dev tasks, env-defect tasks removed, about
60) plus Phase 2's self-mined tasks; dev set = the 33 dev tasks (their Opus trajectories are not trained on); final
check = the OOD set. Thought relabelling with Gemma 31B becomes mandatory (the Opus traces have no thinking); the GLM
variant is the control. Matrix: M0 base · M1 D3 without thoughts · M2 D3 + 31B thoughts · M3 D3 + GLM thoughts
(optional) · alpha sweep · final. No sampling cost in Phase 1. Expert iteration and the mined-task pilot stay as the
route to on-policy data later.
