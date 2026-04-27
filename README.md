# QAnchor

> 基于弱监督学习的金融领域 RAG 检索排序 Pipeline

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Best MRR@10](https://img.shields.io/badge/MRR%4010-0.7758-brightgreen.svg)
![Status](https://img.shields.io/badge/status-Phase1%20complete-green.svg)

QAnchor 是一个面向中文金融年报问答场景的检索排序优化 pipeline。它解决的问题不是"从零训练一个 QA 系统"，而是：**在缺乏文档片段级（chunk-level）相关性标注的情况下，利用已有的问答答案作为弱监督信号，通过 Reverse Mining 自动构造训练数据，微调 Cross-Encoder Reranker，最终以 Gold Eval 量化检索排序增益**。

Pipeline 概览：**年报 PDF 分块 → 多路检索候选 → 用问答答案反推正负例 → 微调排序模型 → 评测排序提升**。

---

## 核心结果

> 微调 Qwen3-Reranker-0.6B 后，MRR@10 从 **0.6115** 提升至 **0.7758**（+26.9%）；对 `Base → Finetuned` 的配对 bootstrap / sign-flip 检验显示提升显著为正。

| 模型配置 | 阶段 | max_length | MRR@10 | NDCG@10 | P@10 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Reranker-0.6B-seq-cls | Base | 768 | 0.6115 | 0.7572 | 0.192 |
| **Qwen3-Reranker-0.6B-seq-cls** | **Finetuned** | **768** | **0.7758 (+26.9%)** | **0.8761 (+15.7%)** | **0.228 (+18.8%)** |
| BGE-v2-m3 | Base | 768 | 0.6443 | 0.7986 | 0.190 |
| BGE-v2-m3 | Finetuned | 768 | 0.7096 (+10.1%) | 0.8522 (+6.7%) | 0.226 (+18.9%) |
| BGE-v2-m3 | Base | 512 | 0.6553 | 0.7840 | 0.182 |
| BGE-v2-m3 | Finetuned | 512 | 0.7103 (+8.4%) | 0.8282 (+5.6%) | 0.224 (+23.1%) |

本次最优为 **Qwen3-Reranker-0.6B-seq-cls Finetuned**。`max_length` 为 reranker 输入序列的最大长度（Query + Document 拼接后的 token 序列，超出会截断）。

完整报告：`data/output/eval/stage1_reranker_comparison_report_20260118_v2.md`
统计显著性报告：`data/output/eval/qwen3-reranker-0.6b/significance_report_qwen3_template_20260117_base_vs_finetuned_20260423.md`

### 模型发布

已发布的 Qwen3 Reranker：
- Merged：[`souflex56/qanchor-reranker-qwen3-0.6b-merged`](https://huggingface.co/souflex56/qanchor-reranker-qwen3-0.6b-merged)
- LoRA：[`souflex56/qanchor-reranker-qwen3-0.6b-lora`](https://huggingface.co/souflex56/qanchor-reranker-qwen3-0.6b-lora)

安装依赖（`pip install -r requirements.txt`）后，可通过 HuggingFace repo id 加载模型。默认推荐使用 `merged` 版本，下方示例即为 merged 版本的用法；`lora` 版本保留为 LoRA / adapter 形态分发，需配合 PEFT 加载。

**Qwen3 Reranker（推荐使用 merged 版本）**：Qwen3 需要拼接 chat template，具体格式可参考 [base model](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls) 的 "Updated Sentence Transformers Usage"。

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("souflex56/qanchor-reranker-qwen3-0.6b-merged")
queries = model.format_queries(["什么是营业收入？"])
docs = model.format_document(["公司2024年营业收入为..."])
scores = model.predict(list(zip(queries, docs)))
```

---

## 问题背景与数据来源

### 数据从哪来

本项目使用 **FinGLM** 年报问答数据作为上游问答信号。这些数据涵盖中国 A 股上市公司年度报告，每条样本包含一个金融领域问题（query）及对应的**人工标注标准答案**（answer）。

FinGLM 数据的清洗、去重、题型分析和公司/年份维度统计见 [`FinGLM-data-eda`](https://github.com/souflex56/FinGLM-data-eda)。本项目沿用其中构建出的标准答案映射，把它作为 Reverse Mining 的答案来源。

仓库中通过以下文件承载这一信号：
- `finglm_data_store/finglm_master.jsonl` — FinGLM 人工标注标准答案库（query ↔ answer 映射）
- `finglm_data_store/` — 数据索引与统计文件

### 为什么仍然是弱监督

本项目**缺乏的是文档片段级（chunk-level）的相关性标注**，而不是完全没有问答答案信号。已有数据是 **question-answer 级别** 的：每条记录告诉了我们"这个问题的答案文本是什么"，但**没有标注"答案出现在 PDF 的哪个具体 chunk"**。

Reverse Mining 的作用，正是把 answer-level 信号反向映射回候选 chunk，自动构造训练所需的**正例与 hard negatives**。因此，本项目属于弱监督排序优化，而非直接使用人工 chunk 标注做强监督训练。

---

## 方法概览

### 问题

金融长文档（A 股年报）+ 无 chunk 级标注数据 → 通用 RAG 方案难以保证检索质量。
- 文档以 PDF 表格为主，结构化分块质量直接影响下游
- 缺乏 chunk 级相关性标注，无法直接微调排序模型
- 跨文档检索噪声大，需要限定检索范围

这个问题定义来自对 FinGLM 年报问答数据的前期梳理：数据里有标准答案和题型/维度统计，但没有直接指向 PDF chunk 的证据标注。

### 解法

采用 Reverse Mining 弱监督 + Listwise 微调：
- 从已有问答答案中自动挖掘训练 triplets（正例 + hard negatives）
- 使用 Listwise Softmax-CE Loss 微调 Cross-Encoder Reranker
- 直接对齐 MRR/NDCG 排序指标

### 与通用 RAG 框架的区别

| 维度 | 通用框架（LangChain/LlamaIndex） | QAnchor |
|------|----------------------------------|---------|
| 分块策略 | 通用 text splitter | PDF 表格感知分块（parent/child 层级） |
| 训练数据 | 依赖人工标注 | Reverse Mining 自动挖掘 |
| 排序优化 | 拿来即用 / zero-shot | LoRA + Listwise 微调 |
| 评测 | 无标准流程 | Gold Eval + 统计显著性验证 |

**适用场景**：领域文档问答、财报检索、缺乏标注数据的 RAG 排序优化。

---

## Pipeline 一览

```text
┌─────────────────────────────────────────────────────────────┐
│                      QAnchor Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  Step 1-2: PDF 分块 + 质检                                   │
│    └── ZenParse 分块 → parent/child 层级结构                 │
├─────────────────────────────────────────────────────────────┤
│  Step 3-5: 三路检索基线                                      │
│    ├── Qwen3-Embedding-0.6B (向量检索)                       │
│    ├── BM25 + jieba (关键词检索)                             │
│    └── RRF 融合 (k=60) → Top-50 候选                         │
├─────────────────────────────────────────────────────────────┤
│  Step 6: Reverse Mining                                      │
│    └── Answer Matching → 正例/负例自动标注                    │
├─────────────────────────────────────────────────────────────┤
│  Step 7-8: Gold Eval + Train/Dev 划分                        │
│    ├── 多标注裁决 (Gemini/Qwen/Codex)                        │
│    └── blacklist 隔离评测集                                  │
├─────────────────────────────────────────────────────────────┤
│  Step 9: Reranker LoRA 训练                                  │
│    ├── Qwen3-Reranker-0.6B-seq-cls/BGE-v2-m3                 │
│    ├── Listwise Softmax-CE Loss                              │
│    └── PEFT + Accelerate + wandb                             │
├─────────────────────────────────────────────────────────────┤
│  Step 10: 评测 (4组对照)                                     │
│    └── MRR@10 / NDCG@10 / P@10                               │
└─────────────────────────────────────────────────────────────┘
```

**说明**：
- Step 3-5 的"三路检索基线"是微调 Reranker 之前的候选召回与排序起点
- Step 6 的 Answer Matching 使用 `finglm_master.jsonl` 中的答案信号进行正例/负例判定
- Step 7 的"多标注裁决"指多个模型分别对 Gold Eval 候选相关性做判定，再汇总裁决

| 模块 | 功能 | 关键代码 |
|------|------|----------|
| 数据分块 | PDF → parent/child 层级 chunks | `src/chunk_manager.py` |
| 多路召回 | Embedding + BM25 + RRF 融合 | `src/embedding_retriever.py` / `src/bm25_retriever.py` |
| Reverse Mining | QA 标准答案 → 训练 triplets | `scripts/06_reverse_mining.py` / `src/answer_matcher.py` |
| Reranker 微调 | LoRA + Listwise Loss | `scripts/09_train_reranker.py` |
| Gold Eval | 多模型裁决 + 评测 | `scripts/10_evaluate.py` / `scripts/07b_adjudicate_gold_eval.py` |

模块详细设计见 [`docs/architecture.md`](docs/architecture.md)。

---

## 项目范围与数据规模

**边界条件（Phase1）**
- 仅覆盖 Type1（单公司、单年份财报问答）
- 仅做**同文档证据排序**（不做跨 PDF 检索）
- 检索范围限制为 query 对应 `pdf_stem` 的 chunks

**数据规模**
- 训练/验证：1,274 训练 triplets、247 验证 triplets
- Gold Eval：50 条黄金评测查询
- Reverse Mining：1,871 条 triplets，3,069 个 hard negatives（平均 12.58 个 / 有效 query）

---

## 快速开始

> 如果只想复现弱监督训练与评测，repo 已预置 retrieval / mining / train / eval 相关中间产物，`git clone` 后可直接运行 Step A 及之后的脚本。如果想从原始 PDF 开始，需要额外下载 PDF 和 chunks。

### 环境安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Pipeline 概览

```
PDF → ① 分块 → ② 三路检索 → ③ Reverse Mining → ④ 微调 → ⑤ 评测
       ⚠️ 需下载      ⚠️ 需下载      ✅ 自带数据      ✅ 自带数据  ✅ 自带数据
```

- `✅ 自带数据`：repo 中已预置该步骤的输入/输出文件，`git clone` 后可直接运行
- `⚠️ 需下载`：PDF 原文和分块结果体积较大，不在 repo 中。可自行跑分块脚本（较慢，见 [docs/reproduction.md](docs/reproduction.md) Step 3）

### Step A: Reverse Mining — 从 QA 答案中自动挖掘训练数据

> **在干什么**：读取 FinGLM 标准答案库（`finglm_master.jsonl`），用 key-value 规则匹配三路检索的 Top-50 结果。匹配成功 → 正例；高排名但未匹配 → hard negative。自动产出 (query, positive, negatives) triplets。
>
> `finglm_master.jsonl` 中的 answer 不是 chunk 标签，而是 Reverse Mining 的匹配依据。

```bash
python scripts/06_reverse_mining.py --stage stage1
```

- **输入**：`data/output/retrieval/hybrid_rrf_top50_stage1.jsonl`（预置）+ `finglm_data_store/finglm_master.jsonl`（预置）
- **输出**：`data/output/mining/mined_triplets_stage1.jsonl`
- **验证**：`head -1 data/output/mining/mined_triplets_stage1.jsonl | python -m json.tool`

### Step B: 训练数据准备 — 过滤噪声 + 划分训练/验证集

> **在干什么**：对上一步的 triplets 做二次过滤——排除评测集 blacklist 中的 query，再筛掉 confidence < 0.7 的噪声样本，最后按 9:1 划分 train/dev。

```bash
python scripts/08_prepare_train_data.py --stage stage1
```

- **输入**：`data/output/mining/mined_triplets_stage1.jsonl` + `config/eval_blacklist.json`（均预置）
- **输出**：`data/output/train/train_triplets_stage1.jsonl`（~1,274 条）、`dev_triplets_stage1.jsonl`（~247 条）
- **验证**：`wc -l data/output/train/*.jsonl`

### 后续步骤

- **微调（Step ④）**：用 Step B 产出的 triplets 训练 Reranker LoRA adapter（需 GPU）。repo 已预置训练好的权重（`data/output/artifacts/reranker/`）
- **评测（Step ⑤）**：用 Gold Eval 对比 Base vs Finetuned，输出 MRR/NDCG/P 指标。预置结果见 [`docs/architecture.md`](docs/architecture.md) 的 SSOT 章节
- 完整复现流程（Step 0-10）见 [`docs/reproduction.md`](docs/reproduction.md)

---

## 项目结构

```
QAnchor/
├── scripts/                  # 全流程脚本（01-10）
├── src/                      # 核心模块（chunk_manager / retriever / answer_matcher / ...）
├── config/
│   ├── weak_supervision_config.yaml   # 主配置
│   ├── zenparse_config.yaml           # 分块配置
│   └── eval_blacklist.json            # 评测隔离列表
├── data/
│   ├── input/                # 原始 PDF 数据
│   └── output/               # 全部产出（chunks / retrieval / mining / train / eval / annotations）
├── docs/                     # 项目文档
└── finglm_data_store/        # FinGLM 问答映射、标准答案索引与统计文件
```

**关键配置文件**：`config/weak_supervision_config.yaml` 统一管理分块、检索、融合参数。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/architecture.md`](docs/architecture.md) | 五大模块详解（Input/Output/Implementation）、SSOT 证据路径、统计稳健性 |
| [`docs/reproduction.md`](docs/reproduction.md) | 完整复现指南（Step 0-10），含所有脚本命令与参数 |
| [`docs/faq.md`](docs/faq.md) | 技术 FAQ：设计决策与核心问题解答 |
| [`docs/future_work.md`](docs/future_work.md) | 后续路线图：数据质量 → 负例策略 → 评测稳健性 |
| [`docs/retrieval_config_contract.md`](docs/retrieval_config_contract.md) | 检索配置契约：融合模式与参数生效规则 |

---

## 设计决策

### 为什么只做 Type1 / 同文档检索

Phase1 仅覆盖 Type1（单公司、单年份财报问答），检索范围限定在 query 对应的 `pdf_stem` 内。这降低了跨文档噪声，使排序优化目标更聚焦——在单个 PDF 的 chunk 集合中重新排好序，而非跨文档召回。

### 为什么选 0.6B 小模型

LoRA 微调在单卡 RTX 4090（24GB）上完成，评测在 Apple M2 Max（MPS）上运行。0.6B 经弱监督微调后 MRR@10 从 0.6115 提升至 0.7758（+26.9%），在性能和工程门槛之间取得了较好的平衡。

### 为什么用 Listwise 而不是 Pairwise

Listwise Softmax-CE Loss 直接对齐 MRR/NDCG 的排序目标，相比 Pairwise 更适合 Top-k 排序场景。原理说明详见 [docs/faq.md](docs/faq.md)。

---

## 致谢

本项目检索评测依赖以下开源项目：
- [ZenParse](https://github.com/AIDC-WX/ZenParse) — PDF 结构化解析
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) — 向量检索基座模型
- [Qwen3-Reranker-0.6B-seq-cls](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls) — Cross-Encoder Reranker 基座模型（最佳）
- [BGE-v2-m3](https://huggingface.co/BAAI/bge-m3) — Cross-Encoder Reranker 对比模型

**可替换其他模型**：Embedding 和 Reranker 的模型路径通过 `config/weak_supervision_config.yaml` 配置。切换 Reranker 时需注意 `--pair-format` 参数——不同模型的输入格式不同（如 Qwen3 用 `qwen3_template`，通用模型用 `hf_pair`），脚本支持 `auto` 自动检测，也可手动指定。详见 [docs/reproduction.md](docs/reproduction.md) Step 7。
