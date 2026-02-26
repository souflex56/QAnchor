# 检索配置契约（Retrieval Config Contract）

> **本文档描述 `scripts/05_three_way_retrieval.py` 的真实生效行为。**
>
> 💡 目的：记录"配置写了什么"与"代码实际做什么"之间的对应关系。

---

## 📋 快速概览

```
检索策略：两路融合（Embedding + BM25）
融合方法：3种（rrf / weighted_sum / max）
默认配置：fusion_method="rrf", embedding_weight=1.0, bm25_weight=1.0
```

---

## 1️⃣ 范围与目标

| 组件 | 文件路径 |
|:---|:---|
| **主脚本** | `scripts/05_three_way_retrieval.py` |
| **稠密检索** | `src/embedding_retriever.py` |
| **稀疏检索** | `src/bm25_retriever.py` |
| **混合融合** | `src/hybrid_fusion.py` |
| **配置文件** | `config/weak_supervision_config.yaml` |

**核心特性：**
- ✅ 检索策略：两路融合（`embedding + bm25`）
- ✅ 融合方法：支持 3 种模式（`rrf` / `weighted_sum` / `max`）
- ✅ 非法 `fusion_method` 不中断流程，输出 warning 后回退到 `rrf`

---

## 2️⃣ 融合模式详解

**统一入口：** `src/hybrid_fusion.py::fuse_two_way(...)`

### 2.1 RRF（Reciprocal Rank Fusion）

**公式：**
```
score = w_e / (rrf_k + rank_e) + w_b / (rrf_k + rank_b)
```

| 参数 | 配置项 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `w_e` | `embedding_weight` | 1.0 | embedding 路权重 |
| `w_b` | `bm25_weight` | 1.0 | BM25 路权重 |
| `rrf_k` | `rrf_k` | 60 | 平滑常数 |
| `missing_rank` | `missing_rank` | 9999 | 单路缺失时的默认排名 |

**特点：**
- 仅使用排名，不使用原始分数
- 对分数尺度差异鲁棒
- 支持加权融合

**示例计算：**
```
chunk_X: embedding_rank=5, bm25_rank=10
权重: w_e=1.0, w_b=1.0, rrf_k=60

score = 1.0/(60+5) + 1.0/(60+10)
      = 0.01538 + 0.01429
      = 0.02967
```

---

### 2.2 weighted_sum

**公式：**
```
# 先按每路做 Min-Max 归一化
norm = (score - min_score) / (max_score - min_score)

# 再加权求和
score = w_e * norm(score_e) + w_b * norm(score_b)
```

**边界处理：**
- 若 `max_score == min_score`，该路归一化结果统一置 `0`
- 缺失该路 score 记为 `0`

**特点：**
- 需要精确控制权重比例时使用
- 分数归一化后尺度统一

---

### 2.3 max

**公式：**
```
score = max(w_e * norm(score_e), w_b * norm(score_b))
```

**特点：**
- 取两路归一化分数的最大值
- 召回优先，适合高召回场景

---

## 3️⃣ 参数生效矩阵

| 配置项 | rrf | weighted_sum | max | 说明 |
|:---|:---:|:---:|:---:|:---|
| `fusion_method` | ✅ | ✅ | ✅ | 非法值 warning + 回退 `rrf` |
| `embedding_weight` | ✅ | ✅ | ✅ | 未配置时默认 1.0 |
| `bm25_weight` | ✅ | ✅ | ✅ | 未配置时默认 1.0 |
| `rrf_k` | ✅ | ❌ | ❌ | 非 RRF 模式会 warning |
| `missing_rank` | ✅ | ❌ | ❌ | 非 RRF 模式会 warning |

---

## 4️⃣ 运行时可观测性

`05_three_way_retrieval.py` 会输出以下关键日志：

| 场景 | 日志级别 | 输出内容 |
|:---|:---:|:---|
| 非法 `fusion_method` | WARN | 回退到 `rrf` |
| `rrf` + 显式配置权重字段 | WARN | 提示启用加权 RRF（即使是 1.0/1.0） |
| `rrf` + 等权（默认） | INFO | 提示使用等权融合 |
| `weighted_sum/max` + 有 `rrf_k` | WARN | 提示参数不生效 |
| `weighted_sum/max` + 等权（默认） | INFO | 提示使用默认等权 |

---

## 5️⃣ 输出产物

### 输出文件命名

```
hybrid_{fusion_method}_top{k}_{stage}.jsonl

示例：
- hybrid_rrf_top50_stage1.jsonl
- hybrid_weighted_sum_top50_stage1.jsonl
- hybrid_max_top50_stage1.jsonl
```

### 结果字段

每个融合结果包含：
- `score` - 融合得分（用于排序）
- `rank` - 融合排名
- `embedding_rank` - Embedding 路原始排名
- `bm25_rank` - BM25 路原始排名
- `embedding_score` - Embedding 路原始得分
- `bm25_score` - BM25 路原始得分
- `chunk_id`, `pdf_stem`, `page_numbers` 等元数据

---

## 6️⃣ Checkpoint 审计

`stats["params"]` 中记录：

```json
{
  "fusion_method_config": "rrf",
  "fusion_method_effective": "rrf",
  "weights_effective": true,
  "score_normalization": null,
  "embedding_weight": 1.0,
  "bm25_weight": 1.0,
  "rrf_k": 60,
  "missing_rank": 9999
}
```

---

## 7️⃣ 兼容性说明

| 场景 | 兼容性 |
|:---|:---|
| `fusion_method=rrf` + 等权（默认） | ✅ 与旧版等权 RRF 一致 |
| `fusion_method=rrf` + 加权 | ⚠️ 新功能：加权 RRF |
| `weighted_sum/max` | ⚠️ 新功能，结果与旧版不可直接对比 |

---

## 8️⃣ 测试说明

### 8.1 单元测试

**文件：** `tests/test_hybrid_fusion.py`

**覆盖内容：**
- RRF 等权与历史公式一致
- RRF 加权后排序变化
- weighted_sum 的 Min-Max 归一化
- max 的归一化后取最大逻辑
- 单路缺失命中时的行为
- `max==min` 边界归一化置 0

```bash
python -m unittest discover -s tests -p "test_hybrid_fusion.py" -v
```

### 8.2 集成测试

**文件：** `tests/test_three_way_retrieval_integration.py`

**覆盖场景：**
- `fusion_method=rrf`
- `fusion_method=weighted_sum`
- `fusion_method=max`
- `fusion_method=unknown`（warning + fallback）

```bash
python -m unittest discover -s tests -p "test_three_way_retrieval_integration.py" -v
```

### 8.3 运行全部测试

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

> **临时文件清理保证：** 集成测试使用临时目录，测试结束后自动删除，不会污染 `data/output/`。

---

## 9️⃣ 架构详解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QAnchor 检索架构 (v2.0)                              │
└─────────────────────────────────────────────────────────────────────────────┘

                                    Query (用户问题)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    ▼                                           ▼
    ╔═══════════════════════════════════╗    ╔═══════════════════════════════════╗
    ║   ① 稠密检索 (Embedding)          ║    ║   ② 稀疏检索 (BM25)              ║
    ╠═══════════════════════════════════╣    ╠═══════════════════════════════════╣
    ║ 文件: embedding_retriever.py      ║    ║ 文件: bm25_retriever.py          ║
    ║ 模型: Qwen3-Embedding-0.6B        ║    ║ 分词: jieba                       ║
    ║ 输出: 向量维度由模型决定            ║    ║ 参数: k1=1.5, b=0.75 (常见默认)  ║
    ║      （当前模型示例为 768 维）      ║    ║ 输出: BM25得分 (未归一化)         ║
    ║ 相似度: 余弦相似度（一般 [-1,1]）   ║    ║                                   ║
    ║ 配置: top_k=50, normalize=true    ║    ║ 配置: top_k=50                    ║
    ╚═══════════════════════════════════╝    ╚═══════════════════════════════════╝
                    │                                           │
                    │ ① Top-50 排名 + 分数                      │ ② Top-50 排名 + 分数
                    │ (embedding_rank, embedding_score)         │ (bm25_rank, bm25_score)
                    │                                           │
                    └─────────────────┬─────────────────────────┘
                                      │
                                      ▼
                    ╔═══════════════════════════════════════════════════════╗
                    ║        ③ Hybrid 融合 (hybrid_fusion.py)             ║
                    ╠═══════════════════════════════════════════════════════╣
                    ║                                                        ║
                    ║   当前配置 (config/weak_supervision_config.yaml):     ║
                    ║   • fusion_method: "rrf"                              ║
                    ║   • embedding_weight: 1.0   (等权)                    ║
                    ║   • bm25_weight: 1.0        (等权)                    ║
                    ║   • rrf_k: 60               (仅RRF生效)                ║
                    ║   • missing_rank: 9999      (仅RRF生效)                ║
                    ║                                                        ║
                    ║   ┌─────────────────────────────────────────────────┐  ║
                    ║   │ 方法1: RRF (默认，推荐)                         │  ║
                    ║   │   score = w_e/(k+r_e) + w_b/(k+r_b)             │  ║
                    ║   │   • 使用排名，不使用原始分数                     │  ║
                    ║   │   • 对分数尺度变化鲁棒                           │  ║
                    ║   │   • 支持加权融合 (w_e ≠ w_b 时)                  │  ║
                    ║   └─────────────────────────────────────────────────┘  ║
                    ║                                                        ║
                    ║   ┌─────────────────────────────────────────────────┐  ║
                    ║   │ 方法2: weighted_sum                              │  ║
                    ║   │   score = w_e·n_e + w_b·n_b                      │  ║
                    ║   │   • Min-Max归一化后加权求和                      │  ║
                    ║   │   • 需要精确控制权重比例时使用                   │  ║
                    ║   └─────────────────────────────────────────────────┘  ║
                    ║                                                        ║
                    ║   ┌─────────────────────────────────────────────────┐  ║
                    ║   │ 方法3: max                                       │  ║
                    ║   │   score = max(w_e·n_e, w_b·n_b)                  │  ║
                    ║   │   • 取两路归一化分数的最大值                     │  ║
                    ║   │   • 召回优先，适合高召回场景                     │  ║
                    ║   └─────────────────────────────────────────────────┘  ║
                    ╚═══════════════════════════════════════════════════════╝
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │  最终输出 (Top-50/20 chunks)  │
                        │  • hybrid_{method}_top*.jsonl │
                        │  • 包含: score, rank         │
                        │  • 保留 emb/bm25 原始分和排名 │
                        └─────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据流详细说明                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ① Embedding 路径:                                                          │
│     Query → Qwen3-Embedding → [0.12, -0.34, ...]                            │
│            （向量维度由模型决定；当前模型示例为 768 维）                     │
│            → 余弦相似度计算 (与每个chunk)                                    │
│            → 得分范围一般为 [-1, 1]                                         │
│            → 排序 → Top-50 排名 + 分数                                      │
│                                                                             │
│  ② BM25 路径:                                                               │
│     Query → jieba分词 → ["如何", "计算", "词频"]                            │
│            → TF-IDF + BM25公式计算                                          │
│            → 得分范围 (未归一化，可能很大)                                  │
│            → 排序 → Top-50 排名 + 分数                                      │
│                                                                             │
│  ③ RRF 融合示例 (等权):                                                      │
│     ┌─────────────────────────────────────────────────────────────────┐   │
│     │ chunk_X: embedding_rank=5, bm25_rank=10                          │   │
│     │ score = 1.0/(60+5) + 1.0/(60+10) = 0.02967                      │   │
│     │                                                                  │   │
│     │ chunk_Y: embedding_rank=1, bm25_rank=50                          │   │
│     │ score = 1.0/(60+1) + 1.0/(60+50) = 0.01644                      │   │
│     │                                                                  │   │
│     │ 最终排序: chunk_X (0.02967) > chunk_Y (0.01644)                  │   │
│     └─────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ④ weighted_sum 融合:                                                      │
│     先对每路做 Min-Max 归一化:                                              │
│     emb_norm = (emb_score - min_emb) / (max_emb - min_emb)                 │
│     bm_norm = (bm_score - min_bm) / (max_bm - min_bm)                      │
│     融合得分 = 1.0 * emb_norm + 1.0 * bm_norm                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔟 BM25 参数说明

| 参数 | 默认值 | 含义 | 是否可配置 |
|:---|:---|:---|:---|
| `k1` | 常见默认 1.5 | 词频饱和度参数 | ❌ 使用库默认 |
| `b` | 常见默认 0.75 | 文档长度归一化参数 | ❌ 使用库默认 |
| `epsilon` | 常见默认 0.25 | IDF 下界 | ❌ 使用库默认 |

> ⚠️ **工程风险**：若 `rank-bm25` 版本升级，默认值可能变化。建议在 requirements.txt 中固定版本。

---

## 📌 版本历史

| 版本 | 日期 | 主要变化 |
|:---|:---|:---|
| v1.0 | 2025-01 | 初始版本，仅支持等权 RRF |
| v2.0 | 2025-02 | 新增加权 RRF、weighted_sum、max 三种融合方法 |
