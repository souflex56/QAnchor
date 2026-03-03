# Future Work: Hard Negative 选择的改进方向

## 概述

本文档探讨QAnchor项目中Hard Negative选择的当前实现、业界标准方法，以及可能的改进方向。

---

## 改进路线图

**核心思路**: 数据质量改进 → Hard Negative 策略优化 → 训练策略提升

```
阶段0: CE 复审（数据去噪）     ← 解决假Hard Negative问题
  ├─ Cross-Encoder 三态分类
  ├─ 隔离不确定样本
  └─ 输出纯净的 pos/verified_neg

阶段1: 混合策略（推荐首选）
  ├─ 50% 检索器排名 + 50% 模型置信度
  └─ 成本低，见效快

阶段2: Curriculum Learning
  ├─ 训练早期用检索器方法
  └─ 训练后期用模型方法

阶段3: Online Hard Negative Mining
  ├─ 每个 batch 动态选择
  └─ 成本高，SOTA 效果
```

**实施顺序**: 必须先完成阶段0（数据去噪），再考虑阶段1-3（Hard Negative策略）。数据质量是基础。

---

## 当前方法评估

QAnchor当前使用**基于检索排名的Hard Negative选择**：

```python
# scripts/06_reverse_mining.py:209
neg_selected = neg_candidates[:neg_ratio * len(pos_chunks)]
```

**逻辑**：选择检索器排名高但不是正例的样本

**具体实现**：
1. 遍历Top50检索结果（按score从高到低排序）
2. 使用规则匹配识别正例
3. 未匹配的样本进入`neg_candidates`（保持原始排序）
4. 取`neg_candidates`的前n个作为Hard Negatives

---

## 业界方法对比

| 方法 | 逻辑 | 计算成本 | 工业界采用 | 学术界采用 | 代表工作 |
|------|------|----------|-----------|-----------|---------|
| **方法1: 检索排名**（QAnchor当前） | 选检索器排名高但非正例的样本 | 1x | ⭐⭐⭐⭐⭐ | ⭐⭐ | Contriever, E5, BGE |
| **方法2: 模型置信度** | 选当前模型最容易误判的样本 | 2x | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ANCE, RocketQA |
| **方法3: Loss值** | 选loss值最大的样本（决策边界） | 3x | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | OHEM, Dense Retriever |

---

## 当前方法的优缺点

### ✅ 优点

1. **简单高效**：不需要额外的模型推理，数据预处理阶段完成
2. **理论基础**：检索器本身就是训练好的相似度模型（BM25/Embedding）
3. **工业验证**：BGE-M3/M3/M3等主流检索模型都使用此方法
4. **可解释性强**：Hard Negative的定义清晰（检索器认为相关但实际不相关）

### ❌ 缺点

1. **不是从重排模型视角选Hard Negative**
   - 用的是检索器（BM25/Embedding）的排序
   - 但训练的是重排模型（Reranker）
   - 两个模型的"难易判断标准"可能不一致

   **例子**：
   ```
   Query: "宁德时代2023年动力电池装机量"

   Rank 1: "2022年装机量120GWh"
   → 检索器认为最相关（关键词完全匹配）
   → 但对重排模型可能很简单（一眼看出年份错误）

   Rank 28: "2023年动力电池装机量165GWh"
   → 检索器排后面（关键词分散）
   → 但这才是真正的正例
   ```

2. **没有利用训练过程中的反馈**
   - Hard Negative选择是静态的（离线生成）
   - 随着模型训练，"hard"的定义应该动态改变

   ```
   Epoch 1: "2022年营收" 可能是hard negative（模型还不懂年份）
   Epoch 5: "2022年营收" 可能太简单了（模型已学会）
   ```

3. **可能选到"假Hard Negative"**（⚠️ **核心问题**）
   - 如果Rank1-10都是正例，Rank11-13就成了Hard Negative
   - 但Rank11可能确实是正确答案，只是标注/匹配问题没被识别

   **例子**：
   ```
   Query: "宁德时代2023年动力电池装机量"
   Answer: "165GWh"

   Top-10 检索结果：
   Rank 1-3: "2022年...120GWh" → 规则匹配到 "2022年" → 正例
   Rank 4-7: "2023年...165GWh" → 规则未匹配（关键词分散）→ 被误判为负例！
   Rank 8-10: "2021年...95GWh" → 规则匹配 → 正例

   结果：Rank4-7的真正正例被当作Hard Negative训练！
   ```

   **影响**：
   - 模型学到错误信号（"正确答案"被当作负例）
   - 严重污染训练数据
   - 模型性能上限被锁死

---

## 阶段0: Cross-Encoder 复审（数据去噪）

**目标**: 解决"假Hard Negative"问题，输出纯净的训练数据。

### 核心思想

使用 Cross-Encoder (CE) 对规则未匹配的候选进行**三态分类**：

| 状态 | 定义 | 处理方式 |
|------|------|---------|
| **Positive** | 规则匹配成功 | 加入正例池 |
| **Uncertain** | 规则未匹配 + CE分数 ≥ 0.75 | **隔离**，不参与训练 |
| **Verified Negative** | 规则未匹配 + CE分数 ≤ 0.35 | 加入负例池（已验证） |
| **Unreviewed Negative** | 规则未匹配 + Rank > 10 | 加入负例池（低风险） |

### 实现逻辑

```python
def classify_with_ce(rule_matched, ce_score, rank):
    """三态分类逻辑"""
    if rule_matched:
        return "positive"
    elif ce_score >= 0.75:  # CE认为高度相关
        return "uncertain"  # 隔离，可能是假负例
    elif ce_score <= 0.35:  # CE明确不相关
        return "verified_negative"
    elif rank > 10:  # 低排名，不太可能是正例
        return "unreviewed_negative"
    else:
        return "unreviewed_negative"  # 默认负例
```

### 数据流变化

**Before (2-pool)**:
```
规则匹配 → Positive
规则未匹配 → Negative (直接训练)
```

**After (4-pool)**:
```
规则匹配 → Positive
规则未匹配 → CE复审 → Uncertain (隔离)
                      → Verified Negative
                      → Unreviewed Negative
```

### 预期效果

| 指标 | Before | After | 提升 |
|------|--------|-------|------|
| 假Hard Negative率 | ~15-30% | <5% | ⬇️ 80%+ |
| 训练数据质量 | 中等 | 高 | ⬆️ 显著 |
| 模型性能上限 | 受限 | 解锁 | ⬆️ 5-10% |

### 实施成本

- **计算成本**: +1x（CE模型推理，可批处理优化）
- **实施难度**: ⭐⭐（中等）
- **风险**: 低（只隔离不确定样本，不影响原有正负例）

---

## 阶段1: 混合策略（低成本，推荐首选）

### 方案1：混合策略（成本低，推荐首选）

**思路**：50%用检索器排名，50%用模型置信度

```python
def select_hybrids_hard_negatives(query, pos_chunks, neg_candidates, model, n=3):
    """混合策略：结合检索器排名和模型置信度"""
    hard_negatives = []

    # 50%用检索器方法（当前方法）
    retrieval_ratio = 0.5
    n_retrieval = int(n * retrieval_ratio)
    hard_negatives.extend(neg_candidates[:n_retrieval])

    # 50%用模型置信度方法
    n_model = n - n_retrieval
    # 计算模型对每个负例的预测分数
    model_scores = []
    for neg in neg_candidates:
        with torch.no_grad():
            score = model.predict(query, pos_chunks[0], neg)
        model_scores.append((score, neg))

    # 选模型分数最高的（模型认为最难区分的）
    model_scores.sort(key=lambda x: x[0], reverse=True)
    hard_negatives.extend([neg for score, neg in model_scores[:n_model]])

    return hard_negatives
```

**优点**：
- 计算成本增加不大（~1.5x）
- 结合了检索器优势和模型视角
- 实现相对简单
- 可以调节hybrid_ratio来平衡两种方法

**缺点**：
- 仍然需要额外的模型推理
- 需要仔细调参（retrieval_ratio）

---

### 方案2：动态Curriculum（中等成本）

**思路**：训练早期用检索器排名，训练后期用模型置信度

```python
def select_curriculum_hard_negatives(query, pos_chunks, neg_candidates, model,
                                     n=3, epoch=0, warmup_epochs=3):
    """Curriculum策略：随训练进度调整难度"""
    if epoch < warmup_epochs:
        # 早期：用检索器方法（简单负例为主）
        return neg_candidates[:n]
    else:
        # 后期：用模型方法（困难负例为主）
        model_scores = []
        for neg in neg_candidates:
            with torch.no_grad():
                score = model.predict(query, pos_chunks[0], neg)
            model_scores.append((score, neg))

        model_scores.sort(key=lambda x: x[0], reverse=True)
        return [neg for score, neg in model_scores[:n]]
```

**优点**：
- 符合Curriculum Learning理论（从简单到困难）
- 随模型能力自动调整难度
- 训练更稳定，收敛更快

**缺点**：
- 需要定义warmup期
- 早期可能浪费一些hard negatives

---

### 方案3：Online Hard Negative Mining（高成本，最先进）

**思路**：每个batch都重新计算loss，选loss最大的样本

```python
def select_online_hard_negatives(query, pos, neg_candidates, model, n=3):
    """Online Hard Example Mining：选loss最大的"""
    losses = []

    for neg in neg_candidates:
        # 计算该负例的loss
        loss = model.compute_loss(query, pos, neg)
        losses.append((loss.item(), neg))

    # 选loss最大的n个（位于决策边界附近）
    losses.sort(key=lambda x: x[0], reverse=True)
    return [neg for loss, neg in losses[:n]]

# 在训练循环中使用
for epoch in range(num_epochs):
    for batch in dataloader:
        query, pos, neg_candidates = batch

        # 每个batch动态选择hard negatives
        hard_negs = select_online_hard_negatives(
            query, pos, neg_candidates, model, n=3
        )

        # 用选定的hard negatives计算最终loss
        loss = model.compute_loss(query, pos, hard_negs)
        loss.backward()
        optimizer.step()
```

**优点**：
- 精确打击模型学不好的边界样本
- 理论基础扎实（对比学习loss反映样本难度）
- 学术界认可度高（顶会论文常用）

**缺点**：
- 计算成本高（~3x，每个负例都要算loss）
- 可能不稳定（loss波动大）
- 实现复杂，需要修改训练循环

---

## 实施建议

### 按场景推荐

| 场景 | 推荐方案 | 理由 | 预期收益 |
|------|----------|------|---------|
| **数据质量差** | **阶段0（CE复审）** | 必须先去噪，否则所有策略都受污染 | +5-15% MRR |
| **快速验证/上线** | 阶段0 + 当前方法 | 基础数据质量保证 | baseline |
| **写顶会论文** | 阶段0 + 阶段2或阶段3 | 完整pipeline，SOTA效果 | +8-20% MRR |
| **大规模生产** | 阶段0 + 阶段1混合 | 平衡效果和成本 | +7-12% MRR |
| **资源受限** | 阶段0 + curriculum | 数据质量优先 + 低成本改进 | +6-10% MRR |

### 实施优先级

**阶段0（必做，数据质量基础）**：
- 实现CE复审，三态分类
- 隔离不确定样本（Uncertain）
- 输出纯净的 pos/verified_neg 数据
- **验证**: 检查uncertain比例是否 >10%（说明数据污染严重）

**阶段1（低成本，快速验证）**：
- 在阶段0基础上，实现混合策略
- 设置hybrid_ratio=0.5（50:50）
- 运行小规模实验验证效果

**阶段2（中期，稳定改进）**：
- 如果阶段1有效，实施Curriculum
- 调优warmup_epochs参数
- 进行完整训练和评测

**阶段3（长期，探索前沿）**：
- 如果需要SOTA效果，实施Online Mining
- 配合其他优化（如更大的batch size）

---

## 代码实现位置

如果要实现改进，主要需要修改以下文件：

### 1. `scripts/06_reverse_mining.py`

**当前代码（第209行）**：
```python
neg_selected = neg_candidates[:neg_ratio * len(pos_chunks)]
```

**改进代码**：
```python
# 新增参数
parser.add_argument("--hard-neg-strategy", type=str, default="retrieval",
                    choices=["retrieval", "hybrid", "curriculum", "online"])
parser.add_argument("--hybrid-ratio", type=float, default=0.5)

# 在mine_triplets函数中实现新策略
if strategy == "retrieval":
    neg_selected = neg_candidates[:n]
elif strategy == "hybrid":
    neg_selected = select_hybrid_hard_negatives(...)
# ...
```

### 2. `scripts/09_train_reranker.py`

**需要添加**：
- 动态Hard Negative选择逻辑
- Online Hard Negative Mining支持
- Curriculum Learning的epoch判断

**示例代码**：
```python
# 在训练循环中
for epoch in range(args.epochs):
    for batch in train_dataloader:
        if args.hard_neg_strategy == "online":
            # 每个batch动态选择
            hard_negs = select_online_hard_negatives(...)
        else:
            # 使用预先生成的hard negatives
            hard_negs = batch['negatives']

        loss = model(query, pos, hard_negs)
        loss.backward()
```

### 3. `config/weak_supervision_config.yaml`

**新增配置项**：
```yaml
hard_negative:
  strategy: "hybrid"  # retrieval | hybrid | curriculum | online

  # 混合策略参数
  hybrid_ratio: 0.5  # 检索器方法占比

  # Curriculum策略参数
  curriculum_warmup: 3  # 使用检索器方法的epoch数

  # Online策略参数
  online_sample_size: 20  # 从多少个候选中选
```

---

## 预期效果

### MRR提升预估

| 方案 | 预期MRR@10提升 | 计算成本增加 | 实施难度 | 风险 |
|------|---------------|-------------|---------|------|
| 当前方法 | baseline (0.7758) | 0x | ⭐ | 低 |
| **阶段0（CE复审）** | **+5-10%** | +1x | ⭐⭐ | 低 |
| 阶段1（混合） | +7-12% | +1.5x | ⭐⭐ | 低 |
| 阶段2（Curriculum） | +8-15% | +2x | ⭐⭐⭐ | 中 |
| 阶段3（Online） | +10-20% | +3x | ⭐⭐⭐⭐ | 高 |

**注**: 阶段0的收益是独立的，可与阶段1-3叠加。

### 其他潜在收益

1. **更好的泛化能力**：模型在难样本上训练，泛化到真实场景时表现更好
2. **更快的收敛**：Curriculum方法可以加速训练收敛
3. **更少的训练轮次**：Online Mining可能减少需要的总epoch数

---

## 参考文献

### 相关论文

1. **KNN-LM** (Khandelwal et al., 2021)
   - 使用retriever作为hard negative来源
   - 展示了检索based方法的有效性

2. **Contriever** (Izacard et al., 2022)
   - 对比学习训练检索模型
   - 使用检索结果作为hard negative
   - arXiv:2112.09118

3. **E5** (Wang et al., 2022)
   - 使用query检索到的非相关文档作为hard negative
   - 提出了困难负例挖掘的框架
   - arXiv:2212.03533

4. **BGE** (Xiao et al., 2023)
   - 混合多种hard negative策略
   - 工业界实践经验
   - GitHub:FlagOpen/FlagEmbedding

5. **OHEM** (Shrivastava et al., 2016)
   - Online Hard Example Mining在目标检测中的应用
   - CVPR 2016

6. **ANCE** (Xiong et al., 2020)
   - 使用模型置信度选择hard negatives
   - Approximate Nearest Neighbor Negative Contrastive Estimation
   - arXiv:2007.00808

7. **RocketQA** (Qu et al., 2021)
   - Curriculum Learning for dense retrieval
   - EMNLP 2021

---

## 附录：实验设计建议

### A/B测试方案

如果要验证改进效果，建议按以下方式设计实验：

| 实验组 | Hard Negative策略 | 目的 |
|-------|------------------|------|
| Baseline | 当前方法（检索排名） | 对比基准 |
| Exp-1 | 混合策略（ratio=0.3） | 验证少量模型方法的收益 |
| Exp-2 | 混合策略（ratio=0.5） | 验证平衡策略 |
| Exp-3 | 混合策略（ratio=0.7） | 验证更多模型方法的收益 |
| Exp-4 | Curriculum（warmup=3） | 验证动态难度调整 |
| Exp-5 | Online Mining | 验证SOTA方法 |

### 评估指标

除了MRR@10，还应该关注：

1. **不同位置的表现**：NDCG@1, NDCG@5, NDCG@10
2. **困难查询的改进**：baseline排名低的query是否有提升
3. **训练效率**：收敛速度、总训练时间
4. **鲁棒性**：不同数据分布下的表现

---

## 结论

QAnchor当前使用的基于检索排名的Hard Negative选择方法是**工业界标准实践**，具有简单、高效、可靠的特点。

**但是**，当前方法存在一个**致命缺陷**：可能将真正的正例误判为Hard Negative（假Hard Negative问题），严重污染训练数据。

**改进路线图**：

1. **阶段0（CE复审）是基础**：必须先解决数据质量问题，否则任何训练策略优化都无法发挥应有效果。
   - 预期收益：+5-10% MRR（独立收益）
   - 实施难度：⭐⭐（中等）
   - 建议：**所有场景都必须先做**

2. **阶段1-3是优化**：在数据质量保证的基础上，再考虑Hard Negative策略优化。
   - 预期收益：+2-10% MRR（叠加收益）
   - 建议：根据场景选择

**总结**：
- 对于**快速上线**：阶段0 + 当前方法（已足够）
- 对于**学术研究**：阶段0 + 阶段2/3（完整pipeline，SOTA效果）
- 对于**工业应用**：阶段0 + 阶段1（性价比最高）

---

*文档版本: v2.0*
*创建日期: 2025-03-02*
*更新日期: 2026-03-03*
*作者: QAnchor Team*
