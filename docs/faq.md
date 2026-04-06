# 技术 FAQ

> QAnchor 项目核心设计决策与技术要点（Q&A 形式）。

---

- **Q：这不就是一个 RAG demo 吗？**
  **A：不是。**我解决的是"金融长文档 + 缺标注数据"的落地难题：通过结构化分块、多路召回与排序闭环，把检索从"能跑通"提升到"有指标保障"。

- **Q：没有标注数据怎么训练 Reranker？**
  **A：用 Reverse Mining 自动构建训练集。**先从 QA 标准答案里抽取关键值，再去检查 Top-50 召回结果里哪些片段包含答案。匹配上的作为正例，高排名但未匹配的作为 hard negatives（neg_ratio=3）。Stage1 共产出 1,871 条 triplets、3,069 个 hard negatives。

  关于”平均负例数”：Reverse Mining 阶段统计的 12.58（= 3,069 / 244 个有效 query）和对照报告附录中的 14.4（训练集按 query 统计的均值）使用了不同的分母——前者是 mining 原始产出按有效 query 求均值，后者是经过 blacklist 过滤 + confidence 筛选后按训练集 query 求均值。两者不矛盾。

- **Q：为什么用 Listwise Loss，而不是 Pairwise？**
  **A：Pairwise 只学"局部胜负"，Listwise 直接对齐"整组排序"。**下面是更细节的解释：
  **1) "拆成 N 组二选一"是什么意思（Pairwise）？**
  假设一个 query 有 1 个正例 P 和 3 个负例 N1/N2/N3。Pairwise 会拆成 3 条训练样本：
  - (P vs N1)
  - (P vs N2)
  - (P vs N3)
  每次只学"P 比某个 Ni 好"。这叫"局部比较"。
  **问题**：它不关心 N1、N2、N3 之间的排序，也不保证 P 一定是全场第一。只要 P 比每个负例"略高一点"，pairwise 就满足了，但这不一定对应检索排序中"正例必须排第一"的目标。
  **2) "压低所有 hard negatives"是什么意思（Listwise）？**
  Listwise 把 P + 所有 hard negatives 当作一个整体候选列表。模型输出一串分数 `[s_pos, s_n1, s_n2, ...]`，然后做 softmax，让正例成为整组里概率最大的一项：
  ```text
  loss = -log_softmax(scores)[0]
  ```
  这会迫使模型**同时把所有负例压下去**，因为只要任何一个负例分数靠得太近，softmax 概率就会被"抢走"。
  3) **为什么说 Listwise 和业务指标对齐？**
  检索评测看的是排名位置（MRR/NDCG），尤其是"正例排第几"。Listwise 的训练目标是"在一组候选里，正例排第一"，与评测目标一致。
  4) **项目中 Listwise 的具体实现？**（`scripts/09_train_reranker.py`）
  - `TripletCollator` 组织候选：`candidates = [pos_text] + neg_texts`，并记录 `group_sizes`。
  - `ListwiseTrainer.compute_loss`：`loss_i = -torch.log_softmax(group_scores, dim=0)[0]`，index 0 即正例。

- **Q：你如何证明增益是真的？**
  **A：两层证据。**先看同口径 Gold Eval：`embedding_only -> hybrid_rrf` 的 MRR@10 从 0.4115 到 0.5756（+39.9%），且 `unjudged_rate=0`。再看配对统计报告：MRR/NDCG 的 bootstrap 95%CI 均为正，说明不是随机抖动；P@10 也上升，但显著性更边界。

- **Q：工程约束下怎么保证检索质量？**
  **A：限定同文档检索 + Embedding/BM25/RRF 三路召回。**避免跨 PDF 噪声，同时提升召回覆盖与精度上限。

- **Q：为什么要做 parent/child 分块？**
  **A：检索粒度和理解粒度需要分离。**child 用于精确召回（`chunk_level=child`），parent 作为更完整上下文（parent_size=4000 / child_size=1200）。

- **Q：为什么不只用向量检索？**
  **A：语义召回和精确匹配各有盲点。**Embedding 擅长语义同义，BM25 擅长年份/数值等精确匹配，RRF 融合提升稳健性。

- **Q：如何保证实验可复现？**
  **A：所有关键配置与指标都固化为 SSOT 文件。**评测结果与配置分别落在 `metrics_comparison_*.json` 与 `eval_config_*.json`，可直接复核。

- **Q：为什么 Phase1 不做跨 PDF 检索？**
  **A：这是问题域约束。**Type1 任务对应单公司单年份财报，跨文档会引入噪声与错误负例，因此限制 `pdf_stem` 能提升训练与评测稳定性。

- **Q：为什么必须用 `qwen3_template`？**
  **A：模板对性能影响显著。**对照报告显示 legacy 模板会显著拉低 Base MRR，而 `qwen3_template` 与 Qwen3 官方格式一致，能稳定释放模型能力。

- **Q：训练成本如何控制？**
  **A：用 LoRA + 小 batch + 梯度累积。**只微调投影层参数（q/k/v/o），再配合小 batch 与梯度累积；检索侧使用 embedding 缓存减少重复编码。

- **Q：如何避免评测数据泄露？**
  **A：用 blacklist + 按 query_id 划分。**Gold Eval 构建时引入 blacklist 隔离；训练集准备阶段会过滤 blacklist 并按 query_id 划分 train/dev。

- **Q：弱监督噪声怎么控制？**
  **A：分两层阈值过滤。**Reverse Mining 阶段用 confidence≥0.5 判正例，训练准备阶段再以 confidence≥0.7 过滤噪声样本。

- **Q：Qwen3 和 BGE 怎么取舍？**
  **A：性能 vs 资源的权衡。**Qwen3-768 在本次评测中 MRR 最高；BGE-512 在资源受限时更稳妥（性能接近、推理成本低）。
