# 检索配置契约（Retrieval Config Contract）

本文档描述当前 `scripts/05_three_way_retrieval.py` 的真实生效行为。

## 1. 范围与目标

- 检索策略仍为两路：`embedding + bm25`。
- 混合融合支持三种模式：
  - `rrf`
  - `weighted_sum`
  - `max`
- 非法 `fusion_method` 不中断流程，输出 warning 后回退到 `rrf`。

## 2. 融合模式定义

统一入口：`src/hybrid_fusion.py::fuse_two_way(...)`

### 2.1 `fusion_method = "rrf"`

公式：

`score = w_e / (rrf_k + rank_e) + w_b / (rrf_k + rank_b)`

- 缺失一路命中时，rank 使用 `missing_rank`。
- `w_e / w_b` 来自 `embedding_weight / bm25_weight`。
- 未显式配置权重时，默认等权（`w_e = 1.0`, `w_b = 1.0`）。

### 2.2 `fusion_method = "weighted_sum"`

先按每 query、每路做 Min-Max 归一化：

`norm = (x - min) / (max - min)`

再加权求和：

`score = w_e * norm(score_e) + w_b * norm(score_b)`

边界：
- 若该路 `max == min`，该路归一化结果统一置 `0`。
- 缺失该路 score 记为 `0`。

### 2.3 `fusion_method = "max"`

同样先做 Min-Max 归一化，再取加权最大：

`score = max(w_e * norm(score_e), w_b * norm(score_b))`

边界规则与 `weighted_sum` 相同。

## 3. 参数生效矩阵

| 参数 | rrf | weighted_sum | max | 说明 |
|:--|:--:|:--:|:--:|:--|
| `hybrid.fusion_method` | ✅ | ✅ | ✅ | 非法值 warning + 回退 `rrf` |
| `hybrid.rrf_k` | ✅ | ❌ | ❌ | 非 RRF 模式会 warning：参数不生效 |
| `hybrid.missing_rank` | ✅ | ❌ | ❌ | 非 RRF 模式会 warning：参数不生效 |
| `hybrid.embedding_weight` | ✅ | ✅ | ✅ | 未配置时默认 1.0 |
| `hybrid.bm25_weight` | ✅ | ✅ | ✅ | 未配置时默认 1.0 |

## 4. 输出兼容与产物命名

- 仍保留融合结果中的兼容字段：
  - `embedding_rank`, `bm25_rank`
  - `embedding_score`, `bm25_score`
  - `score`, `rank`
- 输出文件前缀按生效模式命名：
  - `hybrid_rrf_top*.jsonl`
  - `hybrid_weighted_sum_top*.jsonl`
  - `hybrid_max_top*.jsonl`

## 5. 脚本运行时可观测性

`05_three_way_retrieval.py` 会输出以下关键 warning/info：

1. 非法 `fusion_method`：回退到 `rrf`
2. `rrf` 且显式配置权重：提示启用加权 RRF
3. `weighted_sum/max` 下配置了 `rrf_k/missing_rank`：提示不生效
4. `weighted_sum/max` 且未配置权重：提示使用默认等权

## 6. checkpoint 审计字段

`stats["params"]` 中新增：

- `fusion_method_config`
- `fusion_method_effective`
- `weights_effective`
- `score_normalization`（`weighted_sum/max` 为 `minmax`）
- `embedding_weight`
- `bm25_weight`

## 7. 兼容性说明

- 当 `fusion_method=rrf` 且未配置权重时，排序行为与旧版等权 RRF 保持一致。
- `weighted_sum` / `max` 属于新增能力，结果与旧版（仅 RRF）不可直接等价对比。

## 8. 测试文件说明与用法

### 8.1 `tests/test_hybrid_fusion.py`

作用：
- 覆盖融合核心算法的单元测试（快速、稳定）。
- 验证以下关键点：
  1. `rrf` 等权与历史公式一致
  2. `rrf` 加权后排序会变化
  3. `weighted_sum` 的 Min-Max 归一化生效
  4. `max` 的归一化后取最大逻辑正确
  5. 单路缺失命中时行为正确
  6. `max==min` 边界归一化置 0

运行命令：

```bash
python -m unittest tests/test_hybrid_fusion.py -v
```

### 8.2 `tests/test_three_way_retrieval_integration.py`

作用：
- 覆盖脚本级集成回归（参数路由与产物行为）。
- 覆盖场景：
  1. `fusion_method=rrf`
  2. `fusion_method=weighted_sum`
  3. `fusion_method=max`
  4. `fusion_method=unknown`（warning + fallback 到 `rrf`）
- 同时验证 checkpoint 关键字段与输出文件命名。

运行命令：

```bash
python -m unittest tests/test_three_way_retrieval_integration.py -v
```

### 8.3 一次运行全部测试

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 8.4 临时文件清理保证

- 集成测试使用临时目录作为 `--output-dir` 与 `--checkpoint-path`。
- 测试结束后临时目录自动删除。
- 测试内会校验仓库 `data/output/retrieval` 与 `data/output/checkpoints` 没有新增污染文件。
