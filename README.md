# FinGLM 数据整理总览

## 背景
- FinGLM 原始数据集由 MetaGLM 开源（https://github.com/MetaGLM/FinGLM），基于上市公司年报构建金融问答对。
- 在 SMP 2023 ChatGLM 金融大模型挑战赛中，我们完成了多轮人工标注，共计 10,000 条：
  - 初赛：5,000 条；复赛 A：2,000 条；复赛 B：2,000 条；复赛 C：1,000 条。
- 代表性问答示例：
  - Q: 2019 年中国工商银行财务费用是多少元?  
    A: 2019 年中国工商银行财务费用是 12,345,678.9 元。
  - Q: 工商银行 2019 年营业外支出和营业外收入分别是多少元?  
    A: 营业外支出 12,345,678.9 元，营业外收入 2,345,678.9 元。
  - Q: 中国工商银行 2021 年净利润增长率是多少（保留 2 位小数）?  
    A: 2020 年净利润 12,345,678.90 元，2021 年净利润 22,345,678.90 元，增长率 81.00%。

## 数据位置与一键流程
- 原始数据：`data/input/finglm-data _raw`（pre/A/B/C 四份 answer.json，共 10,000 条）。
- 一键运行：`python scripts/run_all.py`（构建 master → 清洗去重 → 类型/维度 EDA → 子集导出）。
- 核心输出：
  - 主表：`finglm_data_store/finglm_master.jsonl`，统计：`finglm_data_store/finglm_master_stats.json`
  - 清洗后：`finglm_data_store/finglm_master_dedup.jsonl`
  - 子集索引：`finglm_data_store/index/`，目录：`finglm_data_store/dataset_catalog.json`

## 清洗与去重摘要
- 数量变化：10,000 → 9,527（过滤占位回答 473）→ 6,811（去重 2,716）。
- 删除原因计数：placeholder_answer 473；dedup 2,716；empty_question/empty_answers 0。
- 典型占位回答规则：精确短语匹配及正则 `^未查询到\\d{4}年.+的相关信息，无法回答该问题。?$`、`^未查询到.+年报信息，无法回答您的问题。?$`。

## 问题类型分布（基于 6,811 条去重数据）
| 类型 | 说明 | 条数 |
|------|------|------|
| 1 | 简单事实查询 | 2,844 |
| 1-2 | 多值查询 | 677 |
| 2-1 | 计算类 | 2,344 |
| 2-2 | 对比类 | 204 |
| 3-1 | 分析类 | 470 |
| 3-2 | 概念类 | 272 |

## 公司/年份维度分布
- 有年份：6,539；有公司：6,610；完全通用（无公司/年份）：201。
- 交叉占比：单公司单年份 5,092（74.76%）；单公司多年 1,163（17.08%）；多公司单年 310（4.55%）；多公司多年 45（0.66%）；通用 201（2.95%）。

## 子集导出（基于去重后 master）
- `single_company_single_year_core`: 5,021
- `single_company_multi_year`: 1,163
- `multi_company_single_year`: 310
- `multi_company_multi_year`: 45
- `type_3_2_concept`: 272
- `generic_no_company_year`: 201

## 详细统计
- 若需完整表格与字段示例，可参考对应 JSON 版本：`finglm_data_type_analysis.json`、`finglm_qa_dimension_analysis.json`、`finglm_data_store/clean_dedup_report.json`。
