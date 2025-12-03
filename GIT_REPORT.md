# Git Repository Report

**生成时间**: 2025-01-27  
**仓库路径**: `/Users/skybay7/CloudStorage/BaiduYun/LLM-Prj/QAnchor`

## 仓库状态

### Git 基本信息
- **分支**: main
- **提交历史**: 尚未进行首次提交（新仓库）
- **未跟踪文件**: 所有文件均为未跟踪状态

### 项目概览
- **项目名称**: QAnchor - FinGLM 数据处理与分析
- **项目类型**: 金融问答数据集处理管道
- **主要语言**: Python

## 文件统计

### 代码文件统计
- **Python 脚本**: 约 20+ 个
- **数据文件**: JSON/JSONL/CSV 等约 38 个（排除忽略目录）
- **文档文件**: Markdown 报告多个

### 目录结构
```
QAnchor/
├── data/                    # 数据处理模块
│   ├── input/              # 原始输入数据
│   │   ├── finglm_report_pdf/  # [已忽略] 1780 个 PDF 文件 (11GB)
│   │   └── finglm-data _raw/   # 原始问答数据
│   ├── output/             # 输出目录
│   └── *.py                # 数据处理脚本
├── finglm_data_store/       # 处理后的数据存储
│   ├── finglm_master.jsonl      # 主表 (10,000 条)
│   ├── finglm_master_dedup.jsonl # 清洗后 (6,811 条)
│   ├── index/              # 子集索引
│   └── *.json, *.csv       # 统计和映射文件
├── scripts/                # 主流程脚本
│   ├── run_all.py          # 一键执行脚本
│   ├── build_master_table.py
│   ├── clean_and_dedup.py
│   ├── analyze_qa_types.py
│   ├── analyze_qa_dimensions.py
│   └── export_subsets.py
├── Reference/              # [已忽略] 参考代码和文档 (37MB)
├── memory-bank/            # [已忽略] 项目记忆库 (44KB)
└── README.md               # 项目说明
```

## 数据统计

### 主数据集
- **原始数据**: 10,000 条问答对
  - pre: 5,000 条
  - A: 2,000 条
  - B: 2,000 条
  - C: 1,000 条

- **清洗后数据**: 6,811 条
  - 过滤占位答案: 473 条
  - 去重: 2,716 条

### 类型分布（清洗后）
| 类型 | 说明 | 条数 |
|------|------|------|
| 1 | 简单事实查询 | 2,844 |
| 1-2 | 多值查询 | 677 |
| 2-1 | 计算类 | 2,344 |
| 2-2 | 对比类 | 204 |
| 3-1 | 分析类 | 470 |
| 3-2 | 概念类 | 272 |

### 维度分布（清洗后）
- **单公司单年份**: 5,092 (74.76%)
- **单公司多年份**: 1,163 (17.08%)
- **多公司单年份**: 310 (4.55%)
- **多公司多年份**: 45 (0.66%)
- **通用（无公司/年份）**: 201 (2.95%)

## 忽略的目录

以下目录已配置在 `.gitignore` 中，不会被 Git 跟踪：

1. **data/input/finglm_report_pdf/** (11GB)
   - 包含 1,780 个 PDF 年报文件
   - 原因: 文件过大，不适合版本控制

2. **Reference/** (37MB)
   - 包含参考代码和文档
   - 原因: 历史参考材料，不需要版本控制

3. **memory-bank/** (44KB)
   - 包含项目记忆和文档
   - 原因: 临时/内部文档，不需要版本控制

## 建议的首次提交

### 核心文件（建议提交）
- ✅ 所有 Python 脚本 (`scripts/`, `data/*.py`)
- ✅ 配置文件 (`*.json`, `*.jsonl`, `*.csv`)
- ✅ 文档文件 (`*.md`, `README.md`)
- ✅ 数据统计和报告文件

### 不应提交的文件
- ❌ `data/input/finglm_report_pdf/` (已在 .gitignore)
- ❌ `Reference/` (已在 .gitignore)
- ❌ `memory-bank/` (已在 .gitignore)
- ❌ `.DS_Store` (系统文件)
- ❌ `__pycache__/` (Python 缓存)
- ❌ `.vscode/` (IDE 配置)

## 工作流程

### 一键执行
```bash
python scripts/run_all.py
```

### 主要脚本
1. `scripts/build_master_table.py` - 构建主表
2. `scripts/clean_and_dedup.py` - 清洗和去重
3. `scripts/analyze_qa_types.py` - 类型分析
4. `scripts/analyze_qa_dimensions.py` - 维度分析
5. `scripts/export_subsets.py` - 子集导出

## 下一步建议

1. **初始化提交**
   ```bash
   git add .
   git commit -m "Initial commit: FinGLM data processing pipeline"
   ```

2. **创建 .gitignore**（已完成）
   - 已配置忽略大文件和参考目录

3. **建立分支策略**
   - `main`: 稳定版本
   - `dev`: 开发分支（可选）

4. **添加远程仓库**（如需要）
   ```bash
   git remote add origin <repository-url>
   git push -u origin main
   ```

---

*此报告由自动化工具生成，反映当前仓库状态。*

