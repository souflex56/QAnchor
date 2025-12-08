cd /Users/skybay7/CloudStorage/BaiduYun/LLM-Prj/QAnchor
python - <<'PY'
from pathlib import Path
import yaml
from src.data_loader import select_pdf_subset

cfg = yaml.safe_load(Path("config/weak_supervision_config.yaml").read_text())
res = select_pdf_subset(
    stage="stage0",
    summary_path=cfg["data"]["summary"],
    pdf_dir=cfg["data"]["pdf_dir"],
    stage_config=cfg["stages"],
)
print(res["stats"])
print(res["records"][:2])  # 查看前2个样本
PY

我已经按你给的命令在本地测试过，Step 1+2 功能正常。
请现在：
1. 更新 memory-bank/progress.md，增加一条记录；
2. 更新 memory-bank/architecture.md，把本次新增/修改的模块和文件补充进去。

