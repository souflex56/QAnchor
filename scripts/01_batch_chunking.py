#!/usr/bin/env python3
"""Stage-aware PDF 分块脚本，调用 ZenParse runner 生成 chunk JSON。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml

# 确保仓库根目录在 sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import select_pdf_subset


def build_runner_cmd(
    pdf_paths: List[Path],
    zen_root: Path,
    zen_config: Path,
    output_dir: Path,
    use_timestamp: bool = False,
) -> List[str]:
    cmd = [sys.executable, str(zen_root / "runner.py")]
    for pdf in pdf_paths:
        cmd.extend(["-i", str(pdf)])
    cmd.extend(["-o", str(output_dir), "-c", str(zen_config)])
    if use_timestamp:
        cmd.append("--timestamp")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="批量调用 ZenParse，按 Stage 生成 chunk JSON")
    parser.add_argument("--stage", default="stage0", help="阶段名称，默认为 stage0")
    parser.add_argument(
        "--config",
        default="config/weak_supervision_config.yaml",
        help="主配置文件路径",
    )
    parser.add_argument(
        "--zen-root",
        default="Reference/external/ZenParse",
        help="ZenParse 根目录（包含 runner.py）",
    )
    parser.add_argument(
        "--zen-config",
        default=None,
        help="ZenParse 配置文件路径；默认读取主配置中的 zenparse.config",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="若输出文件已存在则跳过（默认开启）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="可选：限制本次处理的 PDF 数量（调试用）",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="为输出文件名添加时间戳后缀，避免覆盖",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    chunk_output = Path(cfg["data"]["chunk_output"])
    chunk_output.mkdir(parents=True, exist_ok=True)

    subset = select_pdf_subset(
        stage=args.stage,
        summary_path=cfg["data"]["summary"],
        pdf_dir=cfg["data"]["pdf_dir"],
        stage_config=cfg["stages"],
    )

    pdf_paths: List[Path] = []
    skipped: List[str] = []
    for rec in subset["records"]:
        if args.limit is not None and len(pdf_paths) >= args.limit:
            break
        pdf_path = Path(rec["pdf_path"])
        out_file = chunk_output / f"{pdf_path.stem}_chunks.json"
        if args.skip_existing and out_file.exists():
            skipped.append(pdf_path.name)
            continue
        pdf_paths.append(pdf_path)

    if not pdf_paths:
        print("没有待处理的 PDF（可能已全部存在或 limit=0）")
        if skipped:
            print(f"已跳过 {len(skipped)} 个已存在文件")
        return

    zen_cfg_path = Path(
        args.zen_config
        or cfg.get("zenparse", {}).get("config")
        or "config/zenparse_config.yaml"
    )

    cmd = build_runner_cmd(
        pdf_paths=pdf_paths,
        zen_root=Path(args.zen_root),
        zen_config=zen_cfg_path,
        output_dir=chunk_output,
        use_timestamp=args.timestamp,
    )

    print(f"[Stage: {args.stage}] 将处理 {len(pdf_paths)} 个 PDF，输出目录: {chunk_output}")
    if skipped:
        print(f"已跳过 {len(skipped)} 个已存在文件: {', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}")
    print("执行命令:", " ".join(cmd))

    subprocess.run(cmd, check=True)
    print("分块完成。")


if __name__ == "__main__":
    main()
