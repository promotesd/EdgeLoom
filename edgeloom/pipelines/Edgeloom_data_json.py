#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

REF_EXTS = (".png", ".jpg", ".jpeg", ".webp")

def find_with_suffixes(dirpath: Path, stem: str, suffixes: Iterable[str]) -> Optional[Path]:
    """在 dirpath 中按给定 stem 和多个后缀查找文件，找到第一个就返回。"""
    for suf in suffixes:
        p = dirpath / f"{stem}{suf}"
        if p.exists():
            return p
    return None

def generate_jsonl(source_dir: Path, target_dir: Path, ref_dir: Path,
                   output_path: Path, prompt: str):
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 收集所有源文件（.jpg）
    src_files = sorted(source_dir.glob("*.jpg"))
    if not src_files:
        print(f"Warning: no .jpg files found in {source_dir}")

    n_total = 0
    n_written = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for src in src_files:
            n_total += 1
            stem = src.stem  # 不带扩展名的文件名

            # 匹配 target: 固定为 .png
            tgt = target_dir / f"{stem}.png"
            if not tgt.exists():
                print(f"Warning: skip {src.name}, missing target {tgt.name}")
                continue

            # 匹配 ref: 允许多种扩展名
            ref = find_with_suffixes(ref_dir, stem, REF_EXTS)
            if ref is None:
                exts = "/".join(REF_EXTS)
                print(f"Warning: skip {src.name}, no matching ref_image {stem}.({exts}) in {ref_dir}")
                continue

            # 绝对路径
            entry = {
                "source": str(src.resolve()),
                "target": str(tgt.resolve()),
                "ref_image": str(ref.resolve()),
                "prompt": prompt,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"✅ Wrote {n_written} entries to {output_path} (from {n_total} source images)")

def main():
    p = argparse.ArgumentParser(
        description="Generate a JSONL pairing source .jpg with target .png, ref_image (png/jpg/jpeg/webp) and a shared prompt."
    )
    p.add_argument("-s", "--source_dir", required=True, type=Path,
                   help="Directory with source .jpg images")
    p.add_argument("-t", "--target_dir", required=True, type=Path,
                   help="Directory with target .png images")
    p.add_argument("-r", "--ref_dir", required=True, type=Path,
                   help="Directory with reference images (same stem; png/jpg/jpeg/webp)")
    p.add_argument("-o", "--output", required=True, type=Path,
                   help="Output JSONL file path")
    p.add_argument("-p", "--prompt", required=True, type=str,
                   help="Prompt text to attach to every entry")
    args = p.parse_args()

    generate_jsonl(args.source_dir, args.target_dir, args.ref_dir, args.output, args.prompt)

if __name__ == "__main__":
    main()
