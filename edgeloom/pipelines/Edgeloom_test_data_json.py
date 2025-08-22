#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Dict, List

# 允许的扩展名（按优先级匹配）
SRC_EXTS = (".jpg", ".png", ".jpeg", ".webp")
REF_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def find_with_suffixes(dirpath: Path, stem: str, suffixes: Iterable[str]) -> Optional[Path]:
    """在 dirpath 中按给定 stem 和多个后缀查找文件，找到第一个就返回。"""
    for suf in suffixes:
        p = dirpath / f"{stem}{suf}"
        if p.exists():
            return p
    return None


def enumerate_unique_by_stem(dirpath: Path, suffixes: Iterable[str]) -> List[Path]:
    """
    返回该目录下唯一 stem 的文件列表（按 suffixes 的优先级选择）。
    例如同一 stem 同时有 .jpg/.png，则优先选择列表前面的后缀。
    """
    chosen: Dict[str, Path] = {}
    for suf in suffixes:
        for p in sorted(dirpath.glob(f"*{suf}")):
            chosen.setdefault(p.stem, p)
    return [chosen[k] for k in sorted(chosen.keys())]


def generate_jsonl(
    source_dir: Path,
    ref_dir: Path,
    output_path: Path,
    prompt: str,
    include_out: bool = False,
    out_ext: str = ".png",
    out_prefix: str = "",
):
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 收集唯一 stem 的 source 文件
    src_files = enumerate_unique_by_stem(source_dir, SRC_EXTS)
    if not src_files:
        print(f"Warning: no images found in {source_dir} with {SRC_EXTS}")

    n_total = 0
    n_written = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for src in src_files:
            n_total += 1
            stem = src.stem

            # 匹配 ref（允许多种扩展名）
            ref = find_with_suffixes(ref_dir, stem, REF_EXTS)
            if ref is None:
                exts = "/".join(REF_EXTS)
                print(f"Warning: skip {src.name}, no matching ref_image {stem}.({exts}) in {ref_dir}")
                continue

            entry = {
                "source": str(src.resolve()),
                "ref_image": str(ref.resolve()),
                "prompt": prompt,
            }

            if include_out:
                # 生成一个可选的输出文件名（推理脚本若检测到 'out' 会直接使用）
                out_name = f"{out_prefix}{stem}__{ref.stem}{out_ext}"
                entry["out"] = out_name

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"✅ Wrote {n_written} entries to {output_path} (from {n_total} source images)")


def main():
    p = argparse.ArgumentParser(
        description="Generate a JSONL for testing (no target): pairs source & ref_image by stem, plus a shared prompt."
    )
    p.add_argument("-s", "--source_dir", required=True, type=Path,
                   help="Directory with source images (edge/hint). Supports .jpg/.png/.jpeg/.webp")
    p.add_argument("-r", "--ref_dir", required=True, type=Path,
                   help="Directory with reference images (same stem). Supports .png/.jpg/.jpeg/.webp")
    p.add_argument("-o", "--output", required=True, type=Path,
                   help="Output JSONL file path")
    p.add_argument("-p", "--prompt", required=True, type=str,
                   help="Prompt text attached to every entry")

    # 可选：是否写出 'out' 字段供推理脚本直接使用
    p.add_argument("--include_out", action="store_true",
                   help="If set, also write an 'out' filename per entry like '<stem_src>__<stem_ref>.png'")
    p.add_argument("--out_ext", type=str, default=".png",
                   help="Extension for 'out' filenames (default: .png)")
    p.add_argument("--out_prefix", type=str, default="",
                   help="Prefix for 'out' filenames (default: empty)")

    args = p.parse_args()

    generate_jsonl(
        source_dir=args.source_dir,
        ref_dir=args.ref_dir,
        output_path=args.output,
        prompt=args.prompt,
        include_out=args.include_out,
        out_ext=args.out_ext,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
