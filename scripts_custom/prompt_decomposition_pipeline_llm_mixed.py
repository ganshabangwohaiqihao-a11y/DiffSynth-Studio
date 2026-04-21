#!/usr/bin/env python3
"""
Prompt Decomposition Pipeline (LLM-Powered) - Mixed Prompt Sources
===================================================================

Use BOTH CSV and XLSX prompt sources to build one unified plan_id -> global_prompt mapping,
then reuse the existing LLM decomposition logic from prompt_decomposition_pipeline_llm.py.

Typical usage (train set):
  python prompt_decomposition_pipeline_llm_mixed.py \
    --metadata /share/home/202230550120/diffusers/output_2026.3.27/chinese_meta_train.json \
    --output /share/home/202230550120/diffusers/output_2026.3.27/chinese_meta_train_enhanced_llm_mixed.json

Fast test without loading LLM (rule-based fallback only):
  python prompt_decomposition_pipeline_llm_mixed.py --no_llm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_CSV_PATH = "/share/home/202230550120/diffusers/土巴兔华工室内装修设计项目-训练数据集 (1).csv"
DEFAULT_XLSX_PATH = "/share/home/202230550120/diffusers/coperation.xlsx"
DEFAULT_METADATA_PATH = "/share/home/202230550120/diffusers/output_2026.3.27/chinese_meta_test.json"
DEFAULT_OUTPUT_PATH = "/share/home/202230550120/diffusers/output_2026.3.27/chinese_meta_test_enhanced_llm_mixed.json"

CSV_PLAN_ID_COLUMN = "方案ID"
CSV_PROMPT_COLUMN = "户型描述、装修需求（布局要求）"

XLSX_PLAN_ID_COLUMN = "plan_id"
XLSX_PLAN_DESC_COLUMN = "户型描述"
XLSX_PROMPT_COLUMN = "装修需求(布局要求)"


def _norm_plan_id(v) -> str:
    if pd.isna(v):
        return ""
    try:
        return str(int(v))
    except Exception:
        return str(v).strip()


def load_mapping_from_csv(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    if CSV_PLAN_ID_COLUMN not in df.columns or CSV_PROMPT_COLUMN not in df.columns:
        raise ValueError(
            f"CSV missing required columns: {CSV_PLAN_ID_COLUMN}, {CSV_PROMPT_COLUMN}"
        )

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        pid = _norm_plan_id(row[CSV_PLAN_ID_COLUMN])
        prompt = str(row[CSV_PROMPT_COLUMN]).strip()
        if pid and prompt and prompt.lower() != "nan":
            mapping[pid] = prompt
    return mapping


def load_mapping_from_xlsx(xlsx_path: str) -> Dict[str, str]:
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    has_plan_id = XLSX_PLAN_ID_COLUMN in df.columns
    has_plan_desc = XLSX_PLAN_DESC_COLUMN in df.columns
    has_prompt = XLSX_PROMPT_COLUMN in df.columns

    if not has_plan_desc and not has_prompt:
        raise ValueError(
            f"XLSX missing both columns: {XLSX_PLAN_DESC_COLUMN} and {XLSX_PROMPT_COLUMN}"
        )

    mapping: Dict[str, str] = {}
    for idx, row in df.iterrows():
        pid = _norm_plan_id(row[XLSX_PLAN_ID_COLUMN]) if has_plan_id else str(idx)
        plan_desc = str(row[XLSX_PLAN_DESC_COLUMN]).strip() if has_plan_desc else ""
        prompt = str(row[XLSX_PROMPT_COLUMN]).strip() if has_prompt else ""

        if plan_desc.lower() == "nan":
            plan_desc = ""
        if prompt.lower() == "nan":
            prompt = ""

        combined = f"{plan_desc}。{prompt}" if plan_desc and prompt else (plan_desc or prompt)
        if pid and combined:
            mapping[pid] = combined
    return mapping


def resolve_mapping_for_metadata(
    metadata: List[Dict],
    csv_map: Dict[str, str],
    xlsx_map: Dict[str, str],
    order: str = "csv_first",
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Resolve plan_id -> prompt for metadata by ordered fallback lookup.

    order:
      - csv_first: csv -> xlsx
      - xlsx_first: xlsx -> csv
    """
    if order not in {"csv_first", "xlsx_first"}:
        raise ValueError("order must be one of: csv_first, xlsx_first")

    primary_name = "csv" if order == "csv_first" else "xlsx"
    primary_map = csv_map if order == "csv_first" else xlsx_map
    secondary_name = "xlsx" if order == "csv_first" else "csv"
    secondary_map = xlsx_map if order == "csv_first" else csv_map

    resolved: Dict[str, str] = {}
    source_used: Dict[str, str] = {}

    meta_plan_ids = {str(x.get("plan_id")) for x in metadata if x.get("plan_id") is not None}
    for pid in meta_plan_ids:
        p = primary_map.get(pid, "")
        if p:
            resolved[pid] = p
            source_used[pid] = primary_name
            continue

        s = secondary_map.get(pid, "")
        if s:
            resolved[pid] = s
            source_used[pid] = secondary_name

    stats = {
        "meta_plan_ids": len(meta_plan_ids),
        "covered": len(resolved),
        "from_csv": sum(1 for v in source_used.values() if v == "csv"),
        "from_xlsx": sum(1 for v in source_used.values() if v == "xlsx"),
        "missing": len(meta_plan_ids) - len(resolved),
    }
    return resolved, stats


def load_metadata(metadata_path: str) -> List[Dict]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt decomposition with mixed CSV+XLSX prompt sources"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    parser.add_argument("--xlsx", default=DEFAULT_XLSX_PATH)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--lookup_order",
        choices=["csv_first", "xlsx_first"],
        default="csv_first",
        help="Ordered fallback lookup: first source miss -> second source",
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Do not load LLM; use existing fallback path in base pipeline",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PROMPT DECOMPOSITION (MIXED SOURCES: CSV + XLSX)")
    print("=" * 70)

    print("\n[1/5] Loading prompt mappings...")
    csv_map = load_mapping_from_csv(args.csv)
    xlsx_map = load_mapping_from_xlsx(args.xlsx)

    print(f"  CSV prompts: {len(csv_map)}")
    print(f"  XLSX prompts: {len(xlsx_map)}")

    print("\n[2/5] Loading metadata...")
    metadata = load_metadata(args.metadata)
    print(f"  Room entries: {len(metadata)}")

    resolved_map, resolve_stats = resolve_mapping_for_metadata(
        metadata,
        csv_map,
        xlsx_map,
        order=args.lookup_order,
    )
    print(f"  Lookup order: {args.lookup_order}")
    print(f"  Unique plan_ids in metadata: {resolve_stats['meta_plan_ids']}")
    print(f"  Covered by prompt mapping: {resolve_stats['covered']}/{resolve_stats['meta_plan_ids']}")
    print(f"  From CSV: {resolve_stats['from_csv']}")
    print(f"  From XLSX: {resolve_stats['from_xlsx']}")
    print(f"  Missing in both: {resolve_stats['missing']}")

    print("\n[3/5] Loading base decomposition pipeline...")
    import prompt_decomposition_pipeline_llm as base

    model, tokenizer = (None, None)
    if not args.no_llm:
        print("\n[4/5] Loading LLM model...")
        model, tokenizer = base.load_model_and_tokenizer()
        if model:
            print("  ✓ LLM loaded")
        else:
            print("  ⚠️ LLM unavailable, fallback mode")
    else:
        print("\n[4/5] Skipping LLM (--no_llm), fallback mode")

    print("\n[5/5] Processing metadata...")
    enhanced, stats = base.process_metadata(metadata, resolved_map, model, tokenizer)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enhanced, f, ensure_ascii=False, indent=2)

    print("\n📊 Stats:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  With global prompt: {stats['with_global_prompt']}")
    print(f"  LLM extracted: {stats['extracted_llm']}")
    print(f"  Fallback extracted: {stats['extracted_fallback']}")
    print(f"  LLM empty/invalid: {stats['llm_failed_empty']}")

    print(f"\n✅ Saved: {output_path}")


if __name__ == "__main__":
    main()
