"""Step 4 — 导出 LLaMA-Factory 的 DPO 训练数据 + dataset_info.json。

用法：
    python scripts/4_to_llamafactory.py --pref data/pref_pairs.jsonl --outdir data/lf_data

产出：
    data/lf_data/agent_pref.json      训练数据（sharegpt + ranking 格式）
    data/lf_data/dataset_info.json    LLaMA-Factory 的数据集注册表

然后把这两个文件拷进 LLaMA-Factory 的 data/ 目录，
config/dpo_qwen15b.yaml 里的 dataset: agent_pref 就能对上。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_jsonl

DATASET_NAME = "agent_pref"
TRAINING_SYSTEM = (
    "你是慢病照护运营助手。你只能使用给定工具完成信息读取、记录、提醒和人工升级；"
    "不得诊断、开药或自行修改处方。执行时必须保留患者约束，完成必要闭环后立即停止。"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref", default="data/pref_pairs.jsonl")
    ap.add_argument("--outdir", default="data/lf_data")
    args = ap.parse_args()

    pairs = load_jsonl(args.pref)
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for p in pairs:
        if not p.get("chosen") or not p.get("rejected"):
            continue
        if p.get("chosen") == p.get("rejected"):
            continue  # chosen/rejected 相同会让 DPO 的 loss 退化
        rows.append({
            "conversations": [{"from": "human", "value": p.get("prompt") or ""}],
            "chosen": {"from": "gpt", "value": p["chosen"]},
            "rejected": {"from": "gpt", "value": p["rejected"]},
            "system": TRAINING_SYSTEM,
        })

    out_path = os.path.join(args.outdir, f"{DATASET_NAME}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    info = {
        DATASET_NAME: {
            "file_name": f"{DATASET_NAME}.json",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "system": "system",
            },
        }
    }
    info_path = os.path.join(args.outdir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # 顺便存一份分类别统计，README 里要用
    from collections import Counter
    dist = Counter(p.get("attr_label") for p in pairs)
    stat_path = os.path.join(args.outdir, "stat.json")
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_pairs": len(rows),
            "by_failure_mode": dict(dist),
            "by_task_type": dict(Counter(p.get("stress") for p in pairs)),
        }, f, ensure_ascii=False, indent=2)

    print(f"训练数据 {len(rows)} 条（原 {len(pairs)} 条，已过滤空值与相同的 chosen/rejected）")
    print("分类别分布：", dict(dist))
    print(f"\n已写入：\n  {out_path}\n  {info_path}\n  {stat_path}")
    print(f"\n下一步：把 {DATASET_NAME}.json 和 dataset_info.json 拷进 LLaMA-Factory 的 data/ 目录，"
          f"然后 llamafactory-cli train config/dpo_qwen15b.yaml")


if __name__ == "__main__":
    main()
