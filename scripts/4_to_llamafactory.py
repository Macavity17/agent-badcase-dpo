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
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_jsonl

DATASET_NAME = "agent_pref"
TRAINING_SYSTEM = (
    "你是慢病照护运营助手。每次只调用一个必要工具，参数必须复用任务或"
    "工具结果中的具体值。必须保留患者约束，不得诊断、开药或自行修改处方。"
    "只有目标中的必要动作全部完成后才调用 finish_task，然后报告关键结果和返回 ID。"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref", default="data/pref_pairs.jsonl")
    ap.add_argument("--outdir", default="data/lf_data")
    ap.add_argument("--train-config", default="config/dpo_qwen15b.yaml")
    args = ap.parse_args()

    pairs = load_jsonl(args.pref)
    os.makedirs(args.outdir, exist_ok=True)

    rows_by_split = {"all": [], "train": [], "eval": []}
    for p in pairs:
        if not p.get("chosen") or not p.get("rejected"):
            continue
        if p.get("chosen") == p.get("rejected"):
            continue  # chosen/rejected 相同会让 DPO 的 loss 退化
        conversations = p.get("context_messages") or [
            {"from": "human", "value": p.get("prompt") or ""}
        ]
        chosen_action = p.get("chosen_action") or {"kind": "final"}
        rejected_action = p.get("rejected_action") or {"kind": "final"}
        row = {
            "conversations": conversations,
            "chosen": {
                "from": "function_call" if chosen_action.get("kind") == "tool" else "gpt",
                "value": _response_value(p["chosen"], chosen_action),
            },
            "rejected": {
                "from": "function_call" if rejected_action.get("kind") == "tool" else "gpt",
                "value": _response_value(p["rejected"], rejected_action),
            },
            "system": p.get("system") or TRAINING_SYSTEM,
        }
        if p.get("tools"):
            row["tools"] = json.dumps(p["tools"], ensure_ascii=False)
        split = p.get("dataset_split")
        rows_by_split[split if split in {"train", "eval"} else "all"].append(row)

    split_mode = bool(rows_by_split["train"] or rows_by_split["eval"])
    outputs = (
        [("agent_pref_train", rows_by_split["train"]),
         ("agent_pref_eval", rows_by_split["eval"])]
        if split_mode else [(DATASET_NAME, rows_by_split["all"])]
    )
    info = {}
    for dataset_name, rows in outputs:
        if not rows:
            raise SystemExit(f"{dataset_name} 为空")
        filename = f"{dataset_name}.json"
        out_path = os.path.join(args.outdir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        columns = {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
            "system": "system",
        }
        if any("tools" in row for row in rows):
            columns["tools"] = "tools"
        info[dataset_name] = {
            "file_name": filename,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": columns,
        }
    info_path = os.path.join(args.outdir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # 顺便存一份分类别统计，README 里要用
    dist = Counter(p.get("attr_label") for p in pairs)
    stat_path = os.path.join(args.outdir, "stat.json")
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_pairs": sum(len(rows) for _, rows in outputs),
            "by_dataset_split": {name: len(rows) for name, rows in outputs},
            "by_failure_mode": dict(dist),
            "by_task_type": dict(Counter(p.get("stress") for p in pairs)),
        }, f, ensure_ascii=False, indent=2)

    total = sum(len(rows) for _, rows in outputs)
    print(f"偏好数据 {total} 条（原 {len(pairs)} 条，已过滤空值与相同的 chosen/rejected）")
    print("数据集切分：", {name: len(rows) for name, rows in outputs})
    print("分类别分布：", dict(dist))
    output_paths = [os.path.join(args.outdir, f"{name}.json") for name, _ in outputs]
    print("\n已写入：\n  " + "\n  ".join(output_paths + [info_path, stat_path]))
    print(f"\n下一步：llamafactory-cli train {args.train_config}")


def _response_value(rendered, action):
    if action.get("kind") != "tool":
        return rendered
    return json.dumps({
        "name": action["tool"],
        "arguments": action.get("args") or {},
    }, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
