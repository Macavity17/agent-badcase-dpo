# Round 1 artifact map

Round 1 is frozen. Do not rename, overwrite, or reuse these paths for later runs.

| Kind | Round-1 path |
|---|---|
| Source tasks | `tasks/tasks.jsonl` |
| Preference pairs | `data/pref_pairs_canonical_v2.jsonl` |
| LLaMA-Factory data | `data/lf_data/` |
| DPO adapter | `outputs/dpo-qwen15b/` |
| Merged model | `outputs/dpo_merged/` |
| Run logs | existing files directly under `runs/` |
| Evaluation | existing files directly under `results/` |

The experiment log is the authoritative record of the exact filenames and
commands used. Round 2 uses dedicated `round2/` subdirectories and must not
write to any path listed above.
