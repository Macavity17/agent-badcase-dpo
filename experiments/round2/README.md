# Round 2 manifest

Round 2 is an independent state-action DPO iteration. It reuses the frozen
round-1 badcase file only as a pinned source; every derived dataset, log,
checkpoint, merged model, and result has a separate path.

| Kind | Round-2 path |
|---|---|
| Teacher workflow | `tasks/teacher_v2_specs.jsonl` |
| Manual pair review | `experiments/round2/pair_review.jsonl` |
| Fixed task split | `experiments/round2/split.json` |
| Blind holdout | `tasks/holdout_v2.jsonl` |
| Derived pairs | `data/round2/pref_pairs.jsonl` |
| Full candidate audit | `data/round2/pref_pairs.audit.jsonl` |
| LLaMA-Factory data | `data/round2/lf_data/` |
| Run logs | `runs/round2/` |
| DPO adapter | `outputs/round2/dpo-adapter/` |
| Merged model | `outputs/round2/merged/` |
| Evaluation | `results/round2/` |

The train/eval split is task-grouped: no `task_id` may occur in both sets.
The holdout is never used by LLaMA-Factory. Do not pass a round-1 path as a
round-2 output argument.
