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
| Base state-action rows | `data/round2/state_eval_base.jsonl` |
| DPO state-action rows | `data/round2/state_eval_dpo.jsonl` |
| Run logs | `runs/round2/` |
| DPO adapter | `outputs/round2/dpo-adapter/` |
| Merged model | `outputs/round2/merged/` |
| State-action comparison | `results/round2/state_action_compare.md` |
| End-to-end comparison | `results/round2/dpo_compare_seed20260904.md` |

The train/eval split is task-grouped: no `task_id` may occur in both sets.
The holdout is never used by LLaMA-Factory. Do not pass a round-1 path as a
round-2 output argument.

Evidence is interpreted in this order: LLaMA-Factory reward separation,
free-generated state-action accuracy, then end-to-end holdout completion.
State-action evaluation uses `tool_choice=auto`; it is diagnostic, while the
untouched holdout completion rate remains the primary product metric.

The completed run used 52 train and 14 task-grouped eval pairs for three
epochs. Eval reward accuracy reached 0.928571 and reward margin 0.054155, but
base and DPO next-action accuracy were both 14.3%, and both completed 0/27
aligned holdout-v2 trajectories. The result is a negative transfer result,
not an uplift claim. The lightweight server evidence archive is
`/root/autodl-tmp/care-agent-evidence-round2-20260903.tar.gz` with SHA-256
`590974ca8bad782eb957b10c06e40f44429c96b02750a4016ab34f534408483c`.
