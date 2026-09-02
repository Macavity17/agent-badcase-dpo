import importlib.util
import os
import re
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from utils import check_completion, load_jsonl


BASELINE_SPEC = importlib.util.spec_from_file_location(
    "baseline", os.path.join(ROOT, "scripts", "1_run_baseline.py")
)
baseline = importlib.util.module_from_spec(BASELINE_SPEC)
BASELINE_SPEC.loader.exec_module(baseline)

PREFERENCE_SPEC = importlib.util.spec_from_file_location(
    "preference", os.path.join(ROOT, "scripts", "3_build_preference.py")
)
preference = importlib.util.module_from_spec(PREFERENCE_SPEC)
PREFERENCE_SPEC.loader.exec_module(preference)

EVALUATE_SPEC = importlib.util.spec_from_file_location(
    "evaluate", os.path.join(ROOT, "scripts", "5_evaluate.py")
)
evaluate = importlib.util.module_from_spec(EVALUATE_SPEC)
EVALUATE_SPEC.loader.exec_module(evaluate)


class CheckerTest(unittest.TestCase):
    def test_layered_context_compresses_old_rounds(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "original task"},
        ]
        steps = []
        for index in range(3):
            messages.extend([
                {"role": "assistant", "content": "", "tool_calls": [{"id": str(index)}]},
                {"role": "tool", "tool_call_id": str(index), "content": f"obs-{index}"},
            ])
            steps.append({
                "tool": f"tool_{index}", "args": {"id": str(index)},
                "observation": f"obs-{index}",
            })

        effective = baseline.apply_layered(
            messages,
            {"goal": "goal", "constraints": ["constraint"]},
            steps,
        )

        self.assertEqual(
            [message["role"] for message in effective],
            ["system", "user", "assistant", "tool", "assistant", "tool"],
        )
        state = effective[1]["content"]
        self.assertIn("tool_0", state)
        self.assertNotIn("tool_1", state)
        self.assertNotIn("tool_2", state)
        self.assertEqual(effective[-3]["content"], "obs-1")
        self.assertEqual(effective[-1]["content"], "obs-2")

    def test_composite_checker(self):
        task = {
            "checker": {
                "type": "all",
                "checks": [
                    {"type": "tool_call_sequence", "tools": ["read", "write"]},
                    {"type": "tool_not_called", "tools": ["delete"]},
                    {"type": "final_contains_all", "values": ["done", "ID-1"]},
                ],
            }
        }
        traj = {
            "tool_calls": [{"name": "read", "args": {}}, {"name": "write", "args": {}}],
            "final_answer": "done: ID-1",
        }
        self.assertTrue(check_completion(task, traj))

    def test_task_set_is_balanced_and_disjoint(self):
        tasks = load_jsonl(os.path.join(ROOT, "tasks", "tasks.jsonl"))
        counts = {}
        families = {"train": set(), "test": set()}
        for task in tasks:
            key = (task["split"], task["stress"])
            counts[key] = counts.get(key, 0) + 1
            families[task["split"]].add(task["scenario_family"])
        self.assertEqual(set(counts.values()), {3, 5})
        self.assertFalse(families["train"] & families["test"])

    def test_frozen_holdout_does_not_reuse_dev_patient_ids(self):
        tasks = load_jsonl(os.path.join(ROOT, "tasks", "tasks.jsonl"))
        dev_tasks = load_jsonl(os.path.join(ROOT, "tasks", "dev_tasks.jsonl"))
        holdout_text = "\n".join(task["goal"] for task in tasks if task["split"] == "test")
        dev_text = "\n".join(task["goal"] for task in dev_tasks)
        patient_pattern = r"P\d{4}"
        self.assertFalse(
            set(re.findall(patient_pattern, holdout_text))
            & set(re.findall(patient_pattern, dev_text))
        )

    def test_canonical_chosen_passes_every_train_checker(self):
        tasks = load_jsonl(os.path.join(ROOT, "tasks", "tasks.jsonl"))
        for task in (task for task in tasks if task["split"] == "train"):
            chosen, info = preference.canonical_chosen(task)
            self.assertIsNotNone(chosen, f"{task['task_id']}: {info}")
            self.assertIn('"name": "finish_task"', chosen["text"])
            self.assertNotIn("工具返回的", chosen["text"])
            self.assertNotIn('"send_at": "19:"', chosen["text"])

    def test_reports_generate_evidence_bounded_conclusions(self):
        before = {
            "n": 27, "tasks": 9, "success": 4, "rate": 4 / 27,
            "avg_steps": 5.8, "avg_ctx": 1234, "tool_call_rate": 1.0,
            "invalid_call_rate": 0.0,
            "by_mode": {mode: (rate, 9) for mode, rate in zip(
                evaluate.MODES, (0.0, 1 / 9, 3 / 9)
            )},
        }
        after = dict(before)
        after.update({"success": 2, "rate": 2 / 27, "avg_ctx": 1230})
        after["by_mode"] = {
            mode: (rate, 9) for mode, rate in zip(evaluate.MODES, (0.0, 1 / 9, 1 / 9))
        }

        dpo_report = evaluate.build_before_after(before, after)
        context_report = evaluate.build_multi_report(
            {"full": before, "layered": after}, ["full", "layered"]
        )

        self.assertNotIn("跑完再写", dpo_report + context_report)
        self.assertIn("下降 7.4pp", dpo_report)
        self.assertIn("格式正确不能替代", dpo_report)
        self.assertIn("只有完成率提高", context_report)


if __name__ == "__main__":
    unittest.main()
