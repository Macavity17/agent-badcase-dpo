import importlib.util
import os
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

        self.assertEqual([message["role"] for message in effective], ["system", "user", "assistant", "tool"])
        state = effective[1]["content"]
        self.assertIn("tool_0", state)
        self.assertIn("tool_1", state)
        self.assertNotIn("tool_2", state)
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


if __name__ == "__main__":
    unittest.main()
