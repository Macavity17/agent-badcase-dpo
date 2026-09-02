import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from utils import check_completion, load_jsonl


class CheckerTest(unittest.TestCase):
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
