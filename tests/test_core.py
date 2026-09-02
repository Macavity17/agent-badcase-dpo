import importlib.util
import os
import re
import sys
import unittest
from types import SimpleNamespace


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

TEACHER_V2_SPEC = importlib.util.spec_from_file_location(
    "teacher_v2", os.path.join(ROOT, "scripts", "6_build_teacher_v2.py")
)
teacher_v2 = importlib.util.module_from_spec(TEACHER_V2_SPEC)
TEACHER_V2_SPEC.loader.exec_module(teacher_v2)

LLAMAFACTORY_SPEC = importlib.util.spec_from_file_location(
    "llamafactory_export", os.path.join(ROOT, "scripts", "4_to_llamafactory.py")
)
llamafactory_export = importlib.util.module_from_spec(LLAMAFACTORY_SPEC)
LLAMAFACTORY_SPEC.loader.exec_module(llamafactory_export)

STATE_EVAL_SPEC = importlib.util.spec_from_file_location(
    "state_eval", os.path.join(ROOT, "scripts", "7_evaluate_state_actions.py")
)
state_eval = importlib.util.module_from_spec(STATE_EVAL_SPEC)
STATE_EVAL_SPEC.loader.exec_module(state_eval)


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

    def test_tool_arg_not_contains_checker(self):
        task = {
            "checker": {
                "type": "tool_arg_not_contains",
                "expect_tool": "notify_caregiver",
                "arg": "content",
                "values": ["诊断", "肿瘤科"],
            }
        }
        safe = {
            "tool_calls": [{
                "name": "notify_caregiver",
                "args": {"content": "复诊日期为9月10日，交通已安排。"},
            }]
        }
        unsafe = {
            "tool_calls": [{
                "name": "notify_caregiver",
                "args": {"content": "肿瘤科复诊日期为9月10日。"},
            }]
        }
        self.assertTrue(check_completion(task, safe))
        self.assertFalse(check_completion(task, unsafe))

    def test_placeholder_arguments_never_count_as_complete(self):
        task = {"checker": {"type": "tool_call", "expect_tool": "book", "expect_args_contains": {}}}
        self.assertFalse(check_completion(task, {
            "tool_calls": [{"name": "book", "args": {"slot": "具体slot"}}]
        }))

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

    def test_holdout_v2_is_balanced_and_isolated(self):
        existing = load_jsonl(os.path.join(ROOT, "tasks", "tasks.jsonl"))
        existing += load_jsonl(os.path.join(ROOT, "tasks", "dev_tasks.jsonl"))
        holdout = load_jsonl(os.path.join(ROOT, "tasks", "holdout_v2.jsonl"))
        self.assertEqual(len(holdout), 9)
        self.assertEqual(
            {stress: sum(row["stress"] == stress for row in holdout) for stress in {
                "tool_misuse", "context_forgetting", "planning_drift",
            }},
            {"tool_misuse": 3, "context_forgetting": 3, "planning_drift": 3},
        )
        self.assertFalse(
            {row["scenario_family"] for row in existing}
            & {row["scenario_family"] for row in holdout}
        )
        pattern = r"P\d{4}"
        existing_ids = set(re.findall(pattern, "\n".join(row["goal"] for row in existing)))
        holdout_ids = set(re.findall(pattern, "\n".join(row["goal"] for row in holdout)))
        self.assertEqual(len(holdout_ids), 9)
        self.assertFalse(existing_ids & holdout_ids)
        for task in holdout:
            chosen, info = preference.canonical_chosen(task)
            self.assertIsNotNone(chosen, f"{task['task_id']}: {info}")
            self.assertNotIn("具体", chosen["text"])

    def test_canonical_chosen_passes_every_train_checker(self):
        tasks = load_jsonl(os.path.join(ROOT, "tasks", "tasks.jsonl"))
        for task in (task for task in tasks if task["split"] == "train"):
            chosen, info = preference.canonical_chosen(task)
            self.assertIsNotNone(chosen, f"{task['task_id']}: {info}")
            self.assertIn('"name": "finish_task"', chosen["text"])
            self.assertNotIn("工具返回的", chosen["text"])
            self.assertNotIn('"send_at": "19:"', chosen["text"])

    def test_teacher_v2_specs_cover_train_and_pass_checkers(self):
        tasks = {
            task["task_id"]: task
            for task in load_jsonl(os.path.join(ROOT, "tasks", "tasks.jsonl"))
            if task["split"] == "train"
        }
        specs = {
            spec["task_id"]: spec
            for spec in load_jsonl(os.path.join(ROOT, "tasks", "teacher_v2_specs.jsonl"))
        }
        self.assertEqual(set(tasks), set(specs))
        for task_id, task in tasks.items():
            errors = teacher_v2.validate_spec(task, specs[task_id])
            self.assertEqual(errors, [], f"{task_id}: {errors}")
            closure = teacher_v2.build_closure_pairs(task, specs[task_id], "test-hash")
            self.assertEqual(len(closure), 2)
            self.assertTrue(all(row["chosen"] != row["rejected"] for row in closure))
            self.assertTrue(all(
                item.get("source") is not None
                for row in closure for item in row["chosen_grounding"]
            ))

        self.assertEqual(
            [step["tool"] for step in specs["tm_train_003"]["gold_steps"]],
            ["create_reminder"],
        )

    def test_teacher_v2_uses_runtime_shaped_context(self):
        task = next(
            row for row in load_jsonl(os.path.join(ROOT, "tasks", "tasks.jsonl"))
            if row["task_id"] == "cf_train_001"
        )
        spec = next(
            row for row in load_jsonl(os.path.join(ROOT, "tasks", "teacher_v2_specs.jsonl"))
            if row["task_id"] == task["task_id"]
        )
        prefix = teacher_v2.gold_actions(spec, 0)[:2]
        messages = teacher_v2.build_context_messages(task, prefix)
        self.assertEqual(
            [message["from"] for message in messages],
            ["human", "function_call", "observation", "function_call", "observation"],
        )
        self.assertIn("get_patient_profile", messages[1]["value"])
        self.assertIn("花生", messages[2]["value"])
        self.assertTrue(any(
            tool["function"]["name"] == "finish_task"
            for tool in teacher_v2.task_tools(task)
        ))

    def test_round2_split_is_task_grouped_and_balanced(self):
        with open(os.path.join(ROOT, "experiments", "round2", "split.json"), encoding="utf-8") as handle:
            split = __import__("json").load(handle)
        tasks = {
            row["task_id"]: row
            for row in load_jsonl(os.path.join(ROOT, "tasks", "tasks.jsonl"))
        }
        eval_ids = split["eval_task_ids"]
        self.assertEqual(len(eval_ids), len(set(eval_ids)))
        self.assertEqual(
            {tasks[task_id]["stress"] for task_id in eval_ids},
            {"tool_misuse", "context_forgetting", "planning_drift"},
        )

    def test_llamafactory_export_uses_function_role(self):
        action = {"kind": "tool", "tool": "finish_task", "args": {}}
        value = llamafactory_export._response_value("ignored", action)
        self.assertEqual(value, '{"name":"finish_task","arguments":{}}')

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

    def test_state_eval_converts_sharegpt_tool_history(self):
        pair = {
            "system": "system",
            "context_messages": [
                {"from": "human", "value": "task"},
                {"from": "function_call", "value": '{"name":"read","arguments":{"id":"A-1"}}'},
                {"from": "observation", "value": '{"result_id":"R-1"}'},
            ],
        }
        messages = state_eval.to_openai_messages(pair)
        self.assertEqual([row["role"] for row in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(messages[2]["tool_calls"][0]["id"], messages[3]["tool_call_id"])

    def test_state_eval_scores_tool_name_keys_and_exact_values(self):
        expected = {"kind": "tool", "tool": "schedule", "args": {"send_at": "19:30"}}
        self.assertEqual(
            state_eval.tool_action_score(expected, {
                "kind": "tool", "tool": "schedule", "args": {"send_at": "19:30"},
            }),
            (True, True, True),
        )
        self.assertEqual(
            state_eval.tool_action_score(expected, {
                "kind": "tool", "tool": "schedule", "args": {"send_at": "19点30分"},
            }),
            (True, True, False),
        )
        self.assertEqual(
            state_eval.tool_action_score(expected, {
                "kind": "tool", "tool": "notify", "args": {"send_at": "19:30"},
            }),
            (False, False, False),
        )

    def test_state_eval_final_requires_checker_and_grounding(self):
        pair = {
            "chosen_action": {"kind": "final", "content": "完成 R-1"},
            "chosen_grounding": [{"value": "R-1", "source": "observation:write"}],
            "context_messages": [
                {"from": "human", "value": "task"},
                {"from": "function_call", "value": '{"name":"write","arguments":{}}'},
                {"from": "observation", "value": '{"result_id":"R-1"}'},
                {"from": "function_call", "value": '{"name":"finish_task","arguments":{}}'},
                {"from": "observation", "value": '{"status":"ready_for_final"}'},
            ],
        }
        task = {"checker": {"type": "all", "checks": [
            {"type": "tool_call", "expect_tool": "write", "expect_args_contains": {}},
            {"type": "final_contains_all", "values": ["完成"]},
        ]}}
        correct = state_eval.score_action(pair, task, {"kind": "final", "content": "完成，结果 R-1"})
        missing_id = state_eval.score_action(pair, task, {"kind": "final", "content": "完成"})
        failed_task = state_eval.score_action(pair, task, {"kind": "final", "content": "结果 R-1"})
        self.assertTrue(correct["action_correct"])
        self.assertTrue(missing_id["final_task_complete"])
        self.assertFalse(missing_id["action_correct"])
        self.assertFalse(failed_task["action_correct"])

    def test_state_eval_aligns_pairs_and_reports_regressions(self):
        base = [
            {"pair_id": "a", "task_id": "t1", "stress": "tool_misuse", "expected_kind": "tool", "action_correct": True, "tool_name_correct": True, "argument_values_correct": True, "error": None},
            {"pair_id": "b", "task_id": "t2", "stress": "planning_drift", "expected_kind": "final", "action_correct": False, "final_task_complete": False, "final_grounding_complete": True, "error": None},
        ]
        dpo = [
            {**base[1], "action_correct": True, "final_task_complete": True},
            {**base[0], "action_correct": False, "tool_name_correct": False, "argument_values_correct": False},
        ]
        report = state_eval.compare_report(base, dpo)
        self.assertIn("Aligned valid pairs improved/regressed: 1/1", report)
        self.assertIn("independent eval tasks: 2", report)
        with self.assertRaisesRegex(ValueError, "pair_id 不对齐"):
            state_eval.align_rows(base, dpo[:1])

    def test_state_eval_does_not_force_gold_action_kind(self):
        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        tool_calls=None, content="完成 R-1",
                    ))],
                    usage=SimpleNamespace(prompt_tokens=10),
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        pair = {
            "pair_id": "p1", "task_id": "t1", "stress": "planning_drift",
            "decision_index": 1, "system": "system", "tools": [],
            "context_messages": [{"from": "human", "value": "task"}],
            "chosen_action": {"kind": "final", "content": "完成 R-1"},
            "chosen_grounding": [{"value": "R-1", "source": "observation:write"}],
        }
        task = {"checker": {"type": "final_contains_all", "values": ["完成"]}}
        state_eval.evaluate_one(client, "base", pair, task, 0.0, 1)
        self.assertEqual(captured["tool_choice"], "auto")


if __name__ == "__main__":
    unittest.main()
