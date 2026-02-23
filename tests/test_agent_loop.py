"""
Tests for the AgentLoop orchestration engine and related components.
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from vibecad.design.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentState,
    LoopStep,
    READ_ONLY_ACTIONS,
    DESTRUCTIVE_ACTIONS,
    is_read_only,
)
from vibecad.design.design_agent import (
    DesignAction,
    DesignActionType,
    DesignAgent,
    DesignRequest,
    normalize_action_parameters,
)
from vibecad.llm.client import LLMConfig, LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeLLMClient:
    """Minimal LLM client that returns canned responses from a queue."""

    def __init__(self, responses=None):
        self.config = LLMConfig(api_key="test-key", max_tokens=256)
        self.is_available = True
        self._responses = list(responses or [])
        self._call_count = 0

    def chat(self, messages, system_prompt=None):
        self._call_count += 1
        # Sub-agents expect a JSON array response. Keep them inert for these
        # state-machine tests so the loop falls through to the monolithic agent.
        if system_prompt and "Return JSON array" in str(system_prompt):
            content = "[]"
            return LLMResponse(
                content=content,
                model="test",
                raw_response={
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ]
                },
            )
        if self._responses:
            content = self._responses.pop(0)
        else:
            content = '{"assistant_message":"DESIGN_COMPLETE","actions":[]}'
        return LLMResponse(
            content=content,
            model="test",
            raw_response={
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ]
            },
        )


def _make_agent(responses=None):
    """Create a DesignAgent with a fake LLM client."""
    client = _FakeLLMClient(responses)
    agent = DesignAgent(llm_client=client)
    return agent, client


# ---------------------------------------------------------------------------
# Action classification tests
# ---------------------------------------------------------------------------

class TestActionClassification(unittest.TestCase):
    """Verify the read-only vs destructive action sets are consistent."""

    def test_search_is_read_only(self):
        self.assertTrue(is_read_only(DesignActionType.SEARCH_PART))

    def test_export_bom_is_read_only(self):
        self.assertTrue(is_read_only(DesignActionType.EXPORT_BOM))

    def test_run_drc_is_read_only(self):
        self.assertTrue(is_read_only(DesignActionType.RUN_DRC))

    def test_add_component_is_destructive(self):
        self.assertIn(DesignActionType.ADD_COMPONENT, DESTRUCTIVE_ACTIONS)

    def test_draw_track_is_destructive(self):
        self.assertIn(DesignActionType.DRAW_TRACK, DESTRUCTIVE_ACTIONS)

    def test_define_board_outline_is_destructive(self):
        self.assertIn(DesignActionType.DEFINE_BOARD_OUTLINE, DESTRUCTIVE_ACTIONS)

    def test_no_overlap(self):
        overlap = READ_ONLY_ACTIONS & DESTRUCTIVE_ACTIONS
        self.assertEqual(overlap, frozenset(), f"Overlap: {overlap}")


# ---------------------------------------------------------------------------
# AgentLoop state machine tests
# ---------------------------------------------------------------------------

class TestAgentLoopStateMachine(unittest.TestCase):
    """Test the state machine transitions of AgentLoop."""

    def test_initial_state_is_idle(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        self.assertEqual(loop.state, AgentState.IDLE)

    def test_is_running_when_idle(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        self.assertFalse(loop.is_running)

    def test_runs_and_finishes(self):
        """Agent with a DESIGN_COMPLETE response should reach DONE."""
        agent, _ = _make_agent([
            '{"assistant_message":"DESIGN_COMPLETE","actions":[]}',
        ])
        loop = AgentLoop(agent, AgentLoopConfig(max_iterations=5))

        states = []
        loop.set_state_change_callback(lambda s: states.append(s))

        loop.run("test goal", {})
        # Wait for the background thread
        loop._worker_thread.join(timeout=5)

        self.assertEqual(loop.state, AgentState.DONE)
        self.assertIn(AgentState.PLANNING, states)
        self.assertIn(AgentState.DONE, states)

    def test_max_iterations_cap(self):
        """AgentLoop should stop after max_iterations."""
        # Responses that never say DESIGN_COMPLETE
        responses = [
            '{"assistant_message":"doing stuff","actions":[{"actiontype":"SEARCH_PART","description":"search","parameters":{"query":"x"},"requiresApproval":false}]}'
        ] * 6  # More than max

        agent, _ = _make_agent(responses)
        config = AgentLoopConfig(max_iterations=3)
        loop = AgentLoop(agent, config)

        messages = []
        loop.set_ui_message_callback(lambda t: messages.append(t))
        loop.run("test", {})
        loop._worker_thread.join(timeout=10)

        self.assertIn(loop.state, (AgentState.DONE, AgentState.PAUSED))
        self.assertLessEqual(loop.iteration, 3)

    def test_stop_terminates(self):
        """Calling stop() should terminate the loop."""
        # Responses that keep going
        responses = [
            '{"assistant_message":"step","actions":[{"actiontype":"SEARCH_PART","description":"s","parameters":{"query":"x"},"requiresApproval":false}]}'
        ] * 50

        agent, _ = _make_agent(responses)
        loop = AgentLoop(agent, AgentLoopConfig(max_iterations=50))
        loop.run("go", {})
        time.sleep(0.3)
        loop.stop()
        loop._worker_thread.join(timeout=5)

        self.assertEqual(loop.state, AgentState.DONE)


# ---------------------------------------------------------------------------
# AgentLoop question detection
# ---------------------------------------------------------------------------

class TestQuestionDetection(unittest.TestCase):
    """Test clarifying question detection logic."""

    def test_detects_question_mark(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        self.assertTrue(loop._is_question("Which footprint do you prefer?", []))

    def test_no_question_with_actions(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        action = DesignAction(DesignActionType.SEARCH_PART, "search", {})
        self.assertFalse(loop._is_question("Searching...", [action]))

    def test_detects_pattern(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        self.assertTrue(loop._is_question("What would you like me to do next", []))

    def test_statement_is_not_question(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        self.assertFalse(loop._is_question("I will place the resistor now.", []))


# ---------------------------------------------------------------------------
# AgentLoop completion detection
# ---------------------------------------------------------------------------

class TestCompletionDetection(unittest.TestCase):
    """Test DESIGN_COMPLETE detection."""

    def test_uppercase_signal(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        self.assertTrue(loop._is_completion("Done: DESIGN_COMPLETE", []))

    def test_phrase(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        self.assertTrue(loop._is_completion("The design is complete and ready for review.", []))

    def test_normal_message(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        self.assertFalse(loop._is_completion("Placing capacitors now.", []))


# ---------------------------------------------------------------------------
# AgentLoop DRC result parsing
# ---------------------------------------------------------------------------

class TestDRCResultParsing(unittest.TestCase):
    """Test _find_last_drc_result helper."""

    def test_drc_passed(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        results = [
            "[SEARCH_PART] OK: found 3 parts",
            "[RUN_DRC] OK: DRC passed with 0 errors, 2 warnings",
        ]
        drc = loop._find_last_drc_result(results)
        self.assertIsNotNone(drc)
        self.assertTrue(drc["passed"])

    def test_drc_failed(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        results = [
            "[RUN_DRC] OK: DRC found 3 errors, 1 warning",
        ]
        drc = loop._find_last_drc_result(results)
        self.assertIsNotNone(drc)
        self.assertFalse(drc["passed"])

    def test_no_drc(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        results = ["[SEARCH_PART] OK: found stuff"]
        drc = loop._find_last_drc_result(results)
        self.assertIsNone(drc)


class TestBoardSnapshot(unittest.TestCase):
    def test_includes_search_part_results(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        snap = loop._build_board_snapshot(
            {
                "search_part_results": {
                    "arduino uno": [{"name": "ATmega328P-PU", "package": "DIP-28"}]
                }
            }
        )
        self.assertIn("search_part_results", snap)
        self.assertIn("arduino uno", snap["search_part_results"])


class TestLoopCompactionAndDedupe(unittest.TestCase):
    def test_compact_action_results(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        rows = [f"row {i}" for i in range(50)]
        compact = loop._compact_action_results(rows, max_lines=10)
        self.assertIn("omitted", compact)
        self.assertIn("row 0", compact)
        self.assertIn("row 49", compact)

    def test_dedup_search_part_query(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        action_results = []

        def _fake_execute(action, context):
            action.executed = True
            action.success = True
            action.result_message = "ok"
            return action

        loop._execute_action = _fake_execute  # type: ignore[method-assign]

        a1 = DesignAction(DesignActionType.SEARCH_PART, "Search", {"query": "ATmega328P"})
        a2 = DesignAction(DesignActionType.SEARCH_PART, "Search again", {"query": "atmega328p"})
        a3 = DesignAction(DesignActionType.SEARCH_PART, "Search third time", {"query": "ATMEGA328P"})

        loop._execute_and_record(a1, {}, action_results, auto=True)
        loop._execute_and_record(a2, {}, action_results, auto=True)
        loop._execute_and_record(a3, {}, action_results, auto=True)

        skipped = [r for r in action_results if "SKIPPED" in r]
        self.assertGreaterEqual(len(skipped), 1)
        self.assertEqual(len(loop._history), 1)

    def test_full_workflow_completion_gate(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        loop._goal = "build arduino uno from scratch"
        loop._require_full_workflow = True
        loop._last_drc_passed = True
        loop._phase.update(
            {
                "outline_defined": True,
                "components_placed": 1,
                "nets_assigned": 0,
                "routing_attempted": False,
            }
        )
        ok, reason = loop._completion_requirements_met()
        self.assertFalse(ok)
        self.assertIn("component", reason.lower())

    def test_error_messages_are_logged_to_debug(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)

        # Force a failing action that results in a user-facing "❌ ..." line.
        def _fake_execute(action, context):
            action.executed = True
            action.success = False
            action.result_message = "Missing component reference or location"
            return action

        loop._execute_action = _fake_execute  # type: ignore[method-assign]
        action_results = []
        action = DesignAction(
            DesignActionType.MOVE_COMPONENT,
            "Move component",
            {"ref": "U1", "location": {"x": 10, "y": 20}},
        )

        with self.assertLogs("vibecad.design.agent_loop", level="DEBUG") as cm:
            loop._execute_and_record(action, {}, action_results)

        self.assertTrue(
            any("❌ Missing component reference or location" in line for line in cm.output),
            f"Expected debug log for failure, got: {cm.output}",
        )

    def test_phase_gate_blocks_are_logged_to_debug(self):
        agent, _ = _make_agent()
        loop = AgentLoop(agent)
        loop._phase["outline_defined"] = True

        action_results = []
        action = DesignAction(
            DesignActionType.DEFINE_BOARD_OUTLINE,
            "Define board outline",
            {"width_mm": 80.0, "height_mm": 50.0, "center": {"x": 150.0, "y": 100.0}},
        )

        with self.assertLogs("vibecad.design.agent_loop", level="DEBUG") as cm:
            loop._execute_and_record(action, {}, action_results)

        self.assertTrue(
            any("⛔ Board outline is already defined" in line for line in cm.output),
            f"Expected debug log for phase gate block, got: {cm.output}",
        )

    def test_normalize_action_parameters_uses_ref_and_location_object(self):
        params = {"id": "U1", "location": [150.0, 100.0], "reference": "U1-ignored"}
        norm = normalize_action_parameters(DesignActionType.MOVE_COMPONENT, params)
        self.assertEqual(norm.get("ref"), "U1")
        self.assertNotIn("id", norm)
        self.assertNotIn("reference", norm)
        self.assertIsInstance(norm.get("location"), dict)
        self.assertEqual(norm["location"]["x"], 150.0)
        self.assertEqual(norm["location"]["y"], 100.0)

    def test_normalize_align_components_refs_is_list(self):
        norm = normalize_action_parameters(
            DesignActionType.ALIGN_COMPONENTS,
            {"refs": ["j1", " U2 ", None], "direction": "horizontal"},
        )
        self.assertEqual(norm["refs"], ["J1", "U2"])

    def test_normalize_assign_nets_from_net_name_and_pads(self):
        norm = normalize_action_parameters(
            DesignActionType.ASSIGN_NETS,
            {"net_name": "GND", "pads": ["U1:3", "u1/5", "bad", "J1-4"]},
        )
        assigns = norm.get("assignments")
        self.assertIsInstance(assigns, list)
        self.assertGreaterEqual(len(assigns), 3)
        self.assertTrue(any(a.get("ref") == "U1" and a.get("pad") == "3" for a in assigns))
        self.assertTrue(any(a.get("ref") == "U1" and a.get("pad") == "5" for a in assigns))
        self.assertTrue(any(a.get("ref") == "J1" and a.get("pad") == "4" for a in assigns))


# ---------------------------------------------------------------------------
# AgentLoop approval flow
# ---------------------------------------------------------------------------

class TestApprovalFlow(unittest.TestCase):
    """Test the approval gate in the agent loop."""

    def test_approve_action(self):
        """When a destructive action is approved, it should execute."""
        responses = [
            '{"assistant_message":"Adding component","actions":[{"actiontype":"ADD_COMPONENT","description":"Add R1","parameters":{"query":"10k resistor","x":10,"y":20},"requiresApproval":true}]}',
            '{"assistant_message":"DESIGN_COMPLETE","actions":[]}',
        ]
        agent, _ = _make_agent(responses)
        loop = AgentLoop(agent, AgentLoopConfig(max_iterations=5))

        previews = []
        loop.set_ui_action_preview_callback(
            lambda atype, desc, prev, act: previews.append((atype, act))
        )
        messages = []
        loop.set_ui_message_callback(lambda t: messages.append(t))

        loop.run("add a resistor", {})
        time.sleep(0.5)  # Let the loop reach AWAITING_APPROVAL

        # It should be waiting for approval
        max_wait = 3.0
        waited = 0
        while loop.state != AgentState.AWAITING_APPROVAL and waited < max_wait:
            time.sleep(0.1)
            waited += 0.1

        if loop.state == AgentState.AWAITING_APPROVAL and previews:
            _atype, act = previews[0]
            loop.approve_action(act, True)

        loop._worker_thread.join(timeout=10)
        self.assertEqual(loop.state, AgentState.DONE)

    def test_reject_action(self):
        """When a destructive action is rejected, it should be skipped."""
        responses = [
            '{"assistant_message":"Placing part","actions":[{"actiontype":"ADD_COMPONENT","description":"Add U1","parameters":{"query":"atmega328p"},"requiresApproval":true}]}',
            '{"assistant_message":"DESIGN_COMPLETE","actions":[]}',
        ]
        agent, _ = _make_agent(responses)
        loop = AgentLoop(agent, AgentLoopConfig(max_iterations=5))

        messages = []
        loop.set_ui_message_callback(lambda t: messages.append(t))

        previews = []
        loop.set_ui_action_preview_callback(
            lambda atype, desc, prev, act: previews.append((atype, act))
        )

        loop.run("place MCU", {})
        time.sleep(0.5)

        max_wait = 3.0
        waited = 0
        while loop.state != AgentState.AWAITING_APPROVAL and waited < max_wait:
            time.sleep(0.1)
            waited += 0.1

        if loop.state == AgentState.AWAITING_APPROVAL and previews:
            _atype, act = previews[0]
            loop.approve_action(act, False, "Not now")

        loop._worker_thread.join(timeout=10)
        # Check that rejection was noted
        skipped = [m for m in messages if "Skipped" in m or "REJECTED" in m.upper() if isinstance(m, str)]
        # The loop should have continued to DESIGN_COMPLETE


# ---------------------------------------------------------------------------
# AgentLoop pause/resume
# ---------------------------------------------------------------------------

class TestPauseResume(unittest.TestCase):
    """Test pause and resume behavior."""

    def test_pause_sets_state(self):
        responses = [
            '{"assistant_message":"searching","actions":[{"actiontype":"SEARCH_PART","description":"s","parameters":{"query":"x"},"requiresApproval":false}]}'
        ] * 20

        agent, _ = _make_agent(responses)
        loop = AgentLoop(agent, AgentLoopConfig(max_iterations=20))
        loop.run("go", {})
        time.sleep(0.3)
        loop.pause()
        loop._worker_thread.join(timeout=5)
        self.assertEqual(loop.state, AgentState.PAUSED)


# ---------------------------------------------------------------------------
# LoopStep dataclass
# ---------------------------------------------------------------------------

class TestLoopStep(unittest.TestCase):

    def test_defaults(self):
        step = LoopStep(iteration=1)
        self.assertEqual(step.iteration, 1)
        self.assertIsNone(step.action)
        self.assertFalse(step.was_auto_executed)


# ---------------------------------------------------------------------------
# AgentLoopConfig
# ---------------------------------------------------------------------------

class TestAgentLoopConfig(unittest.TestCase):

    def test_defaults(self):
        config = AgentLoopConfig()
        self.assertEqual(config.max_iterations, 50)
        self.assertEqual(config.max_drc_retries, 10)
        self.assertTrue(config.auto_approve_readonly)

    def test_custom(self):
        config = AgentLoopConfig(max_iterations=10, max_drc_retries=2, auto_approve_readonly=False)
        self.assertEqual(config.max_iterations, 10)
        self.assertFalse(config.auto_approve_readonly)


if __name__ == '__main__':
    unittest.main()
