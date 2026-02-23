"""
Tests for the sub-agent architecture: base, info_gathering, placement,
routing, verification, and orchestrator.
"""

import math
import unittest
from unittest.mock import MagicMock

from vibecad.design.design_agent import DesignAction, DesignActionType
from vibecad.design.sub_agents.base import SubAgent, SubAgentResult
from vibecad.design.sub_agents.info_gathering import InfoGatheringAgent
from vibecad.design.sub_agents.placement import (
    PlacementAgent,
    _BBox,
    _PlacedComponent,
    _classify_ref,
    _snap,
    MIN_CLEARANCE_MM,
)
from vibecad.design.sub_agents.routing import RoutingAgent
from vibecad.design.sub_agents.verification import VerificationAgent
from vibecad.design.sub_agents.orchestrator import Orchestrator, DesignPhase


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str = "[]", *, available: bool = True):
        self.is_available = available
        self._content = content

    def chat(self, messages, system_prompt=None):
        return _FakeResponse(self._content)


# ---------------------------------------------------------------------------
# BBox spatial helpers
# ---------------------------------------------------------------------------

class TestBBox(unittest.TestCase):
    """Test the axis-aligned bounding box helper."""

    def test_non_overlapping_boxes(self):
        a = _BBox(cx=0, cy=0, hw=5, hh=5)
        b = _BBox(cx=20, cy=0, hw=5, hh=5)
        self.assertFalse(a.overlaps(b, clearance=2.0))

    def test_overlapping_boxes(self):
        a = _BBox(cx=0, cy=0, hw=5, hh=5)
        b = _BBox(cx=8, cy=0, hw=5, hh=5)
        self.assertTrue(a.overlaps(b, clearance=2.0))

    def test_touching_boxes_with_clearance(self):
        """Boxes 10mm apart edge-to-edge, 2mm clearance → should overlap."""
        a = _BBox(cx=0, cy=0, hw=5, hh=5)
        b = _BBox(cx=11, cy=0, hw=5, hh=5)
        # a.right = 5, b.left = 6 → gap = 1 < clearance = 2 → overlap
        self.assertTrue(a.overlaps(b, clearance=2.0))

    def test_sufficient_gap(self):
        a = _BBox(cx=0, cy=0, hw=5, hh=5)
        b = _BBox(cx=13, cy=0, hw=5, hh=5)
        # a.right = 5, b.left = 8 → gap = 3 > clearance = 2
        self.assertFalse(a.overlaps(b, clearance=2.0))

    def test_overlap_vector_pushes_apart(self):
        a = _BBox(cx=0, cy=0, hw=5, hh=5)
        b = _BBox(cx=7, cy=0, hw=5, hh=5)
        dx, dy = a.overlap_vector(b, clearance=2.0)
        # a should be pushed left (negative dx).
        self.assertLess(dx, 0)
        self.assertAlmostEqual(dy, 0.0)

    def test_overlap_vector_zero_when_no_overlap(self):
        a = _BBox(cx=0, cy=0, hw=5, hh=5)
        b = _BBox(cx=20, cy=0, hw=5, hh=5)
        dx, dy = a.overlap_vector(b)
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)

    def test_properties(self):
        b = _BBox(cx=10, cy=20, hw=5, hh=3)
        self.assertAlmostEqual(b.left, 5.0)
        self.assertAlmostEqual(b.right, 15.0)
        self.assertAlmostEqual(b.top, 17.0)
        self.assertAlmostEqual(b.bottom, 23.0)


# ---------------------------------------------------------------------------
# classify_ref
# ---------------------------------------------------------------------------

class TestClassifyRef(unittest.TestCase):
    def test_ic(self):
        self.assertEqual(_classify_ref("U1"), "ic")

    def test_resistor(self):
        self.assertEqual(_classify_ref("R12"), "passive")

    def test_capacitor(self):
        self.assertEqual(_classify_ref("C3"), "passive")

    def test_connector(self):
        self.assertEqual(_classify_ref("J1"), "connector")

    def test_mounting_hole(self):
        self.assertEqual(_classify_ref("H1"), "mounting")

    def test_led(self):
        self.assertEqual(_classify_ref("D1", "LED"), "passive")

    def test_generic(self):
        self.assertEqual(_classify_ref("SW1"), "generic")


# ---------------------------------------------------------------------------
# Grid snap
# ---------------------------------------------------------------------------

class TestSnap(unittest.TestCase):
    def test_snap_to_grid(self):
        self.assertAlmostEqual(_snap(10.0, 1.27), 10.16, places=2)

    def test_snap_exact(self):
        self.assertAlmostEqual(_snap(2.54, 1.27), 2.54, places=4)


# ---------------------------------------------------------------------------
# PlacementAgent.resolve_overlaps
# ---------------------------------------------------------------------------

class TestResolveOverlaps(unittest.TestCase):
    def test_no_overlap_no_change(self):
        comps = [
            {"ref": "U1", "x": 0, "y": 0, "width": 10, "height": 10},
            {"ref": "R1", "x": 30, "y": 0, "width": 5, "height": 5},
        ]
        result = PlacementAgent.resolve_overlaps(comps)
        # Positions should be essentially unchanged (just grid-snapped).
        self.assertAlmostEqual(result[0]["x"], _snap(0), places=1)
        self.assertAlmostEqual(result[1]["x"], _snap(30), places=1)

    def test_overlapping_components_separated(self):
        comps = [
            {"ref": "U1", "x": 0, "y": 0, "width": 10, "height": 10},
            {"ref": "U2", "x": 3, "y": 0, "width": 10, "height": 10},
        ]
        result = PlacementAgent.resolve_overlaps(comps)
        # After resolution, bounding boxes should not overlap.
        b1 = _BBox(result[0]["x"], result[0]["y"], 5, 5)
        b2 = _BBox(result[1]["x"], result[1]["y"], 5, 5)
        self.assertFalse(b1.overlaps(b2, clearance=1.5))

    def test_three_overlapping(self):
        comps = [
            {"ref": "U1", "x": 0, "y": 0, "width": 10, "height": 10},
            {"ref": "C1", "x": 2, "y": 2, "width": 5, "height": 5},
            {"ref": "R1", "x": 1, "y": -1, "width": 4, "height": 4},
        ]
        result = PlacementAgent.resolve_overlaps(comps, clearance_mm=2.0)
        # Verify no pair overlaps.
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                bi = _BBox(result[i]["x"], result[i]["y"],
                           result[i]["width"] / 2, result[i]["height"] / 2)
                bj = _BBox(result[j]["x"], result[j]["y"],
                           result[j]["width"] / 2, result[j]["height"] / 2)
                self.assertFalse(
                    bi.overlaps(bj, clearance=1.5),
                    f"{result[i]['ref']} still overlaps {result[j]['ref']}"
                )

    def test_single_component(self):
        comps = [{"ref": "U1", "x": 10, "y": 10, "width": 5, "height": 5}]
        result = PlacementAgent.resolve_overlaps(comps)
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# PlacementAgent.optimize_layout
# ---------------------------------------------------------------------------

class TestOptimizeLayout(unittest.TestCase):
    def test_basic_layout(self):
        comps = [
            {"ref": "U1", "x": 50, "y": 50, "width": 20, "height": 10, "value": "ATmega328P"},
            {"ref": "J1", "x": 50, "y": 50, "width": 12, "height": 8, "value": "USB-C"},
            {"ref": "C1", "x": 50, "y": 50, "width": 3, "height": 3, "value": "100nF"},
            {"ref": "R1", "x": 50, "y": 50, "width": 3, "height": 2, "value": "10k"},
        ]
        result = PlacementAgent.optimize_layout(
            comps,
            board_width_mm=80,
            board_height_mm=60,
            board_origin_x_mm=0,
            board_origin_y_mm=0,
        )
        self.assertEqual(len(result), 4)
        # All components should be within the board.
        for c in result:
            self.assertGreater(c["x"], 0)
            self.assertLess(c["x"], 80)
            self.assertGreater(c["y"], 0)
            self.assertLess(c["y"], 60)

    def test_no_overlaps_after_optimization(self):
        """All components starting at the same point should end up separated."""
        comps = [
            {"ref": f"R{i}", "x": 40, "y": 30, "width": 4, "height": 2, "value": "10k"}
            for i in range(1, 8)
        ]
        result = PlacementAgent.optimize_layout(
            comps, board_width_mm=80, board_height_mm=60,
        )
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                bi = _BBox(result[i]["x"], result[i]["y"], 2, 1)
                bj = _BBox(result[j]["x"], result[j]["y"], 2, 1)
                self.assertFalse(
                    bi.overlaps(bj, clearance=1.0),
                    f"{result[i]['ref']} overlaps {result[j]['ref']} at "
                    f"({result[i]['x']:.1f},{result[i]['y']:.1f}) vs "
                    f"({result[j]['x']:.1f},{result[j]['y']:.1f})"
                )


# ---------------------------------------------------------------------------
# PlacementAgent.estimate_board_size
# ---------------------------------------------------------------------------

class TestEstimateBoardSize(unittest.TestCase):
    def test_returns_reasonable_size(self):
        comps = [
            {"width": 20, "height": 10},
            {"width": 12, "height": 8},
            {"width": 3, "height": 3},
        ]
        w, h = PlacementAgent.estimate_board_size(comps)
        self.assertGreaterEqual(w, 30)
        self.assertGreaterEqual(h, 20)

    def test_empty_returns_minimum(self):
        w, h = PlacementAgent.estimate_board_size([])
        self.assertEqual(w, 30.0)
        self.assertEqual(h, 30.0)


# ---------------------------------------------------------------------------
# InfoGatheringAgent
# ---------------------------------------------------------------------------

class TestInfoGatheringAgent(unittest.TestCase):
    def test_llm_plan_parses_actions(self):
        llm = _FakeLLM(
            '[{"action_type":"SEARCH_PART","description":"Search for ATmega328P","parameters":{"query":"ATmega328P"}}]'
        )
        agent = InfoGatheringAgent(llm_client=llm)
        result = agent.plan("I need an ATmega328P", {})
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].action_type, DesignActionType.SEARCH_PART)
        self.assertEqual(result.actions[0].parameters.get("query"), "ATmega328P")

    def test_no_components_detected(self):
        agent = InfoGatheringAgent(llm_client=_FakeLLM("[]"))
        result = agent.plan("hello world", {})
        self.assertTrue(result.phase_complete or len(result.actions) == 0)

    def test_handled_types(self):
        agent = InfoGatheringAgent()
        self.assertIn(DesignActionType.SEARCH_PART, agent.HANDLED_ACTION_TYPES)
        self.assertIn(DesignActionType.SEARCH_WEB, agent.HANDLED_ACTION_TYPES)

    def test_extract_primary_goal_text_ignores_feedback(self):
        wrapped = (
            "PHASE: GATHER — Find missing parts and required datasheets.\n"
            "USER GOAL: build arduino uno from scratch\n"
            "FEEDBACK FROM PREVIOUS STEP:\n"
            "[SEARCH_PART] OK: found Package_DIP:DIP-28...\n"
            "[SEARCH_PART] OK: found ESP32-WROOM-32...\n"
        )
        self.assertEqual(
            InfoGatheringAgent._extract_primary_goal_text(wrapped),
            "build arduino uno from scratch",
        )


# ---------------------------------------------------------------------------
# RoutingAgent
# ---------------------------------------------------------------------------

class TestRoutingAgent(unittest.TestCase):
    def test_handled_types(self):
        agent = RoutingAgent()
        self.assertIn(DesignActionType.DEFINE_NET, agent.HANDLED_ACTION_TYPES)
        self.assertIn(DesignActionType.AUTOROUTE_BOARD, agent.HANDLED_ACTION_TYPES)
        self.assertNotIn(DesignActionType.ADD_COMPONENT, agent.HANDLED_ACTION_TYPES)

    def test_plan_without_llm_raises(self):
        from vibecad.llm.client import LLMError
        agent = RoutingAgent()
        with self.assertRaises(LLMError):
            agent.plan("route everything", {})

    def test_plan_with_llm_parses_actions(self):
        llm = _FakeLLM(
            '[{"action_type":"AUTOROUTE_BOARD","description":"Autoroute","parameters":{}}]'
        )
        agent = RoutingAgent(llm_client=llm)
        result = agent.plan("route everything", {})
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].action_type, DesignActionType.AUTOROUTE_BOARD)


# ---------------------------------------------------------------------------
# VerificationAgent
# ---------------------------------------------------------------------------

class TestVerificationAgent(unittest.TestCase):
    def test_handled_types(self):
        agent = VerificationAgent()
        self.assertIn(DesignActionType.RUN_DRC, agent.HANDLED_ACTION_TYPES)
        self.assertIn(DesignActionType.RUN_ERC, agent.HANDLED_ACTION_TYPES)

    def test_plan_without_llm_raises(self):
        from vibecad.llm.client import LLMError
        agent = VerificationAgent()
        with self.assertRaises(LLMError):
            agent.plan("check the board", {})

    def test_analyse_courtyard_errors(self):
        agent = VerificationAgent(llm_client=_FakeLLM("[]"))
        result = agent.plan("fix errors", {
            "drc_errors": "Courtyard overlap between U1 and C2. Board edge clearance violation at R3."
        })
        # No deterministic fix synthesis: VerificationAgent should rely on LLM output.
        self.assertEqual(result.actions, [])

    def test_analyse_short_errors(self):
        agent = VerificationAgent(llm_client=_FakeLLM("[]"))
        result = agent.plan("fix errors", {
            "drc_errors": "Short circuit between GND and VCC on track segment"
        })
        # No deterministic fix synthesis: VerificationAgent should rely on LLM output.
        self.assertEqual(result.actions, [])

    def test_extract_refs(self):
        refs = VerificationAgent._extract_refs(
            "Courtyard overlap U1 / C2, also R3 edge violation"
        )
        self.assertIn("U1", refs)
        self.assertIn("C2", refs)
        self.assertIn("R3", refs)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator(unittest.TestCase):
    def test_initial_phase_is_gather(self):
        orch = Orchestrator()
        self.assertEqual(orch.phase, DesignPhase.GATHER)

    def test_advance_to(self):
        orch = Orchestrator()
        orch.advance_to(DesignPhase.ROUTE)
        self.assertEqual(orch.phase, DesignPhase.ROUTE)

    def test_reset(self):
        orch = Orchestrator()
        orch.advance_to(DesignPhase.VERIFY)
        orch.reset()
        self.assertEqual(orch.phase, DesignPhase.GATHER)

    def test_step_returns_result(self):
        orch = Orchestrator(llm_client=_FakeLLM("[]"))
        result = orch.step("design an Arduino UNO", {})
        self.assertIsInstance(result, SubAgentResult)

    def test_get_agent(self):
        orch = Orchestrator()
        self.assertIsInstance(orch.get_agent("placement"), PlacementAgent)
        self.assertIsInstance(orch.get_agent("routing"), RoutingAgent)
        self.assertIsNone(orch.get_agent("nonexistent"))

    def test_placement_optimization_convenience(self):
        comps = [
            {"ref": "U1", "x": 0, "y": 0, "width": 10, "height": 10},
            {"ref": "U2", "x": 3, "y": 0, "width": 10, "height": 10},
        ]
        orch = Orchestrator()
        result = orch.run_overlap_resolution(comps, clearance_mm=2.0)
        b1 = _BBox(result[0]["x"], result[0]["y"], 5, 5)
        b2 = _BBox(result[1]["x"], result[1]["y"], 5, 5)
        self.assertFalse(b1.overlaps(b2, clearance=1.5))

    def test_phase_auto_advance_gather(self):
        """After 3 gather attempts, orchestrator should advance to PLACE."""
        orch = Orchestrator(llm_client=_FakeLLM("[]"))
        for _ in range(4):
            orch.step("design a board", {})
        self.assertNotEqual(orch.phase, DesignPhase.GATHER)

    def test_place_phase_advances_without_components(self):
        """PLACE phase must not loop forever when no components are placed."""
        orch = Orchestrator(llm_client=_FakeLLM("[]"))
        orch.advance_to(DesignPhase.PLACE)
        # Simulate 6 attempts in PLACE with empty board snapshot
        for _ in range(6):
            orch.step("design an arduino uno", {}, board_snapshot={"components": []})
        # Should have advanced past PLACE
        self.assertNotEqual(orch.phase, DesignPhase.PLACE)

    def test_outline_phase_advances_when_outline_defined(self):
        """OUTLINE phase should advance as soon as the outline is defined."""
        orch = Orchestrator()
        orch.advance_to(DesignPhase.OUTLINE)
        orch._maybe_advance_phase({}, {"outline_defined": True}, None)
        self.assertEqual(orch.phase, DesignPhase.ARRANGE)

    def test_placement_fallback_sets_phase_complete(self):
        """PlacementAgent requires an LLM."""
        from vibecad.llm.client import LLMError
        agent = PlacementAgent()
        with self.assertRaises(LLMError):
            agent.plan("place all components", {})


# ---------------------------------------------------------------------------
# SubAgent base
# ---------------------------------------------------------------------------

class TestSubAgentBase(unittest.TestCase):
    def test_plan_raises(self):
        agent = SubAgent()
        with self.assertRaises(NotImplementedError):
            agent.plan("goal", {})

    def test_can_handle_empty(self):
        agent = SubAgent()
        self.assertFalse(agent.can_handle(frozenset({DesignActionType.ADD_COMPONENT})))

    def test_llm_not_available(self):
        agent = SubAgent()
        self.assertFalse(agent._llm_available())

    def test_llm_chat_import_path(self):
        class _FakeResp:
            content = "ok"

        class _FakeClient:
            is_available = True

            def chat(self, messages, system_prompt=None):
                return _FakeResp()

        agent = SubAgent(llm_client=_FakeClient())
        self.assertEqual(agent._llm_chat("hello"), "ok")


class TestPlacementPromptHandoff(unittest.TestCase):
    def test_prompt_includes_search_handoff(self):
        agent = PlacementAgent()
        prompt = agent._build_prompt(
            "place remaining components",
            {},
            {
                "components": [],
                "search_part_results": {
                    "arduino uno": [
                        {
                            "name": "MCU_Microchip_ATmega:ATmega328P-PU",
                            "package": "DIP-28",
                            "is_footprint_candidate": True,
                        },
                        {
                            "name": "Crystal:Crystal_HC49-U_Vertical",
                            "package": "HC49",
                            "is_footprint_candidate": True,
                        },
                    ]
                },
            },
        )
        self.assertIn("Phase-1 searched candidate parts", prompt)
        self.assertIn("ATmega328P-PU", prompt)


if __name__ == "__main__":
    unittest.main()
