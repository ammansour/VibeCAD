"""
Tests for DRC integration in design_agent.py.
"""

import unittest

from vibecad.design.design_agent import DesignAgent, DesignActionType


class TestDRCReportParsing(unittest.TestCase):
    """Test the text-based DRC report parser.

    _parse_drc_text_report is a @staticmethod returning (errors: List[str], warnings: List[str]).
    """

    def test_parse_clean_report(self):
        report = (
            "** DRC Report **\n"
            "** Found 0 DRC violations **\n"
            "** Found 0 unconnected pads **\n"
        )
        errors, warnings = DesignAgent._parse_drc_text_report(report)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(warnings), 0)

    def test_parse_report_with_errors(self):
        report = (
            "** DRC Report **\n"
            "** Found 3 DRC violations **\n"
            "[Error] Clearance violation at (10mm, 20mm)\n"
            "[Error] Track too narrow at (30mm, 40mm)\n"
            "[Error] Via too small at (50mm, 50mm)\n"
            "** Found 1 unconnected pads **\n"
            "[Warning] Unconnected pad on R1\n"
        )
        errors, warnings = DesignAgent._parse_drc_text_report(report)
        self.assertEqual(len(errors), 3)
        self.assertEqual(len(warnings), 1)

    def test_parse_empty_report(self):
        errors, warnings = DesignAgent._parse_drc_text_report("")
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(warnings), 0)


class TestDRCResultFormatting(unittest.TestCase):
    """Test _format_drc_results helper.

    Signature: _format_drc_results(self, errors: List[str], warnings: List[str]) -> str
    """

    def setUp(self):
        self.agent = DesignAgent(llm_client=None)

    def test_format_passed(self):
        text = self.agent._format_drc_results([], [])
        self.assertIn("passed", text.lower())

    def test_format_with_errors(self):
        text = self.agent._format_drc_results(
            ["Clearance issue", "Track width issue"], []
        )
        self.assertIn("error", text.lower())
        self.assertIn("2", text)

    def test_format_with_warnings(self):
        text = self.agent._format_drc_results(
            [], ["Unconnected pad"]
        )
        self.assertIn("warning", text.lower())


class TestNewActionTypes(unittest.TestCase):
    """Verify new action types exist in the enum."""

    def test_autoroute_board(self):
        self.assertIsNotNone(DesignActionType.AUTOROUTE_BOARD)

    def test_set_layer_count(self):
        self.assertIsNotNone(DesignActionType.SET_LAYER_COUNT)

    def test_add_via(self):
        self.assertIsNotNone(DesignActionType.ADD_VIA)

    def test_add_polygon(self):
        self.assertIsNotNone(DesignActionType.ADD_POLYGON)

    def test_add_text(self):
        self.assertIsNotNone(DesignActionType.ADD_TEXT)

    def test_add_mounting_hole(self):
        self.assertIsNotNone(DesignActionType.ADD_MOUNTING_HOLE)

    def test_align_components(self):
        self.assertIsNotNone(DesignActionType.ALIGN_COMPONENTS)

    def test_define_board_outline(self):
        self.assertIsNotNone(DesignActionType.DEFINE_BOARD_OUTLINE)

    def test_run_drc(self):
        self.assertIsNotNone(DesignActionType.RUN_DRC)


class TestDesignAgentNewHandlers(unittest.TestCase):
    """Verify new handlers are registered in the agent."""

    def setUp(self):
        self.agent = DesignAgent(llm_client=None)

    def test_handler_exists_for_run_drc(self):
        handler = self.agent._get_action_handler(DesignActionType.RUN_DRC)
        self.assertIsNotNone(handler)

    def test_handler_exists_for_add_via(self):
        handler = self.agent._get_action_handler(DesignActionType.ADD_VIA)
        self.assertIsNotNone(handler)

    def test_handler_exists_for_define_board_outline(self):
        handler = self.agent._get_action_handler(DesignActionType.DEFINE_BOARD_OUTLINE)
        self.assertIsNotNone(handler)

    def test_handler_exists_for_add_mounting_hole(self):
        handler = self.agent._get_action_handler(DesignActionType.ADD_MOUNTING_HOLE)
        self.assertIsNotNone(handler)

    def test_handler_exists_for_align_components(self):
        handler = self.agent._get_action_handler(DesignActionType.ALIGN_COMPONENTS)
        self.assertIsNotNone(handler)

    def test_handler_exists_for_add_text(self):
        handler = self.agent._get_action_handler(DesignActionType.ADD_TEXT)
        self.assertIsNotNone(handler)

    def test_handler_exists_for_add_polygon(self):
        handler = self.agent._get_action_handler(DesignActionType.ADD_POLYGON)
        self.assertIsNotNone(handler)

    def test_handler_exists_for_autoroute_board(self):
        handler = self.agent._get_action_handler(DesignActionType.AUTOROUTE_BOARD)
        self.assertIsNotNone(handler)

    def test_handler_exists_for_set_layer_count(self):
        handler = self.agent._get_action_handler(DesignActionType.SET_LAYER_COUNT)
        self.assertIsNotNone(handler)


class TestGenerateDescription(unittest.TestCase):
    """Verify _generate_description covers new action types.

    Signature: _generate_description(self, action_type, params) -> str
    """

    def setUp(self):
        self.agent = DesignAgent(llm_client=None)

    def test_description_for_autoroute(self):
        desc = self.agent._generate_description(DesignActionType.AUTOROUTE_BOARD, {})
        self.assertTrue(len(desc) > 0)

    def test_description_for_set_layer_count(self):
        desc = self.agent._generate_description(DesignActionType.SET_LAYER_COUNT, {'count': 4})
        self.assertTrue(len(desc) > 0)

    def test_description_for_add_via(self):
        desc = self.agent._generate_description(DesignActionType.ADD_VIA, {'x': 10, 'y': 20})
        self.assertTrue(len(desc) > 0)


class TestLocationParsing(unittest.TestCase):
    def test_parse_location_csv(self):
        out = DesignAgent._parse_location_mm("50,25")
        self.assertEqual(out, (50.0, 25.0))

    def test_parse_location_negative(self):
        out = DesignAgent._parse_location_mm("(-70, -40) mm")
        self.assertEqual(out, (-70.0, -40.0))

    def test_parse_location_dict(self):
        out = DesignAgent._parse_location_mm({"x": 0, "y": 45})
        self.assertEqual(out, (0.0, 45.0))

    def test_parse_location_dict_string(self):
        out = DesignAgent._parse_location_mm("{'x': 10, 'y': -10}")
        self.assertEqual(out, (10.0, -10.0))


if __name__ == '__main__':
    unittest.main()
