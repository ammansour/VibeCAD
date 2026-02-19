"""
Tests for VibeCAD checks and parsers.
"""

import unittest
import tempfile
import os
from pathlib import Path


class TestSExprParser(unittest.TestCase):
    """Tests for S-expression parser."""
    
    def test_parse_simple_node(self):
        from vibecad.parsers.sexpr import parse_sexpr
        
        content = '(test_node value1 123)'
        node = parse_sexpr(content)
        
        self.assertEqual(node.name, 'test_node')
        self.assertEqual(node.values[0], 'value1')
        self.assertEqual(node.values[1], 123)
    
    def test_parse_nested_nodes(self):
        from vibecad.parsers.sexpr import parse_sexpr
        
        content = '(parent (child1 value1) (child2 value2))'
        node = parse_sexpr(content)
        
        self.assertEqual(node.name, 'parent')
        self.assertEqual(len(node.children), 2)
        self.assertEqual(node.children[0].name, 'child1')
        self.assertEqual(node.children[1].name, 'child2')
    
    def test_parse_quoted_string(self):
        from vibecad.parsers.sexpr import parse_sexpr
        
        content = '(node "quoted string with spaces")'
        node = parse_sexpr(content)
        
        self.assertEqual(node.values[0], 'quoted string with spaces')
    
    def test_get_child(self):
        from vibecad.parsers.sexpr import parse_sexpr
        
        content = '(parent (child1 value1) (child2 value2))'
        node = parse_sexpr(content)
        
        child = node.get_child('child1')
        self.assertIsNotNone(child)
        self.assertEqual(child.values[0], 'value1')
        
        self.assertIsNone(node.get_child('nonexistent'))


class TestPCBParser(unittest.TestCase):
    """Tests for PCB parser."""
    
    def setUp(self):
        """Create a temporary PCB file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.pcb_path = os.path.join(self.temp_dir, 'test.kicad_pcb')
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parse_minimal_pcb(self):
        from vibecad.parsers.pcb_parser import PCBParser
        
        content = '''(kicad_pcb
            (version 20221018)
            (generator pcbnew)
        )'''
        
        with open(self.pcb_path, 'w') as f:
            f.write(content)
        
        parser = PCBParser(self.pcb_path)
        data = parser.parse()
        
        self.assertEqual(data.version, 20221018)
        self.assertEqual(data.generator, 'pcbnew')
    
    def test_parse_board_outline(self):
        from vibecad.parsers.pcb_parser import PCBParser
        
        content = '''(kicad_pcb
            (version 20221018)
            (generator pcbnew)
            (gr_line (start 0 0) (end 100 0) (layer "Edge.Cuts") (width 0.1))
            (gr_line (start 100 0) (end 100 100) (layer "Edge.Cuts") (width 0.1))
            (gr_line (start 100 100) (end 0 100) (layer "Edge.Cuts") (width 0.1))
            (gr_line (start 0 100) (end 0 0) (layer "Edge.Cuts") (width 0.1))
        )'''
        
        with open(self.pcb_path, 'w') as f:
            f.write(content)
        
        parser = PCBParser(self.pcb_path)
        data = parser.parse()
        
        self.assertTrue(data.has_board_outline)
        self.assertEqual(len(data.board_outline_lines), 4)
    
    def test_no_board_outline(self):
        from vibecad.parsers.pcb_parser import PCBParser
        
        content = '''(kicad_pcb
            (version 20221018)
            (generator pcbnew)
            (gr_line (start 0 0) (end 100 0) (layer "F.Silkscreen") (width 0.1))
        )'''
        
        with open(self.pcb_path, 'w') as f:
            f.write(content)
        
        parser = PCBParser(self.pcb_path)
        data = parser.parse()
        
        self.assertFalse(data.has_board_outline)


class TestBoardOutlineCheck(unittest.TestCase):
    """Tests for board outline checks."""
    
    def test_missing_outline_detected(self):
        from vibecad.parsers.pcb_parser import PCBData
        from vibecad.checks.board_outline import MissingBoardOutlineCheck
        
        # Create empty PCB data
        pcb_data = PCBData(
            version=20221018,
            generator='test',
            general={},
            layers={},
            setup={},
            nets=[],
            footprints=[],
            tracks=[],
            vias=[],
            zones=[],
            board_outline_lines=[],
            board_outline_arcs=[],
            board_outline_circles=[],
            board_outline_rects=[],
            board_outline_polygons=[]
        )
        
        check = MissingBoardOutlineCheck()
        result = check.run(pcb_data=pcb_data)
        
        self.assertFalse(result.passed)
        self.assertEqual(result.error_count, 1)
        self.assertEqual(result.findings[0].rule_id, 'BOARD_OUTLINE_001')
    
    def test_outline_present_passes(self):
        from vibecad.parsers.pcb_parser import PCBData, Line, Point
        from vibecad.checks.board_outline import MissingBoardOutlineCheck
        
        # Create PCB data with outline
        pcb_data = PCBData(
            version=20221018,
            generator='test',
            general={},
            layers={},
            setup={},
            nets=[],
            footprints=[],
            tracks=[],
            vias=[],
            zones=[],
            board_outline_lines=[
                Line(start=Point(0, 0), end=Point(100, 0), layer='Edge.Cuts', width=0.1)
            ],
            board_outline_arcs=[],
            board_outline_circles=[],
            board_outline_rects=[],
            board_outline_polygons=[]
        )
        
        check = MissingBoardOutlineCheck()
        result = check.run(pcb_data=pcb_data)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.error_count, 0)


class TestCheckResult(unittest.TestCase):
    """Tests for CheckResult serialization."""
    
    def test_to_json(self):
        from vibecad.checks.base import CheckResult, Finding, Severity
        import json
        
        result = CheckResult(
            check_id='TEST_001',
            check_name='Test Check',
            description='A test check',
            passed=False,
            findings=[
                Finding(
                    rule_id='TEST_001',
                    severity=Severity.ERROR,
                    message='Test finding',
                    layer='Edge.Cuts',
                    location_x=10.0,
                    location_y=20.0
                )
            ]
        )
        
        json_str = result.to_json()
        data = json.loads(json_str)
        
        self.assertEqual(data['check_id'], 'TEST_001')
        self.assertEqual(data['passed'], False)
        self.assertEqual(len(data['findings']), 1)
        self.assertEqual(data['findings'][0]['rule_id'], 'TEST_001')


class TestLLMClientResponseParsing(unittest.TestCase):
    def test_parse_response_empty_content_raises(self):
        from vibecad.llm.client import LLMClient, LLMConfig, LLMError

        client = LLMClient(LLMConfig(api_key="x"))
        response = {
            "id": "test",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
        }

        with self.assertRaises(LLMError):
            client._parse_response(response)

    def test_parse_response_tool_calls_fallback(self):
        """Providers sometimes return tool_calls with empty message.content."""
        from vibecad.llm.client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(api_key="x"))
        args = '{"assistant_message":"ok","actions":[]}'
        response = {
            "id": "test",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "vibecad_actions", "arguments": args},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }

        parsed = client._parse_response(response)
        self.assertEqual(parsed.content, args)

    def test_parse_response_content_parts_list(self):
        """Some providers return message.content as a list of parts."""
        from vibecad.llm.client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(api_key="x"))
        response = {
            "id": "test",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                    "finish_reason": "stop",
                }
            ],
        }

        parsed = client._parse_response(response)
        self.assertEqual(parsed.content, "hello")

    def test_parse_response_legacy_text_field(self):
        from vibecad.llm.client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(api_key="x"))
        response = {
            "id": "test",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "text": "hello from legacy",
                    "finish_reason": "stop",
                }
            ],
        }

        parsed = client._parse_response(response)
        self.assertEqual(parsed.content, "hello from legacy")

    def test_chat_retries_on_empty_length(self):
        from vibecad.llm.client import LLMClient, LLMConfig, LLMMessage

        class _FakeClient(LLMClient):
            def __init__(self, config):
                super().__init__(config)
                self.calls = 0

            def _make_request(self, url, payload, headers):
                # First response: empty content + finish_reason length
                self.calls += 1
                if self.calls == 1:
                    return {
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": ""},
                                "finish_reason": "length",
                            }
                        ],
                    }
                # Second response: valid content
                return {
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                }

        cfg = LLMConfig(api_key="x", model="openai/gpt-5", max_tokens=8)
        c = _FakeClient(cfg)
        out = c.chat([LLMMessage(role="user", content="hi")])
        self.assertEqual(out.content, "ok")
        self.assertGreaterEqual(c.calls, 2)


class TestDesignAgentLLMActionParsing(unittest.TestCase):
    def test_accepts_actiontype_and_searchpart(self):
        from vibecad.design.design_agent import DesignAgent, DesignActionType
        from vibecad.llm.client import LLMConfig, LLMResponse

        class _FakeClient:
            def __init__(self):
                self.config = LLMConfig(api_key="x", max_tokens=256)

            def chat(self, messages, system_prompt=None):
                content = '{"assistant_message":"Searching…","actions":[{"actiontype":"SEARCHPART","description":"Search for ADS1256","parameters":{"query":"ADS1256"},"requiresApproval":false}]}'
                return LLMResponse(
                    content=content,
                    model="test",
                    raw_response={
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                        ]
                    },
                )

        agent = DesignAgent(llm_client=_FakeClient())
        msg, actions, conf = agent._chat_with_llm("search ADS1256", context={})
        self.assertIn("Searching", msg)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, DesignActionType.SEARCH_PART)
        self.assertEqual(actions[0].parameters.get("query"), "ADS1256")
        self.assertFalse(actions[0].requires_approval)
        self.assertGreater(conf, 0.5)

    def test_salvages_action_from_truncated_json(self):
        from vibecad.design.design_agent import DesignAgent, DesignActionType
        from vibecad.llm.client import LLMConfig, LLMResponse

        class _FakeClient:
            def __init__(self):
                self.config = LLMConfig(api_key="x", max_tokens=256)

            def chat(self, messages, system_prompt=None):
                # Missing closing ] and } to mimic provider truncation.
                content = '{"assistant_message":"ok","actions":[{"actiontype":"SEARCHPART","parameters":{"query":"ADS1256"}}'
                return LLMResponse(
                    content=content,
                    model="test",
                    raw_response={
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "length"}
                        ]
                    },
                )

        agent = DesignAgent(llm_client=_FakeClient())
        msg, actions, conf = agent._chat_with_llm("search ADS1256", context={})
        self.assertTrue(msg)  # any assistant text is fine
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, DesignActionType.SEARCH_PART)
        self.assertEqual(actions[0].parameters.get("query"), "ADS1256")
        self.assertFalse(actions[0].requires_approval)
        self.assertGreater(conf, 0.5)


class TestSettingsPersistence(unittest.TestCase):
    def test_verbose_setting_roundtrip(self):
        from vibecad.config.settings import VibeCADSettings
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "settings.json"
            s = VibeCADSettings(api_key="", api_base="", model="", verbose=True)
            s.save(p)
            loaded = VibeCADSettings.load(p)
            self.assertTrue(loaded.verbose)


class TestLLMExplainer(unittest.TestCase):
    """Tests for LLM explainer."""
    
    def test_offline_explanation(self):
        from vibecad.checks.base import CheckResult, Finding, Severity
        from vibecad.llm.explainer import IssueExplainer, ExplanationRequest
        
        result = CheckResult(
            check_id='TEST_001',
            check_name='Test Check',
            description='A test check',
            passed=False,
            findings=[
                Finding(
                    rule_id='TEST_001',
                    severity=Severity.ERROR,
                    message='Test error finding'
                )
            ]
        )
        
        # Test without LLM (offline mode)
        explainer = IssueExplainer(None)
        request = ExplanationRequest(check_results=[result])
        
        explanation = explainer.explain(request)
        
        self.assertIn('Test Check', explanation.summary)
        self.assertIn('error', explanation.summary.lower())


class TestBOMExporter(unittest.TestCase):
    """Tests for Phase 4 BOM exporter."""
    
    def test_bom_entry_creation(self):
        from vibecad.design.bom_exporter import BOMEntry
        
        entry = BOMEntry(
            references=['R1', 'R2', 'R3'],
            value='10k',
            footprint='0603',
            quantity=3,
            manufacturer='Yageo',
            mpn='RC0603FR-0710KL'
        )
        
        self.assertEqual(entry.quantity, 3)
        self.assertEqual(entry.value, '10k')
        self.assertEqual(entry.manufacturer, 'Yageo')
        self.assertIn('R1', entry.references)
    
    def test_bom_entry_references_str(self):
        from vibecad.design.bom_exporter import BOMEntry
        
        entry = BOMEntry(
            references=['R10', 'R2', 'R1'],
            value='10k',
            footprint='0603',
            quantity=3
        )
        
        # Should sort naturally
        refs_str = entry.references_str()
        self.assertIn('R1', refs_str)
        self.assertIn('R2', refs_str)
        self.assertIn('R10', refs_str)
    
    def test_bom_format_enum(self):
        from vibecad.design.bom_exporter import BOMFormat
        
        self.assertEqual(BOMFormat.CSV_JLCPCB.value, 'csv_jlcpcb')
        self.assertEqual(BOMFormat.CSV_MOUSER.value, 'csv_mouser')
        self.assertEqual(BOMFormat.JSON.value, 'json')
        self.assertEqual(BOMFormat.HTML.value, 'html')
    
    def test_bom_exporter_csv_generic(self):
        from vibecad.design.bom_exporter import BOMExporter, BOMExportRequest, BOMEntry, BOMFormat
        
        entries = [
            BOMEntry(references=['R1'], value='10k', footprint='0603', quantity=1),
            BOMEntry(references=['C1', 'C2'], value='100nF', footprint='0402', quantity=2),
        ]
        
        exporter = BOMExporter()
        request = BOMExportRequest(
            entries=entries,
            format=BOMFormat.CSV_GENERIC,
            project_name='Test Project'
        )
        
        result = exporter.export(request)
        
        self.assertTrue(result.success)
        self.assertEqual(result.total_unique_parts, 2)
        self.assertEqual(result.total_components, 3)
        self.assertIsNotNone(result.preview_data)
        self.assertIn('R1', result.preview_data)
        self.assertIn('100nF', result.preview_data)
    
    def test_bom_exporter_json(self):
        import json
        from vibecad.design.bom_exporter import BOMExporter, BOMExportRequest, BOMEntry, BOMFormat
        
        entries = [
            BOMEntry(references=['U1'], value='STM32F4', footprint='LQFP-64', quantity=1),
        ]
        
        exporter = BOMExporter()
        request = BOMExportRequest(
            entries=entries,
            format=BOMFormat.JSON,
            project_name='MCU Project'
        )
        
        result = exporter.export(request)
        
        self.assertTrue(result.success)
        # Should be valid JSON
        data = json.loads(result.preview_data)
        self.assertEqual(data['project'], 'MCU Project')
        self.assertEqual(len(data['parts']), 1)
        self.assertEqual(data['parts'][0]['value'], 'STM32F4')
    
    def test_bom_preview(self):
        from vibecad.design.bom_exporter import BOMExporter, BOMExportRequest, BOMEntry, BOMFormat
        
        entries = [
            BOMEntry(references=['R1', 'R2'], value='10k', footprint='0603', quantity=2),
            BOMEntry(references=['C1'], value='100nF', footprint='0402', quantity=1),
        ]
        
        exporter = BOMExporter()
        request = BOMExportRequest(
            entries=entries,
            format=BOMFormat.CSV_GENERIC,
            project_name='Preview Test'
        )
        
        preview = exporter.create_preview(request)
        
        self.assertIn('BOM Export Preview', preview)
        self.assertIn('Preview Test', preview)
        self.assertIn('2', preview)  # unique parts
        self.assertIn('3', preview)  # total components


class TestDesignAgent(unittest.TestCase):
    """Tests for Phase 4 design agent."""
    
    def test_action_type_enum(self):
        from vibecad.design.design_agent import DesignActionType
        
        # Verify key action types exist
        self.assertIsNotNone(DesignActionType.SEARCH_PART)
        self.assertIsNotNone(DesignActionType.DRAW_TRACK)
        self.assertIsNotNone(DesignActionType.EXPORT_BOM)
        self.assertIsNotNone(DesignActionType.MOVE_COMPONENT)
        self.assertIsNotNone(DesignActionType.UNKNOWN)
    
    def test_design_action_creation(self):
        from vibecad.design.design_agent import DesignAction, DesignActionType
        
        action = DesignAction(
            action_type=DesignActionType.SEARCH_PART,
            description='Search for USB connector',
            parameters={'query': 'USB-C'}
        )
        
        self.assertEqual(action.action_type, DesignActionType.SEARCH_PART)
        self.assertTrue(action.requires_approval)
        self.assertFalse(action.executed)
    
    def test_design_agent_interpret_search(self):
        from vibecad.design.design_agent import DesignAgent, DesignActionType
        
        agent = DesignAgent(llm_client=None)
        
        request = agent.interpret_request("find a USB-C connector")
        
        self.assertEqual(len(request.interpreted_actions), 1)
        action = request.interpreted_actions[0]
        self.assertEqual(action.action_type, DesignActionType.SEARCH_PART)
        self.assertIn('usb', action.parameters.get('query', '').lower())
    
    def test_design_agent_interpret_bom(self):
        from vibecad.design.design_agent import DesignAgent, DesignActionType
        
        agent = DesignAgent(llm_client=None)
        
        request = agent.interpret_request("export BOM for JLCPCB")
        
        self.assertEqual(len(request.interpreted_actions), 1)
        action = request.interpreted_actions[0]
        self.assertEqual(action.action_type, DesignActionType.EXPORT_BOM)
    
    def test_design_agent_interpret_connect(self):
        from vibecad.design.design_agent import DesignAgent, DesignActionType
        
        agent = DesignAgent(llm_client=None)
        
        request = agent.interpret_request("connect U1 pin 1 to R1 pin 2")
        
        self.assertEqual(len(request.interpreted_actions), 1)
        action = request.interpreted_actions[0]
        self.assertEqual(action.action_type, DesignActionType.DRAW_TRACK)
    
    def test_design_agent_suggestions(self):
        from vibecad.design.design_agent import DesignAgent
        
        agent = DesignAgent(llm_client=None)
        
        suggestions = agent.get_suggestions({})
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)


class TestConnectionManager(unittest.TestCase):
    """Tests for Phase 4 connection manager."""
    
    def test_connection_point_creation(self):
        from vibecad.design.connection_manager import ConnectionPoint
        
        point = ConnectionPoint(
            x=10.0,
            y=20.0,
            layer='F.Cu',
            net_name='VCC',
            component_ref='U1',
            pad_number='1'
        )
        
        self.assertEqual(point.x, 10.0)
        self.assertEqual(point.y, 20.0)
        self.assertEqual(point.net_name, 'VCC')
    
    def test_connection_request_creation(self):
        from vibecad.design.connection_manager import (
            ConnectionRequest, ConnectionPoint, ConnectionType
        )
        
        from_point = ConnectionPoint(x=0, y=0)
        to_point = ConnectionPoint(x=10, y=10)
        
        request = ConnectionRequest(
            from_point=from_point,
            to_point=to_point,
            width_mm=0.25
        )
        
        self.assertEqual(request.connection_type, ConnectionType.PCB_TRACK)
        self.assertEqual(request.width_mm, 0.25)
    
    def test_connection_preview_creation(self):
        from vibecad.design.connection_manager import (
            ConnectionManager, ConnectionRequest, ConnectionPoint
        )
        
        mgr = ConnectionManager()
        
        from_point = ConnectionPoint(x=0, y=0)
        to_point = ConnectionPoint(x=10, y=10)
        
        request = ConnectionRequest(
            from_point=from_point,
            to_point=to_point
        )
        
        preview = mgr.create_preview(request)
        
        self.assertIsNotNone(preview)
        self.assertGreater(preview.total_length_mm, 0)
        self.assertIn('Connection Preview', preview.to_preview_string())


class TestLibraryManagerSearch(unittest.TestCase):
    """Tests for LibraryManager query normalisation and search logic."""

    def test_normalize_query_strips_package_suffix(self):
        from vibecad.design.library_manager import LibraryManager
        variants = LibraryManager._normalize_query("XYZ123-PU")
        self.assertIn("XYZ123-PU", variants)
        self.assertIn("XYZ123", variants)

    def test_normalize_query_no_change_for_clean(self):
        from vibecad.design.library_manager import LibraryManager
        variants = LibraryManager._normalize_query("STM32F103C8T6")
        self.assertEqual(variants[0], "STM32F103C8T6")

    def test_normalize_query_au_suffix(self):
        from vibecad.design.library_manager import LibraryManager
        variants = LibraryManager._normalize_query("STM32F103C8T6-AU")
        self.assertIn("STM32F103C8T6", variants)

    def test_guess_kicad_libraries_pic16(self):
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("PIC16F877A")
        self.assertEqual(libs, ["MCU_Microchip_PIC16"])

    def test_guess_kicad_libraries_stm32(self):
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("STM32F103")
        self.assertEqual(libs, ["MCU_ST_STM32"])

    def test_guess_kicad_libraries_unknown(self):
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("XYZZY99")
        self.assertEqual(libs, [])

    def test_extract_package_from_desc(self):
        from vibecad.design.library_manager import LibraryManager
        pkg = LibraryManager._extract_package_from_desc(
            "20MHz, 32kB Flash, 2kB SRAM, 1kB EEPROM, DIP-28 Keys: AVR"
        )
        self.assertEqual(pkg, "DIP-28")

    def test_extract_package_from_desc_tqfp(self):
        from vibecad.design.library_manager import LibraryManager
        pkg = LibraryManager._extract_package_from_desc("16MHz, TQFP-32")
        self.assertEqual(pkg, "TQFP-32")

    def test_kicad_builtin_source_enum(self):
        from vibecad.design.library_manager import LibrarySource
        self.assertEqual(LibrarySource.KICAD_BUILTIN.value, "kicad_builtin")

    def test_search_parts_sync_with_explicit_source(self):
        """Calling with an unknown prefix returns empty without crashing."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        results = mgr.search_parts_sync("XYZZY_NONEXISTENT_PART", limit=5)
        self.assertEqual(results, [])

    def test_kicad_builtin_preview_summary(self):
        from vibecad.design.library_manager import (
            LibraryManager, LibraryItem, LibrarySource,
        )
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        item = LibraryItem(
            name="MCU_Microchip_PIC16:PIC16F877A",
            manufacturer="(KiCad built-in)",
            mpn="PIC16F877A",
            description="20MHz, 8kB Flash, DIP-40",
            source=LibrarySource.KICAD_BUILTIN,
            package="DIP-40",
        )
        preview = mgr.create_preview_summary(item)
        self.assertIn("KiCad Built-in", preview)
        self.assertIn("PIC16F877A", preview)
        self.assertIn("no download needed", preview)

    def test_kicad_builtin_download_item(self):
        from vibecad.design.library_manager import (
            LibraryManager, LibraryItem, LibrarySource,
        )
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        item = LibraryItem(
            name="MCU_Microchip_PIC16:PIC16F877A",
            manufacturer="(KiCad built-in)",
            mpn="PIC16F877A",
            description="20MHz, 8kB Flash, DIP-40",
            source=LibrarySource.KICAD_BUILTIN,
            package="DIP-40",
        )
        result = mgr.download_item(item, install=True)
        self.assertTrue(result.success)
        self.assertIn("built-in", result.message)
        self.assertIn("No download needed", result.message)


class TestLibraryManagerLocalSearch(unittest.TestCase):
    """Tests for the local KiCad library scanning and token-based search."""

    def test_tokenize_query_basic(self):
        from vibecad.design.library_manager import LibraryManager
        tokens = LibraryManager._tokenize_query("Keystone 590 battery holder")
        self.assertEqual(tokens, ["keystone", "590", "battery", "holder"])

    def test_tokenize_query_dashes_underscores(self):
        from vibecad.design.library_manager import LibraryManager
        tokens = LibraryManager._tokenize_query("BatteryHolder_Keystone-590")
        self.assertEqual(tokens, ["batteryholder", "keystone", "590"])

    def test_tokenize_query_short_tokens_dropped(self):
        from vibecad.design.library_manager import LibraryManager
        tokens = LibraryManager._tokenize_query("a IC b")
        # "a" and "b" are < 2 chars, dropped; "ic" kept
        self.assertEqual(tokens, ["ic"])

    def test_tokenize_query_strips_trailing_punctuation(self):
        from vibecad.design.library_manager import LibraryManager
        tokens = LibraryManager._tokenize_query("ads1256.")
        self.assertEqual(tokens, ["ads1256"])

    def test_score_match_full(self):
        from vibecad.design.library_manager import LibraryManager
        score = LibraryManager._score_match(
            "BatteryHolder_Keystone_590", ["keystone", "590"]
        )
        self.assertAlmostEqual(score, 1.0)

    def test_score_match_partial(self):
        from vibecad.design.library_manager import LibraryManager
        score = LibraryManager._score_match(
            "BatteryHolder_Keystone_3000", ["keystone", "590"]
        )
        self.assertAlmostEqual(score, 0.5)

    def test_score_match_none(self):
        from vibecad.design.library_manager import LibraryManager
        score = LibraryManager._score_match(
            "MCU_ST_STM32F103", ["keystone", "590"]
        )
        self.assertAlmostEqual(score, 0.0)

    def test_guess_kicad_libraries_keyword_battery(self):
        """Keyword 'battery' should map to Battery library."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("battery holder")
        self.assertIn("Battery", libs)

    def test_guess_kicad_libraries_keyword_keystone(self):
        """Manufacturer name 'Keystone' should map to Battery."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("Keystone 590")
        self.assertIn("Battery", libs)

    def test_guess_kicad_libraries_keyword_usb(self):
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("USB Type-C connector")
        self.assertIn("Connector_USB", libs)

    def test_guess_kicad_libraries_keyword_molex(self):
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("Molex connector")
        self.assertIn("Connector_Molex", libs)

    def test_guess_kicad_libraries_prefix_still_works(self):
        """Prefix matching should still take priority."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("PIC16F877A")
        self.assertEqual(libs, ["MCU_Microchip_PIC16"])

    def test_build_local_index_with_fake_tree(self):
        """Build index from a synthetic KiCad-like directory tree."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")

        # Create a temporary KiCad-like tree
        with tempfile.TemporaryDirectory() as tmpdir:
            sym_dir = os.path.join(tmpdir, "symbols")
            fp_dir = os.path.join(tmpdir, "footprints", "Battery.pretty")
            os.makedirs(sym_dir)
            os.makedirs(fp_dir)

            # Minimal .kicad_sym with two symbols
            sym_content = '''(kicad_symbol_lib (version 20220914)
  (symbol "Battery_Cell"
    (property "ki_description" "Single-cell battery")
    (symbol "Battery_Cell_0_1" (polyline (pts (xy 0 0))))
    (symbol "Battery_Cell_1_1" (pin passive line))
  )
  (symbol "BatteryHolder_Keystone_590"
    (property "ki_description" "Keystone 590 coin cell holder")
    (symbol "BatteryHolder_Keystone_590_0_1" (polyline (pts (xy 0 0))))
  )
)'''
            with open(os.path.join(sym_dir, "Battery.kicad_sym"), "w") as f:
                f.write(sym_content)

            # Footprint file (content doesn't matter for indexing, just the filename)
            with open(os.path.join(fp_dir, "BatteryHolder_Keystone_590.kicad_mod"), "w") as f:
                f.write("(footprint)")

            # Patch _detect_kicad_data_dirs to use our temp dir
            mgr._detect_kicad_data_dirs = lambda: [Path(tmpdir)]
            mgr._local_index = None  # force rebuild

            index = mgr._build_local_index()
            names = [entry[1] for entry in index]

            self.assertIn("Battery_Cell", names)
            self.assertIn("BatteryHolder_Keystone_590", names)
            # Sub-symbols should be filtered out
            self.assertNotIn("Battery_Cell_0_1", names)
            self.assertNotIn("Battery_Cell_1_1", names)
            self.assertNotIn("BatteryHolder_Keystone_590_0_1", names)

    def test_search_kicad_local_keystone_590(self):
        """'Keystone 590' should find the battery holder in a local index."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")

        with tempfile.TemporaryDirectory() as tmpdir:
            fp_dir = os.path.join(tmpdir, "footprints", "Battery.pretty")
            os.makedirs(fp_dir)
            with open(os.path.join(fp_dir, "BatteryHolder_Keystone_590.kicad_mod"), "w") as f:
                f.write("(footprint)")
            with open(os.path.join(fp_dir, "BatteryHolder_Keystone_3000.kicad_mod"), "w") as f:
                f.write("(footprint)")

            mgr._detect_kicad_data_dirs = lambda: [Path(tmpdir)]
            mgr._local_index = None

            results = mgr._search_kicad_local("Keystone 590", limit=10)
            self.assertTrue(len(results) >= 1)
            # The exact match should be first
            self.assertIn("590", results[0].mpn)
            self.assertEqual(results[0].source.value, "kicad_builtin")

    def test_search_kicad_local_partial_match(self):
        """A query with only partial token overlap should still return results."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")

        with tempfile.TemporaryDirectory() as tmpdir:
            sym_dir = os.path.join(tmpdir, "symbols")
            os.makedirs(sym_dir)
            sym_content = '(kicad_symbol_lib (symbol "USB_C_Receptacle" ))'
            with open(os.path.join(sym_dir, "Connector_USB.kicad_sym"), "w") as f:
                f.write(sym_content)

            mgr._detect_kicad_data_dirs = lambda: [Path(tmpdir)]
            mgr._local_index = None

            results = mgr._search_kicad_local("USB connector", limit=10)
            self.assertTrue(len(results) >= 1)
            self.assertIn("USB", results[0].mpn)


class TestGitHubCuratedLocalIndex(unittest.TestCase):
    def test_github_curated_dir_index_finds_symbols_and_footprints(self):
        from vibecad.design.library_manager import LibraryManager, LibrarySource

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'repo_root'
            (root / 'kicad' / 'symbols').mkdir(parents=True, exist_ok=True)
            (root / 'kicad' / 'footprints' / 'MyParts.pretty').mkdir(parents=True, exist_ok=True)

            # Minimal symbol lib
            sym_content = '''(kicad_symbol_lib (version 20220914)
  (symbol "My_USB_C"
    (property "ki_description" "USB-C receptacle")
    (symbol "My_USB_C_0_1" (polyline (pts (xy 0 0))))
  )
)'''
            (root / 'kicad' / 'symbols' / 'MySymbols.kicad_sym').write_text(sym_content, encoding='utf-8')

            # Minimal footprint file
            (root / 'kicad' / 'footprints' / 'MyParts.pretty' / 'USB_C_Receptacle.kicad_mod').write_text('(footprint)', encoding='utf-8')

            mgr = LibraryManager(
                kicad_user_lib_path='/tmp',
                enable_github_sources=True,
                github_curated_dirs=[str(root)],
                github_curated_repos=[],
            )

            fp_results = mgr.search_parts_sync('USB_C_Receptacle', source=LibrarySource.GITHUB_CURATED, limit=10)
            self.assertTrue(len(fp_results) >= 1)
            self.assertEqual(fp_results[0].source.value, 'github_curated')
            self.assertTrue(bool(fp_results[0].local_footprint_path) or bool(fp_results[0].local_symbol_path))


class TestKiCadLibraryTables(unittest.TestCase):
    def test_ensure_project_tables_creates_entries(self):
        from vibecad.design.kicad_library_tables import ensure_project_tables

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            fp_dir = project_dir / 'userlib' / 'footprints' / 'VibeCAD.pretty'
            fp_dir.mkdir(parents=True, exist_ok=True)
            (fp_dir / 'TestFootprint.kicad_mod').write_text('(footprint)', encoding='utf-8')

            sym_path = project_dir / 'userlib' / 'symbols' / 'VibeCAD_TestSym.kicad_sym'
            sym_path.parent.mkdir(parents=True, exist_ok=True)
            sym_path.write_text('(kicad_symbol_lib (version 20220914) (symbol "TestSym"))', encoding='utf-8')

            ensure_project_tables(
                project_dir=str(project_dir),
                footprint_lib_dir=str(fp_dir),
                symbol_lib_paths=[str(sym_path)],
            )

            fp_table = (project_dir / 'fp-lib-table').read_text(encoding='utf-8')
            sym_table = (project_dir / 'sym-lib-table').read_text(encoding='utf-8')

            self.assertIn('fp_lib_table', fp_table)
            self.assertIn('"VibeCAD"', fp_table)
            self.assertIn(str(fp_dir), fp_table)

            self.assertIn('sym_lib_table', sym_table)
            self.assertIn('"VibeCAD_TestSym"', sym_table)
            self.assertIn(str(sym_path), sym_table)


class TestEasyEDASearch(unittest.TestCase):
    """Tests for EasyEDA/LCSC integration."""

    def test_easyeda_source_enum(self):
        from vibecad.design.library_manager import LibrarySource
        self.assertEqual(LibrarySource.EASYEDA.value, "easyeda")

    def test_map_lcsc_package_to_local_footprint_ssop28(self):
        """SSOP-28-208mil should resolve to a local Package_SO footprint."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")

        with tempfile.TemporaryDirectory() as tmpdir:
            fp_dir = os.path.join(tmpdir, "footprints", "Package_SO.pretty")
            os.makedirs(fp_dir)
            # Create a matching SSOP-28 footprint file
            fp_path = os.path.join(fp_dir, "SSOP-28_5.3x10.2mm_P0.65mm.kicad_mod")
            with open(fp_path, "w") as f:
                f.write("(footprint)")

            mgr._detect_kicad_data_dirs = lambda: [Path(tmpdir)]
            mgr._local_index = None

            result = mgr._map_lcsc_package_to_local_footprint("SSOP-28-208mil")
            self.assertIsNotNone(result)
            self.assertIn("SSOP-28", result)

    def test_map_lcsc_package_to_local_footprint_tqfp(self):
        """TQFP-32 should resolve to a local Package_QFP footprint."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")

        with tempfile.TemporaryDirectory() as tmpdir:
            fp_dir = os.path.join(tmpdir, "footprints", "Package_QFP.pretty")
            os.makedirs(fp_dir)
            fp_path = os.path.join(fp_dir, "TQFP-32_7x7mm_P0.8mm.kicad_mod")
            with open(fp_path, "w") as f:
                f.write("(footprint)")

            mgr._detect_kicad_data_dirs = lambda: [Path(tmpdir)]
            mgr._local_index = None

            result = mgr._map_lcsc_package_to_local_footprint("TQFP-32")
            self.assertIsNotNone(result)
            self.assertIn("TQFP-32", result)

    def test_map_lcsc_package_unknown_returns_none(self):
        """Unknown package should return None."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        mgr._local_index = []
        result = mgr._map_lcsc_package_to_local_footprint("FOOBAR-99")
        self.assertIsNone(result)

    def test_guess_kicad_libraries_ads_prefix(self):
        """ADS1256 should map to Analog_ADC via prefix matching."""
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("ADS1256")
        self.assertIn("Analog_ADC", libs)


if __name__ == '__main__':
    unittest.main()
