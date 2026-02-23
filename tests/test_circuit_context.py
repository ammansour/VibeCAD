"""
Tests for CircuitContextBuilder and ComponentWebSearch.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# ── Minimal stubs for PCB data structures ──────────────────────

@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0

@dataclass
class Pad:
    number: str = "1"
    pad_type: str = "smd"
    shape: str = "rect"
    at: Point = field(default_factory=Point)
    size: tuple = (1.0, 1.0)
    layers: list = field(default_factory=list)
    net: str = ""
    net_name: str = ""

@dataclass
class Footprint:
    reference: str = ""
    value: str = ""
    library: str = ""
    footprint_name: str = ""
    at: Point = field(default_factory=Point)
    rotation: float = 0.0
    layer: str = "F.Cu"
    pads: list = field(default_factory=list)

@dataclass
class Net:
    number: int = 0
    name: str = ""

@dataclass
class Track:
    start: Point = field(default_factory=Point)
    end: Point = field(default_factory=Point)
    width: float = 0.25
    layer: str = "F.Cu"
    net: int = 0

@dataclass
class Via:
    at: Point = field(default_factory=Point)
    size: float = 0.8
    drill: float = 0.4
    layers: list = field(default_factory=list)
    net: int = 0

@dataclass
class PCBData:
    version: int = 20230101
    generator: str = "test"
    general: dict = field(default_factory=dict)
    layers: dict = field(default_factory=dict)
    setup: dict = field(default_factory=dict)
    nets: list = field(default_factory=list)
    footprints: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    vias: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    board_outline_lines: list = field(default_factory=list)
    board_outline_arcs: list = field(default_factory=list)
    board_outline_circles: list = field(default_factory=list)
    board_outline_rects: list = field(default_factory=list)
    board_outline_polygons: list = field(default_factory=list)

    @property
    def has_board_outline(self):
        return bool(self.board_outline_lines or self.board_outline_arcs or
                     self.board_outline_circles or self.board_outline_rects or
                     self.board_outline_polygons)


# ── Import the modules under test ──────────────────────────────

from vibecad.design.circuit_context import CircuitContextBuilder, CircuitSnapshot
from vibecad.design.component_search import ComponentWebSearch, ComponentInfo, ComponentPrice


# ══════════════════════════════════════════════════════════════
# CircuitContextBuilder Tests
# ══════════════════════════════════════════════════════════════

class TestCircuitContextBuilder:
    """Tests for the smart circuit context builder."""

    def _make_board(self) -> PCBData:
        """Create a test board with R1, R2, U1, C1 and some nets."""
        return PCBData(
            nets=[
                Net(0, ""),
                Net(1, "GND"),
                Net(2, "VCC"),
                Net(3, "SDA"),
                Net(4, "SCL"),
            ],
            footprints=[
                Footprint(
                    reference="R1", value="10k",
                    library="Resistor_SMD", footprint_name="R_0603_1608Metric",
                    at=Point(100, 50), layer="F.Cu",
                    pads=[
                        Pad(number="1", net_name="VCC"),
                        Pad(number="2", net_name="SDA"),
                    ],
                ),
                Footprint(
                    reference="R2", value="4.7k",
                    library="Resistor_SMD", footprint_name="R_0603_1608Metric",
                    at=Point(105, 50), layer="F.Cu",
                    pads=[
                        Pad(number="1", net_name="VCC"),
                        Pad(number="2", net_name="SCL"),
                    ],
                ),
                Footprint(
                    reference="U1", value="STM32F103C8T6",
                    library="MCU_ST_STM32", footprint_name="LQFP-48",
                    at=Point(110, 60), layer="F.Cu",
                    pads=[
                        Pad(number="1", net_name="VCC"),
                        Pad(number="2", net_name="GND"),
                        Pad(number="15", net_name="SDA"),
                        Pad(number="16", net_name="SCL"),
                    ],
                ),
                Footprint(
                    reference="C1", value="100nF",
                    library="Capacitor_SMD", footprint_name="C_0402_1005Metric",
                    at=Point(108, 55), layer="F.Cu",
                    pads=[
                        Pad(number="1", net_name="VCC"),
                        Pad(number="2", net_name="GND"),
                    ],
                ),
            ],
            tracks=[
                Track(net=1),
                Track(net=2),
                Track(net=2),
                Track(net=3),
            ],
            layers={0: ("F.Cu", "signal"), 1: ("B.Cu", "signal")},
        )

    def test_build_snapshot_basic(self):
        """Snapshot should index all components and nets."""
        builder = CircuitContextBuilder()
        board = self._make_board()
        snap = builder.build_snapshot(board)

        assert "R1" in snap.components
        assert "R2" in snap.components
        assert "U1" in snap.components
        assert "C1" in snap.components
        assert snap.components["R1"].value == "10k"
        assert snap.components["U1"].value == "STM32F103C8T6"
        assert snap.track_count == 4
        assert len(snap.nets) >= 4

    def test_component_pad_nets(self):
        """Each component should know what nets its pads connect to."""
        builder = CircuitContextBuilder()
        snap = builder.build_snapshot(self._make_board())

        r1 = snap.components["R1"]
        assert r1.pad_nets.get("1") == "VCC"
        assert r1.pad_nets.get("2") == "SDA"

    def test_extract_refs_from_query(self):
        """Should extract R1, U1, etc. from natural language."""
        builder = CircuitContextBuilder()
        snap = builder.build_snapshot(self._make_board())

        refs = builder._extract_refs("what value should R1 be?", snap)
        assert "R1" in refs

        refs2 = builder._extract_refs("connect U1 to C1", snap)
        assert "U1" in refs2
        assert "C1" in refs2

    def test_extract_refs_by_type(self):
        """Should match 'resistor' to R* components."""
        builder = CircuitContextBuilder()
        snap = builder.build_snapshot(self._make_board())

        refs = builder._extract_refs("what value should the resistor be?", snap)
        assert "R1" in refs
        assert "R2" in refs

    def test_neighbourhood_expansion(self):
        """Asking about R1 should also include U1 (connected via SDA)."""
        builder = CircuitContextBuilder()
        snap = builder.build_snapshot(self._make_board())

        mentioned = {"R1"}
        expanded = builder._expand_neighbourhood(mentioned, set(), snap)
        assert "R1" in expanded
        assert "U1" in expanded
        assert "R2" in expanded
        assert "C1" in expanded

    def test_build_context_for_query(self):
        """Full context string for a query about R1."""
        builder = CircuitContextBuilder()
        snap = builder.build_snapshot(self._make_board())

        ctx = builder.build_context_for_query("what value should R1 be?", snap)

        assert "R1" in ctx
        assert "10k" in ctx
        assert "Board Overview" in ctx
        assert "Component Table" in ctx
        assert "Focused Context" in ctx
        assert "U1" in ctx

    def test_empty_board(self):
        """Should handle an empty board gracefully."""
        builder = CircuitContextBuilder()
        snap = builder.build_snapshot(PCBData())

        ctx = builder.build_context_for_query("anything", snap)
        assert "Components: 0" in ctx

    def test_estimate_tokens(self):
        """Token estimation should be roughly 1 token per 4 chars."""
        builder = CircuitContextBuilder()
        assert builder.estimate_tokens("a" * 400) == 100


# ══════════════════════════════════════════════════════════════
# ComponentWebSearch Tests
# ══════════════════════════════════════════════════════════════

class TestComponentWebSearch:
    """Tests for the component web search module."""

    def test_init(self):
        """Should initialize without errors."""
        searcher = ComponentWebSearch()
        assert searcher is not None

    def test_component_info_to_text(self):
        """ComponentInfo.to_text() should produce readable output."""
        info = ComponentInfo(
            mpn="STM32F103C8T6",
            manufacturer="STMicroelectronics",
            description="ARM Cortex-M3 MCU, 72MHz, 64KB Flash",
            package="LQFP-48",
            datasheet_url="https://example.com/ds.pdf",
            source="lcsc",
            stock=50000,
            prices=[
                ComponentPrice(1, 2.50),
                ComponentPrice(10, 2.10),
            ],
            parameters={"Core": "ARM Cortex-M3", "Flash": "64KB"},
        )
        text = info.to_text()
        assert "STM32F103C8T6" in text
        assert "STMicroelectronics" in text
        assert "LQFP-48" in text
        assert "Cortex-M3" in text
        assert "64KB" in text
        assert "ds.pdf" in text
        assert "50,000" in text

    def test_component_info_to_dict(self):
        """ComponentInfo.to_dict() should produce a clean dict."""
        info = ComponentInfo(
            mpn="LM7805",
            manufacturer="TI",
            source="mouser",
            prices=[ComponentPrice(1, 0.50)],
        )
        d = info.to_dict()
        assert d["mpn"] == "LM7805"
        assert d["manufacturer"] == "TI"
        assert len(d["prices"]) == 1

    @patch('vibecad.design.component_search.urlopen')
    def test_search_lcsc_success(self, mock_urlopen):
        """LCSC search should parse the API response correctly."""
        import json
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps({
            "result": {
                "productList": [
                    {
                        "mpn": "STM32F103C8T6",
                        "manufacturer": "STMicroelectronics",
                        "package": "LQFP-48",
                        "number": "C8734",
                        "description": "ARM MCU",
                        "stock": "50000",
                    }
                ]
            }
        }).encode("utf-8")
        mock_urlopen.return_value = mock_response

        searcher = ComponentWebSearch()
        results = searcher._search_lcsc("STM32F103", 5)
        assert len(results) == 1
        assert results[0].mpn == "STM32F103C8T6"
        assert results[0].manufacturer == "STMicroelectronics"

    @patch('vibecad.design.component_search.urlopen')
    def test_search_lcsc_list_response_is_safe(self, mock_urlopen):
        """Unexpected list-shaped JSON should not crash the parser."""
        import json
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps([]).encode("utf-8")
        mock_urlopen.return_value = mock_response

        searcher = ComponentWebSearch()
        results = searcher._search_lcsc("anything", 5)
        assert results == []

    def test_search_empty_query(self):
        """Empty query should return empty results."""
        searcher = ComponentWebSearch()
        assert searcher.search("") == []
        assert searcher.search(" ") == []

    def test_search_for_llm_context_no_results(self):
        """Should return a user-friendly message when no results."""
        searcher = ComponentWebSearch()
        with patch.object(searcher, '_search_lcsc', return_value=[]):
            result = searcher.search_for_llm_context("nonexistent_part_xyz123")
            assert "No component data found" in result

    def test_enrich_component(self):
        """enrich_component should build a smart query from ref+value+footprint."""
        searcher = ComponentWebSearch()
        mock_info = ComponentInfo(mpn="RC0603FR-0710KL", manufacturer="Yageo", source="lcsc")
        with patch.object(searcher, 'get_component_details', return_value=mock_info) as mock_get:
            result = searcher.enrich_component("R1", "10k", "Resistor_SMD:R_0603_1608Metric")
            assert result is not None
            assert result.mpn == "RC0603FR-0710KL"
            call_args = mock_get.call_args[0][0]
            assert "10k" in call_args
            assert "0603" in call_args


class TestContextIntegration:
    """Tests that verify the context builder + search work together."""

    def test_context_with_web_enrichment(self):
        """Simulate the Q&A flow: build context + web search for mentioned parts."""
        builder = CircuitContextBuilder()
        board = PCBData(
            nets=[Net(1, "GND"), Net(2, "VCC")],
            footprints=[
                Footprint(
                    reference="R1", value="10k",
                    library="Resistor_SMD", footprint_name="R_0603_1608Metric",
                    at=Point(100, 50), layer="F.Cu",
                    pads=[
                        Pad(number="1", net_name="VCC"),
                        Pad(number="2", net_name="GND"),
                    ],
                ),
            ],
            layers={0: ("F.Cu", "signal")},
        )
        snap = builder.build_snapshot(board)

        ctx = builder.build_context_for_query("what value should R1 be?", snap)
        assert "10k" in ctx
        assert "VCC" in ctx
        assert "GND" in ctx

        searcher = ComponentWebSearch()
        web_info = ComponentInfo(
            mpn="RC0603FR-0710KL",
            manufacturer="Yageo",
            description="10k 1% 0603 resistor",
            package="0603",
            source="lcsc",
            prices=[ComponentPrice(1, 0.001)],
        )
        with patch.object(searcher, 'search', return_value=[web_info]):
            web_ctx = searcher.search_for_llm_context("10k 0603", limit=2)
            assert "Yageo" in web_ctx
            assert "0603" in web_ctx
