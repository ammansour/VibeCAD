"""
End-to-end tests for the library search → footprint resolution → placement pipeline.

Each test targets one step so failures are instantly diagnosable.
All tests are offline — external HTTP calls are mocked.
"""

import os
import re
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helper: build a minimal on-disk KiCad library tree with footprints we
# expect the search pipeline to discover for common packages.
# ---------------------------------------------------------------------------

def _make_kicad_tree(root: str) -> Path:
    """Populate *root* with a fake KiCad data directory."""
    base = Path(root)

    # ----- footprints -----
    so_dir = base / "footprints" / "Package_SO.pretty"
    so_dir.mkdir(parents=True)
    for name in [
        "HTSSOP-28-1EP_4.4x9.7mm_P0.65mm_EP2.75x6.2mm",
        "TSSOP-28_4.4x9.7mm_P0.65mm",
        "SSOP-28_5.3x10.2mm_P0.65mm",
        "SOIC-8_3.9x4.9mm_P1.27mm",
    ]:
        (so_dir / f"{name}.kicad_mod").write_text("(footprint)")

    dip_dir = base / "footprints" / "Package_DIP.pretty"
    dip_dir.mkdir(parents=True)
    (dip_dir / "DIP-28_W7.62mm.kicad_mod").write_text("(footprint)")

    qfp_dir = base / "footprints" / "Package_QFP.pretty"
    qfp_dir.mkdir(parents=True)
    (qfp_dir / "TQFP-32_7x7mm_P0.8mm.kicad_mod").write_text("(footprint)")

    # ----- symbols -----
    sym_dir = base / "symbols"
    sym_dir.mkdir(parents=True)
    (sym_dir / "Analog_ADC.kicad_sym").write_text(textwrap.dedent("""\
        (kicad_symbol_lib (version 20211014) (generator kicad_symbol_editor)
          (symbol "ADS1256" (pin_names (offset 0.254)) (in_bom yes) (on_board yes)
            (property "Reference" "U" (id 0))
            (property "Value" "ADS1256" (id 1))
            (property "Footprint" "Package_SO:HTSSOP-28" (id 2))
            (property "ki_description" "Very Low Noise, 24-Bit ADC, TSSOP-28" (id 4))
          )
        )
    """))
    return base


def _make_lib_manager(tree_root: Path):
    """Create a LibraryManager that uses only the fake tree."""
    from vibecad.design.library_manager import LibraryManager
    mgr = LibraryManager(kicad_user_lib_path="/tmp")
    mgr._detect_kicad_data_dirs = lambda: [tree_root]
    mgr._local_index = None        # force rebuild
    mgr._search_cache = {}         # clear cache
    return mgr


# ===================================================================
# Step 1 — Tokenizer & scoring basics
# ===================================================================

class TestTokenizerAndScoring(unittest.TestCase):

    def test_tokenize_ads1256(self):
        from vibecad.design.library_manager import LibraryManager
        tokens = LibraryManager._tokenize_query("ADS1256")
        self.assertEqual(tokens, ["ads1256"])

    def test_tokenize_htssop_28(self):
        from vibecad.design.library_manager import LibraryManager
        tokens = LibraryManager._tokenize_query("HTSSOP-28")
        self.assertEqual(tokens, ["htssop", "28"])

    def test_score_match_htssop_in_footprint_name(self):
        from vibecad.design.library_manager import LibraryManager
        name = "htssop-28-1ep_4.4x9.7mm_p0.65mm_ep2.75x6.2mm"
        score = LibraryManager._score_match(name, ["htssop", "28"])
        self.assertEqual(score, 1.0, "Both tokens should match")

    def test_score_match_ads1256_no_match_in_footprint(self):
        from vibecad.design.library_manager import LibraryManager
        name = "htssop-28-1ep_4.4x9.7mm_p0.65mm_ep2.75x6.2mm"
        score = LibraryManager._score_match(name, ["ads1256"])
        self.assertEqual(score, 0.0, "ADS1256 should NOT match a footprint name")


# ===================================================================
# Step 2 — Package extraction from descriptions
# ===================================================================

class TestPackageExtraction(unittest.TestCase):

    def test_extract_package_hint_tssop_28(self):
        from vibecad.design.library_manager import LibraryManager
        self.assertEqual(LibraryManager._extract_package_hint("TSSOP-28"), "TSSOP-28")

    def test_extract_package_hint_htssop_28(self):
        from vibecad.design.library_manager import LibraryManager
        self.assertEqual(LibraryManager._extract_package_hint("HTSSOP-28"), "HTSSOP-28")

    def test_extract_package_hint_from_description(self):
        from vibecad.design.library_manager import LibraryManager
        desc = "Very Low Noise, 24-Bit ADC, TSSOP-28"
        self.assertEqual(LibraryManager._extract_package_hint(desc), "TSSOP-28")

    def test_extract_package_from_desc_tssop(self):
        from vibecad.design.library_manager import LibraryManager
        desc = "Very Low Noise, 24-Bit ADC, TSSOP-28"
        self.assertEqual(LibraryManager._extract_package_from_desc(desc), "TSSOP-28")

    def test_extract_package_from_desc_htssop(self):
        from vibecad.design.library_manager import LibraryManager
        desc = "ADC IC, HTSSOP-28 package"
        self.assertEqual(LibraryManager._extract_package_from_desc(desc), "HTSSOP-28")

    def test_package_family_tssop(self):
        from vibecad.design.library_manager import LibraryManager
        self.assertEqual(LibraryManager._package_family("TSSOP-28"), "TSSOP")

    def test_package_family_htssop(self):
        from vibecad.design.library_manager import LibraryManager
        self.assertEqual(LibraryManager._package_family("HTSSOP-28"), "HTSSOP")

    def test_preferred_libs_tssop_maps_to_package_so(self):
        from vibecad.design.library_manager import LibraryManager
        libs = LibraryManager._preferred_footprint_libs_for_package("TSSOP-28")
        self.assertIn("Package_SO", libs)

    def test_preferred_libs_htssop_maps_to_package_so(self):
        from vibecad.design.library_manager import LibraryManager
        libs = LibraryManager._preferred_footprint_libs_for_package("HTSSOP-28")
        self.assertIn("Package_SO", libs)


# ===================================================================
# Step 3 — Local KiCad library search
# ===================================================================

class TestLocalKiCadSearch(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tree = _make_kicad_tree(self._tmpdir)
        self._mgr = _make_lib_manager(self._tree)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_local_search_ads1256_finds_symbol(self):
        """Searching 'ADS1256' locally should at least find the symbol."""
        results = self._mgr._search_kicad_local("ADS1256", 20)
        sym_results = [r for r in results if r.local_symbol_path]
        self.assertGreater(len(sym_results), 0,
                           "Should find ADS1256 symbol in local index")

    def test_local_search_htssop_finds_footprint(self):
        """Searching 'HTSSOP-28' locally should find the HTSSOP footprint."""
        results = self._mgr._search_kicad_local("HTSSOP-28", 20)
        fp_results = [r for r in results if r.local_footprint_path]
        self.assertGreater(len(fp_results), 0,
                           "Should find HTSSOP-28 footprint in local index")

    def test_local_search_tssop_finds_footprint(self):
        """Searching 'TSSOP-28' locally should find TSSOP/HTSSOP footprints."""
        results = self._mgr._search_kicad_local("TSSOP-28", 20)
        fp_results = [r for r in results if r.local_footprint_path]
        self.assertGreater(len(fp_results), 0,
                           "Should find TSSOP-28 footprint in local index")

    def test_local_search_ads1256_has_no_footprint(self):
        """'ADS1256' alone should NOT match any footprint (it's a part number)."""
        results = self._mgr._search_kicad_local("ADS1256", 20)
        fp_results = [r for r in results if r.local_footprint_path]
        self.assertEqual(len(fp_results), 0,
                         "ADS1256 should not directly match any footprint name")


# ===================================================================
# Step 4 — resolve_best_footprint_path with and without package hint
# ===================================================================

class TestResolveFootprintPath(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tree = _make_kicad_tree(self._tmpdir)
        self._mgr = _make_lib_manager(self._tree)
        # Disable online sources
        self._mgr._search_kicad_builtin_sync = lambda q, l: []
        self._mgr._search_easyeda_sync = lambda q, l: []
        self._mgr._search_snapeda_sync = lambda q, l: []

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_resolve_with_explicit_package_hint(self):
        """With package_hint='TSSOP-28', should find a Package_SO footprint."""
        result = self._mgr.resolve_best_footprint_path("ADS1256", package_hint="TSSOP-28")
        self.assertIsNotNone(result, "Should resolve with explicit TSSOP-28 hint")
        mpn, fp_path = result
        self.assertIn("TSSOP", fp_path.upper(),
                       "Resolved footprint should be a TSSOP variant")

    def test_resolve_with_htssop_hint(self):
        """With package_hint='HTSSOP-28', should find the HTSSOP footprint."""
        result = self._mgr.resolve_best_footprint_path("ADS1256", package_hint="HTSSOP-28")
        self.assertIsNotNone(result, "Should resolve with explicit HTSSOP-28 hint")
        _, fp_path = result
        self.assertIn("HTSSOP-28", fp_path)

    def test_resolve_without_hint_discovers_package_from_symbol(self):
        """Without a package hint, resolve should discover TSSOP from symbol description."""
        result = self._mgr.resolve_best_footprint_path("ADS1256")
        self.assertIsNotNone(result,
                             "Should auto-discover package from symbol description and resolve")
        _, fp_path = result
        # Should find HTSSOP or TSSOP in Package_SO
        self.assertTrue("TSSOP" in fp_path.upper() or "HTSSOP" in fp_path.upper(),
                        f"Expected TSSOP/HTSSOP footprint, got: {fp_path}")


# ===================================================================
# Step 5 — Full search_parts_sync pipeline (all online mocked)
# ===================================================================

class TestSearchPartsPipelineOffline(unittest.TestCase):
    """Test that search_parts_sync returns useful results even offline."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tree = _make_kicad_tree(self._tmpdir)
        self._mgr = _make_lib_manager(self._tree)
        # Disable all online sources
        self._mgr._search_kicad_builtin_sync = lambda q, l: []
        self._mgr._search_easyeda_sync = lambda q, l: []
        self._mgr._search_snapeda_sync = lambda q, l: []

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_search_ads1256_returns_results(self):
        """search_parts_sync('ADS1256') must not return empty."""
        results = self._mgr.search_parts_sync("ADS1256")
        self.assertGreater(len(results), 0,
                           "ADS1256 must match at least the local symbol")

    def test_search_htssop_returns_footprint(self):
        results = self._mgr.search_parts_sync("HTSSOP-28")
        fp_results = [r for r in results if r.local_footprint_path]
        self.assertGreater(len(fp_results), 0)

    def test_search_tssop_returns_footprint(self):
        results = self._mgr.search_parts_sync("TSSOP-28")
        fp_results = [r for r in results if r.local_footprint_path]
        self.assertGreater(len(fp_results), 0)


# ===================================================================
# Step 6 — _preflight_actions should not drop DOWNLOAD_FOOTPRINT
#           when package hint is available
# ===================================================================

class TestPreflightActions(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tree = _make_kicad_tree(self._tmpdir)
        self._mgr = _make_lib_manager(self._tree)
        self._mgr._search_kicad_builtin_sync = lambda q, l: []
        self._mgr._search_easyeda_sync = lambda q, l: []
        self._mgr._search_snapeda_sync = lambda q, l: []

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_preflight_keeps_download_with_package(self):
        from vibecad.design.design_agent import DesignAgent, DesignAction, DesignActionType
        agent = DesignAgent()
        agent._library_manager = self._mgr

        action = DesignAction(
            action_type=DesignActionType.DOWNLOAD_FOOTPRINT,
            description="Download TSSOP-28 footprint for ADS1256",
            parameters={"part_name": "ADS1256", "package": "TSSOP-28"},
        )
        msg, actions = agent._preflight_actions("OK", [action], {})
        self.assertGreater(len(actions), 0,
                           "Preflight should keep the action when package hint resolves")

    def test_preflight_keeps_download_without_package_if_discoverable(self):
        from vibecad.design.design_agent import DesignAgent, DesignAction, DesignActionType
        agent = DesignAgent()
        agent._library_manager = self._mgr

        action = DesignAction(
            action_type=DesignActionType.DOWNLOAD_FOOTPRINT,
            description="Download footprint for ADS1256",
            parameters={"part_name": "ADS1256"},
        )
        msg, actions = agent._preflight_actions("OK", [action], {})
        # Even without an explicit package param, preflight should either keep it
        # (because resolve discovers the package) or return a helpful message.
        # It should NOT silently drop with a cryptic "no results" error.
        if not actions:
            # Acceptable: it asked for clarification about package
            self.assertIn("package", msg.lower(),
                          "If dropped, should ask for package info")


# ===================================================================
# Step 7 — _handle_download_symbol_or_footprint end-to-end
# ===================================================================

class TestHandleDownloadFootprint(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tree = _make_kicad_tree(self._tmpdir)
        self._mgr = _make_lib_manager(self._tree)
        self._mgr._search_kicad_builtin_sync = lambda q, l: []
        self._mgr._search_easyeda_sync = lambda q, l: []
        self._mgr._search_snapeda_sync = lambda q, l: []

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_download_footprint_with_package_succeeds(self):
        """DOWNLOAD_FOOTPRINT with package='TSSOP-28' should succeed."""
        import asyncio
        from vibecad.design.design_agent import DesignAgent, DesignAction, DesignActionType

        agent = DesignAgent()
        agent._library_manager = self._mgr

        action = DesignAction(
            action_type=DesignActionType.DOWNLOAD_FOOTPRINT,
            description="Download ADS1256 TSSOP-28 footprint",
            parameters={"part_name": "ADS1256", "package": "TSSOP-28"},
        )
        success, message = asyncio.get_event_loop().run_until_complete(
            agent._handle_download_symbol_or_footprint(action, {})
        )
        self.assertTrue(success, f"Download should succeed, got: {message}")

    def test_download_footprint_without_package_gives_useful_response(self):
        """DOWNLOAD_FOOTPRINT without package should still try to resolve or give helpful error."""
        import asyncio
        from vibecad.design.design_agent import DesignAgent, DesignAction, DesignActionType

        agent = DesignAgent()
        agent._library_manager = self._mgr

        action = DesignAction(
            action_type=DesignActionType.DOWNLOAD_FOOTPRINT,
            description="Download ADS1256 footprint",
            parameters={"part_name": "ADS1256"},
        )
        success, message = asyncio.get_event_loop().run_until_complete(
            agent._handle_download_symbol_or_footprint(action, {})
        )
        # Either it auto-discovers the package and succeeds, or it reports
        # a useful "found symbol, need package" message (not just "No results").
        if not success:
            self.assertNotIn("No results found", message,
                             "Should not say 'no results' when symbol exists locally")


# ===================================================================
# Step 8 — _map_lcsc_package_to_local_footprint for HTSSOP/TSSOP
# ===================================================================

class TestLCSCPackageMapping(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tree = _make_kicad_tree(self._tmpdir)
        self._mgr = _make_lib_manager(self._tree)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_lcsc_htssop_28_maps_locally(self):
        result = self._mgr._map_lcsc_package_to_local_footprint("HTSSOP-28")
        self.assertIsNotNone(result, "HTSSOP-28 should map to a local footprint")
        self.assertIn("HTSSOP-28", result)

    def test_lcsc_tssop_28_maps_locally(self):
        result = self._mgr._map_lcsc_package_to_local_footprint("TSSOP-28")
        self.assertIsNotNone(result, "TSSOP-28 should map to a local footprint")
        self.assertIn("TSSOP", result)

    def test_lcsc_ssop_28_maps_locally(self):
        result = self._mgr._map_lcsc_package_to_local_footprint("SSOP-28-208mil")
        self.assertIsNotNone(result, "SSOP-28-208mil should map to a local footprint")


# ===================================================================
# Step 9 — Prefix mapping for ADS parts
# ===================================================================

class TestPrefixMapping(unittest.TestCase):

    def test_ads_prefix_maps_to_analog_adc(self):
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("ADS1256")
        self.assertIn("Analog_ADC", libs)

    def test_ads_prefix_maps_to_analog_adc_lowercase(self):
        from vibecad.design.library_manager import LibraryManager
        mgr = LibraryManager(kicad_user_lib_path="/tmp")
        libs = mgr._guess_kicad_libraries("ads1256")
        self.assertIn("Analog_ADC", libs)



class TestOnlineSearchLogic(unittest.TestCase):
    def test_search_easyeda_parsing(self):
        """Test that _search_easyeda_sync correctly parses LCSC API response."""
        from vibecad.design.library_manager import LibraryManager, LibrarySource
        from unittest.mock import MagicMock, patch
        import json

        mgr = LibraryManager()
        
        # Mock successful EasyEDA response
        mock_resp_data = {
            "result": {
                "productList": [
                    {
                        "mpn": "ADS1256IDB",
                        "number": "C12345",
                        "package": "SSOP-28",
                        "manufacturer": "Texas Instruments",
                        "hasDevice": True
                    }
                ]
            }
        }
        
        with patch('vibecad.design.library_manager.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_resp_data).encode('utf-8')
            # Context manager support
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            # We must clear the cache to ensure the mock is called
            mgr._search_cache = {}
            results = mgr._search_easyeda_sync("ADS1256", limit=10)
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].mpn, "ADS1256IDB")
            self.assertEqual(results[0].source, LibrarySource.EASYEDA)
            self.assertEqual(results[0].category, "SSOP-28")
            # Verify SSL context was passed
            call_args = mock_urlopen.call_args
            self.assertIsNotNone(call_args)
            # The exact kwarg usage depends on impl but we expect 'context'
            kwargs = call_args[1]
            self.assertIn('context', kwargs)


if __name__ == "__main__":
    unittest.main()
