"""
Library Manager for downloading and installing symbols/footprints.

Supports fetching from:
- SnapEDA (primary)
- Ultra Librarian
- Component Search Engine
- Local library paths

All downloads require explicit user approval before installation.
"""

import json
import logging
import os
import platform
import re
import ssl
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple, Iterable
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


class LibrarySource(Enum):
    """Supported library sources."""
    KICAD_BUILTIN = "kicad_builtin"  # Ships with KiCad, no download needed
    SNAPEDA = "snapeda"
    ULTRA_LIBRARIAN = "ultra_librarian"
    COMPONENT_SEARCH = "component_search"
    LOCAL = "local"
    GITHUB_CURATED = "github_curated"  # Keyless ZIP snapshot downloads of known repos
    GITHUB_SEARCH = "github_search"    # Keyless (rate-limited) GitHub search API
    EASYEDA = "easyeda"              # Free, keyless LCSC/EasyEDA Pro search API


#
# Deterministic keyword/prefix library guessing has been disabled.
# It can hide correct matches when the guess is wrong. We prefer a global
# local search, and let the LLM drive specificity via SEARCH_PART and
# explicit "Lib:Footprint" identifiers.
#
KICAD_LIB_PREFIXES: Dict[str, List[str]] = {}
KICAD_LIB_KEYWORDS: Dict[str, List[str]] = {}


@dataclass
class LibraryItem:
    """A downloadable library item (symbol, footprint, or both)."""
    name: str
    manufacturer: str
    mpn: str  # Manufacturer Part Number
    description: str
    source: LibrarySource
    
    # URLs/paths for download
    symbol_url: Optional[str] = None
    footprint_url: Optional[str] = None
    model_3d_url: Optional[str] = None
    
    # Metadata
    datasheet_url: Optional[str] = None
    category: str = ""
    package: str = ""
    
    # Status
    is_downloaded: bool = False
    local_symbol_path: Optional[str] = None
    local_footprint_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'manufacturer': self.manufacturer,
            'mpn': self.mpn,
            'description': self.description,
            'source': self.source.value,
            'symbol_url': self.symbol_url,
            'footprint_url': self.footprint_url,
            'datasheet_url': self.datasheet_url,
            'category': self.category,
            'package': self.package,
        }


@dataclass
class DownloadResult:
    """Result of a library download operation."""
    success: bool
    item: LibraryItem
    message: str
    symbol_path: Optional[str] = None
    footprint_path: Optional[str] = None
    error: Optional[str] = None


# Consolidated list of IC package families recognised across the codebase.
# Used by _extract_package_hint, _extract_package_from_desc, and _package_family.
_PACKAGE_FAMILIES = (
    'HTSSOP', 'TSSOP', 'MSOP', 'LFCSP', 'WQFN',
    'PDIP', 'DIP', 'TQFP', 'LQFP', 'QFN', 'UQFN', 'VQFN',
    'DFN', 'SOIC', 'SOP', 'SO', 'SSOP', 'LSSOP', 'SOT', 'BGA', 'QFP',
    'VFBGA', 'UFBGA', 'DRQFN', 'TFBGA', 'PLCC', 'XSON',
)
# Regex that matches e.g. "TSSOP-28", "QFN-32", etc.
_PACKAGE_RE = re.compile(
    r'\b(' + '|'.join(re.escape(f) + r'-\d+' for f in _PACKAGE_FAMILIES) + r')\b',
    re.IGNORECASE,
)


def _count_footprint_pads(fp_path: str) -> int:
    """Count the number of pads in a .kicad_mod footprint file.

    Reads only the first 64KB to stay fast.  Returns 0 on any error.
    """
    try:
        p = Path(fp_path)
        if not p.is_file():
            return 0
        head = p.read_bytes()[:65536].decode('utf-8', errors='ignore')
        # Each pad is introduced by a top-level '(pad' token.
        return len(re.findall(r'\(pad\s', head))
    except Exception:
        return 0


def _extract_expected_pin_count(query: str, package_hint: str = '') -> int:
    """Guess the expected pin count from a query string or package hint.

    Heuristics:
    1. Explicit package hint like "DIP-28" → 28
    2. Package family name in query like "TQFP-32" → 32
    3. Known MPN patterns (e.g. ATmega328 → 28-pin DIP / 32-pin TQFP)

    Returns 0 if no pin count can be inferred.
    """
    # 1. From package hint
    for text in [package_hint, query]:
        if not text:
            continue
        m = re.search(r'(?:' + '|'.join(re.escape(f) for f in _PACKAGE_FAMILIES) + r')[- ]?(\d+)', text, re.IGNORECASE)
        if m:
            return int(m.group(1))

    # 2. Pin-header / pin-socket patterns: "1x10" → 10, "2x06" → 12
    m = re.search(r'(\d+)x(\d+)', query, re.IGNORECASE)
    if m:
        return int(m.group(1)) * int(m.group(2))

    return 0


_PASSIVE_KIND_UNKNOWN = "unknown"
_PASSIVE_KIND_RESISTOR = "resistor"
_PASSIVE_KIND_CAPACITOR = "capacitor"
_PASSIVE_KIND_INDUCTOR = "inductor"
_PASSIVE_KIND_CRYSTAL = "crystal"


def _infer_value_only_kind(query: str) -> Tuple[str, str]:
    """Infer a passive kind from a value-only query like '100nF' or '10k'."""
    q = (query or "").strip().lower()
    q = q.replace("μ", "u").replace("µ", "u")
    q = re.sub(r'\s+', '', q)
    if not q:
        return _PASSIVE_KIND_UNKNOWN, ""

    # Frequency: "16mhz", "32.768khz"
    if re.fullmatch(r'\d+(?:\.\d+)?(?:mhz|khz|hz)', q):
        return _PASSIVE_KIND_CRYSTAL, q

    # Capacitor: "100nf", "22pf", "10uf"
    if re.fullmatch(r'\d+(?:\.\d+)?(?:pf|nf|uf|f)', q):
        return _PASSIVE_KIND_CAPACITOR, q

    # Inductor: "10uh", "1mh"
    if re.fullmatch(r'\d+(?:\.\d+)?(?:nh|uh|mh|h)', q):
        return _PASSIVE_KIND_INDUCTOR, q

    # Resistor: "10k", "4k7", "330r", "100ohm"
    if re.fullmatch(r'\d+(?:\.\d+)?(?:ohm|k|m|r)', q):
        return _PASSIVE_KIND_RESISTOR, q
    if re.fullmatch(r'\d+[kmr]\d+', q):  # 4k7, 2r2, 1m0
        return _PASSIVE_KIND_RESISTOR, q

    return _PASSIVE_KIND_UNKNOWN, q


def _infer_passive_kind_from_text(query: str) -> str:
    """Infer a passive kind from mixed natural-language text.

    Only returns a kind when the text includes an explicit kind keyword.
    """
    q = (query or "").lower()
    q = q.replace("μ", "u").replace("µ", "u")
    if "capacitor" in q or "cap" in q:
        if re.search(r'\b\d+(?:\.\d+)?(?:pf|nf|uf|f)\b', q.replace(" ", "")):
            return _PASSIVE_KIND_CAPACITOR
        return _PASSIVE_KIND_CAPACITOR
    if "resistor" in q or re.search(r'\b\d+[kmr]\d+\b', q.replace(" ", "")):
        if re.search(r'\b\d+(?:\.\d+)?(?:ohm|k|m|r)\b', q.replace(" ", "")):
            return _PASSIVE_KIND_RESISTOR
        if "resistor" in q:
            return _PASSIVE_KIND_RESISTOR
    if "inductor" in q or "coil" in q:
        return _PASSIVE_KIND_INDUCTOR
    if "crystal" in q or "oscillator" in q or "xtal" in q:
        return _PASSIVE_KIND_CRYSTAL
    return _PASSIVE_KIND_UNKNOWN


def _looks_like_mpn(query: str) -> bool:
    """Heuristic: true when the string resembles a manufacturer part number."""
    q = (query or "").strip()
    if not q or len(q) < 5:
        return False
    # e.g. "ATMEGA328P", "BSC100N10", "USB_C_Receptacle"
    if re.search(r'[A-Za-z]{2,}\d{2,}', q):
        return True
    if re.search(r'\d{2,}[A-Za-z]{2,}', q):
        return True
    if "_" in q and any(ch.isdigit() for ch in q):
        return True
    return False


class LibraryManager:
    """Manages symbol and footprint library operations.
    
    Core principles:
    - All downloads are previewed before execution
    - User must explicitly approve installations
    - Libraries are installed to user-configurable paths
    - Supports multiple sources (SnapEDA, Ultra Librarian, etc.)
    """
    
    # SnapEDA API (requires API key for full access)
    SNAPEDA_API_BASE = "https://www.snapeda.com/api/v1"
    SNAPEDA_SEARCH_URL = "https://www.snapeda.com/api/v1/parts/search"
    
    def __init__(self,
                 kicad_user_lib_path: Optional[str] = None,
                 snapeda_api_key: Optional[str] = None,
                 enable_github_sources: bool = False,
                 github_cache_dir: Optional[str] = None,
                 github_curated_dirs: Optional[List[str]] = None,
                 github_curated_repos: Optional[List[Dict[str, str]]] = None,
                 enable_github_search: bool = False):
        """Initialize the library manager.
        
        Args:
            kicad_user_lib_path: Path to KiCad user library folder
            snapeda_api_key: Optional SnapEDA API key for full access
        """
        self.kicad_user_lib_path = kicad_user_lib_path or self._detect_kicad_lib_path()
        self.snapeda_api_key = snapeda_api_key or os.environ.get('SNAPEDA_API_KEY', '')

        # Optional, keyless providers (disabled by default to keep tests/offline runs stable)
        self.enable_github_sources = bool(enable_github_sources)
        self.enable_github_search = bool(enable_github_search)

        cache_root = Path(github_cache_dir).expanduser() if github_cache_dir else (Path.home() / '.vibecad' / 'lib_cache')
        self.github_cache_dir = cache_root
        self.github_cache_dir.mkdir(parents=True, exist_ok=True)

        self.github_curated_dirs: List[Path] = []
        if github_curated_dirs:
            for d in github_curated_dirs:
                try:
                    p = Path(d).expanduser().resolve()
                    if p.is_dir():
                        self.github_curated_dirs.append(p)
                except Exception:
                    continue

        # Curated repos are optional; default to none to avoid unexpected large downloads.
        # Callers (plugin/CLI) can supply a curated list appropriate for their use case.
        if github_curated_repos is None:
            self.github_curated_repos = []
        else:
            self.github_curated_repos = list(github_curated_repos)
        
        # Callbacks for user approval
        self._on_preview: Optional[Callable] = None
        self._on_progress: Optional[Callable] = None
        
        # Cache of search results
        self._search_cache: Dict[str, List[LibraryItem]] = {}

        # Local KiCad library index (built lazily on first search)
        self._local_index: Optional[List[Tuple[str, str, str, str, Optional[str]]]] = None
        # Each entry: (name_lower, name, library, entry_type "symbol"|"footprint", file_path)

        # Optional GitHub curated cache index
        self._github_index: Optional[List[Tuple[str, str, str, str, Optional[str]]]] = None
    
    def _detect_kicad_lib_path(self) -> str:
        """Detect the KiCad user library path."""
        home = Path.home()

        # 1) Prefer an existing versioned KiCad user dir under Documents.
        docs_root = home / "Documents" / "KiCad"
        if docs_root.is_dir():
            version_dirs: List[Tuple[Tuple[int, ...], Path]] = []
            try:
                for child in docs_root.iterdir():
                    if not child.is_dir():
                        continue
                    if re.match(r'^\d+(?:\.\d+)?$', child.name):
                        # Parse version like "7.0" → (7, 0)
                        try:
                            ver_tuple = tuple(int(p) for p in child.name.split('.'))
                        except Exception:
                            continue
                        version_dirs.append((ver_tuple, child))
            except OSError:
                version_dirs = []

            if version_dirs:
                version_dirs.sort(key=lambda x: x[0], reverse=True)
                return str(version_dirs[0][1])

            # If Documents/KiCad exists but isn't versioned, use it.
            return str(docs_root)

        # 2) Linux-style user data dir
        linux_root = home / ".local" / "share" / "kicad"
        if linux_root.is_dir():
            return str(linux_root)

        # 3) Fallback
        return str(docs_root)
    
    def set_preview_callback(self, callback: Callable):
        """Set callback for previewing downloads before execution."""
        self._on_preview = callback
    
    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates."""
        self._on_progress = callback
    
    async def search_parts(self, query: str,
                          source: Optional[LibrarySource] = None,
                          limit: int = 20) -> List[LibraryItem]:
        """Search for parts across library sources with automatic fallback.

        When *source* is ``None`` the search order is:
        1. Local KiCad library files on disk (fast, offline, comprehensive)
        2. KiCad online symbol index — kicad.github.io (prefix + keyword mapped)
        3. SnapEDA (needs API key)

        Args:
            query: Search query (part number, description, etc.)
            source: Force a specific source, or ``None`` for auto.
            limit: Maximum number of results.

        Returns:
            List of matching LibraryItems.
        """
        return self.search_parts_sync(query, source=source, limit=limit)

    def search_parts_sync(self, query: str,
                         source: Optional[LibrarySource] = None,
                         limit: int = 20) -> List[LibraryItem]:
        """Synchronous search with automatic fallback chain.

        When *source* is ``None`` the search order is:
        1. Local KiCad library files on disk (fast, offline, comprehensive)
        2. KiCad online symbol index — kicad.github.io (prefix + keyword mapped)
        3. SnapEDA (needs API key)
        """
        cache_key = f"{source}:{query}:{limit}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        # Guardrail: value-only queries are extremely ambiguous and tend to
        # match arbitrary MPNs (e.g. "100nF" matching a MOSFET with "100N" in
        # its part number). Prefer returning generic footprint options when
        # available rather than misleading results.
        kind, _norm = _infer_value_only_kind(query)
        if kind != _PASSIVE_KIND_UNKNOWN:
            generic = self._suggest_generic_footprints_for_kind(kind, limit=limit)
            if generic:
                self._search_cache[cache_key] = generic
                return generic

        if source is not None:
            # Caller chose a specific source
            results = self._search_source(source, query, limit)
        else:
            # 1. Local KiCad library scan (fast, works offline)
            results = self._search_kicad_local(query, limit)
            # 2. Online KiCad index (prefix + keyword matching)
            if not results:
                results = self._search_kicad_builtin_sync(query, limit)
            # 3. EasyEDA/LCSC (free, no auth, large DB)
            if not results:
                results = self._search_easyeda_sync(query, limit)
            # 4. GitHub curated repos (optional, keyless)
            if not results and self.enable_github_sources:
                results = self._search_github_curated_local(query, limit)
            # 5. GitHub search (optional, keyless but rate-limited)
            if not results and self.enable_github_sources and self.enable_github_search:
                results = self._search_github_search_sync(query, limit)
            # 6. SnapEDA
            if not results:
                results = self._search_snapeda_sync(query, limit)

        self._search_cache[cache_key] = results
        return results

    def _find_local_footprint_path(self, lib: str, name: str) -> Optional[str]:
        """Return absolute path for a KiCad built-in footprint lib/name, if present."""
        if not lib or not name:
            return None
        try:
            index = self._build_local_index()
        except Exception:
            return None
        target_lib = str(lib).strip()
        target_name = str(name).strip()
        for _, fp_name, fp_lib, etype, file_path in index:
            if etype != "footprint" or not file_path:
                continue
            if fp_lib == target_lib and fp_name == target_name:
                return str(file_path)
        return None

    def _suggest_generic_footprints_for_kind(self, kind: str, limit: int = 20) -> List[LibraryItem]:
        """Return a small set of generic footprints for common value-only queries.

        Only returns footprints that exist in the local KiCad index.
        """
        kind = (kind or "").strip().lower()
        candidates: List[Tuple[str, str, str]] = []  # (lib, name, description)

        # Ordered: common defaults first.
        if kind == _PASSIVE_KIND_RESISTOR:
            candidates = [
                ("Resistor_SMD", "R_0603_1608Metric", "Generic resistor (0603)"),
                ("Resistor_SMD", "R_0402_1005Metric", "Generic resistor (0402)"),
                ("Resistor_SMD", "R_0805_2012Metric", "Generic resistor (0805)"),
                ("Resistor_THT", "R_Axial_DIN0207_L6.3mm_D2.5mm_P7.62mm_Horizontal", "Generic axial resistor (THT)"),
            ]
        elif kind == _PASSIVE_KIND_CAPACITOR:
            candidates = [
                ("Capacitor_SMD", "C_0603_1608Metric", "Generic capacitor (0603)"),
                ("Capacitor_SMD", "C_0402_1005Metric", "Generic capacitor (0402)"),
                ("Capacitor_SMD", "C_0805_2012Metric", "Generic capacitor (0805)"),
                ("Capacitor_THT", "C_Disc_D3.0mm_W1.6mm_P2.50mm", "Generic ceramic disc capacitor (THT)"),
            ]
        elif kind == _PASSIVE_KIND_INDUCTOR:
            candidates = [
                ("Inductor_SMD", "L_0603_1608Metric", "Generic inductor (0603)"),
                ("Inductor_SMD", "L_0805_2012Metric", "Generic inductor (0805)"),
                ("Inductor_THT", "L_Axial_L6.6mm_D2.7mm_P7.62mm_Horizontal_Vishay_IM-1", "Generic axial inductor (THT)"),
            ]
        elif kind == _PASSIVE_KIND_CRYSTAL:
            candidates = [
                ("Crystal", "Crystal_SMD_3225-4Pin_3.2x2.5mm", "Generic crystal (SMD 3225)"),
                ("Crystal", "Crystal_SMD_2016-4Pin_2.0x1.6mm", "Generic crystal (SMD 2016)"),
                ("Crystal", "Crystal_HC49-U_Vertical", "Generic crystal (HC49 THT)"),
            ]
        else:
            return []

        results: List[LibraryItem] = []
        for lib, name, desc in candidates:
            fp_path = self._find_local_footprint_path(lib, name)
            if not fp_path:
                continue
            item = LibraryItem(
                name=f"{lib}:{name}",
                manufacturer="(KiCad built-in)",
                mpn=name,
                description=desc,
                source=LibrarySource.KICAD_BUILTIN,
                category=lib,
                package="",
            )
            item.local_footprint_path = fp_path
            results.append(item)
            if len(results) >= max(1, int(limit)):
                break
        return results

    def _search_source(self, source: LibrarySource, query: str, limit: int) -> List[LibraryItem]:
        """Dispatch to the right backend."""
        if source == LibrarySource.KICAD_BUILTIN:
            # Local scan first, then online index
            results = self._search_kicad_local(query, limit)
            if not results:
                results = self._search_kicad_builtin_sync(query, limit)
            return results
        if source == LibrarySource.SNAPEDA:
            return self._search_snapeda_sync(query, limit)
        if source == LibrarySource.GITHUB_CURATED:
            return self._search_github_curated_local(query, limit)
        if source == LibrarySource.GITHUB_SEARCH:
            return self._search_github_search_sync(query, limit)
        if source == LibrarySource.EASYEDA:
            return self._search_easyeda_sync(query, limit)
        logger.warning(f"Search not implemented for source: {source}")
        return []

    # ------------------------------------------------------------------
    # EasyEDA / LCSC component search (free, no auth required)
    # ------------------------------------------------------------------

    # Mapping from normalised LCSC package family names to KiCad footprint
    # library + name glob patterns.  The tuple is (kicad_library, token) where
    # *token* will be searched in the local footprint index.
    _LCSC_PKG_MAP: Dict[str, Tuple[str, str]] = {
        'SSOP':   ('Package_SO', 'SSOP'),
        'TSSOP':  ('Package_SO', 'TSSOP'),
        'HTSSOP': ('Package_SO', 'HTSSOP'),
        'SOP':    ('Package_SO', 'SOIC'),
        'SOIC':   ('Package_SO', 'SOIC'),
        'SOT':    ('Package_TO_SOT_SMD', 'SOT'),
        'SOT23':  ('Package_TO_SOT_SMD', 'SOT-23'),
        'QFN':    ('Package_DFN_QFN', 'QFN'),
        'DFN':    ('Package_DFN_QFN', 'DFN'),
        'QFP':    ('Package_QFP', 'QFP'),
        'TQFP':   ('Package_QFP', 'TQFP'),
        'LQFP':   ('Package_QFP', 'LQFP'),
        'DIP':    ('Package_DIP', 'DIP'),
        'PDIP':   ('Package_DIP', 'PDIP'),
        'BGA':    ('Package_BGA', 'BGA'),
        'TO':     ('Package_TO_SOT_THT', 'TO'),
        'SC70':   ('Package_TO_SOT_SMD', 'SC-70'),
    }

    def _search_easyeda_sync(self, query: str, limit: int) -> List[LibraryItem]:
        """Search LCSC / EasyEDA Pro for components.  Free, no auth needed.

        This API returns part metadata (MPN, package, manufacturer) plus a
        flag indicating whether EasyEDA has a schematic/footprint model.
        When a model exists we can later download it via ``easyeda2kicad``
        or map its package to a stock KiCad footprint.
        """
        q = (query or '').strip()
        if len(q) < 3:
            return []

        try:
            url = f"https://pro.easyeda.com/api/eda/product/search?keyword={quote_plus(q)}&limit={limit}"
            req = Request(url, headers={'User-Agent': 'VibeCAD/0.4.0'})
            
            # Use a permissive SSL context to avoid certificate errors on some systems
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            with urlopen(req, timeout=15, context=ctx) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            logger.warning("EasyEDA search failed for %r: %s", q, e)
            return []

        products = (data.get('result') or {}).get('productList') or []
        if not products:
            return []

        results: List[LibraryItem] = []
        for p in products:
            mpn = (p.get('mpn') or '').strip()
            lcsc = (p.get('number') or '').strip()
            pkg = (p.get('package') or '').strip()
            mfr = (p.get('manufacturer') or '').strip()
            has_device = bool(p.get('hasDevice', ''))

            if not mpn:
                continue

            item = LibraryItem(
                name=mpn,
                manufacturer=mfr or '(LCSC)',
                mpn=mpn,
                description=f"{mfr} {mpn}, {pkg}" if pkg else f"{mfr} {mpn}",
                source=LibrarySource.EASYEDA,
                category=pkg,
                package=pkg,
            )

            # Try to map the LCSC package to a local KiCad stock footprint
            local_fp = self._map_lcsc_package_to_local_footprint(pkg)
            if local_fp:
                item.local_footprint_path = local_fp

            # Store LCSC number and device UUID for potential easyeda2kicad download
            item.datasheet_url = f"https://www.lcsc.com/product-detail/{lcsc}.html" if lcsc else None

            results.append(item)
            if len(results) >= limit:
                break

        if results:
            logger.info("EasyEDA search for %r returned %d results (top: %s)",
                        q, len(results), results[0].mpn)
        return results

    def _map_lcsc_package_to_local_footprint(self, lcsc_package: str) -> Optional[str]:
        """Map an LCSC package name like 'SSOP-28-208mil' to a local KiCad footprint path.

        Returns the absolute path to the best-matching ``.kicad_mod`` file,
        or ``None`` if no good match is found.
        """
        if not lcsc_package:
            return None

        pkg = lcsc_package.strip().upper()
        # Extract pin count from package name: "SSOP-28-208mil" → 28
        pin_match = re.search(r'(\d+)', pkg)
        pin_count = pin_match.group(1) if pin_match else ''

        # Extract body width hint from LCSC "XXXmil" notation
        # e.g., "SSOP-28-208mil" → 208mil ≈ 5.28mm → match "5.3" in footprint name
        body_width_mm: Optional[float] = None
        mil_match = re.search(r'(\d+)MIL', pkg)
        if mil_match:
            body_width_mm = int(mil_match.group(1)) * 0.0254  # mil to mm

        # Determine package family
        pkg_family = ''
        for fam in sorted(self._LCSC_PKG_MAP.keys(), key=len, reverse=True):
            if pkg.startswith(fam):
                pkg_family = fam
                break

        if not pkg_family:
            return None

        kicad_lib, search_token = self._LCSC_PKG_MAP[pkg_family]

        # Build the local index and search for matching footprints
        index = self._build_local_index()
        if not index:
            return None

        # Find footprints matching the package family and pin count
        search_upper = search_token.upper()
        best_path: Optional[str] = None
        best_score = -1

        for name_lower, name, lib, etype, file_path in index:
            if etype != 'footprint' or not file_path:
                continue
            name_upper = name.upper()
            if search_upper not in name_upper:
                continue
            if pin_count and f'-{pin_count}_' not in name and f'-{pin_count}-' not in name_upper:
                # Also check the name directly: SSOP-28_5.3x10.2mm
                if f'{search_upper}-{pin_count}' not in name_upper:
                    continue

            # Score: prefer exact library match and simpler names
            score = 0
            if lib == kicad_lib:
                score += 100
            # Prefer footprints without thermal pad / extra features
            if 'EP' not in name_upper and 'Thermal' not in name_upper:
                score += 10

            # If we have a body width hint, prefer footprints with matching dimensions
            if body_width_mm is not None:
                dims = re.findall(r'(\d+\.?\d*)x(\d+\.?\d*)mm', name)
                if dims:
                    w = float(dims[0][0])
                    if abs(w - body_width_mm) < 0.3:  # within 0.3mm tolerance
                        score += 50
                    else:
                        score -= 30  # wrong body size

            # Prefer shorter names (simpler, more generic)
            score -= len(name) // 10

            if score > best_score:
                best_score = score
                best_path = file_path

        return best_path

    # ------------------------------------------------------------------
    # GitHub curated repositories (keyless ZIP snapshots)
    # ------------------------------------------------------------------

    def _github_repo_cache_dir(self, owner: str, repo: str, branch: str) -> Path:
        safe = re.sub(r'[^a-zA-Z0-9_.-]+', '_', f"{owner}__{repo}__{branch}")
        return self.github_cache_dir / safe

    def _download_github_repo_zip(self, owner: str, repo: str, branch: str, dest_dir: Path) -> Path:
        """Download and extract a GitHub repo ZIP snapshot into dest_dir.

        Tries the provided branch; if that fails and branch is 'main', also tries 'master' (and vice versa).
        """
        dest_dir.mkdir(parents=True, exist_ok=True)

        def _try_branch(b: str) -> Optional[Path]:
            url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{b}"
            headers = {'User-Agent': 'VibeCAD/0.4.0'}
            req = Request(url, headers=headers)
            zip_path = dest_dir / f"{owner}_{repo}_{b}.zip"
            try:
                with urlopen(req, timeout=30) as resp:
                    content = resp.read()
                with open(zip_path, 'wb') as f:
                    f.write(content)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(dest_dir)
                # Repo zips extract as {repo}-{branch}/
                for child in dest_dir.iterdir():
                    if child.is_dir() and child.name.lower().startswith(repo.lower() + '-'):
                        return child
                # Fallback: first directory
                for child in dest_dir.iterdir():
                    if child.is_dir():
                        return child
            except Exception as e:
                logger.debug("GitHub ZIP download failed for %s/%s@%s: %s", owner, repo, b, e)
                return None
            return None

        root = _try_branch(branch)
        if root is not None:
            return root
        alt = 'master' if branch == 'main' else ('main' if branch == 'master' else '')
        if alt:
            root = _try_branch(alt)
            if root is not None:
                return root
        raise RuntimeError(f"Failed to download GitHub repo {owner}/{repo} (branch {branch})")

    def _ensure_github_curated_cache(self) -> List[Path]:
        """Ensure curated GitHub sources are available locally and return local roots."""
        roots: List[Path] = list(self.github_curated_dirs)
        if not self.enable_github_sources:
            return roots

        # Download ZIP snapshots into cache on-demand, but only if we don't already have them.
        for spec in self.github_curated_repos:
            owner = (spec.get('owner') or '').strip()
            repo = (spec.get('repo') or '').strip()
            branch = (spec.get('branch') or 'main').strip()
            if not owner or not repo:
                continue

            cache_dir = self._github_repo_cache_dir(owner, repo, branch)
            meta_path = cache_dir / '.vibecad_meta.json'
            # Simple TTL to avoid repeated downloads in a session.
            ttl_sec = int(os.environ.get('VIBECAD_GITHUB_CACHE_TTL_SEC', '86400') or '86400')

            extracted_root: Optional[Path] = None
            if cache_dir.is_dir():
                # Find an extracted repo root
                try:
                    for child in cache_dir.iterdir():
                        if child.is_dir() and child.name.lower().startswith(repo.lower() + '-'):
                            extracted_root = child
                            break
                except Exception:
                    extracted_root = None

            should_refresh = True
            if extracted_root is not None and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding='utf-8'))
                    downloaded_at = float(meta.get('downloaded_at', 0) or 0)
                    if downloaded_at and (time.time() - downloaded_at) < ttl_sec:
                        should_refresh = False
                except Exception:
                    should_refresh = True

            if extracted_root is None or should_refresh:
                # Clear old contents (best-effort)
                try:
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
                except Exception:
                    pass
                cache_dir.mkdir(parents=True, exist_ok=True)
                extracted_root = self._download_github_repo_zip(owner, repo, branch, cache_dir)
                try:
                    meta_path.write_text(json.dumps({
                        'owner': owner,
                        'repo': repo,
                        'branch': branch,
                        'downloaded_at': time.time(),
                    }, indent=2, sort_keys=True), encoding='utf-8')
                except Exception:
                    pass

            if extracted_root is not None and extracted_root.is_dir():
                roots.append(extracted_root)

        return roots

    def _build_github_index(self) -> List[Tuple[str, str, str, str, Optional[str]]]:
        """Build a searchable index from curated GitHub sources."""
        if self._github_index is not None:
            return self._github_index

        roots = self._ensure_github_curated_cache()
        entries: List[Tuple[str, str, str, str, Optional[str]]] = []
        if not roots:
            self._github_index = entries
            return entries

        sym_re = re.compile(r'\(symbol\s+"([^"]+)"')
        sub_sym_re = re.compile(r'_\d+_\d+$')

        for root in roots:
            try:
                root_name = root.name
            except Exception:
                root_name = 'github'

            # Symbols: any *.kicad_sym file
            try:
                for sym_file in root.rglob('*.kicad_sym'):
                    lib_name = sym_file.stem
                    try:
                        text = sym_file.read_text(encoding='utf-8', errors='ignore')
                    except OSError:
                        continue
                    seen_syms: set = set()
                    for m in sym_re.finditer(text):
                        sym_name = m.group(1)
                        if sub_sym_re.search(sym_name):
                            continue
                        if sym_name in seen_syms:
                            continue
                        seen_syms.add(sym_name)
                        entries.append((sym_name.lower(), sym_name, lib_name or root_name, 'symbol', str(sym_file)))
            except Exception:
                pass

            # Footprints: any *.pretty/*.kicad_mod
            try:
                for pretty_dir in root.rglob('*.pretty'):
                    if not pretty_dir.is_dir():
                        continue
                    lib_name = pretty_dir.stem
                    for mod_file in pretty_dir.glob('*.kicad_mod'):
                        fp_name = mod_file.stem
                        entries.append((fp_name.lower(), fp_name, lib_name or root_name, 'footprint', str(mod_file)))
            except Exception:
                pass

        logger.info("GitHub curated index: %d entries from %d roots", len(entries), len(roots))
        self._github_index = entries
        return entries

    def _search_index(self,
                      index: List[Tuple[str, str, str, str, Optional[str]]],
                      query: str,
                      limit: int,
                      source: LibrarySource,
                      manufacturer: str) -> List[LibraryItem]:
        if not index:
            return []
        tokens = self._tokenize_query(query)
        if not tokens:
            return []
        extra_tokens_sets: List[List[str]] = []
        for variant in self._normalize_query(query):
            vt = self._tokenize_query(variant)
            if vt != tokens:
                extra_tokens_sets.append(vt)

        scored: List[Tuple[float, int, Tuple[str, str, str, str, Optional[str]]]] = []
        for idx, entry in enumerate(index):
            name_lower, name, lib, etype, file_path = entry
            score = self._score_match(name_lower, tokens)
            for alt in extra_tokens_sets:
                alt_score = self._score_match(name_lower, alt)
                if alt_score > score:
                    score = alt_score
            if score > 0:
                scored.append((score, idx, entry))
        if not scored:
            return []
        scored.sort(key=lambda x: (-x[0], x[2][1]))

        seen: set = set()
        results: List[LibraryItem] = []
        for score, _, (name_lower, name, lib, etype, file_path) in scored:
            key = (lib, name)
            if key in seen:
                continue
            seen.add(key)
            item = LibraryItem(
                name=f"{lib}:{name}",
                manufacturer=manufacturer,
                mpn=name,
                description=f"{etype.title()} from {manufacturer} ({lib})",
                source=source,
                category=lib,
                package='',
            )
            if etype == 'footprint' and file_path:
                item.local_footprint_path = file_path
            elif etype == 'symbol' and file_path:
                item.local_symbol_path = file_path
            results.append(item)
            if len(results) >= limit:
                break
        return results

    def _search_github_curated_local(self, query: str, limit: int) -> List[LibraryItem]:
        """Search curated GitHub repos (cached locally)."""
        try:
            index = self._build_github_index()
            return self._search_index(index, query, limit, LibrarySource.GITHUB_CURATED, '(GitHub curated)')
        except Exception as e:
            logger.debug("GitHub curated search failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # GitHub search (keyless, rate-limited)
    # ------------------------------------------------------------------

    def _search_github_search_sync(self, query: str, limit: int) -> List[LibraryItem]:
        """Search GitHub for KiCad symbol/footprint files.

        This is intentionally conservative: it is rate-limited without a token.
        """
        # Avoid excessive network calls for very short/ambiguous queries.
        q = (query or '').strip()
        if len(q) < 4:
            return []

        # Use tokenised query for better results
        tokens = self._tokenize_query(q)
        if not tokens:
            return []

        joined = ' '.join(tokens[:4])
        # GitHub search API max per_page is 100; keep small.
        per_page = min(max(int(limit), 1), 20)

        def _looks_like_cert_error(exc: Exception) -> bool:
            # SSLCertVerificationError, SSLError, or URLError wrapping one.
            try:
                if isinstance(exc, ssl.SSLCertVerificationError):
                    return True
            except Exception:
                pass
            try:
                reason = getattr(exc, 'reason', None)
                if reason is not None:
                    if isinstance(reason, ssl.SSLCertVerificationError):
                        return True
                    if isinstance(reason, ssl.SSLError) and 'CERTIFICATE_VERIFY_FAILED' in str(reason):
                        return True
            except Exception:
                pass
            return 'CERTIFICATE_VERIFY_FAILED' in str(exc)

        def _ssl_contexts() -> Tuple[Optional[ssl.SSLContext], Optional[ssl.SSLContext]]:
            cafile = (
                os.environ.get("VIBECAD_CA_BUNDLE", "").strip()
                or os.environ.get("REQUESTS_CA_BUNDLE", "").strip()
                or os.environ.get("SSL_CERT_FILE", "").strip()
            )
            if not cafile:
                try:
                    import certifi  # type: ignore

                    cafile = certifi.where() or ""
                except Exception:
                    cafile = ""
            verified: Optional[ssl.SSLContext]
            try:
                verified = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
            except Exception:
                verified = None
            try:
                unverified = ssl._create_unverified_context()
            except Exception:
                unverified = None
            return verified, unverified

        verified_ctx, unverified_ctx = _ssl_contexts()
        allow_insecure = bool(os.environ.get("VIBECAD_SSL_NO_VERIFY", "").strip())

        def _fetch(kind: str) -> List[Dict[str, Any]]:
            # kind: 'kicad_mod' or 'kicad_sym'
            api = 'https://api.github.com/search/code'
            # Narrow to relevant file types. Quoting spaces is handled via quote_plus.
            search_q = f"{joined} extension:{kind}"
            url = f"{api}?q={quote_plus(search_q)}&per_page={per_page}"
            headers = {
                'Accept': 'application/vnd.github+json',
                'User-Agent': 'VibeCAD/0.4.0',
            }
            # Optional token (not required); only helps rate limits.
            tok = os.environ.get('GITHUB_TOKEN', '').strip()
            if tok:
                headers['Authorization'] = f"Bearer {tok}"
            req = Request(url, headers=headers)
            try:
                if verified_ctx is None:
                    with urlopen(req, timeout=15) as resp:
                        return json.loads(resp.read().decode('utf-8')).get('items', [])
                with urlopen(req, timeout=15, context=verified_ctx) as resp:
                    return json.loads(resp.read().decode('utf-8')).get('items', [])
            except Exception as exc:
                if allow_insecure and unverified_ctx is not None and _looks_like_cert_error(exc):
                    logger.warning(
                        "GitHub search API used unverified SSL context (cert verify failed). "
                        "Set VIBECAD_CA_BUNDLE/SSL_CERT_FILE or install certifi to avoid this."
                    )
                    with urlopen(req, timeout=15, context=unverified_ctx) as resp:
                        return json.loads(resp.read().decode('utf-8')).get('items', [])
                raise

        try:
            items_mod = _fetch('kicad_mod')
            items_sym = _fetch('kicad_sym')
        except Exception as e:
            logger.debug("GitHub search API failed: %s", e)
            return []

        results: List[LibraryItem] = []
        seen: set = set()

        def _raw_url(item: Dict[str, Any]) -> Optional[str]:
            try:
                html = item.get('html_url', '')
                # https://github.com/{owner}/{repo}/blob/{ref}/{path}
                m = re.match(r'^https://github.com/([^/]+/[^/]+)/blob/([^/]+)/(.+)$', html)
                if not m:
                    return None
                repo_full = m.group(1)
                ref = m.group(2)
                path = m.group(3)
                return f"https://raw.githubusercontent.com/{repo_full}/{ref}/{path}"
            except Exception:
                return None

        for item in (items_mod + items_sym):
            name = str(item.get('name', '') or '').strip()
            if not name:
                continue
            raw = _raw_url(item)
            if not raw:
                continue
            key = raw
            if key in seen:
                continue
            seen.add(key)

            is_fp = name.endswith('.kicad_mod')
            is_sym = name.endswith('.kicad_sym')
            lib_item = LibraryItem(
                name=name,
                manufacturer='(GitHub search)',
                mpn=Path(name).stem,
                description='GitHub search result',
                source=LibrarySource.GITHUB_SEARCH,
            )
            if is_fp:
                lib_item.footprint_url = raw
            if is_sym:
                lib_item.symbol_url = raw
            results.append(lib_item)
            if len(results) >= limit:
                break

        return results

    # ------------------------------------------------------------------
    # Query normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_query(query: str) -> List[str]:
        """Generate query variants for fuzzy matching.

        E.g. "XYZ123A-PU" → ["XYZ123A-PU", "XYZ123A", "XYZ123"]
        Also normalises SOP↔SOIC so that "SOP-16" queries also match KiCad's
        "SOIC-16" footprint names and vice versa.
        """
        q = query.strip()
        variants: List[str] = [q]

        # Strip common ordering / package suffixes
        # -PU  (DIP), -AU  (TQFP), -MU  (QFN), etc.
        stripped = re.sub(
            r'[-/]?(PU|AU|MU|CU|TU|NU|AN|MN|PH|SS|SN|AUR|MUR)$',
            '', q, flags=re.IGNORECASE,
        )
        if stripped and stripped != q:
            variants.append(stripped)

        # Also strip a trailing single-letter suffix (KiCad uses -P, -A, -M, etc.)
        # Only do this when the query looks like a real part number (has digits),
        # otherwise we can create accidental matches (e.g., "..._PART" → "..._PAR").
        if re.search(r'\d', stripped or q):
            core = re.sub(r'-?[A-Z]$', '', stripped or q, flags=re.IGNORECASE)
            if core and core != stripped and len(core) > 3:
                variants.append(core)

        # SOP ↔ SOIC cross-referencing (KiCad uses both naming conventions)
        for v in list(variants):
            swapped = re.sub(r'\bSOP\b', 'SOIC', v, flags=re.IGNORECASE)
            if swapped != v:
                variants.append(swapped)
            else:
                swapped = re.sub(r'\bSOIC\b', 'SOP', v, flags=re.IGNORECASE)
                if swapped != v:
                    variants.append(swapped)

        return list(dict.fromkeys(variants))  # de-dup, preserve order

    # ------------------------------------------------------------------
    # Token-based matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_query(query: str) -> List[str]:
        """Split a query into lowercase searchable tokens.

        >>> LibraryManager._tokenize_query("Keystone 590 battery holder")
        ['keystone', '590', 'battery', 'holder']
        """
        q = query.strip().lower()
        # Normalise common engineering value formats so that "22 pF" becomes
        # "22pf" (avoids matching random part numbers that contain "22").
        q = q.replace("μ", "u").replace("µ", "u")
        q = re.sub(r'(\d+(?:\.\d+)?)\s*(p|n|u|m)\s*f\b', r'\1\2f', q)
        q = re.sub(r'(\d+(?:\.\d+)?)\s*(n|u|m)\s*h\b', r'\1\2h', q)
        q = re.sub(r'(\d+(?:\.\d+)?)\s*(mhz|khz|hz)\b', r'\1\2', q)
        q = re.sub(r'(\d+)\s*(ohm)\b', r'\1ohm', q)
        q = re.sub(r'(\d+)\s*([kmr])\b', r'\1\2', q)

        tokens = re.split(r'[\s,_\-/]+', q)
        stopwords = {
            'a', 'an', 'the',
            'add', 'place', 'insert', 'put', 'drop', 'create',
            'search', 'find', 'lookup', 'datasheet', 'pinout', 'specs',
            'to', 'on', 'onto', 'into', 'in', 'at',
            'part', 'parts', 'component', 'components',
            'symbol', 'footprint', 'package',
            'kicad', 'library', 'lib',
            'pcb', 'board', 'schematic',
            'mpn', 'pn', 'p/n',
        }

        # Common ordering/package suffixes that are too short and produce
        # pathological substring matches (e.g. token 'pu' matches 'pullback').
        # Keep this list deliberately tight.
        short_noise_tokens = {
            'pu', 'au', 'mu', 'cu', 'tu', 'nu',
            'an', 'mn', 'sn', 'ss',
        }
        out: List[str] = []
        for t in tokens:
            # Strip leading/trailing punctuation so queries like "ads1256." work.
            t = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', t)
            if len(t) < 2:
                continue
            if t in stopwords:
                continue
            if t in short_noise_tokens:
                continue
            out.append(t)
        return out

    def resolve_best_footprint_item(self, query: str, package_hint: Optional[str] = None) -> Optional[LibraryItem]:
        """Return the best footprint *item* candidate.

        Unlike `resolve_best_footprint_path`, this can return items that have a
        `footprint_url` but no local path yet (e.g. GitHub search results).
        """
        q = (query or '').strip()
        if not q:
            return None

        pkg = (package_hint or '').strip().upper() or self._extract_package_hint(q)
        preferred_libs = self._preferred_footprint_libs_for_package(pkg) if pkg else []
        pkg_token = pkg.lower() if pkg else ''

        try:
            candidates = self.search_parts_sync(q, source=None, limit=50)
        except Exception:
            candidates = []
        # Fallback: if we got candidates but NONE have a footprint path/url,
        # the list is symbol-only — retry with the package hint to find real
        # footprint items.
        has_fp_candidates = any(
            getattr(it, 'local_footprint_path', None) or getattr(it, 'footprint_url', None)
            for it in candidates
        )
        if not has_fp_candidates and pkg:
            try:
                candidates = self.search_parts_sync(pkg, source=None, limit=50)
            except Exception:
                candidates = []

        expected_pins = _extract_expected_pin_count(q, pkg)

        best: Optional[Tuple[int, LibraryItem]] = None
        for item in candidates:
            fp_path = getattr(item, 'local_footprint_path', None)
            fp_url = getattr(item, 'footprint_url', None)
            if not fp_path and not fp_url:
                continue

            rank = 0
            name = (getattr(item, 'name', '') or '')
            lib = (getattr(item, 'category', '') or '')

            # Prefer on-disk footprints first (fastest + safest)
            if fp_path:
                rank += 100

            if pkg:
                if preferred_libs and any(pl == lib for pl in preferred_libs):
                    rank += 40
                if pkg_token and pkg_token in name.lower():
                    rank += 20

                fam = self._package_family(pkg)
                if fam:
                    other_fams = ['DFN', 'QFN', 'TQFP', 'LQFP', 'SOIC', 'SSOP', 'SOT', 'BGA']
                    if any(of in name.upper() for of in other_fams) and fam not in name.upper():
                        rank -= 20

            try:
                if getattr(getattr(item, 'source', None), 'value', '') == LibrarySource.KICAD_BUILTIN.value:
                    rank += 5
            except Exception:
                pass

            # ── Pin-count validation: penalise gross mismatches ──────
            if expected_pins > 0 and fp_path:
                actual_pins = _count_footprint_pads(fp_path)
                if actual_pins == 0:
                    # Objective signal: a 0-pad footprint cannot be a real IC.
                    rank -= 200
                elif actual_pins == expected_pins:
                    rank += 30  # exact match bonus
                elif actual_pins < expected_pins * 0.5:
                    rank -= 200  # e.g. DIP-4 for a 28-pin IC
                elif abs(actual_pins - expected_pins) <= 2:
                    rank += 10  # close enough (thermal pad etc.)
                else:
                    rank -= 50  # moderate mismatch

            # Also extract pin count from footprint *name* as a fast heuristic
            # when the file isn't local (fp_url only).
            if expected_pins > 0 and not fp_path and fp_url:
                name_pin_m = re.search(r'-(\d+)[_\s]', name)
                if name_pin_m:
                    name_pins = int(name_pin_m.group(1))
                    if name_pins == expected_pins:
                        rank += 30
                    elif name_pins < expected_pins * 0.5:
                        rank -= 200

            if best is None or rank > best[0]:
                best = (rank, item)

        return best[1] if best else None

    # ------------------------------------------------------------------
    # Footprint resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_package_hint(text: str) -> str:
        """Extract a package hint like DIP-28 or TQFP-32 from free text."""
        if not isinstance(text, str):
            return ''
        t = text.strip()
        if not t:
            return ''
        m = _PACKAGE_RE.search(t)
        if m:
            return m.group(1).upper()
        # Allow "(DIP)" style without pin count.
        m2 = re.search(r'\bDIP\b', t, flags=re.IGNORECASE)
        return 'DIP' if m2 else ''

    @staticmethod
    def _extract_package_from_symbol_file(sym_path: str) -> str:
        """Read a .kicad_sym file and extract package info from properties.

        Looks for:
        - ``(property "Footprint" "Package_SO:HTSSOP-28...")`` → "HTSSOP-28"
        - ``(property "ki_description" "... TSSOP-28 ...")`` → "TSSOP-28"
        """
        try:
            p = Path(sym_path)
            if not p.is_file():
                return ''
            # Read only first 16KB — properties are near the top.
            head = p.read_bytes()[:16384].decode('utf-8', errors='ignore')
            # 1. Try Footprint property (most reliable).
            fp_m = re.search(
                r'\(property\s+"Footprint"\s+"([^"]*)"',
                head,
            )
            if fp_m:
                fp_val = fp_m.group(1)  # e.g. "Package_SO:HTSSOP-28..."
                # Extract the package token after ':'
                if ':' in fp_val:
                    fp_name = fp_val.split(':', 1)[1]
                else:
                    fp_name = fp_val
                pkg = LibraryManager._extract_package_hint(fp_name)
                if not pkg:
                    pkg = LibraryManager._extract_package_from_desc(fp_name)
                if pkg:
                    return pkg
            # 2. Try ki_description property.
            desc_m = re.search(
                r'\(property\s+"ki_description"\s+"([^"]*)"',
                head,
            )
            if desc_m:
                desc_val = desc_m.group(1)
                pkg = LibraryManager._extract_package_from_desc(desc_val)
                if not pkg:
                    pkg = LibraryManager._extract_package_hint(desc_val)
                if pkg:
                    return pkg
        except Exception:
            pass
        return ''

    @staticmethod
    def _package_family(package_hint: str) -> str:
        p = (package_hint or '').upper()
        for fam in _PACKAGE_FAMILIES:
            # Match at start-of-string with optional trailing digits/separator
            if re.match(r'^' + re.escape(fam) + r'(?:$|[\W\d_])', p):
                return fam
        return ''

    @staticmethod
    def _preferred_footprint_libs_for_package(package_hint: str) -> List[str]:
        fam = LibraryManager._package_family(package_hint)
        if fam in {'DIP', 'PDIP'}:
            return ['Package_DIP']
        if fam in {'TQFP', 'LQFP', 'QFP'}:
            return ['Package_QFP']
        if fam in {'QFN', 'UQFN', 'VQFN', 'WQFN', 'DFN', 'LFCSP'}:
            return ['Package_DFN_QFN']
        if fam in {'SOIC', 'SOP', 'SO', 'SSOP', 'LSSOP', 'TSSOP', 'HTSSOP', 'MSOP'}:
            return ['Package_SO']
        if fam in {'SOT'}:
            return ['Package_TO_SOT_SMD', 'Package_TO_SOT_THT']
        if fam in {'BGA'}:
            return ['Package_BGA']
        return []

    def resolve_best_footprint_path(self, query: str, package_hint: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Return (resolved_mpn, local_footprint_path) for the best footprint match.

        This prefers package-specific footprint libraries when a package hint is present.
        When the initial search returns only symbols (no local footprint paths),
        the method extracts the package from symbol descriptions and retries.
        """
        q = (query or '').strip()
        if not q:
            return None

        # If the user asked for a passive by value/role (not an MPN), prefer a
        # generic footprint rather than fuzzy-matching a random part number.
        if not package_hint and not self._extract_package_hint(q) and not _looks_like_mpn(q):
            mixed_kind = _infer_passive_kind_from_text(q)
            if mixed_kind != _PASSIVE_KIND_UNKNOWN:
                generic = self._suggest_generic_footprints_for_kind(mixed_kind, limit=5)
                if generic and getattr(generic[0], 'local_footprint_path', None):
                    chosen = generic[0]
                    return (getattr(chosen, 'mpn', q) or q, str(getattr(chosen, 'local_footprint_path')))

        # Guardrail: value-only queries should not resolve to arbitrary MPNs.
        # If the caller provides no package hint, prefer a generic footprint.
        v_kind, _ = _infer_value_only_kind(q)
        if v_kind != _PASSIVE_KIND_UNKNOWN and not package_hint and not self._extract_package_hint(q):
            generic = self._suggest_generic_footprints_for_kind(v_kind, limit=5)
            if generic and getattr(generic[0], 'local_footprint_path', None):
                chosen = generic[0]
                return (getattr(chosen, 'mpn', q) or q, str(getattr(chosen, 'local_footprint_path')))

        pkg = (package_hint or '').strip().upper() or self._extract_package_hint(q)
        preferred_libs = self._preferred_footprint_libs_for_package(pkg) if pkg else []
        pkg_token = pkg.lower() if pkg else ''
        expected_pins = _extract_expected_pin_count(q, pkg)

        def _candidates(search_q: str) -> List[LibraryItem]:
            try:
                return self.search_parts_sync(search_q, source=None, limit=50)
            except Exception:
                return []

        # If we have a package hint, search for that first (e.g. "DIP-28") because
        # MPN-based search rarely matches KiCad footprint names.
        search_queries: List[str] = []
        if pkg and pkg not in q.upper():
            search_queries.append(pkg)
        elif pkg:
            search_queries.append(pkg)
        search_queries.append(q)

        best: Optional[Tuple[int, LibraryItem]] = None
        # Track symbol-only results so we can extract package from descriptions.
        symbol_only_items: List[LibraryItem] = []

        for sq in search_queries:
            for item in _candidates(sq):
                fp_path = getattr(item, 'local_footprint_path', None)
                if not fp_path:
                    symbol_only_items.append(item)
                    continue

                # Rank package-specific matches higher.
                rank = 0
                name = (getattr(item, 'name', '') or '')
                lib = (getattr(item, 'category', '') or '')
                mpn = (getattr(item, 'mpn', '') or '')

                if pkg:
                    if preferred_libs and any(pl == lib for pl in preferred_libs):
                        rank += 50
                    # Also allow matching in the footprint name itself.
                    if pkg_token and re.search(r'(?:^|[\W_])' + re.escape(pkg_token) + r'(?:$|[\W_])', name.lower()):
                        rank += 25
                    elif pkg_token and pkg_token in name.lower():
                        rank += 10  # Substring match is weaker
                    # If it looks like an obviously different family, penalize.
                    fam = self._package_family(pkg)
                    name_upper = name.upper()
                    fp_name_upper = Path(fp_path).name.upper()
                    fam_pat = r'(?:^|[\W_])' + re.escape(fam) + r'(?:$|[\W_])' if fam else None
                    fam_in_name = fam_pat and (re.search(fam_pat, name_upper) or re.search(fam_pat, fp_name_upper))
                    if fam and not fam_in_name:
                        # Example: requested DIP but candidate contains DFN
                        other_fams = ['DFN', 'QFN', 'TQFP', 'LQFP', 'SOIC', 'SSOP', 'SOT', 'BGA']
                        if any(re.search(r'(?:^|[\W_])' + of + r'(?:$|[\W_])', name_upper) for of in other_fams):
                            rank -= 20

                # Prefer results coming from local KiCad libraries (already indexed) over external.
                try:
                    if getattr(getattr(item, 'source', None), 'value', '') == LibrarySource.KICAD_BUILTIN.value:
                        rank += 5
                except Exception:
                    pass

                # ── Pin-count validation ──────────────────────────
                if expected_pins > 0 and fp_path:
                    actual_pins = _count_footprint_pads(fp_path)
                    if actual_pins == 0:
                        rank -= 200
                    elif actual_pins == expected_pins:
                        rank += 30
                    elif actual_pins < expected_pins * 0.5:
                        rank -= 200
                    elif abs(actual_pins - expected_pins) <= 2:
                        rank += 10
                    else:
                        rank -= 50

                if best is None or rank > best[0]:
                    best = (rank, item)

        # If we found no footprints but did find symbol-only results, try to
        # discover the package from their descriptions, then re-search by package.
        if best is None and symbol_only_items and not pkg:
            discovered_pkgs: List[str] = []
            for sym in symbol_only_items:
                # 1. Try the item's own description / package fields.
                desc = getattr(sym, 'description', '') or ''
                p = self._extract_package_from_desc(desc)
                if not p:
                    p = self._extract_package_hint(desc)
                if not p:
                    p = (getattr(sym, 'package', '') or '').strip().upper()
                    if p and not re.match(r'^[A-Z]+-\d+$', p):
                        p = self._extract_package_hint(p) or self._extract_package_from_desc(p)
                # 2. If the item has a local symbol path, read the file to
                #    extract the Footprint property and ki_description.
                if not p:
                    sym_path = getattr(sym, 'local_symbol_path', None)
                    if sym_path:
                        p = self._extract_package_from_symbol_file(sym_path)
                if p and p not in discovered_pkgs:
                    discovered_pkgs.append(p)
            # Re-search using discovered packages.
            for dp in discovered_pkgs:
                dp_preferred = self._preferred_footprint_libs_for_package(dp)
                dp_token = dp.lower()
                for item in _candidates(dp):
                    fp_path = getattr(item, 'local_footprint_path', None)
                    if not fp_path:
                        continue
                    rank = 0
                    name = (getattr(item, 'name', '') or '')
                    lib = (getattr(item, 'category', '') or '')
                    if dp_preferred and any(pl == lib for pl in dp_preferred):
                        rank += 50
                    if dp_token and dp_token in name.lower():
                        rank += 25
                    try:
                        if getattr(getattr(item, 'source', None), 'value', '') == LibrarySource.KICAD_BUILTIN.value:
                            rank += 5
                    except Exception:
                        pass
                    # Pin-count validation for discovered-package re-search
                    dp_pins = _extract_expected_pin_count('', dp)
                    if dp_pins > 0 and fp_path:
                        actual_pins = _count_footprint_pads(fp_path)
                        if actual_pins == 0:
                            rank -= 200
                        elif actual_pins == dp_pins:
                            rank += 30
                        elif actual_pins < dp_pins * 0.5:
                            rank -= 200
                        elif abs(actual_pins - dp_pins) <= 2:
                            rank += 10
                        else:
                            rank -= 50
                    if best is None or rank > best[0]:
                        best = (rank, item)
                if best is not None:
                    break  # Found footprints for a discovered package.

        if best is None:
            return None
        chosen = best[1]
        return (getattr(chosen, 'mpn', q) or q, str(getattr(chosen, 'local_footprint_path')))

    @staticmethod
    def _score_match(name: str, tokens: List[str]) -> float:
        """Score how well *name* matches the query tokens (0.0 – 1.0).

        Each token that appears on a word boundary in *name* gets a full
        point.  A token that only appears as a substring (e.g. "dip" inside
        "cerdip") gets a partial 0.3 score.  This prevents false positives
        like CERDIP matching DIP.
        """
        if not tokens:
            return 0.0
        name_lower = name.lower()
        score = 0.0
        for t in tokens:
            # Whole-word / boundary match (handles separators like - _ . :)
            if re.search(r'(?:^|[\W_])' + re.escape(t) + r'(?:$|[\W_])', name_lower):
                score += 1.0
            elif t in name_lower:
                score += 0.3  # Substring-only match is much weaker
        return score / len(tokens)

    # ------------------------------------------------------------------
    # Local KiCad library scanning
    # ------------------------------------------------------------------

    def _detect_kicad_data_dirs(self) -> List[Path]:
        """Find KiCad data directories containing ``symbols/`` and/or ``footprints/``."""
        dirs: List[Path] = []

        def _add_dir_if_exists(p: Path) -> None:
            if p.is_dir():
                dirs.append(p)

        def _add_if_has_libs(p: Path, dest: List[Path]) -> None:
            if (p / 'symbols').is_dir() or (p / 'footprints').is_dir():
                dest.append(p)

        def _add_versioned_subdirs(p: Path, dest: List[Path]) -> None:
            try:
                for child in p.iterdir():
                    if not child.is_dir():
                        continue
                    if re.match(r'^\d+(?:\.\d+)?$', child.name):
                        _add_if_has_libs(child, dest)
            except OSError:
                return

        # 1. Environment variables (most reliable — set by KiCad itself)
        for env_var in ('KICAD8_SYMBOL_DIR', 'KICAD7_SYMBOL_DIR',
                        'KICAD_SYMBOL_DIR'):
            val = os.environ.get(env_var)
            if val and Path(val).is_dir():
                _add_dir_if_exists(Path(val).parent)

        for env_var in ('KICAD8_FOOTPRINT_DIR', 'KICAD7_FOOTPRINT_DIR',
                        'KICAD_FOOTPRINT_DIR'):
            val = os.environ.get(env_var)
            if val and Path(val).is_dir():
                _add_dir_if_exists(Path(val).parent)

        # 1b. User library path (symbols/footprints under KiCad user dir)
        try:
            user_lib = Path(self.kicad_user_lib_path)
            if user_lib.is_dir():
                _add_dir_if_exists(user_lib)
        except Exception:
            pass

        # 2. Platform-specific default install locations
        system = platform.system()
        if system == 'Darwin':
            # macOS — try multiple KiCad versions / install methods
            # Common layouts:
            # - /Applications/KiCad.app/Contents/SharedSupport/kicad
            # - /Applications/KiCad/KiCad.app/Contents/SharedSupport/kicad
            # - /Applications/KiCad*/KiCad.app/Contents/SharedSupport/kicad
            app_roots: List[Path] = []
            # Direct app bundles
            for app_name in ('KiCad', 'KiCad 8.0', 'KiCad-8.0', 'KiCad 9.0', 'KiCad-9.0'):
                app_roots.append(Path(f'/Applications/{app_name}.app'))
            # Folder-contained KiCad.app bundles (seen with some installers)
            app_roots.append(Path('/Applications/KiCad/KiCad.app'))
            # Best-effort glob for other folder names
            try:
                for p in Path('/Applications').glob('KiCad*/KiCad.app'):
                    app_roots.append(p)
            except Exception:
                pass

            seen_app: set = set()
            for app in app_roots:
                try:
                    app_resolved = app.resolve()
                except Exception:
                    app_resolved = app
                if app_resolved in seen_app:
                    continue
                seen_app.add(app_resolved)
                shared = app / 'Contents' / 'SharedSupport'
                if shared.is_dir():
                    nested = shared / 'kicad'
                    if nested.is_dir():
                        _add_dir_if_exists(nested)
                        _add_versioned_subdirs(nested, dirs)
                    _add_dir_if_exists(shared)
                    _add_versioned_subdirs(shared, dirs)
            for p in ('/usr/local/share/kicad', '/opt/homebrew/share/kicad'):
                p = Path(p)
                if p.is_dir():
                    _add_dir_if_exists(p)
                    _add_versioned_subdirs(p, dirs)
        elif system == 'Linux':
            for p in ('/usr/share/kicad', '/usr/local/share/kicad',
                      str(Path.home() / '.local' / 'share' / 'kicad')):
                p = Path(p)
                if p.is_dir():
                    _add_dir_if_exists(p)
                    _add_versioned_subdirs(p, dirs)
        elif system == 'Windows':
            for ver in ('9.0', '8.0', '7.0'):
                for base in ('C:/Program Files/KiCad',
                             'C:/Program Files (x86)/KiCad'):
                    p = Path(f'{base}/{ver}/share/kicad')
                    if p.is_dir():
                        _add_dir_if_exists(p)
                        _add_versioned_subdirs(p, dirs)

        # De-duplicate by resolved path
        seen: set = set()
        result: List[Path] = []
        expanded: List[Path] = []
        for d in list(dirs):
            _add_if_has_libs(d, expanded)
            kicad_dir = d / 'kicad'
            if kicad_dir.is_dir():
                _add_if_has_libs(kicad_dir, expanded)
                _add_versioned_subdirs(kicad_dir, expanded)
            _add_versioned_subdirs(d, expanded)

        for d in expanded or dirs:
            try:
                resolved = d.resolve()
            except OSError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                result.append(d)
        return result

    def _build_local_index(self) -> List[Tuple[str, str, str, str, Optional[str]]]:
        """Build a searchable index of local KiCad symbols and footprints.

        Returns a list of ``(name_lower, name, library, entry_type)`` tuples
        where *entry_type* is ``"symbol"`` or ``"footprint"``.

        The index is cached in ``self._local_index`` for the session lifetime.
        """
        if self._local_index is not None:
            return self._local_index

        entries: List[Tuple[str, str, str, str, Optional[str]]] = []
        data_dirs = self._detect_kicad_data_dirs()
        if not data_dirs:
            logger.debug("No local KiCad data directories found")
            self._local_index = entries
            return entries

        # 1. Symbols — parse .kicad_sym files
        sym_re = re.compile(r'\(symbol\s+"([^"]+)"')
        # Sub-symbols follow the pattern ParentName_N_N — filter them out.
        sub_sym_re = re.compile(r'_\d+_\d+$')

        for data_dir in data_dirs:
            sym_dir = data_dir / 'symbols'
            if sym_dir.is_dir():
                for sym_file in sorted(sym_dir.glob('*.kicad_sym')):
                    lib_name = sym_file.stem  # e.g. "Battery"
                    try:
                        text = sym_file.read_text(encoding='utf-8', errors='ignore')
                    except OSError:
                        continue
                    # Collect all symbol names, skip sub-symbol variants
                    seen_syms: set = set()
                    for m in sym_re.finditer(text):
                        sym_name = m.group(1)
                        if sub_sym_re.search(sym_name):
                            continue
                        if sym_name not in seen_syms:
                            seen_syms.add(sym_name)
                            # Include library name in search index for better context matching
                            search_text = f"{lib_name} {sym_name}".lower()
                            entries.append(
                                (search_text, sym_name, lib_name, 'symbol', str(sym_file))
                            )

            # 2. Footprints — list .kicad_mod files in .pretty dirs
            fp_dir = data_dir / 'footprints'
            if fp_dir.is_dir():
                for pretty_dir in sorted(fp_dir.glob('*.pretty')):
                    lib_name = pretty_dir.stem  # e.g. "Battery"
                    for mod_file in sorted(pretty_dir.glob('*.kicad_mod')):
                        fp_name = mod_file.stem  # e.g. "BatteryHolder_Keystone_590"
                        # Include library name in search index
                        search_text = f"{lib_name} {fp_name}".lower()
                        entries.append(
                            (search_text, fp_name, lib_name, 'footprint', str(mod_file))
                        )

        logger.info("Local KiCad index: %d symbols + footprints from %d data dirs",
                     len(entries), len(data_dirs))
        self._local_index = entries
        return entries

    def _search_kicad_local(self, query: str, limit: int) -> List[LibraryItem]:
        """Search locally installed KiCad libraries using token-based matching.

        Tokenises the query (e.g. ``"Keystone 590"`` → ``["keystone","590"]``)
        and scores every symbol/footprint name by the fraction of tokens that
        appear in it.  Results are sorted best-first and de-duplicated so that
        a matching footprint and its companion symbol both appear.
        """
        index = self._build_local_index()
        if not index:
            return []

        tokens = self._tokenize_query(query)
        if not tokens:
            return []

        # Also try the MPN-normalised variants as extra token sets
        extra_tokens_sets: List[List[str]] = []
        for variant in self._normalize_query(query):
            vt = self._tokenize_query(variant)
            if vt != tokens:
                extra_tokens_sets.append(vt)

        # Score every entry globally (no deterministic keyword/prefix filtering).
        scored: List[Tuple[float, int, Tuple[str, str, str, str, Optional[str]]]] = []
        for idx, entry in enumerate(index):
            name_lower, name, lib, etype, file_path = entry
            score = self._score_match(name_lower, tokens)
            # Try alternative token sets and keep the best score
            for alt in extra_tokens_sets:
                alt_score = self._score_match(name_lower, alt)
                if alt_score > score:
                    score = alt_score
            if score > 0:
                scored.append((score, idx, entry))

        if not scored:
            return []

        # Sort: highest score first, then shorter name for ties (shorter = more precise)
        scored.sort(key=lambda x: (-x[0], len(x[2][1]), x[2][1]))

        # Build LibraryItem results, de-duplicating by (library, name)
        seen: set = set()
        results: List[LibraryItem] = []
        for score, _, (name_lower, name, lib, etype, file_path) in scored:
            key = (lib, name)
            if key in seen:
                continue
            seen.add(key)

            item = LibraryItem(
                name=f"{lib}:{name}",
                manufacturer='(KiCad built-in)',
                mpn=name,
                description=f"{etype.title()} from KiCad {lib} library",
                source=LibrarySource.KICAD_BUILTIN,
                category=lib,
                package='',
            )
            # For local search results, attach the on-disk path so callers can load/place footprints.
            if etype == 'footprint' and file_path:
                item.local_footprint_path = file_path
            elif etype == 'symbol' and file_path:
                item.local_symbol_path = file_path
            results.append(item)
            if len(results) >= limit:
                break

        if results:
            logger.info("Local KiCad search for %r returned %d results "
                        "(top: %s, score=%.2f)",
                        query, len(results), results[0].name, scored[0][0])
        return results

    def _guess_kicad_libraries(self, query: str) -> List[str]:
        """Return candidate KiCad library names for a query.

        Deterministic keyword/prefix guessing is intentionally disabled. If you
        want to constrain the search, have the LLM provide an explicit
        "LibName:SymbolName" or "LibName:FootprintName" identifier instead.
        """
        return []

    # ------------------------------------------------------------------
    # KiCad built-in library search (kicad.github.io)
    # ------------------------------------------------------------------

    def _search_kicad_builtin_sync(self, query: str, limit: int) -> List[LibraryItem]:
        """Search the official KiCad libraries via kicad.github.io."""
        variants = self._normalize_query(query)
        candidate_libs = self._guess_kicad_libraries(query)

        if not candidate_libs:
            logger.debug("No KiCad library mapping for query %r", query)
            return []

        all_results: List[LibraryItem] = []

        # KiCad's embedded Python on macOS can lack a usable certificate store,
        # causing HTTPS fetches to fail with CERTIFICATE_VERIFY_FAILED. For the
        # public, read-only KiCad symbol index pages, fall back to an unverified
        # context when verification fails.
        verified_ctx: Optional[ssl.SSLContext] = None
        unverified_ctx: Optional[ssl.SSLContext] = None
        try:
            cafile = ""
            try:
                import certifi  # type: ignore

                cafile = certifi.where() or ""
            except Exception:
                cafile = ""

            verified_ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
        except Exception:
            verified_ctx = None
        try:
            unverified_ctx = ssl._create_unverified_context()
        except Exception:
            unverified_ctx = None

        fetched_any = False
        fetch_errors: List[str] = []

        for lib_name in candidate_libs:
            if len(all_results) >= limit:
                break
            def _looks_like_cert_error(exc: Exception) -> bool:
                # Depending on the Python build, cert failures can surface as
                # SSLCertVerificationError, SSLError, or URLError wrapping one.
                try:
                    if isinstance(exc, ssl.SSLCertVerificationError):
                        return True
                except Exception:
                    pass
                try:
                    reason = getattr(exc, 'reason', None)
                    if reason is not None:
                        if isinstance(reason, ssl.SSLCertVerificationError):
                            return True
                        if isinstance(reason, ssl.SSLError) and 'CERTIFICATE_VERIFY_FAILED' in str(reason):
                            return True
                except Exception:
                    pass
                return 'CERTIFICATE_VERIFY_FAILED' in str(exc)

            try:
                url = f"https://kicad.github.io/symbols/{lib_name}"
                req = Request(url, headers={'User-Agent': 'VibeCAD/0.4.0'})
                try:
                    if verified_ctx is None:
                        with urlopen(req, timeout=15) as resp:
                            html = resp.read().decode('utf-8')
                            fetched_any = True
                    else:
                        with urlopen(req, timeout=15, context=verified_ctx) as resp:
                            html = resp.read().decode('utf-8')
                            fetched_any = True
                except Exception as exc:
                    if (unverified_ctx is None) or (not _looks_like_cert_error(exc)):
                        raise

                    with urlopen(req, timeout=15, context=unverified_ctx) as resp:
                        html = resp.read().decode('utf-8')
                        fetched_any = True
                        logger.warning(
                            "KiCad built-in index fetch used unverified SSL context (cert verify failed): %s",
                            exc,
                        )
            except Exception as exc:
                fetch_errors.append(f"{lib_name}: {exc}")
                logger.debug("Failed to fetch KiCad lib page %s: %s", lib_name, exc)
                continue

            # Parse HTML table rows: <td>SymbolName</td> <td>Description: ...</td>
            row_re = re.compile(
                r'<td[^>]*>\s*([A-Za-z0-9][^<]{1,60}?)\s*</td>'
                r'\s*<td[^>]*>\s*Description:\s*([^<]+)',
                re.DOTALL,
            )

            seen: set = set()
            for match in row_re.finditer(html):
                sym_name = match.group(1).strip()
                desc = match.group(2).strip()

                sym_lower = sym_name.lower()
                # Check each variant
                matched = False
                for variant in variants:
                    vl = variant.lower()
                    if vl in sym_lower or sym_lower.startswith(vl):
                        matched = True
                        break
                if not matched:
                    continue
                if sym_name in seen:
                    continue
                seen.add(sym_name)

                pkg = self._extract_package_from_desc(desc)
                item = LibraryItem(
                    name=f"{lib_name}:{sym_name}",
                    manufacturer='(KiCad built-in)',
                    mpn=sym_name,
                    description=desc[:200],
                    source=LibrarySource.KICAD_BUILTIN,
                    category=lib_name,
                    package=pkg,
                )
                all_results.append(item)
                if len(all_results) >= limit:
                    break

        if all_results:
            logger.info("KiCad built-in search for %r returned %d results", query, len(all_results))
        elif candidate_libs and not fetched_any and fetch_errors:
            # Nothing could be fetched; log and fall back without failing the search.
            logger.warning(
                "KiCad built-in index fetch failed (falling back). Details: %s",
                fetch_errors[0],
            )
            return []
        return all_results

    @staticmethod
    def _extract_package_from_desc(desc: str) -> str:
        """Pull the package designator out of a KiCad description string."""
        m = _PACKAGE_RE.search(desc)
        return m.group(1) if m else ''
    
    def _search_snapeda_sync(self, query: str, limit: int) -> List[LibraryItem]:
        """Search SnapEDA for parts (synchronous)."""
        try:
            url = f"{self.SNAPEDA_SEARCH_URL}?q={quote_plus(query)}&limit={limit}"
            
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'VibeCAD/0.4.0',
            }
            
            if self.snapeda_api_key:
                headers['Authorization'] = f'Token {self.snapeda_api_key}'
            
            request = Request(url, headers=headers)

            def _looks_like_cert_error(exc: Exception) -> bool:
                try:
                    if isinstance(exc, ssl.SSLCertVerificationError):
                        return True
                except Exception:
                    pass
                try:
                    reason = getattr(exc, 'reason', None)
                    if reason is not None:
                        if isinstance(reason, ssl.SSLCertVerificationError):
                            return True
                        if isinstance(reason, ssl.SSLError) and 'CERTIFICATE_VERIFY_FAILED' in str(reason):
                            return True
                except Exception:
                    pass
                return 'CERTIFICATE_VERIFY_FAILED' in str(exc)

            cafile = (
                os.environ.get("VIBECAD_CA_BUNDLE", "").strip()
                or os.environ.get("REQUESTS_CA_BUNDLE", "").strip()
                or os.environ.get("SSL_CERT_FILE", "").strip()
            )
            if not cafile:
                try:
                    import certifi  # type: ignore

                    cafile = certifi.where() or ""
                except Exception:
                    cafile = ""
            verified_ctx: Optional[ssl.SSLContext]
            try:
                verified_ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
            except Exception:
                verified_ctx = None
            try:
                unverified_ctx = ssl._create_unverified_context()
            except Exception:
                unverified_ctx = None
            allow_insecure = bool(os.environ.get("VIBECAD_SSL_NO_VERIFY", "").strip())

            try:
                if verified_ctx is None:
                    with urlopen(request, timeout=15) as response:
                        data = json.loads(response.read().decode('utf-8'))
                else:
                    with urlopen(request, timeout=15, context=verified_ctx) as response:
                        data = json.loads(response.read().decode('utf-8'))
            except Exception as exc:
                if allow_insecure and unverified_ctx is not None and _looks_like_cert_error(exc):
                    logger.warning(
                        "SnapEDA search used unverified SSL context (cert verify failed). "
                        "Set VIBECAD_CA_BUNDLE/SSL_CERT_FILE or install certifi to avoid this."
                    )
                    with urlopen(request, timeout=15, context=unverified_ctx) as response:
                        data = json.loads(response.read().decode('utf-8'))
                else:
                    raise

            # SnapEDA returns {"error": "not logged in"} without auth
            if data.get('error'):
                err_msg = data['error']
                if 'not logged in' in err_msg.lower() or 'auth' in err_msg.lower():
                    logger.warning(
                        "SnapEDA API requires authentication. "
                        "Set SNAPEDA_API_KEY env var or request an API key "
                        "at https://www.snapeda.com/get-api/"
                    )
                else:
                    logger.warning("SnapEDA API error: %s", err_msg)
                return []
            
            results = []
            for item in data.get('results', []):
                lib_item = LibraryItem(
                    name=item.get('part_number', ''),
                    manufacturer=item.get('manufacturer', {}).get('name', ''),
                    mpn=item.get('part_number', ''),
                    description=item.get('short_description', ''),
                    source=LibrarySource.SNAPEDA,
                    symbol_url=item.get('_links', {}).get('symbol', {}).get('href'),
                    footprint_url=item.get('_links', {}).get('footprint', {}).get('href'),
                    model_3d_url=item.get('_links', {}).get('models', {}).get('href'),
                    datasheet_url=item.get('_links', {}).get('datasheet', {}).get('href'),
                    category=item.get('category', {}).get('name', ''),
                    package=item.get('package', ''),
                )
                results.append(lib_item)
            
            return results
            
        except Exception as e:
            logger.exception(f"SnapEDA search failed: {e}")
            return []
    
    async def _search_snapeda(self, query: str, limit: int) -> List[LibraryItem]:
        """Search SnapEDA for parts (async)."""
        # For now, delegate to sync version
        # TODO: Use aiohttp for true async
        return self._search_snapeda_sync(query, limit)
    
    def download_item(self, item: LibraryItem,
                     install: bool = False,
                     project_dir: Optional[str] = None) -> DownloadResult:
        """Download a library item.
        
        Args:
            item: The LibraryItem to download
            install: Whether to install to KiCad library path
        
        Returns:
            DownloadResult with paths or error
        """
        # KiCad built-in parts need no download
        if item.source == LibrarySource.KICAD_BUILTIN:
            return DownloadResult(
                success=True,
                item=item,
                message=(
                    f"✅ {item.mpn} is available in KiCad's built-in library.\n"
                    f"Library reference: {item.name}\n"
                    f"Package: {item.package}\n\n"
                    f"No download needed — open the symbol chooser in the "
                    f"schematic editor and search for \"{item.mpn}\"."
                ),
                symbol_path=item.name,  # e.g. "MCU_Microchip_ATmega:ATmega328P-P"
            )

        temp_dir = tempfile.mkdtemp(prefix="vibecad_lib_")
        symbol_path: Optional[str] = None
        footprint_path: Optional[str] = None
        
        try:
            # Obtain symbol (either local cached path or download URL)
            if getattr(item, 'local_symbol_path', None) and Path(str(item.local_symbol_path)).exists():
                symbol_path = str(item.local_symbol_path)
            elif item.symbol_url:
                symbol_path = self._download_file(
                    item.symbol_url,
                    temp_dir,
                    f"{item.mpn}.kicad_sym"
                )
            
            # Obtain footprint (either local cached path or download URL)
            if getattr(item, 'local_footprint_path', None) and Path(str(item.local_footprint_path)).exists():
                footprint_path = str(item.local_footprint_path)
            elif item.footprint_url:
                footprint_path = self._download_file(
                    item.footprint_url,
                    temp_dir,
                    f"{item.mpn}.kicad_mod"
                )
            
            # Install to KiCad library if requested
            if install and self.kicad_user_lib_path:
                if symbol_path:
                    installed_sym = self._install_symbol(symbol_path, item.mpn)
                    if installed_sym:
                        symbol_path = installed_sym
                
                if footprint_path:
                    installed_fp = self._install_footprint(footprint_path, item.mpn)
                    if installed_fp:
                        footprint_path = installed_fp

                # Update project library tables if a project directory is provided.
                if project_dir:
                    try:
                        from .kicad_library_tables import ensure_project_tables
                        ensure_project_tables(
                            project_dir=project_dir,
                            footprint_lib_dir=os.path.join(self.kicad_user_lib_path, 'footprints', 'VibeCAD.pretty'),
                            symbol_lib_paths=[symbol_path] if symbol_path else [],
                        )
                    except Exception as e:
                        logger.debug("Failed to update KiCad library tables: %s", e)
            
            item.is_downloaded = True
            item.local_symbol_path = symbol_path
            item.local_footprint_path = footprint_path
            
            return DownloadResult(
                success=True,
                item=item,
                message=f"Downloaded {item.mpn} from {item.source.value}",
                symbol_path=symbol_path,
                footprint_path=footprint_path,
            )
            
        except Exception as e:
            logger.exception(f"Download failed for {item.mpn}")
            return DownloadResult(
                success=False,
                item=item,
                message=f"Download failed: {e}",
                error=str(e),
            )
    
    def _download_file(self, url: str, dest_dir: str, filename: str) -> Optional[str]:
        """Download a file from URL."""
        try:
            headers = {'User-Agent': 'VibeCAD/0.4.0'}
            
            if self.snapeda_api_key and 'snapeda.com' in url:
                headers['Authorization'] = f'Token {self.snapeda_api_key}'
            
            request = Request(url, headers=headers)
            
            dest_path = os.path.join(dest_dir, filename)
            
            with urlopen(request, timeout=30) as response:
                content = response.read()
                
                # Check if it's a zip file
                if content[:4] == b'PK\x03\x04':
                    zip_path = dest_path + '.zip'
                    with open(zip_path, 'wb') as f:
                        f.write(content)
                    
                    # Extract the zip
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(dest_dir)
                    
                    # Find the extracted file
                    for f in os.listdir(dest_dir):
                        if f.endswith(('.kicad_sym', '.kicad_mod', '.lib', '.pretty')):
                            return os.path.join(dest_dir, f)
                else:
                    with open(dest_path, 'wb') as f:
                        f.write(content)
                    return dest_path
            
            return None
            
        except Exception as e:
            logger.exception(f"File download failed: {e}")
            return None
    
    def _install_symbol(self, source_path: str, name: str) -> Optional[str]:
        """Install a symbol to the KiCad library."""
        try:
            symbols_dir = os.path.join(self.kicad_user_lib_path, "symbols")
            os.makedirs(symbols_dir, exist_ok=True)
            
            dest_path = os.path.join(symbols_dir, f"VibeCAD_{name}.kicad_sym")
            
            import shutil
            shutil.copy2(source_path, dest_path)
            
            logger.info(f"Installed symbol to {dest_path}")
            return dest_path
            
        except Exception as e:
            logger.exception(f"Symbol installation failed: {e}")
            return None
    
    def _install_footprint(self, source_path: str, name: str) -> Optional[str]:
        """Install a footprint to the KiCad library."""
        try:
            footprints_dir = os.path.join(self.kicad_user_lib_path, "footprints", "VibeCAD.pretty")
            os.makedirs(footprints_dir, exist_ok=True)
            
            dest_path = os.path.join(footprints_dir, f"{name}.kicad_mod")
            
            import shutil
            shutil.copy2(source_path, dest_path)
            
            logger.info(f"Installed footprint to {dest_path}")
            return dest_path
            
        except Exception as e:
            logger.exception(f"Footprint installation failed: {e}")
            return None
    
    def get_installed_libraries(self) -> Dict[str, List[str]]:
        """Get list of installed VibeCAD libraries."""
        result = {'symbols': [], 'footprints': []}
        
        try:
            symbols_dir = os.path.join(self.kicad_user_lib_path, "symbols")
            if os.path.exists(symbols_dir):
                for f in os.listdir(symbols_dir):
                    if f.startswith("VibeCAD_") and f.endswith(".kicad_sym"):
                        result['symbols'].append(f)
            
            footprints_dir = os.path.join(self.kicad_user_lib_path, "footprints", "VibeCAD.pretty")
            if os.path.exists(footprints_dir):
                for f in os.listdir(footprints_dir):
                    if f.endswith(".kicad_mod"):
                        result['footprints'].append(f)
        except Exception as e:
            logger.exception(f"Error listing libraries: {e}")
        
        return result
    
    def create_preview_summary(self, item: LibraryItem) -> str:
        """Create a human-readable preview of what will be installed."""
        # KiCad built-in — nothing to download
        if item.source == LibrarySource.KICAD_BUILTIN:
            lines = [
                f"📦 KiCad Built-in Library Part",
                f"",
                f"Part: {item.mpn}",
                f"Library: {item.name}",
                f"Description: {item.description}",
            ]
            if item.package:
                lines.append(f"Package: {item.package}")
            lines += [
                f"",
                f"✅ This part ships with KiCad — no download needed.",
                f"Open the symbol chooser and search for \"{item.mpn}\".",
            ]
            return "\n".join(lines)

        lines = [
            f"📦 Library Download Preview",
            f"",
            f"Part: {item.name}",
            f"Manufacturer: {item.manufacturer}",
            f"MPN: {item.mpn}",
            f"Source: {item.source.value}",
            f"Category: {item.category}",
            f"Package: {item.package}",
            f"",
            f"Will download:",
        ]
        
        if item.symbol_url:
            lines.append(f"  ✓ Symbol (.kicad_sym)")
        if item.footprint_url:
            lines.append(f"  ✓ Footprint (.kicad_mod)")
        if item.model_3d_url:
            lines.append(f"  ✓ 3D Model")
        
        lines.append(f"")
        lines.append(f"Install location: {self.kicad_user_lib_path}")
        
        if item.datasheet_url:
            lines.append(f"")
            lines.append(f"📄 Datasheet: {item.datasheet_url}")
        
        return "\n".join(lines)
