"""
Autorouter integration — delegates routing to Freerouting via DSN/SES export/import.

Provides:
  - DSN export from current board
  - Freerouting invocation (headless, subprocess)
  - SES import of routed results
  - Optional grid-based A* fallback autorouter with obstacle avoidance
"""

import heapq
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import pcbnew
    PCBNEW_AVAILABLE = True
except ImportError:
    PCBNEW_AVAILABLE = False


@dataclass
class AutorouteResult:
    """Result of an autorouting attempt."""
    success: bool
    message: str
    method: str = ""        # "freerouting", "manhattan", or "direct"
    manual_required: bool = False
    dsn_path: str = ""
    tracks_added: int = 0
    vias_added: int = 0
    unrouted_remaining: int = 0


def _common_freerouting_locations() -> List[Path]:
    roots = [
        Path.home() / ".vibecad" / "freerouting.jar",
        Path.home() / "Library" / "Application Support" / "kicad" / "scripting" / "plugins",
        Path.home() / ".local" / "share" / "kicad" / "scripting" / "plugins",
        Path.home() / ".kicad_plugins",
    ]
    roots.extend(_kicad_third_party_roots())

    deduped: List[Path] = []
    seen: Set[str] = set()
    for path in roots:
        key = str(path)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _iter_version_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    try:
        children = [child for child in root.iterdir() if child.is_dir()]
    except Exception:
        return []
    return sorted(children, key=lambda path: path.name, reverse=True)


def _kicad_third_party_roots() -> List[Path]:
    roots: List[Path] = []
    for base in (
        Path.home() / "Documents" / "KiCad",
        Path.home() / ".local" / "share" / "kicad",
        Path(os.environ.get("DOCUMENTS", Path.home() / "Documents")) / "KiCad",
    ):
        for version_dir in _iter_version_dirs(base):
            roots.append(version_dir / "3rdparty")
            roots.append(version_dir / "3rdparty" / "plugins")

    import platform
    if platform.system() == "Windows":
        prefs_root = Path(os.environ.get("APPDATA", "")) / "kicad"
    else:
        prefs_root = Path.home() / "Library" / "Preferences" / "kicad"
        
    for version_dir in _iter_version_dirs(prefs_root):
        pcbnew_json = version_dir / "pcbnew.json"
        if not pcbnew_json.is_file():
            continue
        try:
            data = json.loads(pcbnew_json.read_text())
        except Exception:
            continue
        for row in list(data.get("action_plugins") or []):
            if not isinstance(row, dict):
                continue
            for key in row.keys():
                raw = str(key or "")
                if "freerouting" not in raw.lower():
                    continue
                plugin_path = raw.split("/plugin.py/", 1)[0].strip()
                if plugin_path:
                    roots.append(Path(plugin_path))
                    roots.append(Path(plugin_path).parent)
        roots.append(Path.home() / "Documents" / "KiCad" / version_dir.name / "3rdparty")
        roots.append(Path.home() / "Documents" / "KiCad" / version_dir.name / "3rdparty" / "plugins")

    return roots


def _kicad_pcm_stale_hint() -> str:
    import platform
    if platform.system() == "Windows":
        prefs_root = Path(os.environ.get("APPDATA", "")) / "kicad"
    else:
        prefs_root = Path.home() / "Library" / "Preferences" / "kicad"
        
    stale_paths: List[str] = []
    if not prefs_root.is_dir():
        return ""
    for version_dir in _iter_version_dirs(prefs_root):
        pcbnew_json = version_dir / "pcbnew.json"
        if not pcbnew_json.is_file():
            continue
        try:
            data = json.loads(pcbnew_json.read_text())
        except Exception:
            continue
        for row in list(data.get("action_plugins") or []):
            if not isinstance(row, dict):
                continue
            for key in row.keys():
                raw = str(key or "")
                if "freerouting" not in raw.lower():
                    continue
                plugin_path = raw.split("/plugin.py/", 1)[0].strip()
                if plugin_path and (not Path(plugin_path).exists()):
                    stale_paths.append(plugin_path)
    if not stale_paths:
        return ""
    unique = sorted(set(stale_paths))
    return (
        "KiCad PCM metadata references Freerouting, but the plugin payload is missing at "
        f"{unique[0]}. Reinstall the Freerouting PCM package in KiCad."
    )


def _find_freerouting_jar() -> Optional[str]:
    """Locate a Freerouting executable or JAR.

    Search order:
      1. FREEROUTING_JAR environment variable
      2. ~/.vibecad/freerouting.jar
      3. common KiCad plugin directories
      4. 'freerouting' on PATH
    """
    env_jar = os.environ.get("FREEROUTING_JAR")
    if env_jar and Path(env_jar).is_file():
        return env_jar

    for candidate in _common_freerouting_locations():
        if candidate.is_file():
            return str(candidate)
        if not candidate.is_dir():
            continue
        for pattern in ("*freerouting*.jar", "*Freerouting*.jar"):
            for match in candidate.rglob(pattern):
                if match.is_file():
                    return str(match)
        for pattern in ("*freerouting*.app", "*Freerouting*.app"):
            for match in candidate.rglob(pattern):
                launcher = match / "Contents" / "MacOS" / match.stem
                if launcher.is_file() and os.access(launcher, os.X_OK):
                    return str(launcher)
        for pattern in ("*freerouting*", "*Freerouting*"):
            for match in candidate.rglob(pattern):
                if not match.is_file():
                    continue
                if match.suffix.lower() in {".py", ".json", ".md", ".txt", ".zip"}:
                    continue
                if os.access(match, os.X_OK):
                    return str(match)

    fr = shutil.which("freerouting")
    if fr:
        return fr
    
    import platform
    if platform.system() == "Windows":
        fr_exe = shutil.which("freerouting.exe")
        if fr_exe:
            return fr_exe

    return None


def _java_available() -> bool:
    """Check if Java runtime is available."""
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _java_runtime_major_version() -> Optional[int]:
    """Return installed Java major version (e.g. 17, 21), if detectable."""
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None

    text = "\n".join(
        [
            str(result.stderr or "").strip(),
            str(result.stdout or "").strip(),
        ]
    )
    if not text.strip():
        return None

    import re

    token = ""
    m = re.search(r'version\s+"([^"]+)"', text)
    if m:
        token = str(m.group(1) or "").strip()
    if not token:
        m = re.search(r"\b(\d+(?:\.\d+){0,2})\b", text)
        if m:
            token = str(m.group(1) or "").strip()
    if not token:
        return None

    if token.startswith("1."):
        parts = token.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
    first = token.split(".", 1)[0].strip()
    if first.isdigit():
        return int(first)
    return None


def _class_major_to_java_major(class_major: int) -> Optional[int]:
    if class_major < 45:
        return None
    return int(class_major - 44)


def _class_major_from_bytes(blob: bytes) -> Optional[int]:
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 8:
        return None
    if blob[0:4] != b"\xca\xfe\xba\xbe":
        return None
    try:
        return int.from_bytes(blob[6:8], "big")
    except Exception:
        return None


def _jar_required_java_major(jar_path: str) -> Optional[int]:
    """Estimate minimum Java major required by a jar's class files."""
    try:
        with zipfile.ZipFile(jar_path, "r") as zf:
            names = list(zf.namelist())
            if not names:
                return None

            candidates: List[str] = []
            try:
                manifest = zf.read("META-INF/MANIFEST.MF").decode("utf-8", errors="ignore")
            except Exception:
                manifest = ""
            if manifest:
                import re

                m = re.search(r"(?mi)^Main-Class:\s*([^\r\n]+)\s*$", manifest)
                if m:
                    main_class = str(m.group(1) or "").strip()
                    if main_class:
                        candidates.append(main_class.replace(".", "/") + ".class")

            if not candidates:
                candidates = [
                    name
                    for name in names
                    if name.endswith(".class") and (not name.startswith("META-INF/versions/"))
                ]
            if not candidates:
                candidates = [name for name in names if name.endswith(".class")]
            if not candidates:
                return None

            max_major = 0
            for name in candidates:
                try:
                    major = _class_major_from_bytes(zf.read(name)[:8])
                except Exception:
                    major = None
                if major is None:
                    continue
                max_major = max(max_major, int(major))
            if max_major <= 0:
                return None
            return _class_major_to_java_major(max_major)
    except Exception:
        return None


def _freerouting_java_compatibility_issue(jar_path: str) -> str:
    runtime_major = _java_runtime_major_version()
    if runtime_major is None:
        return "Java runtime version could not be detected. Install Java to use Freerouting."
    required_major = _jar_required_java_major(jar_path)
    if required_major is None:
        return ""
    if runtime_major < required_major:
        return (
            f"Installed Java runtime is {runtime_major}, but Freerouting requires Java {required_major}+ "
            f"for {Path(jar_path).name}."
        )
    return ""


def export_dsn(board_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
    """Export the current board to Specctra DSN format.

    Args:
        board_path: Path to the .kicad_pcb file.
        output_path: Optional output DSN path. Defaults to <board>.dsn.

    Returns:
        (success, dsn_path_or_error)
    """
    if not PCBNEW_AVAILABLE:
        return False, "pcbnew not available"

    if not output_path:
        output_path = str(Path(board_path).with_suffix(".dsn"))

    try:
        board = pcbnew.LoadBoard(board_path)
        if board is None:
            return False, "Failed to load board"

        export_fn = getattr(pcbnew, "ExportSpecctraDSN", None)
        if not callable(export_fn):
            return False, "pcbnew.ExportSpecctraDSN not available in this KiCad version"

        export_fn(board, output_path)

        if Path(output_path).exists():
            return True, output_path
        return False, "DSN export produced no file"

    except Exception as e:
        return False, f"DSN export failed: {e}"


def import_ses(board_path: str, ses_path: str) -> Tuple[bool, str]:
    """Import Specctra SES (routed) file back into the board.

    Args:
        board_path: Path to the .kicad_pcb file.
        ses_path: Path to the .ses file.

    Returns:
        (success, message)
    """
    if not PCBNEW_AVAILABLE:
        return False, "pcbnew not available"

    try:
        import_fn = getattr(pcbnew, "ImportSpecctraSES", None)
        if not callable(import_fn):
            return False, "pcbnew.ImportSpecctraSES not available in this KiCad version"

        import_fn(ses_path)
        return True, "SES import successful"

    except Exception as e:
        return False, f"SES import failed: {e}"


def run_freerouting(dsn_path: str, timeout_seconds: int = 300) -> Tuple[bool, str]:
    """Run Freerouting on a DSN file in headless mode.

    Args:
        dsn_path: Path to the .dsn file.
        timeout_seconds: Max time for routing (default 5 minutes).

    Returns:
        (success, ses_path_or_error)
    """
    jar = _find_freerouting_jar()
    if not jar:
        return False, "Freerouting not found. Install it at ~/.vibecad/freerouting.jar or set FREEROUTING_JAR."

    ses_path = str(Path(dsn_path).with_suffix(".ses"))
    ses_file = Path(ses_path)

    # Prevent stale session files from prior runs from being mistaken as new output.
    try:
        ses_file.unlink(missing_ok=True)
    except Exception:
        pass

    launch_time = time.time()

    # Determine if it's a JAR or executable
    if jar.endswith(".jar"):
        if not _java_available():
            return False, "Java runtime not found. Install Java to use Freerouting."
        compat_issue = _freerouting_java_compatibility_issue(jar)
        if compat_issue:
            return False, compat_issue
        cmd = ["java", "-jar", jar, "-de", dsn_path, "-do", ses_path, "-mp", "20"]
    else:
        cmd = [jar, "-de", dsn_path, "-do", ses_path, "-mp", "20"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if ses_file.exists() and ses_file.stat().st_size > 0:
            return True, ses_path
        elif result.returncode == 0:
            return False, f"Freerouting completed but no SES file produced. stdout: {result.stdout[:500]}"
        else:
            return False, f"Freerouting failed (rc={result.returncode}): {result.stderr[:500]}"

    except subprocess.TimeoutExpired:
        # Some Freerouting builds can emit SES and then linger waiting for UI
        # shutdown; treat a fresh, non-empty SES as a successful route.
        try:
            if ses_file.exists() and ses_file.stat().st_size > 0 and ses_file.stat().st_mtime >= (launch_time - 1.0):
                return True, ses_path
        except Exception:
            pass
        return False, f"Freerouting timed out after {timeout_seconds}s"
    except Exception as e:
        return False, f"Freerouting execution failed: {e}"


def _manual_freerouting_message(reason: str, board_path: str, dsn_path: str) -> str:
    board_name = Path(board_path).name or board_path
    return (
        f"{reason}\n"
        f"Manual Freerouting handoff:\n"
        f"1. Open or keep the board loaded in KiCad: {board_name}\n"
        f"2. Use the exported DSN file: {dsn_path}\n"
        f"3. Run Freerouting manually on that DSN and generate a routed .ses session file.\n"
        f"4. Import the resulting .ses back into KiCad for the same board.\n"
        "To enable automatic Freerouting in VibeCAD later, set FREEROUTING_JAR, "
        "place the jar at ~/.vibecad/freerouting.jar, or install a `freerouting` executable on PATH."
    )


def autoroute(
    board_path: str,
    timeout_seconds: int = 300,
    allow_manhattan_fallback: bool = False,
) -> AutorouteResult:
    """Full autoroute pipeline: export DSN → run Freerouting → import SES.

    Args:
        board_path: Path to the .kicad_pcb file.
        timeout_seconds: Max time for Freerouting.
        allow_manhattan_fallback: Whether to fall back to the internal router.

    Returns:
        AutorouteResult with success status and details.
    """
    ok, dsn_path = export_dsn(board_path)
    if not ok:
        return AutorouteResult(
            success=False,
            message=f"DSN export failed: {dsn_path}",
            method="freerouting",
        )
    logger.info("AUTOROUTE: exported DSN %s for %s", dsn_path, Path(board_path).name or board_path)

    jar = _find_freerouting_jar()
    if not jar:
        stale_hint = _kicad_pcm_stale_hint()
        reason = "Freerouting was not found on this system."
        if stale_hint:
            reason = f"{reason} {stale_hint}"
        logger.info("AUTOROUTE: Freerouting not installed; leaving manual DSN handoff at %s", dsn_path)
        manual_msg = _manual_freerouting_message(
            reason,
            board_path,
            dsn_path,
        )
        if not allow_manhattan_fallback:
            return AutorouteResult(
                success=False,
                message=manual_msg,
                method="freerouting",
                manual_required=True,
                dsn_path=dsn_path,
            )
        logger.warning("Freerouting not found; falling back to internal router.")
        try:
            Path(dsn_path).unlink(missing_ok=True)
        except Exception:
            pass
        return _grid_route_all(board_path)

    if jar.endswith(".jar"):
        if not _java_available():
            logger.info("AUTOROUTE: Freerouting jar found but Java is unavailable; manual DSN handoff at %s", dsn_path)
            manual_msg = _manual_freerouting_message(
                "Freerouting was found but Java is not available to run the jar.",
                board_path,
                dsn_path,
            )
            if not allow_manhattan_fallback:
                return AutorouteResult(
                    success=False,
                    message=manual_msg,
                    method="freerouting",
                    manual_required=True,
                    dsn_path=dsn_path,
                )
            logger.warning("Java unavailable for Freerouting; falling back to internal router.")
            try:
                Path(dsn_path).unlink(missing_ok=True)
            except Exception:
                pass
            return _grid_route_all(board_path)

        compat_issue = _freerouting_java_compatibility_issue(jar)
        if compat_issue:
            logger.warning("AUTOROUTE: Java/Freerouting compatibility issue: %s", compat_issue)
            manual_msg = _manual_freerouting_message(
                compat_issue,
                board_path,
                dsn_path,
            )
            if not allow_manhattan_fallback:
                return AutorouteResult(
                    success=False,
                    message=manual_msg,
                    method="freerouting",
                    manual_required=True,
                    dsn_path=dsn_path,
                )
            logger.warning("Freerouting Java compatibility issue; falling back to internal router.")
            try:
                Path(dsn_path).unlink(missing_ok=True)
            except Exception:
                pass
            return _grid_route_all(board_path)

    logger.info("AUTOROUTE: launching Freerouting with %s", jar)
    ok, ses_result = run_freerouting(dsn_path, timeout_seconds)
    if not ok:
        failure_message = str(ses_result or "")
        logger.info("AUTOROUTE: Freerouting did not complete; manual DSN handoff remains at %s", dsn_path)
        manual_msg = _manual_freerouting_message(
            f"Freerouting failed: {failure_message}",
            board_path,
            dsn_path,
        )
        if not allow_manhattan_fallback:
            return AutorouteResult(
                success=False,
                message=manual_msg,
                method="freerouting",
                manual_required=True,
                dsn_path=dsn_path,
            )
        logger.warning("Freerouting failed: %s. Falling back to internal router.", ses_result)
        try:
            Path(dsn_path).unlink(missing_ok=True)
        except Exception:
            pass
        return _grid_route_all(board_path)

    ok, import_msg = import_ses(board_path, ses_result)
    try:
        Path(dsn_path).unlink(missing_ok=True)
        Path(ses_result).unlink(missing_ok=True)
    except Exception:
        pass

    if ok:
        logger.info("AUTOROUTE: Freerouting session imported successfully")
        return AutorouteResult(
            success=True,
            message="Board routed successfully via Freerouting",
            method="freerouting",
        )

    return AutorouteResult(
        success=False,
        message=f"SES import failed: {import_msg}",
        method="freerouting",
    )


# ── Grid-based A* autorouter ─────────────────────────────────────────────────

_SQRT2 = 1.4142135623730951

# 8-directional moves: (dx, dy, cost)
_DIRECTIONS = [
    ( 0, -1, 1.0),     # N
    ( 1, -1, _SQRT2),  # NE
    ( 1,  0, 1.0),     # E
    ( 1,  1, _SQRT2),  # SE
    ( 0,  1, 1.0),     # S
    (-1,  1, _SQRT2),  # SW
    (-1,  0, 1.0),     # W
    (-1, -1, _SQRT2),  # NW
]


def _rebuild_connectivity(board) -> None:
    """Rebuild net/connectivity data after programmatic edits."""
    try:
        if hasattr(board, 'BuildListOfNets'):
            board.BuildListOfNets()
    except Exception:
        pass
    try:
        conn = getattr(board, 'GetConnectivity', None)
        if callable(conn):
            c = conn()
            for name in ('RecalculateRatsnest', 'Recalculate', 'Rebuild', 'Build'):
                fn = getattr(c, name, None)
                if callable(fn):
                    try:
                        fn()
                        break
                    except Exception:
                        continue
    except Exception:
        pass


def _pad_net_name(pad) -> str:
    """Best-effort net name from a pad object."""
    if pad is None:
        return ""
    try:
        fn = getattr(pad, 'GetNetname', None)
        if callable(fn):
            return str(fn() or '')
    except Exception:
        pass
    try:
        net = pad.GetNet()
        if net is not None:
            for attr in ('GetNetname', 'GetNetName', 'GetName'):
                g = getattr(net, attr, None)
                if callable(g):
                    v = g()
                    if v:
                        return str(v)
    except Exception:
        pass
    return ""


def _octile_heuristic(ax: int, ay: int, bx: int, by: int) -> float:
    """Octile distance — admissible heuristic for 8-directional grids."""
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return max(dx, dy) + (_SQRT2 - 1.0) * min(dx, dy)


def _astar(
    blocked: Set[Tuple[int, int]],
    cols: int,
    rows: int,
    start: Tuple[int, int],
    end: Tuple[int, int],
    max_iterations: int = 200_000,
) -> Optional[List[Tuple[int, int]]]:
    """A* pathfinding on a 2-D grid with 8-directional movement.

    Returns list of (gx, gy) waypoints from *start* to *end*, or ``None``
    if no path exists within *max_iterations* expansions.
    """
    if start == end:
        return [start]

    sx, sy = start
    ex, ey = end

    open_set: list = [(0.0, 0, sx, sy)]
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    counter = 1
    iterations = 0

    while open_set:
        iterations += 1
        if iterations > max_iterations:
            return None

        _f, _, cx, cy = heapq.heappop(open_set)
        current = (cx, cy)

        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        current_g = g_score.get(current)
        if current_g is None:
            continue

        for dx, dy, cost in _DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= cols or ny < 0 or ny >= rows:
                continue
            nb = (nx, ny)
            if nb in blocked:
                continue

            new_g = current_g + cost
            old_g = g_score.get(nb)
            if old_g is not None and new_g >= old_g:
                continue

            g_score[nb] = new_g
            h = _octile_heuristic(nx, ny, ex, ey)
            came_from[nb] = current
            heapq.heappush(open_set, (new_g + h, counter, nx, ny))
            counter += 1

    return None


def _simplify_path(
    path: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Merge collinear consecutive segments into single segments."""
    if not path or len(path) <= 2:
        return list(path)
    result = [path[0]]
    for i in range(1, len(path) - 1):
        dx1 = path[i][0] - path[i - 1][0]
        dy1 = path[i][1] - path[i - 1][1]
        dx2 = path[i + 1][0] - path[i][0]
        dy2 = path[i + 1][1] - path[i][1]
        if (dx1, dy1) != (dx2, dy2):
            result.append(path[i])
    result.append(path[-1])
    return result


def _mark_rect(
    target: Set[Tuple[int, int]],
    cx: int, cy: int,
    half_w: int, half_h: int,
    cols: int, rows: int,
) -> None:
    """Mark a rectangular area of grid cells."""
    x0 = max(0, cx - half_w)
    x1 = min(cols - 1, cx + half_w)
    y0 = max(0, cy - half_h)
    y1 = min(rows - 1, cy + half_h)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            target.add((x, y))


def _mark_line_thick(
    target: Set[Tuple[int, int]],
    x0: int, y0: int,
    x1: int, y1: int,
    half_w: int,
    cols: int, rows: int,
) -> None:
    """Mark cells along a line segment with a given half-width."""
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy), 1)
    for s in range(steps + 1):
        t = s / steps
        cx = int(round(x0 + dx * t))
        cy = int(round(y0 + dy * t))
        _mark_rect(target, cx, cy, half_w, half_w, cols, rows)


def _grid_route_all(board_path: str) -> AutorouteResult:
    """Grid-based A* autorouter with obstacle avoidance.

    Improvements over the previous naive L-shaped router:
      * Discretises the board into a fine grid.
      * Marks pads of *other* nets (+ clearance) as obstacles.
      * Marks previously-routed track corridors as obstacles.
      * Routes nets shortest-first using A* (cardinal + 45° diagonals).
      * Merges collinear segments to reduce track count.
      * Reports per-net success/failure.
    """
    if not PCBNEW_AVAILABLE:
        return AutorouteResult(
            success=False,
            message="pcbnew not available",
            method="grid-astar",
        )

    try:
        board = pcbnew.GetBoard()
        if board is None:
            board = pcbnew.LoadBoard(board_path)
        if board is None:
            return AutorouteResult(
                success=False,
                message="No active board found",
                method="grid-astar",
            )

        _rebuild_connectivity(board)

        to_mm = getattr(pcbnew, 'ToMM', None)
        from_mm = getattr(pcbnew, 'FromMM', None)
        if not callable(to_mm) or not callable(from_mm):
            return AutorouteResult(
                success=False,
                message="pcbnew missing ToMM/FromMM",
                method="grid-astar",
            )

        # ── Collect pad information ──────────────────────────────────
        # Per-pad: (x_mm, y_mm, width_mm, height_mm, net_code, pad_obj)
        all_pads: List[Tuple[float, float, float, float, int, object]] = []
        net_pads: Dict[int, List[Tuple[float, float, float, float, object]]] = {}
        net_names: Dict[int, str] = {}

        for fp in board.GetFootprints():
            try:
                pads_iter = fp.Pads()
            except Exception:
                pads_iter = []
            for pad in pads_iter:
                try:
                    nc = int(pad.GetNetCode())
                except Exception:
                    nc = 0
                try:
                    pos = pad.GetPosition()
                    px, py = to_mm(pos.x), to_mm(pos.y)
                except Exception:
                    continue
                try:
                    sz = pad.GetSize()
                    sw, sh = to_mm(sz.x), to_mm(sz.y)
                except Exception:
                    sw, sh = 1.0, 1.0

                all_pads.append((px, py, sw, sh, nc, pad))
                if nc > 0:
                    net_pads.setdefault(nc, []).append((px, py, sw, sh, pad))
                    if nc not in net_names:
                        nn = _pad_net_name(pad)
                        if nn:
                            net_names[nc] = nn

        # ── Identify routable nets (≥ 2 pads) ───────────────────────
        routable = {nc: pads for nc, pads in net_pads.items() if len(pads) >= 2}

        if not routable:
            return AutorouteResult(
                success=False,
                message=(
                    "No routable nets found (pads appear unconnected / netlist missing). "
                    "Use ASSIGN_NETS or KiCad's Update PCB from Schematic."
                ),
                method="grid-astar",
                tracks_added=0,
            )

        # ── Read design settings ─────────────────────────────────────
        clearance_mm = 0.2
        track_width_mm = 0.25
        try:
            ds = board.GetDesignSettings()
            try:
                clearance_mm = max(to_mm(ds.GetDefault().GetClearance()), 0.1)
            except Exception:
                clearance_mm = 0.2
            try:
                track_width_mm = to_mm(ds.GetCurrentTrackWidth())
            except Exception:
                track_width_mm = 0.25
        except Exception:
            pass

        # ── Set up grid ──────────────────────────────────────────────
        # Resolution: ~half a track width, clamped between 0.1 and 0.5 mm
        grid_mm = max(0.1, min(0.5, round(track_width_mm / 2, 3)))
        clearance_cells = max(1, int(clearance_mm / grid_mm + 0.99))

        # Board bounding box
        try:
            bb = board.GetBoardEdgesBoundingBox()
            if bb.GetWidth() <= 0 or bb.GetHeight() <= 0:
                bb = board.GetBoundingBox()
        except Exception:
            try:
                bb = board.GetBoundingBox()
            except Exception:
                bb = None
        if bb is None or bb.GetWidth() <= 0:
            return AutorouteResult(
                success=False,
                message="Cannot determine board bounds",
                method="grid-astar",
            )

        margin_mm = 2.0
        origin_x = to_mm(bb.GetX()) - margin_mm
        origin_y = to_mm(bb.GetY()) - margin_mm
        board_w = to_mm(bb.GetWidth()) + 2 * margin_mm
        board_h = to_mm(bb.GetHeight()) + 2 * margin_mm

        cols = int(round(board_w / grid_mm)) + 1
        rows = int(round(board_h / grid_mm)) + 1

        # Safety cap for very large boards
        if cols * rows > 500_000:
            scale = (cols * rows / 500_000) ** 0.5
            grid_mm *= scale
            cols = int(round(board_w / grid_mm)) + 1
            rows = int(round(board_h / grid_mm)) + 1
            clearance_cells = max(1, int(clearance_mm / grid_mm + 0.99))

        def mm_to_grid(mx: float, my: float) -> Tuple[int, int]:
            gx = int(round((mx - origin_x) / grid_mm))
            gy = int(round((my - origin_y) / grid_mm))
            return max(0, min(gx, cols - 1)), max(0, min(gy, rows - 1))

        def grid_to_mm(gx: int, gy: int) -> Tuple[float, float]:
            return origin_x + gx * grid_mm, origin_y + gy * grid_mm

        # ── Build per-net pad cells & obstacle map ───────────────────
        # pad_obstacles_by_net[nc]: cells blocked by this net's pads + clearance
        # net_landing[nc]: small set of cells AT the pad (always kept free for own net)
        pad_obstacles_by_net: Dict[int, Set[Tuple[int, int]]] = {}
        net_landing: Dict[int, Set[Tuple[int, int]]] = {}

        for px, py, sw, sh, nc, _pad_obj in all_pads:
            gx, gy = mm_to_grid(px, py)
            # Pad + clearance zone (obstacle for other nets)
            half_w = max(1, int((sw / 2 + clearance_mm) / grid_mm + 0.99))
            half_h = max(1, int((sh / 2 + clearance_mm) / grid_mm + 0.99))
            obstacle_cells: Set[Tuple[int, int]] = set()
            _mark_rect(obstacle_cells, gx, gy, half_w, half_h, cols, rows)
            pad_obstacles_by_net.setdefault(nc, set()).update(obstacle_cells)

            if nc > 0:
                # Landing zone (just the pad footprint, no clearance)
                lhw = max(0, int(sw / 2 / grid_mm + 0.5))
                lhh = max(0, int(sh / 2 / grid_mm + 0.5))
                landing: Set[Tuple[int, int]] = set()
                _mark_rect(landing, gx, gy, lhw, lhh, cols, rows)
                net_landing.setdefault(nc, set()).update(landing)

        # Mark existing tracks on the board as obstacles (pre-existing routing)
        existing_obstacles: Set[Tuple[int, int]] = set()
        for track in board.GetTracks():
            try:
                s = track.GetStart()
                e = track.GetEnd()
                tw = to_mm(track.GetWidth())
                sx, sy = mm_to_grid(to_mm(s.x), to_mm(s.y))
                ex, ey = mm_to_grid(to_mm(e.x), to_mm(e.y))
                hw = max(1, int((tw / 2 + clearance_mm) / grid_mm + 0.99))
                _mark_line_thick(existing_obstacles, sx, sy, ex, ey, hw, cols, rows)
            except Exception:
                continue

        # ── Sort nets: shortest total span first ─────────────────────
        def _net_span(pads):
            xs = [p[0] for p in pads]
            ys = [p[1] for p in pads]
            return (max(xs) - min(xs)) + (max(ys) - min(ys))

        sorted_nets = sorted(routable.items(), key=lambda kv: _net_span(kv[1]))

        # ── Route each net ───────────────────────────────────────────
        track_half_cells = max(1, int((track_width_mm / 2 + clearance_mm) / grid_mm + 0.99))
        routed_obstacles: Set[Tuple[int, int]] = set()
        tracks_added = 0
        nets_routed = 0
        nets_failed: List[str] = []
        track_width_iu = int(from_mm(track_width_mm))
        f_cu = getattr(pcbnew, 'F_Cu', 0)

        for nc, pads in sorted_nets:
            # Build obstacle set: everything *except* this net's own pads
            blocked: Set[Tuple[int, int]] = set()
            blocked.update(existing_obstacles)
            blocked.update(routed_obstacles)
            for other_nc, obs in pad_obstacles_by_net.items():
                if other_nc != nc:
                    blocked.update(obs)

            # Ensure this net's landing cells are reachable
            for cell in net_landing.get(nc, set()):
                blocked.discard(cell)

            # Pad grid positions (centre cell of each pad)
            pad_positions: List[Tuple[int, int, object]] = []
            for px, py, _sw, _sh, pad_obj in pads:
                gx, gy = mm_to_grid(px, py)
                pad_positions.append((gx, gy, pad_obj))

            # Net object for assigning to tracks
            net_obj = None
            try:
                net_obj = pads[0][4].GetNet()
            except Exception:
                net_obj = None

            # Route using nearest-neighbour (Prim-like) ordering
            connected = {0}
            unconnected = set(range(1, len(pad_positions)))
            net_tracks: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            net_path_cells: List[Tuple[int, int]] = []
            net_ok = True

            while unconnected:
                best_path = None
                best_cost = float('inf')
                best_to = -1

                # Try every (connected, unconnected) pair; A* only the closest candidates
                candidates: List[Tuple[float, int, int]] = []
                for ci in connected:
                    for ui in unconnected:
                        sx, sy = pad_positions[ci][0], pad_positions[ci][1]
                        ex, ey = pad_positions[ui][0], pad_positions[ui][1]
                        est = _octile_heuristic(sx, sy, ex, ey)
                        candidates.append((est, ci, ui))
                candidates.sort()

                for est, ci, ui in candidates:
                    if est >= best_cost:
                        break  # remaining candidates are farther
                    sx, sy = pad_positions[ci][0], pad_positions[ci][1]
                    ex, ey = pad_positions[ui][0], pad_positions[ui][1]
                    path = _astar(blocked, cols, rows, (sx, sy), (ex, ey))
                    if path is not None:
                        cost = len(path)
                        if cost < best_cost:
                            best_cost = cost
                            best_path = path
                            best_to = ui

                if best_path is None:
                    net_name = net_names.get(nc) or f"net {nc}"
                    nets_failed.append(net_name)
                    net_ok = False
                    break

                connected.add(best_to)
                unconnected.discard(best_to)

                # Record path cells for obstacle marking
                net_path_cells.extend(best_path)

                # Simplify → track segments
                simplified = _simplify_path(best_path)
                for i in range(len(simplified) - 1):
                    mm0 = grid_to_mm(simplified[i][0], simplified[i][1])
                    mm1 = grid_to_mm(simplified[i + 1][0], simplified[i + 1][1])
                    net_tracks.append((mm0, mm1))

                # Mark path as obstacle for the *next* net
                for gx, gy in best_path:
                    _mark_rect(
                        routed_obstacles, gx, gy,
                        track_half_cells, track_half_cells,
                        cols, rows,
                    )

            # Create PCB track objects for this net
            for (x0, y0), (x1, y1) in net_tracks:
                track = pcbnew.PCB_TRACK(board)
                track.SetStart(pcbnew.VECTOR2I(int(from_mm(x0)), int(from_mm(y0))))
                track.SetEnd(pcbnew.VECTOR2I(int(from_mm(x1)), int(from_mm(y1))))
                track.SetWidth(track_width_iu)
                track.SetLayer(f_cu)
                try:
                    if net_obj is not None and hasattr(track, 'SetNet'):
                        track.SetNet(net_obj)
                except Exception:
                    try:
                        track.SetNetCode(nc)
                    except Exception:
                        pass
                if hasattr(track, 'thisown'):
                    track.thisown = False
                board.Add(track)
                tracks_added += 1

            if net_ok:
                nets_routed += 1

        # ── Refresh board view ───────────────────────────────────────
        try:
            pcbnew.Refresh()
        except Exception:
            pass

        # ── Build result message ─────────────────────────────────────
        skipped_single: List[str] = []
        for nc, pads in net_pads.items():
            if len(pads) < 2:
                skipped_single.append(net_names.get(nc) or f"netcode {nc}")

        parts: List[str] = [
            f"Routing complete: {tracks_added} tracks for {nets_routed}/{len(routable)} net(s)"
        ]
        if nets_failed:
            parts.append(
                f"Could not route {len(nets_failed)} net(s): {', '.join(nets_failed[:5])}"
            )
        if skipped_single:
            parts.append(
                f"Skipped {len(skipped_single)} single-pad net(s): "
                + ", ".join(sorted(set(skipped_single))[:5])
            )

        return AutorouteResult(
            success=nets_routed > 0,
            message=". ".join(parts),
            method="grid-astar",
            tracks_added=tracks_added,
            unrouted_remaining=len(nets_failed),
        )

    except Exception as e:
        logger.exception("Grid routing failed")
        return AutorouteResult(
            success=False,
            message=f"Grid routing failed: {e}",
            method="grid-astar",
        )
