"""User settings persistence for VibeCAD.

Settings live inside the installed ``vibecad/`` package directory so a copied
plugin folder is self-contained. We still support one-time fallback loading
from the legacy repo-root ``vibecad_settings.json`` and from
``~/.vibecad/settings.json`` to migrate existing users.

We intentionally do not write environment variables; we load settings and
configure the LLM client directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_REPO_SETTINGS_FILENAME = "vibecad_settings.json"
_LOCAL_API_KEY_FILENAME = "vibecad_api_key.local.txt"


def _bundle_root() -> Path:
    # .../vibecad/config/settings.py -> package root one level up
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    # .../vibecad/config/settings.py -> checkout root two levels up
    return Path(__file__).resolve().parents[2]


def _looks_like_vibecad_repo(path: Path) -> bool:
    p = path.expanduser()
    return (
        p.exists()
        and p.is_dir()
        and (p / "vibecad").is_dir()
        and (p / "pyproject.toml").exists()
    )


def _preferred_repo_root() -> Optional[Path]:
    candidates: list[Path] = []

    env_root = str(os.environ.get("VIBECAD_REPO_ROOT", "") or "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.append(Path.home() / "Documents" / "GitHub" / "VibeCAD")
    candidates.append(_repo_root())
    candidates.append(Path.cwd())

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_vibecad_repo(candidate):
            return candidate
    return None


def default_settings_path() -> Path:
    """Primary settings path used by the plugin bundle."""
    return _bundle_root() / _REPO_SETTINGS_FILENAME


def default_api_key_override_path() -> Path:
    """Optional bundle-local API key override file checked by the plugin."""
    return _bundle_root() / _LOCAL_API_KEY_FILENAME


def _legacy_repo_settings_path() -> Optional[Path]:
    preferred_root = _preferred_repo_root()
    if preferred_root is not None:
        return preferred_root / _REPO_SETTINGS_FILENAME
    return None


def _legacy_settings_path() -> Path:
    # Backward-compat path used by older versions.
    return Path.home() / ".vibecad" / "settings.json"


def _default_settings_path() -> Path:
    return default_settings_path()


def _storage_path_for_save(path: Path, *, base_dir: Path) -> str:
    """Return a path string suitable for persisting in settings JSON.

    Relative values are preserved. Absolute values are rewritten relative to
    *base_dir* when possible so bundle-local credentials remain portable.
    """
    raw = str(path or "").strip()
    if not raw:
        return ""

    p = Path(raw).expanduser()
    if not p.is_absolute():
        return raw

    try:
        resolved = p.resolve()
    except Exception:
        resolved = p

    try:
        return str(resolved.relative_to(base_dir.resolve()))
    except Exception:
        try:
            bundled_copy = base_dir.resolve() / resolved.name
            if _is_service_account_json(bundled_copy):
                return bundled_copy.name
        except Exception:
            pass
    return str(resolved)


def _is_service_account_json(path: Path) -> bool:
    """Return True if a JSON file looks like a GCP service account key."""
    if not path.exists() or not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    return (
        str(data.get("type", "")).strip().lower() == "service_account"
        and bool(str(data.get("client_email", "")).strip())
        and bool(str(data.get("private_key", "")).strip())
    )


def _default_vertex_credentials_path(base_dir: Optional[Path] = None) -> str:
    """Best-effort default service-account path for Vertex credentials."""
    # Explicit env override takes priority if set.
    env_path = str(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if _is_service_account_json(p):
            return str(p)

    # Auto-detect service-account key files next to the bundle settings first.
    storage_base = base_dir or _bundle_root()
    search_roots: list[Path] = []
    for candidate in (storage_base, _bundle_root(), _preferred_repo_root()):
        if candidate is None:
            continue
        if candidate in search_roots:
            continue
        search_roots.append(candidate)

    for root in search_roots:
        try:
            for cand in sorted(root.glob("*.json")):
                if _is_service_account_json(cand):
                    return _storage_path_for_save(cand, base_dir=storage_base)
        except Exception:
            pass
    return ""


def _read_api_key_override(path: Optional[Path] = None) -> str:
    """Read API key override from a local text file.

    Accepted first non-empty/non-comment line formats:
      - sk-... (raw key)
      - VIBECAD_API_KEY=sk-...
      - API_KEY=sk-...
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(path)
    else:
        candidates.append(default_api_key_override_path())
        legacy_root = _preferred_repo_root()
        if legacy_root is not None:
            candidates.append(legacy_root / _LOCAL_API_KEY_FILENAME)
        candidates.append(Path.home() / ".vibecad" / _LOCAL_API_KEY_FILENAME)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        try:
            for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k_norm = str(k or "").strip().upper()
                    if k_norm in {"VIBECAD_API_KEY", "API_KEY"}:
                        line = str(v or "").strip()
                line = line.strip().strip('"').strip("'")
                if line:
                    return line
        except Exception:
            continue
    return ""


# Recognised provider identifiers (stored in settings.json as "llm_provider").
LLM_PROVIDER_OPENROUTER = "openrouter"
LLM_PROVIDER_VERTEX = "vertex"
LLM_PROVIDER_GITHUB = "github"
LLM_PROVIDER_OPENAI = "openai"
LLM_PROVIDER_CUSTOM = "custom"

# Preset API base URLs for each provider.
_PROVIDER_API_BASES: Dict[str, str] = {
    LLM_PROVIDER_OPENROUTER: "https://openrouter.ai/api/v1",
    LLM_PROVIDER_VERTEX: "",   # built dynamically from project/location
    LLM_PROVIDER_GITHUB: "https://models.github.ai/inference",
    LLM_PROVIDER_OPENAI: "https://api.openai.com/v1",
    LLM_PROVIDER_CUSTOM: "",
}

# Default model suggestions per provider (shown as hint text).
_PROVIDER_MODEL_HINTS: Dict[str, str] = {
    LLM_PROVIDER_OPENROUTER: "google/gemini-2.0-flash-001",
    LLM_PROVIDER_VERTEX: "google/gemini-2.0-flash-001",
    LLM_PROVIDER_GITHUB: "openai/gpt-4.1",
    LLM_PROVIDER_OPENAI: "gpt-4o",
    LLM_PROVIDER_CUSTOM: "",
}

# In-chat model selector options.
LLM_MODEL_GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"
LLM_MODEL_GEMINI_3_1_PRO_PREVIEW = "google/gemini-3.1-pro-preview"
LLM_MODEL_CHOICES = [
    ("Gemini 3 Flash Preview", LLM_MODEL_GEMINI_3_FLASH_PREVIEW),
    ("Gemini 3.1 Pro Preview", LLM_MODEL_GEMINI_3_1_PRO_PREVIEW),
]
DEFAULT_LLM_MODEL = LLM_MODEL_GEMINI_3_FLASH_PREVIEW


def normalize_llm_model_choice(model: str) -> str:
    """Return a stable model id for the chat model selector.

    Empty values fall back to the default flash preview model. Known preview
    aliases are normalized to their canonical `google/...` ids. Unknown model
    strings are preserved so existing custom configs are not silently lost.
    """
    raw = str(model or "").strip()
    if not raw:
        return DEFAULT_LLM_MODEL

    lowered = raw.lower()
    if "gemini-3.1-pro-preview" in lowered:
        return LLM_MODEL_GEMINI_3_1_PRO_PREVIEW
    if "gemini-3-flash-preview" in lowered:
        return LLM_MODEL_GEMINI_3_FLASH_PREVIEW
    return raw


@dataclass
class VibeCADSettings:
    api_key: str = ""
    github_token: str = ""
    api_base: str = ""
    model: str = DEFAULT_LLM_MODEL
    verbose: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    verify_ssl: bool = True
    ca_bundle_path: str = ""

    # Provider selection (one of the LLM_PROVIDER_* constants above).
    llm_provider: str = LLM_PROVIDER_OPENROUTER

    # Vertex AI specific
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_credentials_path: str = ""

    # Design agent UX
    thinking_output_enabled: bool = True
    yolo_auto_apply: bool = False

    # Inference quality
    enable_thinking: bool = False   # Send thinking/reasoning token budget to the model
    thinking_budget: Optional[int] = None  # Token budget for reasoning (None → 8000 default)
    top_p: Optional[float] = None   # Nucleus sampling (None → provider default)

    # DigiKey API (used by SPEC agent for real datasheet lookups)
    digikey_client_id: str = ""
    digikey_client_secret: str = ""

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "VibeCADSettings":
        return cls(
            api_key=str(data.get("api_key", "") or ""),
            github_token=str(data.get("github_token", "") or ""),
            api_base=str(data.get("api_base", "") or ""),
            model=normalize_llm_model_choice(str(data.get("model", "") or "")),
            verbose=bool(data.get("verbose", False)),
            temperature=(float(data["temperature"]) if "temperature" in data and data["temperature"] is not None else None),
            max_tokens=(int(data["max_tokens"]) if "max_tokens" in data and data["max_tokens"] is not None else None),
            timeout=(int(data["timeout"]) if "timeout" in data and data["timeout"] is not None else None),
            verify_ssl=(bool(data.get("verify_ssl", True))),
            ca_bundle_path=str(data.get("ca_bundle_path", "") or ""),
            llm_provider=str(data.get("llm_provider", LLM_PROVIDER_OPENROUTER) or LLM_PROVIDER_OPENROUTER),
            vertex_project=str(data.get("vertex_project", "") or ""),
            vertex_location=str(data.get("vertex_location", "us-central1") or "us-central1"),
            vertex_credentials_path=str(data.get("vertex_credentials_path", "") or ""),
            thinking_output_enabled=bool(data.get("thinking_output_enabled", True)),
            yolo_auto_apply=bool(data.get("yolo_auto_apply", False)),
            enable_thinking=bool(data.get("enable_thinking", False)),
            thinking_budget=(int(data["thinking_budget"]) if "thinking_budget" in data and data["thinking_budget"] is not None else None),
            top_p=(float(data["top_p"]) if "top_p" in data and data["top_p"] is not None else None),
            digikey_client_id=str(data.get("digikey_client_id", "") or ""),
            digikey_client_secret=str(data.get("digikey_client_secret", "") or ""),
        )

    @staticmethod
    def _apply_api_key_file_override(settings: "VibeCADSettings", *, enabled: bool) -> "VibeCADSettings":
        if not enabled:
            return settings
        api_key_override = _read_api_key_override()
        if api_key_override:
            settings.api_key = api_key_override
        return settings

    @staticmethod
    def _apply_vertex_credentials_default(
        settings: "VibeCADSettings",
        *,
        base_dir: Optional[Path] = None,
    ) -> "VibeCADSettings":
        if not str(getattr(settings, "vertex_credentials_path", "") or "").strip():
            auto_path = _default_vertex_credentials_path(base_dir=base_dir)
            if auto_path:
                settings.vertex_credentials_path = auto_path
        return settings

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "VibeCADSettings":
        target = path or _default_settings_path()
        allow_api_key_override = path is None
        candidates = [target]
        if path is None:
            legacy_repo = _legacy_repo_settings_path()
            if legacy_repo is not None and legacy_repo != target:
                candidates.append(legacy_repo)
            legacy_home = _legacy_settings_path()
            if legacy_home != target:
                candidates.append(legacy_home)

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                settings = cls._from_dict(data)
                settings = cls._apply_vertex_credentials_default(settings, base_dir=target.parent)
                # One-time auto-migration from legacy user-home file into repo.
                if path is None and candidate != target:
                    try:
                        settings.save(target)
                    except Exception:
                        pass
                settings = cls._apply_api_key_file_override(settings, enabled=allow_api_key_override)
                return settings
            except Exception:
                continue

        settings = cls._apply_vertex_credentials_default(cls(), base_dir=target.parent)
        settings = cls._apply_api_key_file_override(settings, enabled=allow_api_key_override)
        return settings

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or _default_settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        creds = str(data.get("vertex_credentials_path", "") or "").strip()
        if creds:
            data["vertex_credentials_path"] = _storage_path_for_save(Path(creds), base_dir=p.parent)
        p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return p

    def to_llm_overrides(self) -> Dict[str, Any]:
        """Return only values that were explicitly set."""
        out: Dict[str, Any] = {}
        if self.api_key.strip():
            out["api_key"] = self.api_key.strip()
        if self.api_base.strip():
            out["api_base"] = self.api_base.strip()
        if self.model.strip():
            out["model"] = self.model.strip()
        if self.temperature is not None:
            out["temperature"] = float(self.temperature)
        if self.max_tokens is not None:
            out["max_tokens"] = int(self.max_tokens)
        if self.timeout is not None:
            out["timeout"] = int(self.timeout)
        if self.verify_ssl is False:
            out["verify_ssl"] = False
        if self.ca_bundle_path.strip():
            out["ca_bundle"] = self.ca_bundle_path.strip()
        if self.enable_thinking:
            out["enable_thinking"] = True
            out["thinking_budget"] = int(self.thinking_budget) if self.thinking_budget is not None else 8000
        if self.top_p is not None:
            out["top_p"] = float(self.top_p)
        return out
