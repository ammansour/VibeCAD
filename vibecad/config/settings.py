"""User settings persistence for VibeCAD.

Settings are stored outside the KiCad project so you can configure API
credentials once and reuse them across boards.

We intentionally do not write environment variables; we load settings and
configure the LLM client directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _default_settings_path() -> Path:
    # macOS/Linux/Windows compatible user home location
    return Path.home() / ".vibecad" / "settings.json"


@dataclass
class VibeCADSettings:
    api_key: str = ""
    api_base: str = ""
    model: str = ""
    verbose: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    verify_ssl: bool = True
    ca_bundle_path: str = ""

    # Design agent UX
    thinking_output_enabled: bool = True
    yolo_auto_apply: bool = False

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "VibeCADSettings":
        p = path or _default_settings_path()
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            return cls(
                api_key=str(data.get("api_key", "") or ""),
                api_base=str(data.get("api_base", "") or ""),
                model=str(data.get("model", "") or ""),
                verbose=bool(data.get("verbose", False)),
                temperature=(float(data["temperature"]) if "temperature" in data and data["temperature"] is not None else None),
                max_tokens=(int(data["max_tokens"]) if "max_tokens" in data and data["max_tokens"] is not None else None),
                timeout=(int(data["timeout"]) if "timeout" in data and data["timeout"] is not None else None),
                verify_ssl=(bool(data.get("verify_ssl", True))),
                ca_bundle_path=str(data.get("ca_bundle_path", "") or ""),
                thinking_output_enabled=bool(data.get("thinking_output_enabled", True)),
                yolo_auto_apply=bool(data.get("yolo_auto_apply", False)),
            )
        except Exception:
            return cls()

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or _default_settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
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
        return out
