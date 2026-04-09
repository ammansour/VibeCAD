"""
Google Vertex AI LLM client — same interface as LLMClient.

Uses the Vertex AI OpenAI-compatible REST endpoint for Gemini models:
  https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/
      locations/{location}/endpoints/openapi/chat/completions

Authentication (tried in order):
  1. google-auth library (Application Default Credentials or service account JSON path)
  2. Manual service-account JWT bearer flow (openssl + HTTPS token exchange)
  3. gcloud CLI:  gcloud auth print-access-token
"""

from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .client import LLMClient, LLMConfig, LLMError, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_VENDOR_DIRNAME = "_vendor_py"
_SYSTEM_CA_BUNDLE_CANDIDATES = [
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/usr/local/share/ca-certificates",
]

# Default Gemini model available on Vertex AI
DEFAULT_VERTEX_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_VERTEX_LOCATION = "us-central1"


# ── Access-token helpers ────────────────────────────────────────────────────

def _repo_root() -> Path:
    # .../vibecad/llm/vertex_client.py -> repo root two levels up
    return Path(__file__).resolve().parents[2]


def _bundle_root() -> Path:
    # .../vibecad/llm/vertex_client.py -> package root one level up
    return Path(__file__).resolve().parents[1]


def _vendor_dir() -> Path:
    return _repo_root() / _VENDOR_DIRNAME


def _add_vendor_to_syspath() -> None:
    p = str(_vendor_dir())
    if p not in sys.path:
        sys.path.insert(0, p)


def _google_auth_importable() -> bool:
    try:
        import google.auth  # type: ignore  # noqa: F401
        import google.auth.transport.requests  # type: ignore  # noqa: F401
        import google.oauth2.service_account  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_google_auth_ready() -> bool:
    _add_vendor_to_syspath()
    return _google_auth_importable()


def _resolve_credentials_json_path(credentials_json_path: str = "") -> str:
    """Resolve bundle-relative credential paths to an absolute filesystem path."""
    raw = str(credentials_json_path or "").strip()
    if not raw:
        return ""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    return str((_bundle_root() / p).resolve())


def _resolve_ca_bundle(ca_bundle: str = "") -> str:
    """Return the best available CA bundle path, if any."""
    candidates = []

    explicit = str(ca_bundle or "").strip()
    if explicit:
        candidates.append(explicit)

    for env_name in ("VIBECAD_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        env_path = str(os.environ.get(env_name, "") or "").strip()
        if env_path:
            candidates.append(env_path)

    _add_vendor_to_syspath()
    try:
        import certifi  # type: ignore

        certifi_path = str(certifi.where() or "").strip()
        if certifi_path:
            candidates.append(certifi_path)
    except Exception:
        pass

    candidates.extend(_SYSTEM_CA_BUNDLE_CANDIDATES)

    for candidate in candidates:
        try:
            if candidate and os.path.exists(candidate):
                return candidate
        except Exception:
            continue
    return ""


def _build_ssl_context(verify_ssl: bool = True, ca_bundle: str = "") -> Optional[ssl.SSLContext]:
    """Build an SSL context honoring VibeCAD's TLS settings."""
    try:
        if not verify_ssl:
            return ssl._create_unverified_context()

        bundle_path = _resolve_ca_bundle(ca_bundle)
        if bundle_path:
            if os.path.isdir(bundle_path):
                return ssl.create_default_context(capath=bundle_path)
            return ssl.create_default_context(cafile=bundle_path)
        return ssl.create_default_context()
    except Exception as e:
        logger.debug("Vertex SSL context setup failed: %s", e)
        return None


def _requests_verify_arg(verify_ssl: bool = True, ca_bundle: str = "") -> object:
    """Return the `requests.Session.verify` value for the current TLS settings."""
    if not verify_ssl:
        return False
    bundle_path = _resolve_ca_bundle(ca_bundle)
    return bundle_path or True


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _sign_rs256_with_openssl(private_key_pem: str, signing_input: bytes) -> Optional[bytes]:
    """Sign bytes with RSA-SHA256 via openssl, returning signature bytes."""
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as keyf:
            keyf.write(private_key_pem)
            key_path = keyf.name
        with tempfile.NamedTemporaryFile("wb", delete=False) as inf:
            inf.write(signing_input)
            in_path = inf.name
        try:
            result = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", key_path, in_path],
                capture_output=True,
                timeout=20,
            )
            if result.returncode != 0:
                return None
            return bytes(result.stdout or b"")
        finally:
            try:
                os.unlink(key_path)
            except Exception:
                pass
            try:
                os.unlink(in_path)
            except Exception:
                pass
    except Exception:
        return None


def _token_via_service_account_manual(
    credentials_json_path: str = "",
    verify_ssl: bool = True,
    ca_bundle: str = "",
) -> Optional[str]:
    """Use service-account JWT bearer flow without google-auth dependency."""
    path = _resolve_credentials_json_path(credentials_json_path)
    if not path:
        return None
    logger.info("Vertex manual SA token: attempting openssl JWT flow")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Vertex manual SA token: could not read credentials JSON: %s", e)
        return None

    try:
        if str(data.get("type", "")).strip().lower() != "service_account":
            return None
        client_email = str(data.get("client_email", "") or "").strip()
        private_key = str(data.get("private_key", "") or "").strip()
        token_uri = str(data.get("token_uri", "") or "").strip() or "https://oauth2.googleapis.com/token"
        if not client_email or not private_key:
            return None
    except Exception:
        return None

    now = int(time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    payload = {
        "iss": client_email,
        "scope": " ".join(_SCOPES),
        "aud": token_uri,
        "iat": now,
        "exp": now + 3600,
    }

    try:
        header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        signing_input = f"{_b64url(header_json)}.{_b64url(payload_json)}".encode("ascii")
        signature = _sign_rs256_with_openssl(private_key, signing_input)
        if not signature:
            logger.warning("Vertex manual SA token: openssl signing failed")
            return None
        assertion = signing_input.decode("ascii") + "." + _b64url(signature)
    except Exception as e:
        logger.warning("Vertex manual SA token: JWT signing build failed: %s", e)
        return None

    post_data = urlencode({
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": assertion,
    }).encode("utf-8")
    req = Request(
        token_uri,
        data=post_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        ssl_context = _build_ssl_context(verify_ssl=verify_ssl, ca_bundle=ca_bundle)
        with urlopen(req, timeout=25, context=ssl_context) as resp:
            body = (resp.read() or b"").decode("utf-8", errors="replace")
        token_data = json.loads(body)
        token = str(token_data.get("access_token", "") or "").strip()
        if token:
            logger.info("Vertex manual SA token: token exchange succeeded")
            return token
    except Exception as e:
        logger.warning("Vertex manual SA token exchange failed: %s", e)
    return None


def _token_via_google_auth(
    credentials_json_path: str = "",
    verify_ssl: bool = True,
    ca_bundle: str = "",
) -> Optional[str]:
    """Try to get an access token via the google-auth library.

    If *credentials_json_path* is non-empty, load that service account file.
    Relative paths are resolved against the bundled ``vibecad/`` package root.
    Otherwise, fall back to Application Default Credentials (ADC).
    """
    if not _ensure_google_auth_ready():
        return None

    try:
        path = _resolve_credentials_json_path(credentials_json_path)
        if path:
            import google.oauth2.service_account as sa  # type: ignore
            import google.auth.transport.requests as tr  # type: ignore
            import requests  # type: ignore

            session = requests.Session()
            session.verify = _requests_verify_arg(verify_ssl=verify_ssl, ca_bundle=ca_bundle)
            creds = sa.Credentials.from_service_account_file(
                path,
                scopes=_SCOPES,
            )
            req = tr.Request(session=session)
            creds.refresh(req)
            return str(creds.token or "")
        else:
            import google.auth  # type: ignore
            import google.auth.transport.requests as tr  # type: ignore
            import requests  # type: ignore

            session = requests.Session()
            session.verify = _requests_verify_arg(verify_ssl=verify_ssl, ca_bundle=ca_bundle)
            creds, _ = google.auth.default(scopes=_SCOPES)
            req = tr.Request(session=session)
            creds.refresh(req)
            return str(creds.token or "")
    except ImportError:
        return None
    except Exception as e:
        logger.debug("google-auth token fetch failed: %s", e)
        return None


def _token_via_gcloud() -> Optional[str]:
    """Shell out to `gcloud auth print-access-token` as a last resort.

    Tries both the bare command (relies on PATH) and several well-known
    absolute install locations (Homebrew, system) so this works when
    invoked from hosts like KiCad that don't inherit the user's shell PATH.
    """
    import os
    import sys
    _GCLOUD_CANDIDATES = [
        "gcloud.cmd" if sys.platform == "win32" else "gcloud",  # relies on PATH
        "/opt/homebrew/share/google-cloud-sdk/bin/gcloud",  # Homebrew (Apple Silicon)
        "/usr/local/share/google-cloud-sdk/bin/gcloud",     # Homebrew (Intel)
        "/usr/lib/google-cloud-sdk/bin/gcloud",             # apt/system
        os.path.expanduser("~/google-cloud-sdk/bin/gcloud"),
    ]
    if sys.platform == "win32":
        _GCLOUD_CANDIDATES.extend([
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
            os.path.expandvars(r"%ProgramFiles%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
            os.path.expandvars(r"%LocalAppData%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
        ])

    # Ensure the SDK's own Python works even when CLOUDSDK_PYTHON isn't set.
    env = os.environ.copy()
    if "CLOUDSDK_PYTHON" not in env:
        for py in [
            "/opt/homebrew/opt/python@3.12/libexec/bin/python3",
            "/opt/homebrew/opt/python@3.11/libexec/bin/python3",
            "/usr/bin/python3",
        ]:
            if os.path.isfile(py):
                env["CLOUDSDK_PYTHON"] = py
                break

    for candidate in _GCLOUD_CANDIDATES:
        try:
            result = subprocess.run(
                [candidate, "auth", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=15,
                env=env,
            )
            token = (result.stdout or "").strip()
            if token and result.returncode == 0:
                return token
            stderr = (result.stderr or "").strip()
            if stderr:
                logger.debug("gcloud (%s) stderr: %s", candidate, stderr)
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.debug("gcloud (%s) failed: %s", candidate, e)
            continue
    return None


def get_vertex_access_token(
    credentials_json_path: str = "",
    verify_ssl: bool = True,
    ca_bundle: str = "",
) -> str:
    """Return a valid Vertex AI access token, raising LLMError if none found."""
    token = _token_via_google_auth(
        credentials_json_path,
        verify_ssl=verify_ssl,
        ca_bundle=ca_bundle,
    )
    if token:
        return token

    token = _token_via_service_account_manual(
        credentials_json_path,
        verify_ssl=verify_ssl,
        ca_bundle=ca_bundle,
    )
    if token:
        return token

    token = _token_via_gcloud()
    if token:
        return token

    raise LLMError(
        "Vertex AI: could not obtain an access token.\n"
        "Ensure one of the following:\n"
        "  • Service-account JSON is present and readable (vertex_credentials_path)\n"
        "  • openssl is available for service-account fallback token signing\n"
        "  • google-auth is already available (system Python or bundled _vendor_py)\n"
        "  • google-auth is installed and Application Default Credentials are set up\n"
        "    (run: gcloud auth application-default login)\n"
        "  • A service account JSON path is configured in VibeCAD settings\n"
        "  • gcloud CLI is installed and authenticated (gcloud auth login)"
    )


# ── Client class ────────────────────────────────────────────────────────────

class VertexAIClient:
    """LLM client backed by Vertex AI's OpenAI-compatible endpoint.

    Drop-in replacement for LLMClient — exposes the same public interface:
      chat(), explain_simple(), is_available, config
    """

    def __init__(
        self,
        project: str,
        location: str = DEFAULT_VERTEX_LOCATION,
        model: str = DEFAULT_VERTEX_MODEL,
        credentials_json_path: str = "",
        verify_ssl: bool = True,
        ca_bundle: str = "",
        temperature: float = 0.3,
        max_tokens: int = 16384,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        activity_timeout: int = 60,
        enable_thinking: bool = False,
        thinking_budget: int = 8000,
    ):
        self._project = project.strip()
        self._location = location.strip() or DEFAULT_VERTEX_LOCATION
        _raw_model = model.strip() or DEFAULT_VERTEX_MODEL
        # Vertex AI OpenAI-compatible endpoint requires "<publisher>/<model>".
        # Auto-prefix with "google/" when the user omits it.
        self._model = _raw_model if "/" in _raw_model else f"google/{_raw_model}"
        self._credentials_json_path = _resolve_credentials_json_path(credentials_json_path)
        self._verify_ssl = bool(verify_ssl)
        self._ca_bundle = str(ca_bundle or "").strip()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        # Per-chunk timeout used by the SSE streaming path. The request stays
        # alive as long as a token arrives within this many seconds, so total
        # generation time is unbounded. Falls back to _timeout if not set.
        self._activity_timeout = activity_timeout if activity_timeout > 0 else timeout
        self._enable_thinking = enable_thinking
        self._thinking_budget = max(1, int(thinking_budget)) if thinking_budget else 8000

        # Cached token + expiry so we don't re-auth every call.
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0.0

        # Expose a config-like object so code that reads client.config.model works.
        self.config = _VertexConfigProxy(
            model=self._model,
            api_base=self._base_url,
            project=self._project,
            location=self._location,
        )

    @property
    def _base_url(self) -> str:
        # For "global" (required by Gemini 3 preview models) the hostname has
        # no location prefix.  Regional locations use {location}-aiplatform.
        if self._location == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{self._location}-aiplatform.googleapis.com"
        return (
            f"https://{host}/v1"
            f"/projects/{self._project}/locations/{self._location}/endpoints/openapi"
        )

    @property
    def is_available(self) -> bool:
        return bool(self._project)

    def _get_token(self) -> str:
        """Return a valid access token, refreshing if near expiry."""
        # Tokens typically last 1 hour; refresh 5 min early.
        if self._cached_token and time.time() < self._token_expires_at - 300:
            return self._cached_token
        token = get_vertex_access_token(
            self._credentials_json_path,
            verify_ssl=self._verify_ssl,
            ca_bundle=self._ca_bundle,
        )
        self._cached_token = token
        self._token_expires_at = time.time() + 3600  # assume 1h lifetime
        return token

    def chat(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Send a chat completion request to Vertex AI.

        Raises LLMError on failure.
        """
        if not self.is_available:
            raise LLMError("Vertex AI: no project configured.")

        # Build the full message list (include system prompt as system message)
        all_messages: List[LLMMessage] = []
        if system_prompt:
            all_messages.append(LLMMessage(role="system", content=system_prompt))
        all_messages.extend(messages)

        url = f"{self._base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in all_messages],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._enable_thinking:
            # Gemini: thinking_budget at the top level of the request body.
            # response_format is NOT compatible with thinking on Gemini — omit it.
            payload["thinking_budget"] = self._thinking_budget
            logger.debug("Vertex AI: thinking enabled (budget=%d)", self._thinking_budget)
        elif response_format:
            payload["response_format"] = response_format

        last_error: Optional[LLMError] = None
        for attempt in range(self._max_retries):
            try:
                token = self._get_token()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                }
                # Reuse LLMClient's low-level request/parse helpers via a
                # temporary client with a dummy api_key.
                _tmp = LLMClient(LLMConfig(
                    api_key=token,
                    api_base=self._base_url,
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    timeout=self._timeout,
                    verify_ssl=self._verify_ssl,
                    ca_bundle=self._ca_bundle,
                ))
                response = _tmp._make_request(url, payload, headers)
                return _tmp._parse_response(response)
            except LLMError as e:
                last_error = e
                logger.warning(
                    "Vertex AI attempt %d/%d failed: %s",
                    attempt + 1, self._max_retries, e,
                )
                code = int(getattr(e, "status_code", 0) or 0)
                if code == 401:
                    # Force token refresh on next attempt.
                    self._cached_token = None
                    self._token_expires_at = 0.0
                if code == 429 and getattr(e, "retry_after", None):
                    delay = min(30.0, float(e.retry_after))
                    if delay > 30:
                        raise
                    time.sleep(delay)
                    continue
                if attempt < self._max_retries - 1:
                    time.sleep(min(15.0, self._retry_delay * (attempt + 1)))

        raise last_error or LLMError("Vertex AI: request failed after retries")

    def explain_simple(self, prompt: str) -> str:
        response = self.chat([LLMMessage(role="user", content=prompt)])
        return response.content


class _VertexConfigProxy:
    """Minimal config-like namespace so code reading client.config.model works."""
    def __init__(self, model: str, api_base: str, project: str, location: str):
        self.model = model
        self.api_base = api_base
        self.project = project
        self.location = location
        self.is_configured = True
