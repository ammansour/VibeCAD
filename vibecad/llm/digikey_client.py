"""DigiKey Product Information API v4 client.

Provides two things used by the SPEC agent's datasheet pass:
  1. ``get_datasheet_url(mpn)``  — resolves an MPN to a PDF URL via the
     DigiKey /media endpoint.
  2. ``fetch_datasheet_text(url)`` — downloads the PDF and extracts plain text
     (≤ 8 000 chars) suitable for inclusion in the step-2 LLM prompt.

Auth uses OAuth2 client-credentials (2-legged).  No user login is required.
Tokens are cached in memory for their lifetime (~3600 s) and refreshed lazily.

Quick-start:
  Register a free app at https://developer.digikey.com → My Apps → Create App.
  You only need the "Product Information" API subscription.
  Set client_id / client_secret in VibeCAD settings and you're done.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://api.digikey.com/v1/oauth2/token"


def _ssl_context():
    """Return an SSL context that actually works inside KiCad's embedded Python.

    KiCad ships its own Python interpreter which often lacks the OS CA bundle.
    Resolution order:
      1. certifi (pip-installable, most reliable)
      2. macOS system trust store at /etc/ssl/cert.pem
      3. Unverified context (logs a warning — still encrypted, just no cert check)
    """
    import ssl
    # ── 1. certifi ──────────────────────────────────────────────────────────
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        logger.debug("DigiKey SSL: using certifi CA bundle")
        return ctx
    except ImportError:
        pass
    except Exception as e:
        logger.debug("DigiKey SSL: certifi failed: %s", e)

    # ── 2. macOS / common system paths ──────────────────────────────────────
    import os
    _SYSTEM_CA_PATHS = [
        "/etc/ssl/cert.pem",                              # macOS / Homebrew Python
        "/etc/ssl/certs/ca-certificates.crt",             # Debian/Ubuntu
        "/etc/pki/tls/certs/ca-bundle.crt",               # RHEL/CentOS
        "/usr/local/share/ca-certificates",               # custom installs
    ]
    for ca_path in _SYSTEM_CA_PATHS:
        if os.path.exists(ca_path):
            try:
                ctx = ssl.create_default_context(cafile=ca_path)
                logger.debug("DigiKey SSL: using system CA bundle at %s", ca_path)
                return ctx
            except Exception as e:
                logger.debug("DigiKey SSL: %s failed: %s", ca_path, e)

    # ── 3. Unverified fallback ───────────────────────────────────────────────
    logger.warning(
        "DigiKey SSL: no CA bundle found (certifi not installed, system bundle missing). "
        "Falling back to unverified HTTPS. Install certifi to fix: pip install certifi"
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
_MEDIA_URL   = "https://api.digikey.com/products/v4/search/{product_number}/media"
_DETAIL_URL  = "https://api.digikey.com/products/v4/search/{product_number}/productdetails"
_KEYWORD_URL = "https://api.digikey.com/products/v4/search/keyword"

# Maximum characters of PDF text forwarded to the LLM prompt.
_MAX_PDF_CHARS = 8_000
# Timeout for HTTP calls (seconds).
_HTTP_TIMEOUT  = 15


class DigiKeyClient:
    """Lightweight DigiKey API v4 client with lazy token refresh."""

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id.strip()
        self._client_secret = client_secret.strip()
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._ssl_ctx = _ssl_context()   # built once, reused for all requests
        self._token_lock = threading.Lock()  # prevents duplicate token fetches from parallel threads

    # ── Auth ─────────────────────────────────────────────────────────────────

    def _ensure_token(self) -> bool:
        """Fetch / refresh the Bearer token if needed.  Returns True on success."""
        if not self._client_id or not self._client_secret:
            logger.debug("DigiKey: credentials not configured, skipping.")
            return False
        # Fast path (no lock): token already valid.
        if self._access_token and time.time() < self._token_expiry - 30:
            return True
        # Slow path: acquire lock so only one thread actually fetches.
        with self._token_lock:
            # Re-check inside the lock — another thread may have refreshed.
            if self._access_token and time.time() < self._token_expiry - 30:
                return True
            try:
                import urllib.request, urllib.parse
                payload = urllib.parse.urlencode({
                    "client_id":     self._client_id,
                    "client_secret": self._client_secret,
                    "grant_type":    "client_credentials",
                }).encode()
                req = urllib.request.Request(
                    _TOKEN_URL,
                    data=payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )
                with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=_HTTP_TIMEOUT) as resp:
                    import json
                    data = json.loads(resp.read().decode())
                self._access_token = data["access_token"]
                self._token_expiry  = time.time() + int(data.get("expires_in", 3600))
                logger.info("DigiKey: OAuth token obtained (expires in %ss)", data.get("expires_in"))
                return True
            except Exception as exc:
                logger.warning("DigiKey: token fetch failed: %s", exc)
                self._access_token = None
                return False

    def _get(self, url: str) -> Optional[dict]:
        """GET a DigiKey API endpoint and return parsed JSON, or None on error."""
        if not self._ensure_token():
            return None
        try:
            import urllib.request, json
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization":      f"Bearer {self._access_token}",
                    "X-DIGIKEY-Client-Id": self._client_id,
                    "Accept":             "application/json",
                },
            )
            with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=_HTTP_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            logger.warning("DigiKey: GET %s failed: %s", url, exc)
            return None

    def _post(self, url: str, body: dict) -> Optional[dict]:
        """POST a JSON body to a DigiKey API endpoint and return parsed JSON."""
        if not self._ensure_token():
            return None
        try:
            import urllib.request, json
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization":       f"Bearer {self._access_token}",
                    "X-DIGIKEY-Client-Id": self._client_id,
                    "Content-Type":        "application/json",
                    "Accept":              "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=_HTTP_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            logger.warning("DigiKey: POST %s failed: %s", url, exc)
            return None

    def _resolve_digikey_pn(self, mpn: str) -> Optional[str]:
        """Use keyword search to find the DigiKey catalog number for an MPN.

        The /media and /productdetails endpoints require the DigiKey-internal
        product number (e.g. 'FDN340POSCT-ND'), not the manufacturer MPN.
        This method resolves the mapping for parts where the MPN string doesn't
        match DigiKey's catalog key directly.
        Returns the DigiKey part number, or None if not found.
        """
        result = self._post(_KEYWORD_URL, {"Keywords": mpn, "Limit": 1})
        if not result:
            return None
        products = result.get("Products") or []
        if not products or not isinstance(products, list):
            return None
        pn = str(products[0].get("DigiKeyPartNumber") or "").strip()
        return pn if pn else None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_datasheet_url(self, mpn: str) -> Optional[str]:
        """Return a direct PDF URL for the given MPN, or None if not found.

        Strategy:
          1. Try /media endpoint → look for MediaType containing "Datasheet".
          2. Fall back to /productdetails → DatasheetUrl field.
        """
        if not mpn:
            return None
        # Encode the MPN for use in a URL path
        import urllib.parse
        encoded = urllib.parse.quote(mpn, safe="")

        # ── Attempt 1: media endpoint ─────────────────────────────────────
        media_data = self._get(_MEDIA_URL.format(product_number=encoded))
        if media_data and isinstance(media_data.get("MediaLinks"), list):
            for link in media_data["MediaLinks"]:
                if not isinstance(link, dict):
                    continue
                media_type = str(link.get("MediaType") or "").lower()
                if "datasheet" in media_type:
                    url = str(link.get("Url") or "").strip()
                    if url.startswith("http"):
                        logger.info("DigiKey: datasheet URL for %s → %s", mpn, url)
                        return url

        # ── Attempt 2: product details ────────────────────────────────────
        detail_data = self._get(_DETAIL_URL.format(product_number=encoded))
        if detail_data:
            product = detail_data.get("Product") or {}
            url = str(product.get("DatasheetUrl") or "").strip()
            if url.startswith("http"):
                logger.info("DigiKey: datasheet URL (detail) for %s → %s", mpn, url)
                return url

        # ── Attempt 3: keyword search → resolve DigiKey PN → retry ───────
        # Many MPNs (e.g. FDN340P) don't match DigiKey's catalog key directly.
        # Resolve via keyword search first, then retry media/detail endpoints.
        dk_pn = self._resolve_digikey_pn(mpn)
        if dk_pn and dk_pn.upper() != mpn.upper():
            logger.debug("DigiKey: resolved %s → %s, retrying endpoints", mpn, dk_pn)
            dk_encoded = __import__('urllib.parse', fromlist=['parse']).parse.quote(dk_pn, safe="")
            media_data2 = self._get(_MEDIA_URL.format(product_number=dk_encoded))
            if media_data2 and isinstance(media_data2.get("MediaLinks"), list):
                for link in media_data2["MediaLinks"]:
                    if not isinstance(link, dict):
                        continue
                    if "datasheet" in str(link.get("MediaType") or "").lower():
                        url = str(link.get("Url") or "").strip()
                        if url.startswith("http"):
                            logger.info("DigiKey: datasheet URL (resolved PN) for %s → %s", mpn, url)
                            return url
            detail_data2 = self._get(_DETAIL_URL.format(product_number=dk_encoded))
            if detail_data2:
                product = detail_data2.get("Product") or {}
                url = str(product.get("DatasheetUrl") or "").strip()
                if url.startswith("http"):
                    logger.info("DigiKey: datasheet URL (resolved PN detail) for %s → %s", mpn, url)
                    return url

        logger.debug("DigiKey: no datasheet URL found for %s", mpn)
        return None

    def fetch_datasheet_text(self, pdf_url: str) -> str:
        """Download a PDF and return up to _MAX_PDF_CHARS of extracted text.

        Tries pdfplumber first (better table extraction), then pymupdf (fitz),
        then a naive byte-stream regex on raw PDF text objects as a last resort.
        Returns an empty string if none of these work.
        """
        if not pdf_url:
            return ""
        raw_bytes = self._download_bytes(pdf_url)
        if not raw_bytes:
            return ""

        # ── pdfplumber ────────────────────────────────────────────────────
        try:
            import io, pdfplumber
            text_parts = []
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                for page in pdf.pages[:6]:           # first 6 pages is plenty
                    t = page.extract_text() or ""
                    text_parts.append(t)
                    if sum(len(p) for p in text_parts) >= _MAX_PDF_CHARS:
                        break
            text = "\n".join(text_parts)
            if text.strip():
                return text[:_MAX_PDF_CHARS]
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("DigiKey: pdfplumber failed for %s: %s", pdf_url, exc)

        # ── pymupdf (fitz) ────────────────────────────────────────────────
        try:
            import io, fitz  # type: ignore
            doc = fitz.open(stream=io.BytesIO(raw_bytes), filetype="pdf")
            text_parts = []
            for i, page in enumerate(doc):
                if i >= 6:
                    break
                text_parts.append(page.get_text())
                if sum(len(p) for p in text_parts) >= _MAX_PDF_CHARS:
                    break
            text = "\n".join(text_parts)
            if text.strip():
                return text[:_MAX_PDF_CHARS]
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("DigiKey: pymupdf failed for %s: %s", pdf_url, exc)

        # ── Naive fallback: regex over raw PDF BT..ET objects ────────────
        try:
            raw_text = raw_bytes.decode("latin-1", errors="replace")
            chunks = re.findall(r"BT\s+(.*?)\s+ET", raw_text, re.DOTALL)
            tokens = re.findall(r"\(([^)]{1,200})\)", " ".join(chunks[:200]))
            text = " ".join(tokens)
            if text.strip():
                return text[:_MAX_PDF_CHARS]
        except Exception as exc:
            logger.debug("DigiKey: raw PDF parse failed: %s", exc)

        return ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _download_bytes(self, url: str) -> bytes:
        """Download a URL and return its bytes, but only if it looks like a real PDF.

        Pre-filters known redirect/login URL patterns (TI suppproductinfo, Widen CDN
        with token params, etc.) that reliably return HTML rather than a PDF, saving
        a round-trip. Also rejects responses whose Content-Type is text/html or whose
        body does not start with the PDF magic bytes %PDF.
        """
        # ── Pre-filter: known non-PDF URL patterns ──────────────────────────
        _REDIRECT_PATTERNS = (
            "suppproductinfo",       # TI redirect page
            "gotoUrl=",              # TI encoded redirect
            "widen.net",             # Widen CDN (requires browser session/cookie)
            "literature.cdn",        # some CDN redirect patterns
        )
        for pat in _REDIRECT_PATTERNS:
            if pat in url:
                logger.debug("DigiKey: skipping known redirect URL for %s", url)
                return b""
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 VibeCAD-DatasheetFetcher/1.0",
                    "Accept":     "application/pdf,application/octet-stream,*/*",
                },
            )
            with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=_HTTP_TIMEOUT) as resp:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" in content_type:
                    logger.debug("DigiKey: skipping HTML response for %s (content-type: %s)", url, content_type)
                    return b""
                data = resp.read(1_500_000)   # cap at ~1.5 MB
            # Validate PDF magic bytes
            if not data.startswith(b"%PDF"):
                logger.debug("DigiKey: response for %s is not a PDF (starts with %r)", url, data[:8])
                return b""
            return data
        except Exception as exc:
            logger.warning("DigiKey: failed to download %s: %s", url, exc)
            return b""
