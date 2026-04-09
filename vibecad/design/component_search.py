"""
Component Web Search for VibeCAD.

Provides free, keyless web search for electronic component data:
- Pricing & availability (LCSC/EasyEDA, Mouser affiliate, DigiKey)
- Datasheet URLs
- Component specifications & parameters
- Alternative/equivalent parts

Fallback chain (all free, no API key required):
1. LCSC/EasyEDA API — richest free data, no auth
2. Octopart public search — scrapes publicly available component pages
3. Generic web fetch — grabs datasheet PDFs from known URLs

For richer data, optional API keys can be configured:
- NEXAR_CLIENT_ID + NEXAR_CLIENT_SECRET — Octopart/Nexar GraphQL (1000 free queries/month)
- DIGIKEY_CLIENT_ID — DigiKey Product Information API (free tier)
- MOUSER_API_KEY — Mouser Search API (free tier, 1000 req/day)
"""

import json
import logging
import os
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ── SSL helper (reused from library_manager pattern) ────────────
def _make_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx

_SSL_CTX = _make_ssl_context()


# ── Data classes ────────────────────────────────────────────────

@dataclass
class ComponentPrice:
    """Price break for a component."""
    quantity: int
    unit_price: float
    currency: str = "USD"

    def __repr__(self) -> str:
        return f"{self.quantity}+ @ ${self.unit_price:.4f}"


@dataclass
class ComponentInfo:
    """Unified component information from any source."""
    mpn: str                                        # Manufacturer Part Number
    manufacturer: str = ""
    description: str = ""
    package: str = ""
    category: str = ""
    datasheet_url: str = ""
    product_url: str = ""
    image_url: str = ""
    source: str = ""                                # Which API returned this
    lcsc_number: str = ""
    stock: Optional[int] = None
    prices: List[ComponentPrice] = field(default_factory=list)
    parameters: Dict[str, str] = field(default_factory=dict)  # e.g. {"Resistance": "10kΩ"}
    alternatives: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Human-readable summary for LLM context or UI display."""
        lines = [f"**{self.mpn}** by {self.manufacturer}"]
        if self.description:
            lines.append(f"  {self.description}")
        if self.package:
            lines.append(f"  Package: {self.package}")
        if self.parameters:
            params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
            lines.append(f"  Specs: {params}")
        if self.datasheet_url:
            lines.append(f"  Datasheet: {self.datasheet_url}")
        if self.prices:
            price_str = " | ".join(str(p) for p in self.prices[:4])
            lines.append(f"  Price: {price_str}")
        if self.stock is not None:
            lines.append(f"  Stock: {self.stock:,}")
        if self.product_url:
            lines.append(f"  Link: {self.product_url}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mpn": self.mpn,
            "manufacturer": self.manufacturer,
            "description": self.description,
            "package": self.package,
            "datasheet_url": self.datasheet_url,
            "product_url": self.product_url,
            "source": self.source,
            "stock": self.stock,
            "prices": [{"qty": p.quantity, "price": p.unit_price, "currency": p.currency} for p in self.prices],
            "parameters": self.parameters,
        }


# ── Main Search Class ──────────────────────────────────────────

class ComponentWebSearch:
    """Multi-source component search with datasheet + pricing retrieval.

    All primary methods are synchronous (stdlib only).  The class uses
    a fallback chain so it always returns *something* even without API keys.
    """

    def __init__(self):
        # Optional API keys (from env or settings)
        self.nexar_client_id = os.environ.get("NEXAR_CLIENT_ID", "")
        self.nexar_client_secret = os.environ.get("NEXAR_CLIENT_SECRET", "")
        self.mouser_api_key = os.environ.get("MOUSER_API_KEY", "")
        self.digikey_client_id = os.environ.get("DIGIKEY_CLIENT_ID", "")

        # Token cache for Nexar OAuth
        self._nexar_token: Optional[str] = None
        self._nexar_token_expires: float = 0

    # ── Public API ──────────────────────────────────────────────

    def search(self, query: str, limit: int = 5) -> List[ComponentInfo]:
        """Search for a component across all available sources.

        Returns results from the first source that succeeds, merged if
        multiple sources return data.
        """
        query = (query or "").strip()
        if len(query) < 2:
            return []

        results: List[ComponentInfo] = []

        # 1. LCSC/EasyEDA (free, keyless, richest data on Chinese-sourced parts)
        try:
            lcsc_results = self._search_lcsc(query, limit)
            results.extend(lcsc_results)
        except Exception as e:
            logger.debug("LCSC search failed: %s", e)

        # 2. Mouser (if API key configured)
        if self.mouser_api_key and len(results) < limit:
            try:
                mouser_results = self._search_mouser(query, limit - len(results))
                results.extend(mouser_results)
            except Exception as e:
                logger.debug("Mouser search failed: %s", e)

        # 3. Nexar/Octopart (if credentials configured)
        if self.nexar_client_id and self.nexar_client_secret and len(results) < limit:
            try:
                nexar_results = self._search_nexar(query, limit - len(results))
                results.extend(nexar_results)
            except Exception as e:
                logger.debug("Nexar search failed: %s", e)

        return results[:limit]

    def search_github(self, query: str, limit: int = 5) -> List[ComponentInfo]:
        """Search GitHub repositories for KiCad-related assets (repo-level, no auth).

        This is used for symbol/footprint discovery queries where component
        commerce APIs (LCSC/Mouser/etc.) are the wrong source.
        """
        q = (query or "").strip()
        if len(q) < 2:
            return []
        # Remove common web-search syntax / boilerplate to improve GitHub repo search.
        q = re.sub(r"\bsite:github\.com\b", " ", q, flags=re.I)
        q = re.sub(r"\bkicad\b", " KiCad ", q, flags=re.I)
        q = re.sub(r"\b(symbol|footprint|library|libraries)\b", " ", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
        if not q:
            q = "KiCad footprint"

        url = f"https://api.github.com/search/repositories?q={quote_plus(q + ' KiCad')}&per_page={max(1, min(int(limit), 20))}"
        req = Request(url, headers={
            "User-Agent": "VibeCAD/0.4.0",
            "Accept": "application/vnd.github+json",
        })
        try:
            with urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug("GitHub search failed: %s", e)
            return []

        items = data.get("items") or []
        if not isinstance(items, list):
            return []
        out: List[ComponentInfo] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            full_name = str(it.get("full_name") or "").strip()
            html_url = str(it.get("html_url") or "").strip()
            desc = str(it.get("description") or "").strip()
            owner = it.get("owner") if isinstance(it.get("owner"), dict) else {}
            owner_login = str(owner.get("login") or "").strip()
            if not full_name or not html_url:
                continue
            out.append(
                ComponentInfo(
                    mpn=full_name,
                    manufacturer=owner_login or "github",
                    description=desc or "GitHub repository (possible KiCad assets)",
                    source="github",
                    product_url=html_url,
                )
            )
            if len(out) >= limit:
                break
        return out

    def get_datasheet_url(self, mpn: str) -> Optional[str]:
        """Try to find a datasheet URL for a given MPN.

        Tries multiple free sources in order.
        """
        # 1. LCSC product page often has datasheet
        try:
            results = self._search_lcsc(mpn, 1)
            if results and results[0].datasheet_url:
                return results[0].datasheet_url
        except Exception:
            pass

        # 2. Alldatasheet.com (publicly accessible)
        try:
            url = self._search_alldatasheet(mpn)
            if url:
                return url
        except Exception:
            pass

        return None

    def get_component_details(self, mpn: str) -> Optional[ComponentInfo]:
        """Get detailed info for a specific MPN."""
        results = self.search(mpn, limit=3)
        # Find exact or closest MPN match
        mpn_upper = mpn.upper().replace("-", "").replace(" ", "")
        for r in results:
            r_mpn = r.mpn.upper().replace("-", "").replace(" ", "")
            if r_mpn == mpn_upper or mpn_upper in r_mpn or r_mpn in mpn_upper:
                return r
        return results[0] if results else None

    def search_for_llm_context(self, query: str, limit: int = 3) -> str:
        """Search and return a compact text block suitable for LLM context injection."""
        results = self.search(query, limit)
        if not results:
            return f"No component data found for '{query}'."
        return "\n\n".join(r.to_text() for r in results)

    # ── LCSC / EasyEDA (free, keyless) ──────────────────────────

    def _search_lcsc(self, query: str, limit: int) -> List[ComponentInfo]:
        """Search LCSC/EasyEDA Pro API. Free, no authentication."""
        url = f"https://pro.easyeda.com/api/eda/product/search?keyword={quote_plus(query)}&limit={limit}"
        req = Request(url, headers={"User-Agent": "VibeCAD/0.4.0"})
        with urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # The EasyEDA API usually returns a dict, but under some error modes it
        # can return a list (or other JSON) which previously crashed with:
        #   "'list' object has no attribute 'get'"
        if not isinstance(data, dict):
            return []

        result = data.get("result") or {}
        if isinstance(result, list):
            products = result
        elif isinstance(result, dict):
            products = result.get("productList") or []
        else:
            products = []

        if not isinstance(products, list):
            products = []
        results: List[ComponentInfo] = []

        for p in products:
            if not isinstance(p, dict):
                continue
            mpn = (p.get("mpn") or "").strip()
            if not mpn:
                continue

            lcsc = (p.get("number") or "").strip()
            pkg = (p.get("package") or "").strip()
            mfr = (p.get("manufacturer") or "").strip()
            desc = (p.get("description") or "").strip()
            stock_val = p.get("stock")
            
            info = ComponentInfo(
                mpn=mpn,
                manufacturer=mfr,
                description=desc or f"{mfr} {mpn} {pkg}".strip(),
                package=pkg,
                source="lcsc",
                lcsc_number=lcsc,
                stock=int(stock_val) if str(stock_val or "").strip().isdigit() else None,
                product_url=f"https://www.lcsc.com/product-detail/{lcsc}.html" if lcsc else "",
                datasheet_url="",
            )

            # Extract price breaks if available
            price_list = p.get("price") or p.get("priceList") or []
            if isinstance(price_list, dict):
                price_list = list(price_list.values())
            if isinstance(price_list, list):
                for pb in price_list:
                    if not isinstance(pb, dict):
                        continue
                    try:
                        qty = int(pb.get("ladder") or pb.get("quantity") or 0)
                        price = float(pb.get("productPrice") or pb.get("price") or 0)
                        if qty > 0 and price > 0:
                            info.prices.append(ComponentPrice(
                                quantity=qty, unit_price=price, currency="USD",
                            ))
                    except (ValueError, TypeError):
                        pass

            # Extract parameters/attributes
            attrs = p.get("attributes") or p.get("paramList") or {}
            if isinstance(attrs, dict):
                info.parameters = {k: str(v) for k, v in attrs.items() if v}
            elif isinstance(attrs, list):
                for attr in attrs:
                    if isinstance(attr, dict):
                        k = attr.get("key") or attr.get("name") or ""
                        v = attr.get("value") or ""
                        if k and v:
                            info.parameters[k] = str(v)

            # Try to get datasheet URL from LCSC product detail
            ds = p.get("datasheet") or p.get("datasheetUrl") or ""
            if ds:
                info.datasheet_url = ds

            results.append(info)

        return results

    # ── Mouser (free API key, 1000 req/day) ─────────────────────

    def _search_mouser(self, query: str, limit: int) -> List[ComponentInfo]:
        """Search Mouser API. Requires MOUSER_API_KEY env var."""
        if not self.mouser_api_key:
            return []

        url = f"https://api.mouser.com/api/v1/search/keyword?apiKey={self.mouser_api_key}"
        payload = json.dumps({
            "SearchByKeywordRequest": {
                "keyword": query,
                "records": limit,
                "startingRecord": 0,
                "searchOptions": "",
                "searchWithYourSignUpLanguage": "",
            }
        }).encode("utf-8")

        req = Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "User-Agent": "VibeCAD/0.4.0",
        })

        with urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        parts = (data.get("SearchResults") or {}).get("Parts") or []
        results: List[ComponentInfo] = []

        for p in parts:
            mpn = (p.get("ManufacturerPartNumber") or "").strip()
            if not mpn:
                continue

            info = ComponentInfo(
                mpn=mpn,
                manufacturer=(p.get("Manufacturer") or "").strip(),
                description=(p.get("Description") or "").strip(),
                source="mouser",
                product_url=(p.get("ProductDetailUrl") or "").strip(),
                datasheet_url=(p.get("DataSheetUrl") or "").strip(),
                image_url=(p.get("ImagePath") or "").strip(),
            )

            # Stock
            avail = p.get("Availability") or ""
            stock_match = re.search(r"(\d[\d,]*)", str(avail).replace(",", ""))
            if stock_match:
                try:
                    info.stock = int(stock_match.group(1))
                except ValueError:
                    pass

            # Prices
            for pb in (p.get("PriceBreaks") or []):
                try:
                    qty = int(pb.get("Quantity") or 0)
                    price_str = (pb.get("Price") or "").replace("$", "").replace(",", "").strip()
                    price = float(price_str) if price_str else 0
                    currency = pb.get("Currency") or "USD"
                    if qty > 0 and price > 0:
                        info.prices.append(ComponentPrice(qty, price, currency))
                except (ValueError, TypeError):
                    pass

            results.append(info)

        return results

    # ── Nexar / Octopart (free tier: 1000 queries/month) ───────

    def _get_nexar_token(self) -> Optional[str]:
        """Get an OAuth2 token for the Nexar API."""
        if self._nexar_token and time.time() < self._nexar_token_expires:
            return self._nexar_token

        if not self.nexar_client_id or not self.nexar_client_secret:
            return None

        try:
            payload = urlencode({
                "grant_type": "client_credentials",
                "client_id": self.nexar_client_id,
                "client_secret": self.nexar_client_secret,
            }).encode("utf-8")

            req = Request(
                "https://identity.nexar.com/connect/token",
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            self._nexar_token = data.get("access_token")
            self._nexar_token_expires = time.time() + int(data.get("expires_in", 3600)) - 60
            return self._nexar_token
        except Exception as e:
            logger.warning("Nexar token acquisition failed: %s", e)
            return None

    def _search_nexar(self, query: str, limit: int) -> List[ComponentInfo]:
        """Search Nexar/Octopart GraphQL API."""
        token = self._get_nexar_token()
        if not token:
            return []

        graphql_query = """
        query SearchParts($q: String!, $limit: Int!) {
          supSearch(q: $q, limit: $limit) {
            results {
              part {
                mpn
                manufacturer { name }
                shortDescription
                bestDatasheet { url }
                bestImage { url }
                specs { attribute { name } displayValue }
                sellers {
                  company { name }
                  offers {
                    inventoryLevel
                    prices { quantity price currency }
                  }
                }
              }
            }
          }
        }
        """

        payload = json.dumps({
            "query": graphql_query,
            "variables": {"q": query, "limit": limit},
        }).encode("utf-8")

        req = Request(
            "https://api.nexar.com/graphql",
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "VibeCAD/0.4.0",
            },
        )

        with urlopen(req, timeout=20, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results_data = (
            (data.get("data") or {}).get("supSearch") or {}
        ).get("results") or []
        results: List[ComponentInfo] = []

        for r in results_data:
            part = r.get("part") or {}
            mpn = (part.get("mpn") or "").strip()
            if not mpn:
                continue

            info = ComponentInfo(
                mpn=mpn,
                manufacturer=((part.get("manufacturer") or {}).get("name") or "").strip(),
                description=(part.get("shortDescription") or "").strip(),
                source="nexar",
                datasheet_url=((part.get("bestDatasheet") or {}).get("url") or "").strip(),
                image_url=((part.get("bestImage") or {}).get("url") or "").strip(),
            )

            # Specs
            for spec in (part.get("specs") or []):
                attr_name = ((spec.get("attribute") or {}).get("name") or "").strip()
                disp_val = (spec.get("displayValue") or "").strip()
                if attr_name and disp_val:
                    info.parameters[attr_name] = disp_val

            # Pricing from best seller
            for seller in (part.get("sellers") or []):
                for offer in (seller.get("offers") or []):
                    inv = offer.get("inventoryLevel")
                    if inv is not None and info.stock is None:
                        try:
                            info.stock = int(inv)
                        except (ValueError, TypeError):
                            pass
                    for pb in (offer.get("prices") or []):
                        try:
                            qty = int(pb.get("quantity") or 0)
                            price = float(pb.get("price") or 0)
                            cur = pb.get("currency") or "USD"
                            if qty > 0 and price > 0:
                                info.prices.append(ComponentPrice(qty, price, cur))
                        except (ValueError, TypeError):
                            pass
                    if info.prices:
                        break  # First seller with prices is enough
                if info.prices:
                    break

            results.append(info)

        return results

    # ── Alldatasheet fallback ───────────────────────────────────

    def _search_alldatasheet(self, mpn: str) -> Optional[str]:
        """Search alldatasheet.com for a datasheet PDF link.

        Returns the URL of the datasheet page (not the PDF itself, which
        requires browser interaction), or None.
        """
        try:
            url = f"https://www.alldatasheet.com/view.jsp?Searchword={quote_plus(mpn)}&sField=2"
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; VibeCAD/0.4.0)"})
            with urlopen(req, timeout=10, context=_SSL_CTX) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Look for a direct PDF link or the product page link
            pdf_match = re.search(r'href="(https?://[^"]*\.pdf[^"]*)"', html, re.IGNORECASE)
            if pdf_match:
                return pdf_match.group(1)

            # Return the search results page itself
            return url
        except Exception:
            return None

    # ── Convenience: enrich circuit context with web data ───────

    def enrich_component(self, reference: str, value: str, footprint: str = "") -> Optional[ComponentInfo]:
        """Try to find web data for a component on the board.

        Uses the component value (e.g., "10k") and footprint to search.
        """
        # Build a smart query from what we know
        query_parts = []
        if value and value not in ("~", reference):
            query_parts.append(value)
        if footprint:
            # Extract package hint from footprint name
            # e.g., "Resistor_SMD:R_0603_1608Metric" → "0603"
            pkg_match = re.search(r'(\d{4})(?:_|Metric)', footprint)
            if pkg_match:
                query_parts.append(pkg_match.group(1))
        if not query_parts:
            query_parts.append(reference)

        query = " ".join(query_parts)
        return self.get_component_details(query)
