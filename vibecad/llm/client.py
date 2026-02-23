"""
Provider-agnostic LLM client using OpenAI-compatible API.

Supports any LLM provider with an OpenAI-compatible API:
- OpenAI
- Azure OpenAI
- Anthropic (via proxy)
- Local models (Ollama, LM Studio, etc.)
- Any other OpenAI-compatible endpoint
"""

import os
import json
import logging
import ssl
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


logger = logging.getLogger(__name__)

# Guardrail: prevent pathological token requests from user settings
# from exhausting provider context windows.
MAX_COMPLETION_TOKENS = 8192
# Some providers return an empty completion (finish_reason="length") when the
# input prompt consumes most/all of the context window. Keep prompts bounded.
MAX_PROMPT_CHARS = int(os.environ.get("VIBECAD_MAX_PROMPT_CHARS", "40000") or "40000")
FALLBACK_PROMPT_CHARS = int(os.environ.get("VIBECAD_FALLBACK_PROMPT_CHARS", "12000") or "12000")


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    
    # API configuration
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    
    # Request parameters
    temperature: float = 0.3  # Low temperature for consistent explanations
    max_tokens: int = 2048
    timeout: int = 30

    # TLS / SSL
    verify_ssl: bool = True
    ca_bundle: str = ""
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def from_environment(cls) -> 'LLMConfig':
        """Create config from environment variables."""
        def _parse_bool(value: str, default: bool) -> bool:
            if value is None:
                return default
            v = str(value).strip().lower()
            if v in ("1", "true", "yes", "y", "on"):
                return True
            if v in ("0", "false", "no", "n", "off"):
                return False
            return default

        # Primary config
        api_key = os.environ.get('VIBECAD_API_KEY', '').strip()
        api_base = os.environ.get('VIBECAD_API_BASE', '').strip()
        model = os.environ.get('VIBECAD_MODEL', '').strip()

        # GitHub Models convenience:
        # If the user provides GITHUB_TOKEN, allow it to function as the API key.
        github_token = os.environ.get('GITHUB_TOKEN', '').strip()
        if not api_key and github_token:
            api_key = github_token

        # If using a GitHub token and no base/model were explicitly set,
        # default to GitHub Models OpenAI-compatible endpoint and a GPT-5 model.
        if github_token and not api_base:
            api_base = 'https://models.github.ai/inference'
        if github_token and not model:
            model = 'openai/gpt-5'

        # OpenAI default base/model if still unset
        if not api_base:
            api_base = 'https://api.openai.com/v1'
        if not model:
            model = 'gpt-4o-mini'

        return cls(
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=float(os.environ.get('VIBECAD_TEMPERATURE', '0.3')),
            max_tokens=int(os.environ.get('VIBECAD_MAX_TOKENS', '2048')),
            timeout=int(os.environ.get('VIBECAD_TIMEOUT', '30')),
            verify_ssl=_parse_bool(os.environ.get('VIBECAD_SSL_VERIFY', ''), True),
            ca_bundle=os.environ.get('VIBECAD_CA_BUNDLE', '').strip(),
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if the LLM is properly configured."""
        return bool(self.api_key)


@dataclass
class LLMMessage:
    """A message in the conversation."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {'role': self.role, 'content': self.content}


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Optional[Dict[str, Any]] = None


class LLMClient:
    """Provider-agnostic LLM client using OpenAI-compatible API.
    
    This client is designed to be used ONLY for explanation and documentation
    purposes. It should never be used to generate design modifications.
    """
    
    # System prompt that enforces the constraints
    SYSTEM_PROMPT = """You are an expert PCB design review assistant for KiCad. Your role is to:

1. EXPLAIN detected issues in clear, plain English
2. ANSWER questions about existing design data
3. SUMMARIZE review results for documentation

CRITICAL CONSTRAINTS - You must NEVER:
- Suggest creating or modifying nets, footprints, or layouts
- Infer electrical specifications not explicitly provided in the data
- Assume or reference online datasheet information unless the user provides it
- Generate any design modifications or PCB changes

RESPONSE REQUIREMENTS:
- Always reference specific facts from the provided data (component refs, net names, rule IDs)
- Explain WHY an issue matters for manufacturing or functionality
- Suggest follow-up CHECKS the user might run (not actions to take)
- Be concise but thorough

When explaining issues, structure your response as:
1. What was detected (referencing specific rule IDs and components)
2. Why this matters (impact on manufacturing, reliability, or functionality)
3. Suggested follow-up checks (not fixes)"""

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM client.
        
        Args:
            config: LLM configuration. If None, loads from environment.
        """
        self.config = config or LLMConfig.from_environment()
        
    @property
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        return self.config.is_configured
    
    def chat(self, messages: List[LLMMessage], 
             system_prompt: Optional[str] = None) -> LLMResponse:
        """Send a chat completion request with retries.

        Raises LLMError on failure.
        """
        if not self.is_available:
            raise LLMError("LLM not configured. Set VIBECAD_API_KEY environment variable.")

        def _truncate_text(s: str, limit: int) -> str:
            if not isinstance(s, str):
                s = str(s)
            if limit <= 0 or len(s) <= limit:
                return s
            # Preserve both ends: the start tends to contain instructions, the end
            # tends to contain the most recent board state / question.
            keep = max(2000, limit // 2)
            head = s[:keep]
            tail = s[-keep:]
            return head + "\n\n[... VibeCAD truncated prompt for context-window safety ...]\n\n" + tail

        prompt_limit = MAX_PROMPT_CHARS
        all_messages = [LLMMessage(role='system', content=_truncate_text(system_prompt or self.SYSTEM_PROMPT, prompt_limit))]
        for m in messages:
            all_messages.append(LLMMessage(role=m.role, content=_truncate_text(m.content, prompt_limit)))
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"
        prompt_chars = sum(len(m.content or "") for m in all_messages)

        # Build payload
        token_key = 'max_completion_tokens' if self._prefers_max_completion_tokens() else 'max_tokens'
        requested_tokens = int(self.config.max_tokens)
        token_budget = max(256, min(requested_tokens, MAX_COMPLETION_TOKENS))
        if requested_tokens > MAX_COMPLETION_TOKENS:
            logger.warning(
                "Requested max_tokens=%d exceeds safety cap; clamping to %d",
                requested_tokens,
                MAX_COMPLETION_TOKENS,
            )
        payload: Dict[str, Any] = {
            'model': self.config.model,
            'messages': [m.to_dict() for m in all_messages],
            token_key: token_budget,
        }
        temp = self._temperature_payload_value()
        if temp is not None:
            payload['temperature'] = temp

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.config.api_key}',
            'User-Agent': 'VibeCAD',
        }
        try:
            host = urlparse(self.config.api_base).netloc.lower()
        except Exception:
            host = ''
        if host.endswith('models.github.ai'):
            headers['Accept'] = 'application/vnd.github+json'
            headers['X-GitHub-Api-Version'] = '2022-11-28'
        # OpenRouter extension: explicitly disable "reasoning" payloads.
        # Some free routed models return text only in a separate `reasoning` field
        # and leave `message.content` empty, which breaks OpenAI-compatible parsing.
        if 'openrouter.ai' in host and 'include_reasoning' not in payload:
            payload['include_reasoning'] = False

        # Retry loop (simple exponential backoff, capped at 30s)
        import time
        last_error: Optional[LLMError] = None
        saw_empty_length = False
        for attempt in range(self.config.max_retries):
            try:
                response = self._make_request(url, payload, headers)
                return self._parse_response(response)
            except LLMError as e:
                last_error = e
                logger.warning(
                    "LLM attempt %d/%d failed: %s (model=%s token_budget=%s prompt_chars=%s)",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                    self.config.model,
                    token_budget,
                    prompt_chars,
                )

                # On empty-content / finish_reason=length, bump token budget once
                msg = str(e)
                if "Empty LLM response content" in msg and "finish_reason='length'" in msg:
                    # Two common causes:
                    # 1) Provider uses a different response schema (parser issue).
                    # 2) Prompt consumed nearly all context; provider returns an empty completion.
                    #
                    # First retry: *reduce* max tokens to improve odds that the request
                    # fits within the context window and yields at least some visible output.
                    if not saw_empty_length:
                        # First recovery: shrink the prompt aggressively. If the router/model
                        # has a small context window, MAX_PROMPT_CHARS may still be too large.
                        if FALLBACK_PROMPT_CHARS > 0 and prompt_limit > FALLBACK_PROMPT_CHARS:
                            prompt_limit = FALLBACK_PROMPT_CHARS
                            for i, m in enumerate(all_messages):
                                all_messages[i] = LLMMessage(role=m.role, content=_truncate_text(m.content, prompt_limit))
                            payload['messages'] = [m.to_dict() for m in all_messages]
                            prompt_chars = sum(len(m.content or "") for m in all_messages)

                        token_budget = min(token_budget, 256)
                        saw_empty_length = True
                    else:
                        # Subsequent retry: increase budget (helps when the model did produce
                        # visible output but got cut off, yet parser saw empty).
                        if token_budget < 4096:
                            token_budget = min(8192, max(512, token_budget * 2))

                    payload[token_key] = token_budget
                    # Try both keys for maximum compatibility with OpenAI-style proxies.
                    payload['max_tokens'] = token_budget
                    payload['max_completion_tokens'] = token_budget

                # Rate limit: respect Retry-After but cap at 30s
                if e.status_code == 429 and e.retry_after is not None:
                    delay = min(30.0, max(1.0, float(e.retry_after)))
                    if float(e.retry_after) > 30:
                        raise  # Too long, bail out
                    time.sleep(delay)
                    continue

                if attempt < self.config.max_retries - 1:
                    time.sleep(min(15.0, self.config.retry_delay * (attempt + 1)))

        raise last_error or LLMError("Request failed after retries")

    def _prefers_max_completion_tokens(self) -> bool:
        """Return True if this model/provider expects max_completion_tokens."""
        model = (self.config.model or "").strip().lower()
        if not model:
            return False

        # GPT-5 models frequently require `max_completion_tokens`.
        if "gpt-5" in model:
            return True

        # Some providers for newer reasoning models also prefer completion tokens.
        if model.startswith("o1") or model.startswith("o3"):
            return True

        return False

    def _temperature_payload_value(self) -> Optional[float]:
        """Return the temperature to send, or None to omit the parameter.

        Some models (notably GPT-5 via GitHub Models) only support the default
        temperature value. In those cases we omit `temperature` entirely.
        """
        model = (self.config.model or "").strip().lower()
        if not model:
            return self.config.temperature

        # GPT-5 on GitHub Models commonly rejects non-default temperatures.
        if "gpt-5" in model:
            # If the user explicitly set 1.0, we can send it; otherwise omit.
            try:
                if abs(float(self.config.temperature) - 1.0) < 1e-9:
                    return 1.0
            except Exception:
                pass
            return None

        # Default behavior: send whatever was configured.
        return self.config.temperature
    
    def _make_request(self, url: str, payload: Dict[str, Any], 
                      headers: Dict[str, str]) -> Dict[str, Any]:
        """Make HTTP request to the LLM API."""
        data = json.dumps(payload).encode('utf-8')
        request = Request(url, data=data, headers=headers, method='POST')

        # Build SSL context
        ssl_context = None
        try:
            if not self.config.verify_ssl:
                ssl_context = ssl._create_unverified_context()
            else:
                cafile = (self.config.ca_bundle or '').strip()
                if not cafile:
                    try:
                        import certifi
                        cafile = certifi.where()
                    except Exception:
                        cafile = ''
                ssl_context = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
        except Exception:
            ssl_context = None

        try:
            with urlopen(request, timeout=self.config.timeout, context=ssl_context) as response:
                return json.loads(response.read().decode('utf-8'))
        except HTTPError as e:
            body = e.read().decode('utf-8') if e.fp else ''
            retry_after = None
            try:
                ra = e.headers.get('Retry-After')
                if ra is not None:
                    retry_after = float(ra) or None
            except Exception:
                pass
            raise LLMError(f"HTTP {e.code}: {body}", status_code=int(e.code), retry_after=retry_after)
        except URLError as e:
            raise LLMError(f"Connection error: {e.reason}")
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON response: {e}")
    
    def _parse_response(self, response: Dict[str, Any]) -> LLMResponse:
        """Parse API response into LLMResponse."""
        def _coerce(value: Any) -> str:
            """Coerce various content shapes into a plain string."""
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "".join(
                    _coerce(item.get('text', item.get('content', item))) if isinstance(item, dict) else _coerce(item)
                    for item in value if item is not None
                )
            if isinstance(value, dict):
                # Some providers wrap text like {"text": {"value": "..."}} or {"value": "..."}.
                if 'text' in value:
                    return _coerce(value.get('text'))
                if 'content' in value:
                    return _coerce(value.get('content'))
                if 'value' in value:
                    return _coerce(value.get('value'))
                # Unknown dict shape: preserve signal rather than dropping to "".
                try:
                    return json.dumps(value, ensure_ascii=False, sort_keys=True)
                except Exception:
                    return str(value)
            return str(value)

        def _extract_json_snippet(text: str) -> str:
            """Extract a valid JSON array/object from arbitrary text.

            Returns a canonical JSON string (json.dumps) or "" if none found.
            """
            if not isinstance(text, str):
                text = str(text)
            s = text.strip()
            if not s:
                return ""

            def _try_extract(op: str, cl: str) -> str:
                start = s.find(op)
                if start < 0:
                    return ""
                end = s.rfind(cl)
                while end > start:
                    snippet = s[start:end + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, (list, dict)):
                            return json.dumps(obj, ensure_ascii=False)
                    except Exception:
                        pass
                    end = s.rfind(cl, start, end)
                return ""

            return _try_extract("[", "]") or _try_extract("{", "}")

        try:
            # Some "OpenAI compatible" proxies return HTTP 200 with an error payload.
            # Surface that as a hard failure.
            if isinstance(response.get('error'), dict):
                err = response.get('error') or {}
                msg = err.get('message') or err.get('type') or str(err)
                raise LLMError(f"Provider error: {msg}")

            choices = response.get('choices')
            if not isinstance(choices, list) or not choices:
                raise KeyError("choices")

            choice = choices[0] or {}
            if not isinstance(choice, dict):
                raise KeyError("choices[0]")

            message = choice.get('message') or {}
            content = _coerce(message.get('content') if isinstance(message, dict) else None)

            # Fallbacks: legacy text field, delta (streaming proxies), tool-call args
            if not content.strip():
                content = _coerce(choice.get('text'))
            if not content.strip() and isinstance(choice.get('delta'), dict):
                content = _coerce(choice['delta'].get('content'))
            if not content.strip() and isinstance(message, dict):
                # tool-call-only responses
                for tc_key in ('tool_calls', 'function_call'):
                    tc = message.get(tc_key)
                    if tc:
                        if isinstance(tc, list) and tc:
                            fn = (tc[0] or {}).get('function', {})
                            content = _coerce(fn.get('arguments'))
                        elif isinstance(tc, dict):
                            content = _coerce(tc.get('arguments'))
                        if content.strip():
                            break
            if not content.strip():
                content = _coerce(response.get('output_text'))

            # OpenRouter/free routed models sometimes return all text in `message.reasoning`
            # and leave `message.content` empty. We never surface reasoning verbatim; we only
            # salvage a JSON snippet from it (the agents already require JSON-only output).
            if not content.strip() and isinstance(message, dict):
                reasoning_text = _coerce(message.get('reasoning')) or _coerce(message.get('reasoning_details'))
                recovered = _extract_json_snippet(reasoning_text)
                if recovered:
                    content = recovered

            if not content.strip():
                finish_reason = choice.get('finish_reason')
                # Log a minimal, non-sensitive shape summary for debugging provider incompatibilities.
                try:
                    msg_keys = list(message.keys()) if isinstance(message, dict) else []
                    content_len = 0
                    reasoning_len = 0
                    try:
                        content_len = len(_coerce(message.get('content')) if isinstance(message, dict) else "")
                    except Exception:
                        content_len = 0
                    try:
                        if isinstance(message, dict):
                            reasoning_len = len(_coerce(message.get('reasoning')) or _coerce(message.get('reasoning_details')))
                    except Exception:
                        reasoning_len = 0
                    logger.debug(
                        "LLM empty content: model=%r finish_reason=%r choice_keys=%s message_keys=%s content_len=%d reasoning_len=%d usage=%r",
                        response.get('model', self.config.model),
                        finish_reason,
                        sorted(list(choice.keys())) if isinstance(choice, dict) else [],
                        sorted(msg_keys),
                        content_len,
                        reasoning_len,
                        response.get('usage', {}),
                    )
                except Exception:
                    pass
                raise LLMError(
                    f"Empty LLM response content (finish_reason={finish_reason!r}). "
                    "Provider may not be returning chat.completions-compatible 'message.content'."
                )

            return LLMResponse(
                content=content,
                model=response.get('model', self.config.model),
                usage=response.get('usage', {}),
                raw_response=response,
            )
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response format: {e}")
    
    def explain_simple(self, prompt: str) -> str:
        """Simple interface for single-turn explanation.
        
        Args:
            prompt: The prompt to send
        
        Returns:
            The assistant's response text
        """
        response = self.chat([LLMMessage(role='user', content=prompt)])
        return response.content


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
