# VibeCAD - KiCad 7 LLM-Assisted Design Review Plugin

A KiCad 7 Python plugin that integrates an LLM for **explanation, querying, and review assistance** — not autonomous design.

## Architecture

- **Deterministic Python code** performs all parsing and rule checks
- **LLM is only used to:**
  1. Explain detected issues in plain English
  2. Answer user questions about existing design data
  3. Summarize review results for documentation

## LLM Constraints

The LLM must **NEVER**:
- Create or modify nets, footprints, or layouts
- Infer electrical specs not present in the project files
- Fetch or assume online datasheet data unless explicitly provided by the user

All LLM responses must reference specific detected facts (component refs, net names, rule IDs).

## Installation

1. Copy the `vibecad` folder to your KiCad plugins directory:
   - **Linux**: `~/.local/share/kicad/7.0/scripting/plugins/`
   - **macOS**: `~/Library/Preferences/kicad/7.0/scripting/plugins/`
   - **Windows**: `%APPDATA%\kicad\7.0\scripting\plugins\`

2. Configure an OpenAI-compatible API key:
   ```bash
   export VIBECAD_API_KEY="your-api-key"
   export VIBECAD_API_BASE="https://api.openai.com/v1"  # Optional, defaults to OpenAI
   ```

### Using GitHub Models (recommended)

VibeCAD supports GitHub Models via the OpenAI-compatible endpoint.

```bash
export GITHUB_TOKEN="<YOUR_GITHUB_PAT>"
export VIBECAD_API_BASE="https://models.github.ai/inference"
export VIBECAD_MODEL="openai/gpt-5"
```

Notes:
- If `VIBECAD_API_KEY` is not set, VibeCAD will automatically use `GITHUB_TOKEN`.
- `VIBECAD_API_BASE` and `VIBECAD_MODEL` will default to the GitHub Models values above when `GITHUB_TOKEN` is present.
- Your GitHub token must be able to call GitHub Models (PAT/Fine-grained PAT with `models: read`).

Note: Some models (including `openai/gpt-5`) expect `max_completion_tokens` rather than `max_tokens`. VibeCAD handles this automatically.

Note: Some models (including `openai/gpt-5` on GitHub Models) only support the default `temperature` value. If you set a different temperature, VibeCAD will omit the parameter to avoid request errors.

### TLS / certificates (macOS)

Some KiCad Python environments on macOS may fail HTTPS certificate verification (e.g. `CERTIFICATE_VERIFY_FAILED`).

- Preferred: keep TLS verification on and provide a CA bundle.
- Workaround: disable TLS verification.

Environment variables:
```bash
export VIBECAD_SSL_VERIFY="true"     # or "false"
export VIBECAD_CA_BUNDLE="/etc/ssl/cert.pem"  # optional
```

You can also set these in the plugin ⚙ Settings.

3. Restart KiCad and access via **Tools → External Plugins → VibeCAD**

## Features

### Deterministic Checks
- Missing board outline detection
- (More checks to be added)

### LLM-Assisted Explanations
- Plain English explanations of detected issues
- Context-aware responses referencing specific components
- Suggested follow-up checks (not actions)

## Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## License

MIT License
