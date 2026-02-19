from __future__ import annotations

from pathlib import Path


def main() -> None:
    path = Path("vibecad/design/design_agent.py")
    text = path.read_text(encoding="utf-8")

    start = text.find("def _get_action_handler")
    if start < 0:
        raise SystemExit("Could not find def _get_action_handler")

    end = text.find("async def _handle_assign_nets", start)
    if end < 0:
        end = text.find("\n    async def", start)
    if end < 0:
        raise SystemExit("Could not find end of _get_action_handler block")

    block = text[start:end]
    lines = block.splitlines(True)

    needle = "DesignActionType.DEFINE_NET: self._handle_define_net,"
    anchor = "DesignActionType.ASSIGN_NETS: self._handle_assign_nets,"

    # Remove duplicates (keep first)
    idxs = [i for i, ln in enumerate(lines) if needle in ln]
    if len(idxs) > 1:
        for i in reversed(idxs[1:]):
            del lines[i]

    # Insert if missing
    idxs = [i for i, ln in enumerate(lines) if needle in ln]
    if not idxs:
        inserted = False
        for i, ln in enumerate(lines):
            if anchor in ln:
                indent = ln[: len(ln) - len(ln.lstrip(" "))]
                lines.insert(i + 1, f"{indent}{needle}\n")
                inserted = True
                break
        if not inserted:
            raise SystemExit("Could not find ASSIGN_NETS mapping inside _get_action_handler")

    new_block = "".join(lines)
    if new_block == block:
        print("No changes needed (DEFINE_NET dispatch already correct).")
        return

    path.write_text(text[:start] + new_block + text[end:], encoding="utf-8")
    print("Patched DEFINE_NET dispatch in _get_action_handler.")


if __name__ == "__main__":
    main()
