from __future__ import annotations

from pathlib import Path


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "vibecad" / "design" / "design_agent.py"
    text = path.read_text(encoding="utf-8")
    orig = text

    # 1) Enum: add DEFINE_NET
    text = text.replace(
        "    ASSIGN_NETS = auto()\n    \n    # BOM operations\n",
        "    ASSIGN_NETS = auto()\n    DEFINE_NET = auto()\n    \n    # BOM operations\n",
    )

    # 2) System prompt: add DEFINE_NET guidance + large-board guidance
    old_prompt_snip = (
        "Connectivity prerequisite:\n"
        "- Do NOT attempt routing (DRAW_TRACK/AUTOROUTE_BOARD) until pads have nets assigned.\n"
        "- If pads are unconnected (autorouter reports \"No routable nets found\"), you MUST propose either:\n"
        "    1) ASSIGN_NETS with explicit ref/pad → net mappings, OR\n"
        "    2) instruct the user to run KiCad's \"Update PCB from Schematic\" to import the netlist,\n"
        "    then retry AUTOROUTE_BOARD.\n\n"
        "DRC iteration:\n"
    )
    new_prompt_snip = (
        "Connectivity prerequisite:\n"
        "- Do NOT attempt routing (DRAW_TRACK/AUTOROUTE_BOARD) until pads have nets assigned.\n"
        "- If pads are unconnected (autorouter reports \"No routable nets found\"), you MUST propose either:\n"
        "    1) DEFINE_NET to assign a net name to multiple pads at once, OR\n"
        "    2) ASSIGN_NETS with explicit ref/pad → net mappings, OR\n"
        "    3) instruct the user to run KiCad's \"Update PCB from Schematic\" to import the netlist,\n"
        "    then retry AUTOROUTE_BOARD.\n\n"
        "Large / complex boards:\n"
        "- Work in batches. Propose at most ~10 ADD_COMPONENT actions per iteration.\n"
        "- After each batch, verify progress (components placed count, nets assigned ratio) and continue.\n"
        "- If the schematic has few/no net labels, use LOOKUP_DATASHEET / SEARCH_WEB to pull pinout hints for key ICs/connectors, then use DEFINE_NET to name critical nets (GND, 5V, 3V3, USB_D+/D-, SPI, I2C) so routing can proceed.\n\n"
        "DRC iteration:\n"
    )
    if old_prompt_snip in text:
        text = text.replace(old_prompt_snip, new_prompt_snip)

    # 3) Advertise tool
    if "DesignActionType.DEFINE_NET," not in text:
        text = text.replace(
            "        DesignActionType.ASSIGN_NETS,\n",
            "        DesignActionType.ASSIGN_NETS,\n        DesignActionType.DEFINE_NET,\n",
            1,
        )

    # 4) LLM parameter keys
    text = text.replace(
        "- ASSIGN_NETS: { \"assignments\": [{\"ref\":\"U1\",\"pad\":\"1\",\"net\":\"GND\"}, {\"ref\":\"U1\",\"pad\":\"2\",\"net\":\"VCC\"}] }\n",
        "- ASSIGN_NETS: { \"assignments\": [{\"ref\":\"U1\",\"pad\":\"1\",\"net\":\"GND\"}, {\"ref\":\"U1\",\"pad\":\"2\",\"net\":\"VCC\"}] }\n"
        "- DEFINE_NET: { \"net\": \"<net name>\", \"pads\": [\"U1/1\", \"J1/2\", {\"ref\":\"U2\",\"pad\":\"14\"}] }\n",
    )

    # 5) Handler map
    if "DesignActionType.DEFINE_NET" not in text:
        text = text.replace(
            "            DesignActionType.ASSIGN_NETS: self._handle_assign_nets,\n",
            "            DesignActionType.ASSIGN_NETS: self._handle_assign_nets,\n            DesignActionType.DEFINE_NET: self._handle_define_net,\n",
        )

    # 6) Insert handler implementation after _handle_assign_nets
    if "async def _handle_define_net" not in text:
        insert_point = "return True, msg\n\n    async def _handle_add_component"
        if insert_point not in text:
            raise SystemExit("Could not find insertion point after _handle_assign_nets")

        define_net_impl = """

    async def _handle_define_net(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        \"\"\"Assign a single net name to multiple pads.

        Parameters:
            net: net name (e.g., \"GND\")
            pads: list of pad specs, either:
                - strings like \"U1/1\" (ref/pad)
                - objects like {\"ref\": \"U1\", \"pad\": \"1\"}
        \"\"\"
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, \"pcbnew not available (this action must run inside KiCad)\"

        params = action.parameters or {}
        net_name = str(params.get('net', '') or '').strip()
        pads = params.get('pads')
        if not net_name:
            return False, \"Missing 'net' (e.g., {net:'GND', pads:[...]})\"
        if not isinstance(pads, list) or not pads:
            return False, \"Missing 'pads' list (e.g., {net:'GND', pads:['U1/1','J1/2']})\"

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, \"No active board found\"

        def _find_footprint(ref: str):
            ref_u = (ref or '').strip().upper()
            if not ref_u:
                return None
            try:
                fn = getattr(board, 'FindFootprintByReference', None)
                if callable(fn):
                    fp = fn(ref_u)
                    if fp is not None:
                        return fp
            except Exception:
                pass
            try:
                for fp in board.GetFootprints():
                    try:
                        if str(fp.GetReference()).upper() == ref_u:
                            return fp
                    except Exception:
                        continue
            except Exception:
                return None
            return None

        def _find_pad(fp, pad_number: str):
            if fp is None:
                return None
            pn = str(pad_number)
            try:
                f = getattr(fp, 'FindPadByNumber', None)
                if callable(f):
                    p = f(pn)
                    if p is not None:
                        return p
            except Exception:
                pass
            try:
                for p in fp.Pads():
                    try:
                        if str(p.GetNumber()) == pn:
                            return p
                    except Exception:
                        continue
            except Exception:
                return None
            return None

        def _find_or_create_net(name_in: str):
            name = (name_in or '').strip()
            if not name:
                return None
            try:
                n = board.FindNet(name)
                if n is not None:
                    return n
            except Exception:
                pass
            try:
                net_item = pcbnew.NETINFO_ITEM(board, name)
                add = getattr(board, 'Add', None)
                if callable(add):
                    board.Add(net_item)
                else:
                    an = getattr(board, 'AddNet', None)
                    if callable(an):
                        an(net_item)
                return net_item
            except Exception:
                return None

        net_obj = _find_or_create_net(net_name)
        if net_obj is None:
            return False, f\"{net_name}: net create/find failed\"

        assigned = 0
        errors: List[str] = []
        invalid: List[str] = []

        for item in pads:
            ref = ''
            pad_num = ''
            if isinstance(item, str):
                spec = item.strip()
                if '/' not in spec:
                    invalid.append(spec)
                    continue
                ref, pad_num = spec.split('/', 1)
            elif isinstance(item, dict):
                ref = str(item.get('ref', '') or '')
                pad_num = str(item.get('pad', '') or '')
            else:
                invalid.append(str(item))
                continue

            ref = ref.strip().upper()
            pad_num = pad_num.strip()
            if not ref or not pad_num:
                invalid.append(str(item))
                continue

            fp = _find_footprint(ref)
            if fp is None:
                errors.append(f\"{ref}: footprint not found\")
                continue
            pad = _find_pad(fp, pad_num)
            if pad is None:
                errors.append(f\"{ref}/{pad_num}: pad not found\")
                continue

            try:
                if hasattr(pad, 'SetNet'):
                    pad.SetNet(net_obj)
                else:
                    try:
                        pad.SetNetCode(int(net_obj.GetNet()))
                    except Exception:
                        pass
                assigned += 1
            except Exception as e:
                errors.append(f\"{ref}/{pad_num}: set net failed ({e})\")

        # Rebuild connectivity so subsequent routing sees updated net codes.
        try:
            if hasattr(board, 'BuildListOfNets'):
                board.BuildListOfNets()
        except Exception:
            pass
        try:
            conn = getattr(board, 'GetConnectivity', None)
            if callable(conn):
                c = conn()
                for m in ('RecalculateRatsnest', 'Recalculate', 'Rebuild', 'Build'):
                    fn = getattr(c, m, None)
                    if callable(fn):
                        try:
                            fn()
                            break
                        except Exception:
                            continue
        except Exception:
            pass

        if assigned <= 0:
            suffix = (\" Errors: \" + \"; \".join(errors[:5])) if errors else \"\"
            if invalid:
                suffix += (\" Invalid: \" + \"; \".join(invalid[:5]))
            return False, f\"No pads were assigned to net '{net_name}'.\" + suffix

        msg = f\"Defined net '{net_name}' on {assigned} pad(s).\"
        if errors:
            msg += f\" Warnings: {len(errors)} item(s) could not be assigned.\"
        if invalid:
            msg += f\" Ignored {len(invalid)} invalid pad spec(s).\"
        return True, msg
"""

        text = text.replace(insert_point, "return True, msg" + define_net_impl + "\n\n    async def _handle_add_component")

    if text == orig:
        raise SystemExit("No changes applied (file already patched or patterns did not match)")

    path.write_text(text, encoding="utf-8")
    print(f"Patched {path}")


if __name__ == "__main__":
    main()
