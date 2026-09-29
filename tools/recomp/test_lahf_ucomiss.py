"""
Self-check for `lahf` and jcc after an SSE scalar compare (comiss/ucomiss).

Run: py -3 tools/recomp/test_lahf_ucomiss.py   (or via pytest)

MSVC asks "is this float non-zero" as

    ucomiss xmm1, xmm0  ; xmm0 = 0.0  -- ZF/PF/CF <- compare
    lahf                ; AH <- SF:ZF:0:AF:0:PF:1:CF
    test ah, 44h        ; ZF | PF
    jp   nonzero        ; even parity: neither (greater/less) or both (NaN)

The comiss branch of _make_condition had no answer for js, so
_lahf_from_flags gave up and lahf stayed a comment: AH kept whatever eax
held, and the branch went either way depending on stale register bits.
Amped 2's scene graph (sub_000BD8E0) uses exactly this to decide whether a
node has a position, rotation or scale; one GUI camera lost its translation
and looked sideways, blanking most of the menus. jp/jnp were also constant
"ordered", which is wrong for NaN.

This checks, against x86's definition, the AH that `lahf` produces and every
jcc condition after ucomiss, for less / equal / greater / unordered.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.recomp.lifter import _lahf_from_flags, _make_condition  # noqa: E402


class _Op:
    def __init__(self, reg):
        self.type = "reg"
        self.reg = reg
        self.size = 16
        self.mem_size = None
        self.imm = 0
        self.text = reg


OPS = [_Op("xmm1"), _Op("xmm0")]

# (a, b) for less / equal / greater / unordered, and x86's (ZF, PF, CF).
CASES = [("1.0", "2.0", (0, 0, 1)), ("2.0", "2.0", (1, 0, 0)),
         ("3.0", "2.0", (0, 0, 0)), ("NAN", "2.0", (1, 1, 1))]


def _jcc_truth(jcc, flags):
    zf, pf, cf = flags
    return {
        "je": zf, "jne": not zf, "jb": cf, "jae": not cf,
        "ja": not cf and not zf, "jbe": cf or zf,
        "jp": pf, "jnp": not pf, "js": 0, "jns": 1,
    }[jcc]


def _find_cc():
    for name in ("clang", "gcc", "cc"):
        p = shutil.which(name)
        if p:
            return ("gnu", p)
    p = shutil.which("cl")
    if p:
        return ("msvc", p)
    return None


def test_lahf_is_lifted_after_ucomiss():
    for setter in ("ucomiss", "comiss", "ucomisd", "comisd"):
        stmt = _lahf_from_flags(setter, OPS)
        assert stmt is not None, f"lahf after {setter} must be lifted"
        assert stmt.startswith("eax = ") and "_fca" in stmt


def test_lahf_and_sse_jcc_match_hardware():
    cc = _find_cc()
    if not cc:
        print("  SKIP no C compiler")
        return
    lahf = re.sub(r"/\*.*?\*/", "", _lahf_from_flags("ucomiss", OPS))
    jccs = ["je", "jne", "jb", "jae", "ja", "jbe", "jp", "jnp", "js", "jns"]
    conds = {j: re.sub(r"/\*.*?\*/", "", _make_condition(j, "ucomiss", OPS)[0])
             for j in jccs}

    lines = ["#include <stdio.h>", "#include <stdint.h>", "#include <math.h>",
             "static uint32_t eax; static double _fca, _fcb;",
             "int main(void) { int bad = 0;"]
    for i, (a, b, _) in enumerate(CASES):
        lines += [f" _fca = {a}; _fcb = {b}; eax = 0xDEAD00EFu;",
                  "  " + lahf,
                  f'  printf("ah {i} %u\\n", (eax >> 8) & 0xFFu);',
                  "  if ((eax & 0xFFFF00FFu) != 0xDEAD00EFu) bad = 1;"]
        for j, c in conds.items():
            lines.append(f'  printf("{j} {i} %d\\n", ({c}) ? 1 : 0);')
    lines += [' printf("bad %d\\n", bad); return 0; }']

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "t.c")
        exe = os.path.join(tmp, "t.exe")
        open(src, "w").write("\n".join(lines))
        if cc[0] == "gnu":
            cmd = [cc[1], "-w", src, "-o", exe, "-lm"]
        else:
            cmd = [cc[1], "/nologo", "/w", src, "/Fe" + exe, "/Fo" + tmp + "\\"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
        out = subprocess.run([exe], capture_output=True, text=True).stdout

    seen = 0
    for line in out.splitlines():
        parts = line.split()
        if parts[0] == "bad":
            assert parts[1] == "0", "lahf wrote outside AH"
            continue
        i = int(parts[1])
        flags = CASES[i][2]
        if parts[0] == "ah":
            zf, pf, cf = flags
            want = 0x02 | (zf << 6) | (pf << 2) | cf
            ah = int(parts[2])
            assert ah & 0xC5 == want & 0xC5, (CASES[i], hex(ah))
        else:
            assert bool(int(parts[2])) == bool(_jcc_truth(parts[0], flags)), \
                (parts[0], CASES[i])
        seen += 1
    assert seen == len(CASES) * (1 + len(jccs))


if __name__ == "__main__":
    test_lahf_is_lifted_after_ucomiss()
    test_lahf_and_sse_jcc_match_hardware()
    print("OK")
