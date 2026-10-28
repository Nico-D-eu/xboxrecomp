"""
Self-check for `lahf` after an FPU compare-to-EFLAGS.

Run: py -3 tools/recomp/test_lahf_fcomi.py   (or via pytest)

MSVC's P6 isnan idiom is

    fucomip st, st(1)   ; ZF/PF/CF <- compare (unordered: all three set)
    lahf                ; AH <- SF:ZF:0:AF:0:PF:1:CF
    test ah, 44h        ; ZF | PF
    jnp  ordered

`lahf` used to lift to a comment, so AH kept whatever eax held and the parity
of `AH & 0x44` never matched the compare -- every number read as NaN. Amped 2's
Lua raised "table index is NaN" on the first numeric table key and aborted its
boot. The FPU branch conditions also read unordered (g_fp_cmp == 2) as "not
equal / not below / not parity", the opposite of the hardware.

This checks, against x86's definition, the AH that `lahf` produces and the
jcc conditions after fucomip, for less / equal / greater / unordered.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.recomp.lifter import _lahf_from_flags, _make_condition  # noqa: E402

# x86 truth after FCOMI/FUCOMI: (ZF, PF, CF); SF is cleared.
TRUTH = {-1: (0, 0, 1), 0: (1, 0, 0), 1: (0, 0, 0), 2: (1, 1, 1)}
# The low EFLAGS byte for each outcome, which the lifted compare leaves in _fa
# (RECOMP_FCMP_CC >> 8) since the upstream v0.16 merge; lahf and the jccs
# read it there.
FA = {cmp: (zf << 6) | (pf << 2) | cf for cmp, (zf, pf, cf) in TRUTH.items()}


def _expected_ah(cmp):
    zf, pf, cf = TRUTH[cmp]
    return 0x02 | (zf << 6) | (pf << 2) | cf


def _jcc_truth(jcc, cmp):
    zf, pf, cf = TRUTH[cmp]
    return {
        "je": zf, "jne": not zf, "jb": cf, "jae": not cf,
        "ja": not cf and not zf, "jbe": cf or zf,
        "jp": pf, "jnp": not pf,
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


def test_lahf_is_lifted_after_fucomip():
    stmt = _lahf_from_flags("fucomip", [])
    assert stmt is not None, "lahf after fucomip must be lifted, not dropped"
    assert "_fa" in stmt and stmt.startswith("eax = ")


def test_lahf_without_setter_falls_back():
    assert _lahf_from_flags(None, []) is None


def test_lahf_and_fpu_jcc_match_hardware():
    cc = _find_cc()
    if not cc:
        print("  SKIP no C compiler")
        return
    lahf = _lahf_from_flags("fucomip", [])
    jccs = ["je", "jne", "jb", "jae", "ja", "jbe", "jp", "jnp"]
    conds = {j: _make_condition(j, "fucomip", [])[0] for j in jccs}
    # Strip the trailing C comment the lifter appends; keep the expression.
    lahf_expr = re.sub(r"/\*.*?\*/", "", lahf)

    lines = ["#include <stdio.h>", "#include <stdint.h>",
             "static uint32_t eax, _fa; static int g_fp_cmp;",
             "int main(void) { int bad = 0; int cmps[4] = {-1, 0, 1, 2};",
             "  unsigned fa[4] = {%uu, %uu, %uu, %uu};" % (FA[-1], FA[0], FA[1], FA[2]),
             " for (int i = 0; i < 4; i++) { g_fp_cmp = cmps[i]; _fa = fa[i];"
             " eax = 0xDEAD00EFu;",
             "  " + lahf_expr,
             '  printf("ah %d %u\\n", g_fp_cmp, (eax >> 8) & 0xFFu);',
             "  if ((eax & 0xFFFF00FFu) != 0xDEAD00EFu) bad = 1;"]
    for j, c in conds.items():
        c = re.sub(r"/\*.*?\*/", "", c)
        lines.append(f'  printf("{j} %d %d\\n", g_fp_cmp, ({c}) ? 1 : 0);')
    lines += [" }", ' printf("bad %d\\n", bad); return 0; }']

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "t.c")
        exe = os.path.join(tmp, "t.exe")
        open(src, "w").write("\n".join(lines))
        if cc[0] == "gnu":
            cmd = [cc[1], "-w", src, "-o", exe]
        else:
            cmd = [cc[1], "/nologo", "/w", src, "/Fe" + exe, "/Fo" + tmp + "\\"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
        out = subprocess.run([exe], capture_output=True, text=True).stdout

    for line in out.splitlines():
        parts = line.split()
        if parts[0] == "bad":
            assert parts[1] == "0", "lahf wrote outside AH"
        elif parts[0] == "ah":
            cmp, ah = int(parts[1]), int(parts[2])
            assert ah & 0xC5 == _expected_ah(cmp) & 0xC5, (cmp, hex(ah))
            # The idiom itself: `test ah, 44h; jnp` is taken when exactly one
            # of ZF/PF is set -- equal and ordered. For `n == n`, the only
            # outcomes are equal and unordered, which is why it tests isnan.
            parity_odd = bin(ah & 0x44).count("1") % 2 == 1
            assert parity_odd == (cmp == 0), (cmp, hex(ah))
        else:
            jcc, cmp, taken = parts[0], int(parts[1]), int(parts[2])
            assert bool(taken) == bool(_jcc_truth(jcc, cmp)), (jcc, cmp, taken)


if __name__ == "__main__":
    test_lahf_is_lifted_after_fucomip()
    test_lahf_without_setter_falls_back()
    test_lahf_and_fpu_jcc_match_hardware()
    print("OK")
