"""`in` and `out` reach the runtime instead of vanishing.

Lifted as no-ops, `in al, dx` left AL holding whatever it held before.
D3D's vblank ISR reads the video encoder's field pin that way (port 0x80C0,
bit 5), so the field it recorded never changed, D3DDevice_GetDisplayFieldStatus
said "odd" for ever, and the XMV player -- which starts its clock on an even
field -- never showed Burnout 3's first intro frame.

The expected text is the merged lifter's (amped2-fixes + upstream v0.16):
ports are passed as uint16_t, to the one runtime definition in kernel_hal.c.
"""
import unittest

from .disasm import Instruction, Operand
from .lifter import Lifter


def _lift(mnemonic, op_str, operands):
    lifter = Lifter()
    insn = Instruction(0x00356B4C, 1, mnemonic, op_str, "",
                       operands=list(operands))
    return lifter, " ".join(lifter.lift_instruction(insn))


def _reg(name):
    return Operand(type="reg", reg=name)


def _imm(value):
    return Operand(type="imm", imm=value)


class PortIoTest(unittest.TestCase):

    def test_in_al_dx_writes_only_al(self):
        lifter, out = _lift("in", "al, dx", [_reg("al"), _reg("dx")])
        self.assertIn("SET_LO8(eax, xbox_PortIn((uint16_t)LO16(edx), 1));", out)
        self.assertEqual(lifter.unimplemented, {})

    def test_in_eax_imm_port(self):
        _, out = _lift("in", "eax, 0x61", [_reg("eax"), _imm(0x61)])
        self.assertIn("eax = xbox_PortIn((uint16_t)0x61, 4);", out)

    def test_out_dx_al(self):
        lifter, out = _lift("out", "dx, al", [_reg("dx"), _reg("al")])
        self.assertIn("xbox_PortOut((uint16_t)LO16(edx), LO8(eax), 1);", out)
        self.assertNotIn("RECOMP_UNIMPL", out)
        self.assertEqual(lifter.unimplemented, {})

    def test_helpers_are_declared_for_generated_code(self):
        import pathlib
        hdr = (pathlib.Path(__file__).resolve().parents[2] / "templates" /
               "runtime" / "recomp_types.h").read_text(encoding="utf-8")
        self.assertIn("uint32_t xbox_PortIn(uint16_t port, int size);", hdr)
        self.assertIn("void xbox_PortOut(uint16_t port, uint32_t value, int size);", hdr)


if __name__ == "__main__":
    unittest.main()
