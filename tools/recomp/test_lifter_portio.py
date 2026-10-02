"""in/out lift to the runtime's port model.

D3D's vblank handler reads the TV encoder's field pin with `in al, dx`
(dx = 0x80C0). Left untranslated, al kept its old value, the field never
changed, and XMV, which starts a movie only on an even field, never started
one. These pin the translation: the register and immediate port forms go to
xbox_PortIn/xbox_PortOut with the right width, and the string forms stay
unimplemented.
"""
import unittest

from .disasm import Instruction, Operand
from .lifter import Lifter


def _lift(mnemonic, op_str, operands):
    lifter = Lifter()
    insn = Instruction(0x00012340, 2, mnemonic, op_str, "",
                       operands=list(operands))
    return lifter, " ".join(lifter.lift_instruction(insn))


def _reg(name):
    return Operand(type="reg", reg=name)


def _imm(v):
    return Operand(type="imm", imm=v)


class PortIoTest(unittest.TestCase):

    def test_in_al_dx(self):
        lifter, out = _lift("in", "al, dx", [_reg("al"), _reg("dx")])
        self.assertIn("SET_LO8(eax, xbox_PortIn((uint16_t)LO16(edx), 1));", out)
        self.assertNotIn("RECOMP_UNIMPL", out)
        self.assertNotIn("in", lifter.unimplemented)

    def test_in_eax_imm(self):
        _, out = _lift("in", "eax, 0x32", [_reg("eax"), _imm(0x32)])
        self.assertIn("eax = xbox_PortIn((uint16_t)0x32, 4);", out)

    def test_out_dx_ax(self):
        _, out = _lift("out", "dx, ax", [_reg("dx"), _reg("ax")])
        self.assertIn("xbox_PortOut((uint16_t)LO16(edx), LO16(eax), 2);", out)

    def test_string_form_stays_unimplemented(self):
        _, out = _lift("insb", "byte ptr es:[edi], dx", [])
        self.assertIn("RECOMP_UNIMPL", out)


if __name__ == "__main__":
    unittest.main()
