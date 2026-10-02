"""engine.follows_padding: an aligned seed after int3 padding is a boundary.

Amped 2's vtables point at small prologue-less methods placed right after a
switch's jump tables (0x0006FA20, 0x00070CA0). The sweep decodes the tables
as instructions and runs through the padding, so the seed lands
mid-instruction and was rejected. The padding is the evidence instead.
"""
import unittest
from types import SimpleNamespace

from .engine import DisasmEngine

BASE = 0x00010000


class _Image:
    def __init__(self, data):
        self.section = SimpleNamespace(virtual_addr=BASE, executable=True)
        self.data = data

    def get_section_at_va(self, va):
        return self.section if BASE <= va < BASE + len(self.data) else None

    def get_section_data(self, section):
        return self.data


def _engine(data):
    return DisasmEngine(_Image(data))


class FollowsPaddingTest(unittest.TestCase):

    def test_aligned_after_int3_padding(self):
        # Amped 2 at 0x0006FA1C: jump-table bytes, three int3, the method.
        data = bytes(0x1C) + bytes.fromhex("05cccccc66837 92c1b".replace(" ", ""))
        self.assertTrue(_engine(data).follows_padding(BASE + 0x20))

    def test_unaligned_is_not(self):
        data = bytes(0x1D) + b"\xcc\xcc\x66\x83"
        self.assertFalse(_engine(data).follows_padding(BASE + 0x1F))

    def test_one_int3_is_not_enough(self):
        data = bytes(0x1F) + b"\xcc\x66\x83"
        self.assertFalse(_engine(data).follows_padding(BASE + 0x20))

    def test_no_padding(self):
        data = bytes(0x1E) + b"\x8b\x44\x66\x83"
        self.assertFalse(_engine(data).follows_padding(BASE + 0x20))


if __name__ == "__main__":
    unittest.main()
