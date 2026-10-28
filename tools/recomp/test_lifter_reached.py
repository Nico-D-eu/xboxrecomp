"""A condition lifted as the `_flags` fallback carries a coverage marker.

The fallback tests a variable nothing assigns, so the branch is a guess. With
RECOMP_COVERAGE the marker reports the guest address the first time a run
reaches it (RECOMP_REACHED in recomp_types.h); without it the marker compiles
to nothing. These pin that every fallback shape -- jcc, setcc, cmovcc and
loope/loopne -- carries it with its own address, and that a condition the
lifter traced does not: a marker there would bury the guesses among sound
code.
"""
from tools.recomp import config
from tools.recomp.translator import FunctionTranslator

BASE = 0x10000


def translate(hexbytes):
    image = bytes.fromhex(hexbytes)
    config._install([config.Section('.text', BASE, len(image), 0, len(image), True)],
                    entry_point=BASE, kernel_thunk_addr=BASE, origin='reached-test')
    db = {BASE: {'start': hex(BASE), 'end': BASE + len(image),
                 '_addr': BASE, 'size': len(image)}}
    return FunctionTranslator(image, db).translate_function(BASE, db[BASE])


def marker(va):
    return f'RECOMP_REACHED(RECOMP_REACHED_FLAGS, 0x{va:08X}u);'


def test_untraced_jcc_is_marked():
    # je +1; nop; ret -- nothing before the je set the flags.
    code = translate('7401' '90' 'c3')
    assert '_flags /* je' in code, code
    assert marker(BASE) in code, code


def test_untraced_setcc_is_marked():
    # nop; sete al; ret
    code = translate('90' '0f94c0' 'c3')
    assert '_flags /* sete */' in code, code
    assert marker(BASE + 1) in code, code


def test_untraced_cmovcc_is_marked():
    # nop; cmove eax, edi; ret
    code = translate('90' '0f44c7' 'c3')
    assert '_flags /* cmove */' in code, code
    assert marker(BASE + 1) in code, code


def test_untraced_loope_is_marked():
    # top: nop; loope top; ret
    code = translate('90' 'e1fd' 'c3')
    assert '(ecx != 0) && (_flags)' in code, code
    assert marker(BASE + 1) in code, code


def test_traced_conditions_are_not_marked():
    # Negative control. cmp eax, edx; je +1; nop; sete al; ret -- both read
    # the cmp's flags, which the lifter traces.
    code = translate('39d0' '7401' '90' '0f94c0' 'c3')
    assert 'RECOMP_REACHED' not in code, code


def test_marker_is_a_no_op_without_coverage():
    # The header half: off unless the project asks for it, so harnesses that
    # compile lifted code need no recomp_reached of their own.
    import os
    header = os.path.join(os.path.dirname(__file__), '..', '..',
                          'templates', 'runtime', 'recomp_types.h')
    with open(header, encoding='utf-8') as f:
        text = f.read()
    off = text.index('#else', text.index('#ifdef RECOMP_COVERAGE'))
    assert '#define RECOMP_REACHED(_kind, _va) ((void)0)' in text[off:], text[off:off + 200]
