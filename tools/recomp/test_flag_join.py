"""Comparison snapshots must survive a join of different CMP operands."""
from tools.recomp import config
from tools.recomp.translator import FunctionTranslator, _merge_flag_states
from tools.recomp.disasm import Operand

BASE = 0x10000

def translate_join(consumer=bytes.fromhex('0f95c0c3')):
    # test ecx,ecx; jz alternate; cmp eax,edx; jmp join; nop;
    # alternate: cmp ebx,esi; join: consumer.
    image = bytes.fromhex('85c9740539d0eb039039f3') + consumer
    config._install([config.Section('.text', BASE, len(image), 0, len(image), True)],
                              entry_point=BASE, kernel_thunk_addr=BASE,
                              origin='flag-join-test')
    db = {BASE: {'start': hex(BASE), 'end': BASE + len(image),
                 '_addr': BASE, 'size': len(image)}}
    return FunctionTranslator(image, db).translate_function(BASE, db[BASE])

def test_different_cmp_operands_join_for_setne():
    code = translate_join()
    assert 'CMP_NE(_fa, _fb)' in code, code
    assert '_flags /* setne */' not in code

def test_different_cmp_operands_join_for_cmovne():
    code = translate_join(bytes.fromhex('0f45c7c3'))
    assert 'if (CMP_NE(_fa, _fb)) eax = edi;' in code, code

def test_unknown_path_is_not_guessed():
    assert _merge_flag_states([None, ('cmp', [])]) is None

def test_mixed_operations_are_not_guessed():
    a = Operand(type='reg', reg='eax')
    b = Operand(type='reg', reg='edx')
    assert _merge_flag_states([('cmp', [a, b]), ('test', [a, b])]) is None

def test_mixed_widths_merge_to_a_marker():
    wide = [Operand(type='reg', reg='eax'), Operand(type='reg', reg='edx')]
    narrow = [Operand(type='reg', reg='al'), Operand(type='reg', reg='dl')]
    merged = _merge_flag_states([('cmp', wide), ('cmp', narrow)])
    assert merged == ('__cmp_mixed_width', wide)
    # and it keeps merging with further cmps of either width
    assert _merge_flag_states([merged, ('cmp', narrow)]) == merged
    assert _merge_flag_states([('test', wide), ('cmp', narrow)]) is None


def test_mixed_width_marker_answers_only_width_independent_conditions():
    from tools.recomp.lifter import _make_condition
    ops = [Operand(type='reg', reg='eax'), Operand(type='imm', imm=7)]
    for jcc, macro in (('jne', 'CMP_NE(_fa, _fb)'), ('jb', 'CMP_B(_fa, _fb)'),
                       ('jle', 'CMP_LE(_fas, _fbs)')):
        cond = _make_condition(jcc, '__cmp_mixed_width', ops)
        assert cond and macro in cond[0], (jcc, cond)
    for jcc in ('js', 'jns', 'jo', 'jp'):
        assert _make_condition(jcc, '__cmp_mixed_width', ops) is None, jcc
    assert _make_condition('jl', '__test_mixed_width', ops) is None


def test_join_of_16_and_32_bit_cmp_for_jne():
    # Amped 2 0x003165D9: cmp eax,7; jmp join / cmp word [esi+30h],7; join: jne
    # test ecx,ecx; jz alt; cmp eax,7; jmp join; alt: cmp word [esi+30h],7;
    # join: jne out; xor edi,edi; out: ret
    image = bytes.fromhex('85c974058' '3f807eb05' '66837e3007' '7502' '31ff' 'c3')
    config._install([config.Section('.text', BASE, len(image), 0, len(image), True)],
                    entry_point=BASE, kernel_thunk_addr=BASE, origin='flag-join-test')
    db = {BASE: {'start': hex(BASE), 'end': BASE + len(image),
                 '_addr': BASE, 'size': len(image)}}
    code = FunctionTranslator(image, db).translate_function(BASE, db[BASE])
    assert '_flags /* jne' not in code, code
    assert 'CMP_NE(_fa, _fb)' in code, code


def _translate(image):
    config._install([config.Section('.text', BASE, len(image), 0, len(image), True)],
                    entry_point=BASE, kernel_thunk_addr=BASE, origin='flag-join-test')
    db = {BASE: {'start': hex(BASE), 'end': BASE + len(image),
                 '_addr': BASE, 'size': len(image)}}
    return FunctionTranslator(image, db).translate_function(BASE, db[BASE])


def test_zero_flag_join_across_registers():
    # D3D 0x0031E1E0: test ecx,ecx; jz alt; test esi,esi; jmp join;
    # alt: or esi,80000000h; join: je out; xor edi,edi; out: ret
    code = _translate(bytes.fromhex('85c97404' '85f6eb06' '81ce00000080'
                                    '7402' '31ff' 'c3'))
    assert '_flags /* je' not in code, code
    assert '(_fa == 0)' in code, code


def test_zero_flag_join_does_not_answer_sign():
    # same shape with js at the join: SF is not merged
    code = _translate(bytes.fromhex('85c97404' '85f6eb06' '81ce00000080'
                                    '7802' '31ff' 'c3'))
    assert '_flags /* js' in code, code


def test_sbb_signed_relations_use_the_exact_result():
    # 0x0022D2FB: sub eax,edi; sbb edx,ecx; js out; jg out; xor edi,edi; out: ret
    code = _translate(bytes.fromhex('29f8' '19ca' '7804' '7f02' '31ff' 'c3'))
    assert '_flags /* jg' not in code, code
    assert '(_fbs > 0)' in code, code
    assert '_fbs = (_e > 0) - (_e < 0)' in code, code


def test_unreachable_padding_is_not_a_predecessor():
    # 0x00274D10 shape: cmp edi,ecx; jge join; xor eax,eax; jmp out;
    # pad: lea esi,[esi] (falls into join, reached by nothing); join: jne out;
    # xor edi,edi; out: ret
    code = _translate(bytes.fromhex('39cf' '7d06' '31c0' 'eb08' '8d36' '8d3f'
                                    '7502' '31ff' 'c3'))
    assert '_flags /* jne' not in code, code
