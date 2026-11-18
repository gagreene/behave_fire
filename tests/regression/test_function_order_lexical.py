"""Lexical directives and Rust literal boundaries must retain their actual meaning."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / 'tools/check_function_order.py'


def _checker():
    spec = importlib.util.spec_from_file_location('order_lexical', TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problems(tmp_path, name, source, labels=True):
    path = tmp_path / name
    path.write_text(source)
    module = _checker()
    problems = []
    (module.check_python if path.suffix == '.py' else module.check_rust)(path, problems, require_labels=labels)
    return problems


# ---- checker token and scope ownership tests ----

def test_helper_classes_retain_ordinary_method_order(tmp_path):
    assert _problems(tmp_path, 'test_helpers.py', 'class Helper:\n def zebra(self): pass\n def alpha(self): pass\n')


def test_labels_from_other_scopes_do_not_cover_module_tests(tmp_path):
    source = 'class Helper:\n # ---- sample tests ----\n def helper(self): pass\ndef test_a(): pass\n'
    assert any('not in a labelled group' in item for item in _problems(tmp_path, 'test_scope.py', source))


def test_literal_marker_is_not_an_exemption(tmp_path):
    source = 'def zebra(value="# order-exempt: sample text"): pass\ndef alpha(): pass\n'
    assert any('out of order' in item for item in _problems(tmp_path, 'module.py', source))


def test_marker_in_another_indentation_scope_does_not_exempt_a_definition(tmp_path):
    source = 'def _first():\n pass\n # order-exempt: belongs to first\ndef zebra(): pass\ndef alpha(): pass\n'
    assert any('out of order' in item for item in _problems(tmp_path, 'module.py', source))


@pytest.mark.parametrize('literal', ['"x"', 'r#"x"#', 'b"x"', 'br#"x"#', 'c"x"', 'cr#"x"#', "'x'", "b'x'"])
def test_literal_suffix_is_one_rust_token(literal):
    before = f'fn f() {{ m!({literal}suffix); }}'
    after = f'fn f() {{ m!({literal} suffix); }}'
    with pytest.raises(ValueError):
        _checker().prove_reorder(before, after, '.rs')


@pytest.mark.parametrize('suffix', ['℘', '℮', 'a\u0301'])
@pytest.mark.parametrize('literal', ['"x"', 'r#"x"#', 'b"x"', 'cr#"x"#', "'x'", '1', '1.0'])
def test_unicode_literal_suffix_boundaries_are_preserved(literal, suffix):
    with pytest.raises(ValueError):
        _checker().prove_reorder(f'fn f(){{m!({literal}{suffix});}}', f'fn f(){{m!({literal} {suffix});}}', '.rs')


def test_unicode_identifier_boundaries_are_preserved():
    with pytest.raises(ValueError):
        _checker().prove_reorder('fn f(){m!(℘foo);}', 'fn f(){m!(℘ foo);}', '.rs')


def test_raw_c_string_scanners_agree():
    source = 'fn f() { let _s = cr#"x" } "y"#; }'
    _checker().prove_reorder(source, source, '.rs')


def test_literal_text_does_not_supply_a_label(tmp_path):
    source = 'SAMPLE = """\n# ---- example tests ----\n"""\ndef test_a(): pass\n'
    assert any('not in a labelled group' in item for item in _problems(tmp_path, 'test_string.py', source))


def test_literal_text_does_not_change_label_order(tmp_path):
    source = ('SAMPLE = """\n# ---- zebra tests ----\n# ---- alpha tests ----\n"""\n'
              '# ---- real tests ----\ndef test_a(): pass\n')
    assert not _problems(tmp_path, 'test_string.py', source)


# ---- rust same-line attributes tests ----

@pytest.mark.parametrize('attribute', ['#[inline]', '#[doc = "[not end]"] #[inline]',
                                     '#[cfg(all(\n test, feature="x"\n))]'])
def test_same_line_rust_attributes_keep_declarations_visible(tmp_path, attribute):
    source = f'{attribute} fn zebra() {{}}\n{attribute} fn alpha() {{}}'
    assert any('out of order' in item for item in _problems(tmp_path, 'lib.rs', source))
    _checker().prove_reorder(source, f'{attribute} fn alpha() {{}}\n{attribute} fn zebra() {{}}', '.rs')


def test_same_line_rust_test_attribute_keeps_test_rules(tmp_path):
    problems = _problems(tmp_path, 'lib.rs', '#[test] fn sample() {}\nfn helper() {}')
    assert any('not in a labelled group' in item for item in problems)
    assert any('after the first test' in item for item in problems)


# ---- test class ordering and grouping tests ----

@pytest.mark.parametrize('header', ['class TestMath:', 'class Math(unittest.TestCase):'])
def test_test_classes_keep_group_order_instead_of_alphabetizing_tests(tmp_path, header):
    source = header + '\n # ---- arithmetic tests ----\n def test_z(self): pass\n def test_a(self): pass\n'
    assert not _problems(tmp_path, 'test_math.py', source)


def test_test_classes_require_labels_and_early_helpers(tmp_path):
    source = 'class TestMath:\n def test_a(self): pass\n def zebra_helper(self): pass\n'
    problems = _problems(tmp_path, 'test_math.py', source)
    assert any('not in a labelled group' in item for item in problems)
    assert any('after the first test' in item for item in problems)


def test_test_labels_are_checked_per_class(tmp_path):
    source = ('class TestOne:\n # ---- zebra tests ----\n def test_a(self): pass\n'
              'class TestTwo:\n # ---- alpha tests ----\n def test_a(self): pass\n')
    assert not _problems(tmp_path, 'test_classes.py', source)


@pytest.mark.parametrize('prefix', [' __test__ = False\n', ' def __init__(self): pass\n',
                                   ' __test__: bool = False\n', ' def __new__(cls): pass\n'])
def test_uncollected_test_named_classes_retain_ordinary_order(tmp_path, prefix):
    source = 'class TestData:\n' + prefix + ' def test_z(self): pass\n def test_a(self): pass\n'
    assert any('out of order' in item for item in _problems(tmp_path, 'test_helper.py', source, labels=False))
