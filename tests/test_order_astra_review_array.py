"""Counterexamples from the independent Astra review of the declaration checker."""

import importlib.util
from pathlib import Path

import pytest


def _checker():
    path = Path(__file__).resolve().parents[1] / 'tools/check_function_order.py'
    spec = importlib.util.spec_from_file_location('astra_order', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problems(tmp_path, source, labels=False):
    path = tmp_path / 'lib.rs'
    path.write_text(source, encoding='utf-8')
    problems = []
    _checker().check_rust(path, problems, require_labels=labels)
    return problems


@pytest.mark.parametrize('wrapper', ['{}', 'class C:\n{}'])
@pytest.mark.parametrize('compound', ['if True:', 'try:', 'for _ in [1]:', 'with manager():'])
def test_duplicate_python_bindings_keep_order_across_compound_statements(wrapper, compound):
    first = 'def f(): return 1\n'
    second = compound + '\n def f(): return 2\n'
    if compound == 'try:':
        second += 'finally:\n pass\n'
    before, after = first + second, second + first
    if wrapper.startswith('class'):
        before, after = ('class C:\n' + ''.join(' ' + line + '\n' for line in source.splitlines())
                         for source in (before, after))
    with pytest.raises(ValueError):
        _checker().prove_reorder(before, after, '.py')


def test_duplicate_python_bindings_still_allow_independent_reorders():
    before = 'def z(): pass\ndef f(): return 1\nif True:\n def f(): return 2\ndef a(): pass\n'
    after = 'def a(): pass\ndef f(): return 1\nif True:\n def f(): return 2\ndef z(): pass\n'
    _checker().prove_reorder(before, after, '.py')


@pytest.mark.parametrize('container', ['mod data', 'impl S'])
def test_rust_attribute_tokens_are_never_normalized(container):
    before = '#[doc = stringify!({ fn zebra() {} fn alpha() {} })]\n' + container + ' {}'
    after = before.replace('fn zebra() {} fn alpha() {}', 'fn alpha() {} fn zebra() {}')
    with pytest.raises(ValueError):
        _checker().prove_reorder(before, after, '.rs')


@pytest.mark.parametrize('header', ['#[doc = stringify!({ fn z() {} })]\nmod data',
                                  'impl S<{1}>', 'impl<const N: usize> S<N>',
                                  'impl S<{1}> where S<{1}>: Sized', 'mod r#data',
                                  'unsafe /* header */ impl Trait for S', "impl<'a> S<'a>"])
def test_rust_container_bodies_are_checked_and_reordered(tmp_path, header):
    before = header + ' { fn zebra() {} fn alpha() {} }'
    after = header + ' { fn alpha() {} fn zebra() {} }'
    assert any('out of order' in problem for problem in _problems(tmp_path, before))
    _checker().prove_reorder(before, after, '.rs')


@pytest.mark.parametrize('source', [
    'fn zebra() { let _ = "// order-exempt: sample text"; }\nfn alpha() {}',
    '#[doc = "// order-exempt: sample text"]\nfn zebra() {}\nfn alpha() {}',
    'fn alpha() {} // order-exempt: belongs to alpha\nfn zebra() {}\nfn beta() {}',
])
def test_rust_exemptions_do_not_come_from_literals_or_previous_items(tmp_path, source):
    assert any('out of order' in problem for problem in _problems(tmp_path, source))


@pytest.mark.parametrize('source', [
    '// order-exempt: compile-time dependency\nfn zebra() {}\nfn alpha() {}',
    'fn zebra() {} // order-exempt: compile-time dependency\nfn alpha() {}',
    '// order-exempt: compile-time dependency\n#[inline]\nfn zebra() {}\nfn alpha() {}',
])
def test_rust_exemptions_remain_attached_to_their_declaration(tmp_path, source):
    assert not _problems(tmp_path, source)


def test_rust_external_module_declarations_remain_unchanged():
    before = 'pub mod data;\nfn zebra() {}\nfn alpha() {}'
    after = 'pub mod data;\nfn alpha() {}\nfn zebra() {}'
    _checker().prove_reorder(before, after, '.rs')


def test_rust_label_order_is_enforced_with_missing_labels_disabled(tmp_path):
    source = '// ---- zebra tests ----\n#[test] fn a() {}\n// ---- alpha tests ----\n#[test] fn b() {}'
    assert any('labels are not alphabetical' in problem for problem in _problems(tmp_path, source))


def test_rust_labels_from_other_scopes_cannot_cover_tests(tmp_path):
    source = 'mod data { // ---- example tests ----\n#[test] fn a() {} }\n#[test] fn b() {}'
    assert any("test 'b' is not in a labelled group" in problem
               for problem in _problems(tmp_path, source, labels=True))


def test_rust_labels_inside_attributes_are_not_scope_labels(tmp_path):
    source = '#[cfg(any(\n// ---- example tests ----\nunix))]\n#[test] fn a() {}'
    assert any('not in a labelled group' in problem for problem in _problems(tmp_path, source, labels=True))


def test_rust_labels_in_independent_modules_are_checked_separately(tmp_path):
    source = ('mod a {\n// ---- zebra tests ----\n#[test] fn a() {} }\n'
              'mod b {\n// ---- alpha tests ----\n#[test] fn b() {} }')
    assert not _problems(tmp_path, source, labels=True)


def test_rust_literal_lines_are_not_group_labels(tmp_path):
    source = ('const S: &str = r#"\n// ---- zebra tests ----\n// ---- alpha tests ----\n"#;\n'
              '// ---- real tests ----\n#[test] fn a() {}')
    assert not _problems(tmp_path, source, labels=True)


@pytest.mark.parametrize('header', ['pub\nfn {}() {{}}', 'fn r#{}() {{}}',
                                  'pub(crate)\nunsafe\nfn {}() {{}}',
                                  'pub /* visibility */ (crate) unsafe /* qualifier */ fn {}() {{}}',
                                  "fn {}<'a, 'b>(_: &'a str, _: &'b str) {{}}",
                                  "fn {}() -> &'static str {{ \"\" }}"])
def test_rust_multiline_and_raw_function_headers_keep_identity(tmp_path, header):
    before = header.format('zebra') + '\n' + header.format('alpha')
    after = header.format('alpha') + '\n' + header.format('zebra')
    assert any('out of order' in problem for problem in _problems(tmp_path, before))
    _checker().prove_reorder(before, after, '.rs')


@pytest.mark.parametrize('value', ['1 << 2', '1 < 2', '{1} + {2}', 'S::<{1}> {}'])
def test_rust_statement_initializers_cannot_hide_following_functions(tmp_path, value):
    prefix = f'const X: T = {value};\n'
    before = prefix + 'fn zebra() {}\nfn alpha() {}'
    after = prefix + 'fn alpha() {}\nfn zebra() {}'
    assert any('out of order' in problem for problem in _problems(tmp_path, before))
    _checker().prove_reorder(before, after, '.rs')


def test_rust_test_attributes_keep_multiline_and_raw_names_visible(tmp_path):
    source = '#[test]\npub\nfn r#case() {}\nfn helper() {}'
    problems = _problems(tmp_path, source, labels=True)
    assert any("test 'case' is not in a labelled group" in problem for problem in problems)
    assert any('after the first test' in problem for problem in problems)
