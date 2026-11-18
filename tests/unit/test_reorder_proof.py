"""Reorder proofs must reject changes beyond declaration order."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / 'tools/check_function_order.py'


def _proof(before, after, suffix):
    spec = importlib.util.spec_from_file_location('order_proof', TOOL)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker.prove_reorder(before, after, suffix)


@pytest.mark.parametrize('suffix,before,after', [
    ('.py', 'def z(x=1):\n return x\ndef a():\n return 2\n', 'def a():\n return 2\ndef z(x=1):\n return x\n'),
    ('.rs', 'fn z(x: f64) -> f64 { x }\npub fn a() {}', 'pub fn a() {}\nfn z(x: f64) -> f64 { x }'),
    ('.rs', 'impl A { fn z() {} fn a() {} }', 'impl A { fn a() {} fn z() {} }'),
    ('.py', 'class A:\n def z(self): pass\n def a(self): pass\n',
     'class A:\n def a(self): pass\n def z(self): pass\n'),
    ('.rs', '//! module\n#![allow(dead_code)]\nfn z(){} fn a(){}',
     '//! module\n#![allow(dead_code)]\nfn a(){} fn z(){}'),
])
def test_declaration_reordering_preserves_identity(suffix, before, after):
    _proof(before, after, suffix)


@pytest.mark.parametrize('suffix,before,after', [
    ('.py', 'def f(x=1): return x', 'def f(x=2): return x'),
    ('.py', '@one\ndef f(): pass', '@two\ndef f(): pass'),
    ('.py', 'def f(): return 1', 'def f(): return 2'),
    ('.py', 'class A:\n def f(self): pass\nclass B: pass', 'class A: pass\nclass B:\n def f(self): pass'),
    ('.py', 'X=1\ndef f(): pass', 'X=2\ndef f(): pass'),
    ('.py', 'X=True', 'X=1'),
    ('.py', 'X=1', 'X=1.0'),
    ('.py', 'class A: x=1', 'class A: x=1.0'),
    ('.py', 'def f(x): # type: (int) -> int\n return x\ndef g(x): # type: (str) -> str\n return x',
     'def f(x): # type: (str) -> str\n return x\ndef g(x): # type: (int) -> int\n return x'),
    ('.py', '@property\ndef f(): return 1\n@f.setter\ndef f(x): pass',
     '@f.setter\ndef f(x): pass\n@property\ndef f(): return 1'),
    ('.rs', '#[cfg(test)]\nfn f() {}', '#[cfg(feature="x")]\nfn f() {}'),
    ('.rs', 'fn f(x: u32) {}', 'fn f(x: u64) {}'),
    ('.rs', 'fn f() { let s = "a b"; }', 'fn f() { let s = "ab"; }'),
    ('.rs', 'const X:u8=1; fn f(){}', 'const X:u8=2; fn f(){}'),
    ('.rs', 'mod a { fn f(){} } mod b {}', 'mod a {} mod b { fn f(){} }'),
    ('.rs', 'fn f(){let x=1e-3;}', 'fn f(){let x=1e - 3;}'),
    ('.rs', 'fn f(){m!(?&);}', 'fn f(){m!(? &);}'),
    ('.rs', 'macro_rules! c { ($x:literal)=>{1}; ($x:tt $y:tt)=>{2}; } fn f()->u8 {c!(1.)}',
     'macro_rules! c { ($x:literal)=>{1}; ($x:tt $y:tt)=>{2}; } fn f()->u8 {c!(1 .)}'),
    ('.py', '# order-exempt: dependency\ndef z(): pass\ndef a(): pass',
     'def z(): pass\n# order-exempt: dependency\ndef a(): pass'),
])
def test_non_order_changes_fail(suffix, before, after):
    with pytest.raises(ValueError):
        _proof(before, after, suffix)


@pytest.mark.parametrize('suffix,source', [('.py', 'def test_a(): pass\ndef helper(): pass'),
                                          ('.rs', '#[test]\nfn a() {} fn helper() {}')])
def test_skipping_labels_keeps_helper_enforcement(tmp_path, suffix, source):
    spec = importlib.util.spec_from_file_location('order_toggle', TOOL)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    path = tmp_path / ('test_case' + suffix)
    path.write_text(source)
    problems = []
    method = checker.check_python if suffix == '.py' else checker.check_rust
    method(path, problems, require_labels=False)
    assert any('after the first test' in item for item in problems)
    assert not any('not in a labelled group' in item for item in problems)


def test_rust_functions_after_same_line_semicolons_are_checked(tmp_path):
    spec = importlib.util.spec_from_file_location('order_semicolon', TOOL)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    path = tmp_path / 'lib.rs'
    path.write_text('const X:u8=1; fn zebra() {}\nfn alpha() {}')
    problems = []
    checker.check_rust(path, problems, require_labels=False)
    assert any('out of order' in item for item in problems)
