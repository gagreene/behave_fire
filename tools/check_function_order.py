"""Check the function-order style rule (exit status 1 when a module breaks it).

Rule: the functions of a module (and the methods of a class or ``impl`` block, and the definitions inside a
conditional block or an inline Rust ``mod``) are in alphabetical order, case insensitive. Python: dunders first, then
other names starting with an underscore, then public names; same-named consecutive definitions (a property and its
setter) count once. Rust: functions without ``pub`` first, then ``pub`` functions. Test files: helpers first in the same
order, then the tests in groups that start with a ``# ---- <label> ----`` (Rust: ``// ---- <label> ----``) comment;
every test sits in a group, the labels are alphabetical (``other tests`` last) and the tests keep their order inside a
group.

An import-time dependency (a decorator, or a function called in a default argument or at module level) may stay above
its user: put ``# order-exempt: <reason>`` (Rust: ``// order-exempt: <reason>``) on the line above the definition or on
its ``def`` / ``fn`` line. The reason is required; a marker without one does not exempt anything.

Malformed Rust (unbalanced braces, an unterminated string or comment) is reported as a problem, like a Python syntax
error.

Usage: python tools/check_function_order.py [path ...]   (default: src tools tests rust/crates)
"""
import argparse
import ast
import io
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path

SKIP = {'.venv', 'target', '__pycache__', '.git', 'node_modules', 'build', 'dist'}
LABEL = re.compile(r'^\s*(?:#|//) ---- (.*) ----\s*$', re.M)
EXEMPT = re.compile(r'(?:#|//)\s*order-exempt:\s*(\S.*)$')

FN_RE = re.compile(r'^(?P<vis>pub(?:\([^)]*\))?\s+)?'
                   r'(?:(?:const|async|unsafe|extern(?:\s+"[^"]*")?)\s+)*fn\s+(?P<name>\w+)')
MOD_RE = re.compile(r'^(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{')


def _python_comments(source):
    """Retain every Python comment while declarations are normalized.

    :param source: Python source text.
    :return: Comment text and multiplicity.
    """
    return Counter(token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
                   if token.type == tokenize.COMMENT)


def _python_exemptions(tree, lines):
    """Bind import-dependency exemption markers to their original qualified definitions.

    :param tree: Parsed Python module.
    :param lines: Original source lines.
    :return: Qualified definition identities mapped to exemption marker text.
    """
    result = {}
    counts = Counter()
    pending = [(tree, ())]
    while pending:
        node, scope = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = (*scope, node.name)
            counts[scope] += 1
            start = min([node.lineno] + [item.lineno for item in node.decorator_list])
            candidates = ([lines[start - 2]] if start > 1 else []) + [lines[node.lineno - 1]]
            result[scope, counts[scope]] = tuple(match.group(0) for line in candidates
                                                if (match := EXEMPT.search(line)))
        pending.extend((child, scope) for child in reversed(list(ast.iter_child_nodes(node))))
    return result


def _python_value(value):
    """Preserve AST fields while normalizing declaration lists outside function bodies.

    :param value: Parsed AST value or statement list.
    :return: Structural value retaining scope, defaults, annotations, and decorators.
    """
    if isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ast.dump(value, include_attributes=False)
    if isinstance(value, list):
        if any(isinstance(item, ast.stmt) for item in value):
            groups = {}
            other = []
            for item in value:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    groups.setdefault(item.name, []).append(_python_value(item))
                else:
                    other.append(_python_value(item))
            return other, sorted(groups.items())
        return [_python_value(item) for item in value]
    if isinstance(value, ast.AST):
        return type(value).__name__, [(name, _python_value(item)) for name, item in ast.iter_fields(value)]
    return type(value).__name__, repr(value)


def _rust_tokens(text):
    """Tokenize Rust without discarding string contents, attributes, or comments.

    :param text: Rust item source validated by the item scanner.
    :return: Lexical tokens with insignificant whitespace removed.
    """
    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        start = i
        if text.startswith('//', i):
            end = text.find('\n', i)
            i = len(text) if end < 0 else end
        elif text.startswith('/*', i):
            depth = 1
            i += 2
            while i < len(text) and depth:
                if text.startswith('/*', i):
                    depth += 1
                    i += 2
                elif text.startswith('*/', i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise ValueError('Unterminated Rust comment')
        else:
            raw = re.match(r'(?:br|cr|r)(#*)"', text[i:])
            quoted = re.match(r'(?:b|c)?"', text[i:])
            char = re.match(r"b?'(?:\\.|[^'\\])+'", text[i:])
            if raw:
                close = '"' + raw.group(1)
                end = text.find(close, i + raw.end())
                if end < 0:
                    raise ValueError('Unterminated Rust raw string')
                i = end + len(close)
            elif quoted:
                i += quoted.end()
                while i < len(text) and text[i] != '"':
                    i += 2 if text[i] == '\\' else 1
                if i >= len(text):
                    raise ValueError('Unterminated Rust string')
                i += 1
            elif char:
                i += char.end()
            else:
                word = re.match(r'(?:r#)?[^\W\d]\w*|\d[\d_]*\.(?![.\w])'
                                r'|\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?[\d_]+)?\w*'
                                r'|::|->|=>|\.\.=|\.\.|<<=?|>>=?|&&|\|\||[=!<>+*/%&|^-]=', text[i:])
                i += word.end() if word else 1
        tokens.append(text[start:i])
    return tuple(tokens)


def _rust_value(text):
    """Normalize function order independently inside each Rust impl/module scope.

    :param text: One Rust scope, retaining its surrounding item identities.
    :return: Ordered non-function items and name-grouped function tokens.
    """
    # Inner module documentation and attributes belong to the container, not its first function.
    tokens = _rust_tokens(text)
    index = 0
    while index < len(tokens):
        if tokens[index].startswith('//!'):
            index += 1
        elif tokens[index:index + 3] == ('#', '!', '['):
            index += 3
            depth = 1
            while index < len(tokens) and depth:
                depth += (tokens[index] == '[') - (tokens[index] == ']')
                index += 1
            if depth:
                raise ValueError('Unterminated Rust inner attribute')
        else:
            break
    end_prefix = 0
    for token in tokens[:index]:
        end_prefix = text.index(token, end_prefix) + len(token)
    functions = {}
    other = [('container', tokens[:index])]
    text = text[end_prefix:]
    for start, end in scan_items(text):
        chunk = text[start:end]
        info = classify(chunk)
        if info[0] == 'fn':
            functions.setdefault(info[1], []).append(_rust_tokens(chunk))
        elif info[1] and (re.match(r'(?:unsafe\s+)?impl\b', info[1]) or MOD_RE.match(info[1])):
            span = body_span(chunk)
            if span is None:
                raise ValueError('Malformed Rust container')
            other.append((_rust_tokens(chunk[:span[0]]), _rust_value(chunk[span[0]:span[1]]),
                          _rust_tokens(chunk[span[1]:])))
        else:
            other.append(_rust_tokens(chunk))
    return other, sorted(functions.items())


def blocks_of(node):
    """The statement lists nested directly in a compound statement (branches, handlers, cases), not function bodies.

    :param node: AST node to inspect.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    out = []
    for attr in ('body', 'orelse', 'finalbody'):
        value = getattr(node, attr, None)
        if isinstance(value, list):
            out.append(value)
    out.extend(handler.body for handler in getattr(node, 'handlers', []))
    out.extend(case.body for case in getattr(node, 'cases', []))
    return out


def body_span(chunk):
    """Offsets of the text strictly inside the outermost braces of an item (first '{' to its matching '}').

    :param chunk: One Rust item including attributes and trivia.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    depth = 0
    i = 0
    n = len(chunk)
    opening = None
    while i < n:
        c = chunk[i]
        if chunk.startswith('//', i):
            j = chunk.find('\n', i)
            i = n if j < 0 else j + 1
            continue
        if chunk.startswith('/*', i):
            j = chunk.find('*/', i + 2)
            i = n if j < 0 else j + 2
            continue
        m = re.match(r'b?r(#*)"', chunk[i:i + 12])
        if m and (i == 0 or not (chunk[i - 1].isalnum() or chunk[i - 1] == '_')):
            close = '"' + m.group(1)
            j = chunk.find(close, i + len(m.group(0)))
            i = n if j < 0 else j + len(close)
            continue
        if c == '"':
            i += 1
            while i < n and chunk[i] != '"':
                i += 2 if chunk[i] == '\\' else 1
            i += 1
            continue
        if c == "'":
            m = re.match(r"'(\\.[^']*|[^\\'])'", chunk[i:i + 12])
            i += len(m.group(0)) if m else 1
            continue
        if c in '{':
            if opening is None and depth == 0:
                opening = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and opening is not None:
                return opening + 1, i
        i += 1
    return None


def check_py_body(nodes, scope, is_test, lines, problems, path, require_labels=True):
    """Inspect check py body according to the function-order rule.

    :param nodes: AST statements in this scope.
    :param scope: Qualified Python scope name.
    :param is_test: Whether test-section rules apply.
    :param lines: Original source lines.
    :param problems: Mutable diagnostic list.
    :param path: Source file to inspect.
    :param require_labels: Require test labels; helper placement is always checked.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    entries = []
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            if py_exempt(lines, start, n.lineno):
                continue
            if not entries or entries[-1][0] != n.name:
                entries.append((n.name, start))
    names = [name for name, _start in entries]
    if is_test:
        names = [n for n in names if not n.startswith('test')]
    expected = sorted(names, key=py_key)
    if names != expected:
        first = next(a for a, b in zip(names, expected, strict=True) if a != b)
        problems.append(f'{path}: {scope}: functions out of order near {first!r} '
                        f'(expected {expected[names.index(first)]!r})')
    if is_test:
        check_py_test_section(entries, lines, problems, path, require_labels=require_labels)


def check_py_test_section(entries, lines, problems, path, require_labels=True):
    """Every test in a labelled group, and every helper before the first test.

    :param entries: Definition names and source positions.
    :param lines: Original source lines.
    :param problems: Mutable diagnostic list.
    :param path: Source file to inspect.
    :param require_labels: Require test labels; helper placement is always checked.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    label_lines = [i + 1 for i, line in enumerate(lines) if LABEL.match(line)]
    tests = [(name, start) for name, start in entries if name.startswith('test')]
    for name, start in tests:
        if require_labels and not any(label < start for label in label_lines):
            problems.append(f'{path}: test {name!r} is not in a labelled group (# ---- <label> ----)')
    if tests:
        first_test = min(start for _name, start in tests)
        for name, start in entries:
            if not name.startswith('test') and start > first_test:
                problems.append(f'{path}: helper {name!r} is defined after the first test')


def check_python(path, problems, require_labels=True):
    """Inspect check python according to the function-order rule.

    :param path: Source file to inspect.
    :param problems: Mutable diagnostic list.
    :param require_labels: Require test labels; helper placement is always checked.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    text = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    lines = text.splitlines()
    is_test = path.name.startswith('test_')
    where = path.as_posix()
    scopes = []
    py_scopes(tree.body, '<module>', is_test, scopes)
    for nodes, scope, scope_is_test in scopes:
        check_py_body(nodes, scope, scope_is_test, lines, problems, where, require_labels=require_labels)
    if is_test:
        labels_in_order(text, problems, where)


def check_rust(path, problems, require_labels=True):
    """Inspect check rust according to the function-order rule.

    :param path: Source file to inspect.
    :param problems: Mutable diagnostic list.
    :param require_labels: Require test labels; helper placement is always checked.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    text = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    header = re.match(r'((?://![^\n]*\n|\s*\n)*)', text)
    try:
        check_rust_items(text[header.end():], path.as_posix(), problems, require_labels=require_labels)
    except ValueError as error:
        problems.append(f'{path.as_posix()}: malformed Rust: {error}')
    labels_in_order(text, problems, path.as_posix())


def check_rust_items(text, where, problems, require_labels=True):
    """Inspect check rust items according to the function-order rule.

    :param text: Source text in the current scope.
    :param where: Qualified scope used in diagnostics.
    :param problems: Mutable diagnostic list.
    :param require_labels: Require test labels; helper placement is always checked.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    spans = scan_items(text)
    infos = [classify(text[a:b]) for a, b in spans]
    fns = []
    for (a, _b), info in zip(spans, infos, strict=True):
        if info[0] == 'fn' and not rust_exempt(text[a:a + info[5]], info[6]):
            fns.append((info[1], info[2], info[3], a + info[5]))
    helpers = [f for f in fns if not f[2]] if any(f[2] for f in fns) else fns
    keys = [sort_key(n, pub) for n, pub, _t, _a in helpers]
    if keys != sorted(keys):
        problems.append(f'{where}: functions out of order: {[n for n, _p, _t, _a in helpers][:6]}')
    check_rust_test_section(text, spans, infos, fns, where, problems, require_labels=require_labels)
    for (a, b), info in zip(spans, infos, strict=True):
        code = info[1] or ''
        if info[0] == 'other' and (re.match(r'(unsafe\s+)?impl\b', code) or MOD_RE.match(code)
                                   or any(x.startswith('#[cfg(') and 'test' in x for x in info[4])):
            span = body_span(text[a:b])
            if span:
                check_rust_items(text[a:b][span[0]:span[1]], where + ' > ' + code[:40], problems,
                                 require_labels=require_labels)
            elif '{' in code:
                raise ValueError(f'item {code[:40]!r} has no closing brace')


def check_rust_test_section(text, spans, infos, fns, where, problems, require_labels=True):
    """Every #[test] after a label in this scope, and every helper before the first test.

    :param text: Source text in the current scope.
    :param spans: Rust item source spans.
    :param infos: Classified Rust items.
    :param fns: Collected Rust function metadata.
    :param where: Qualified scope used in diagnostics.
    :param problems: Mutable diagnostic list.
    :param require_labels: Require test labels; helper placement is always checked.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    labels = []
    for (a, _b), info in zip(spans, infos, strict=True):
        labels.extend(a + m.start() for m in LABEL.finditer(text[a:a + info[5]]))
    tests = [(name, a) for name, _pub, is_test, a in fns if is_test]
    for name, a in tests:
        if require_labels and not any(label < a for label in labels):
            problems.append(f'{where}: test {name!r} is not in a labelled group (// ---- <label> ----)')
    if tests:
        first_test = min(a for _name, a in tests)
        for name, _pub, is_test, a in fns:
            if not is_test and a > first_test:
                problems.append(f'{where}: helper {name!r} is defined after the first test')


def classify(chunk):
    """('fn', name, is_pub, is_test, attrs, code offset, code line) or ('other', code, False, False, attrs, ...).

    :param chunk: One Rust item including attributes and trivia.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    lines = chunk.split('\n')
    attrs = []
    code = None
    depth = 0
    offset = 0
    k = 0
    while k < len(lines):
        s, depth = trim_comments(lines[k], depth)
        if s.startswith(('#[', '#![')):
            attrs.append(s)
            # multi-line attribute
            while attrs[-1].count('[') > attrs[-1].count(']') and k + 1 < len(lines):
                k += 1
                attrs[-1] += lines[k].strip()
        elif s and not s.startswith('//'):
            code = s
            break
        offset += len(lines[k]) + 1
        k += 1
    if code is None:
        return 'other', None, False, False, attrs, len(chunk), ''
    m = FN_RE.match(code)
    if m:
        return ('fn', m.group('name'), bool(m.group('vis')), any(a.startswith('#[test') for a in attrs), attrs,
                offset, lines[k])
    return 'other', code, False, False, attrs, offset, lines[k]


def labels_in_order(text, problems, where):
    """Inspect labels in order according to the function-order rule.

    :param text: Source text in the current scope.
    :param problems: Mutable diagnostic list.
    :param where: Qualified scope used in diagnostics.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    labels = [x for x in LABEL.findall(text) if x != 'helpers']
    if labels != sorted(labels, key=lambda s: (s == 'other tests', s)):
        problems.append(f'{where}: test group labels are not alphabetical: {labels[:6]}')


def main(argv):
    """Inspect main according to the function-order rule.

    :param argv: Command-line arguments excluding the program name.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-labels', action='store_true', help='Also reject tests missing a group label.')
    parser.add_argument('--prove-reorder', nargs=2, type=Path, metavar=('BEFORE', 'AFTER'))
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args(argv)
    if args.prove_reorder:
        before, after = args.prove_reorder
        if before.suffix != after.suffix:
            parser.error('Proof inputs must have the same supported suffix')
        prove_reorder(before.read_text(encoding='utf-8-sig'),
                      after.read_text(encoding='utf-8-sig'), before.suffix)
        print('Declaration-only reorder verified')
        return 0
    roots = [Path(a) for a in args.paths] or [
        Path(p) for p in ('src', 'tools', 'tests', 'rust/crates') if Path(p).exists()
    ]
    problems = []
    for root in roots:
        for path in sorted(root.rglob('*') if root.is_dir() else [root]):
            if SKIP & set(path.parts) or path.suffix not in ('.py', '.rs'):
                continue
            (check_python if path.suffix == '.py' else check_rust)(path, problems, require_labels=args.require_labels)
    for problem in problems:
        print(problem)
    print(f'{len(problems)} function-order problem(s)')
    return 1 if problems else 0


def prove_reorder(before, after, suffix):
    """Reject changes beyond declaration order within each fully qualified scope.

    :param before: Original source text.
    :param after: Reordered source text.
    :param suffix: Supported .py or .rs suffix.
    :return: None; raises ValueError for any structural, token, or comment change.
    """
    if suffix == '.py':
        before_tree, after_tree = ast.parse(before, type_comments=True), ast.parse(after, type_comments=True)
        left, right = _python_value(before_tree), _python_value(after_tree)
        same = (left == right and _python_comments(before) == _python_comments(after)
                and _python_exemptions(before_tree, before.splitlines())
                == _python_exemptions(after_tree, after.splitlines()))
    elif suffix == '.rs':
        same = _rust_value(before) == _rust_value(after)
    else:
        raise ValueError('Unsupported reorder-proof language')
    if not same:
        raise ValueError('Source differs beyond declaration order within its original scope')


def py_exempt(lines, start, def_line):
    """True when the line above the definition (or its decorator) or the ``def`` line carries a reasoned marker.

    :param lines: Original source lines.
    :param start: First decorator or declaration line.
    :param def_line: Declaration line number.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    candidates = [lines[start - 2]] if start >= 2 else []
    candidates.append(lines[def_line - 1])
    return any(EXEMPT.search(line) for line in candidates)


def py_key(name):
    """Inspect py key according to the function-order rule.

    :param name: Definition name.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    dunder = name.startswith('__') and name.endswith('__') and len(name) > 4
    return (0 if dunder else 1 if name.startswith('_') else 2, name.lower(), name)


def py_scopes(nodes, scope, is_test, out):
    """(statements, scope name, in a test module) for a body, its conditional blocks and its class bodies.

    :param nodes: AST statements in this scope.
    :param scope: Qualified Python scope name.
    :param is_test: Whether test-section rules apply.
    :param out: Mutable collected scope list.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    out.append((nodes, scope, is_test))
    for n in nodes:
        if isinstance(n, ast.ClassDef):
            py_scopes(n.body, n.name, False, out)
        elif not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for block in blocks_of(n):
                py_scopes(block, f'{scope} > {type(n).__name__.lower()}', is_test, out)


def rust_exempt(trivia, code_line):
    """Inspect rust exempt according to the function-order rule.

    :param trivia: Leading Rust comments and whitespace.
    :param code_line: Rust declaration line.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    return any(EXEMPT.search(line) for line in trivia.split('\n') + [code_line])


def scan_items(text):
    """Split ``text`` into items: list of (start, end) offsets (end exclusive), leading trivia included.

    Raises ValueError for unbalanced delimiters or an unterminated string or comment.

    :param text: Source text in the current scope.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    items = []
    n = len(text)
    i = 0
    start = 0
    depth = 0
    item_has_code = False
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j + 1
            continue
        if text.startswith('/*', i):
            level = 1
            i += 2
            while i < n and level:
                if text.startswith('/*', i):
                    level += 1
                    i += 2
                elif text.startswith('*/', i):
                    level -= 1
                    i += 2
                else:
                    i += 1
            if level:
                raise ValueError('unterminated block comment')
            continue
        m = re.match(r'b?r(#*)"', text[i:i + 12])
        if m and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')):
            close = '"' + m.group(1)
            j = text.find(close, i + len(m.group(0)))
            if j < 0:
                raise ValueError('unterminated raw string')
            i = j + len(close)
            item_has_code = True
            continue
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == '\\' else 1
            if i >= n:
                raise ValueError('unterminated string')
            i += 1
            item_has_code = True
            continue
        if c == "'":
            m = re.match(r"'(\\.[^']*|[^\\'])'", text[i:i + 12])
            if m:
                i += len(m.group(0))
                item_has_code = True
                continue
            i += 1
            continue
        if c in '([{':
            depth += 1
            item_has_code = True
        elif c in ')]}':
            depth -= 1
            if depth < 0:
                raise ValueError('unbalanced closing delimiter')
            if depth == 0 and c == '}':
                k = i + 1
                while k < n and text[k] in ' \t':
                    k += 1
                if k < n and text[k] == ';':
                    i = k
                    c = ';'
                else:
                    # end of a braced item; whatever follows on the line starts the next item
                    items.append((start, i + 1))
                    start = i + 1
                    item_has_code = False
                    i += 1
                    continue
        if c == ';' and depth == 0 and item_has_code:
            i += 1
            items.append((start, i))
            start = i
            item_has_code = False
            continue
        if not c.isspace():
            item_has_code = True
        i += 1
    if depth:
        raise ValueError('unbalanced opening delimiter')
    if start < n and text[start:].strip():
        items.append((start, n))
    return items


def sort_key(name, is_pub):
    """Inspect sort key according to the function-order rule.

    :param name: Definition name.
    :param is_pub: Whether the Rust item is public.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    return (1 if is_pub else 0, name.lower(), name)


def trim_comments(line, depth):
    """Drop leading block comments from ``line``; ``depth`` > 0 means an earlier line left a block comment open.

    :param line: Rust source line to inspect.
    :param depth: Open block-comment depth.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    s = line.strip()
    while True:
        if depth:
            i = 0
            while i < len(s) and depth:
                if s.startswith('/*', i):
                    depth += 1
                    i += 2
                elif s.startswith('*/', i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            s = s[i:].strip()
            if depth:
                return '', depth
        elif s.startswith('/*'):
            depth = 1
            s = s[2:]
        else:
            return s, 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
