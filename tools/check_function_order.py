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
from functools import lru_cache
from pathlib import Path

SKIP = {'.venv', 'target', '__pycache__', '.git', 'node_modules', 'build', 'dist'}
LABEL = re.compile(r'^\s*(?:#|//) ---- (.*) ----\s*$', re.M)
EXEMPT = re.compile(r'(?:#|//)\s*order-exempt:\s*(\S.*)$')
RUST_CHARACTER = re.compile(r"b?'(?:\\(?:[^\r\nxu]|x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]+\})|[^'\\\r\n])'")
RUST_NUMBER_OR_OPERATOR = re.compile(r'\d[\d_]*\.(?![.\w])'
                                    r'|\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?[\d_]+)?\w*'
                                    r'|::|->|=>|\.\.=|\.\.|<<=?|>>=?|&&|\|\||[=!<>+*/%&|^-]=')
RUST_QUOTED_STRING = re.compile(r'(?:b|c)?"')
RUST_RAW_STRING = re.compile(r'(?:br|cr|r)(#*)"')
RUST_WHITESPACE = frozenset('\t\n\v\f\r \u0085\u200e\u200f\u2028\u2029')

def _python_comment_lines(source):
    """Extract real Python comment tokens, excluding all string contents.

    :param source: Python source text.
    :return: Line numbers mapped to comment text and indentation.
    """
    return {token.start[0]: (token.string, token.start[1])
            for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT}


def _python_comments(source):
    """Retain every Python comment while declarations are normalized.

    :param source: Python source text.
    :return: Comment text and multiplicity.
    """
    return Counter(token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
                   if token.type == tokenize.COMMENT)


def _python_duplicate_bindings(tree):
    """Retain duplicate function binding order across compound statements sharing a namespace.

    :param tree: Parsed Python module.
    :return: Qualified duplicate names mapped to their declaration sequence.
    """
    bindings = {}
    pending = [(tree, ())]
    while pending:
        node, scope = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings.setdefault((*scope, node.name), []).append(ast.dump(node, include_attributes=False))
            continue
        if isinstance(node, ast.ClassDef):
            scope = (*scope, node.name)
        pending.extend((child, scope) for child in reversed(list(ast.iter_child_nodes(node))))
    return {name: values for name, values in bindings.items() if len(values) > 1}


def _python_exemptions(tree, lines):
    """Bind import-dependency exemption markers to their original qualified definitions.

    :param tree: Parsed Python module.
    :param lines: Original source lines.
    :return: Qualified definition identities mapped to exemption marker text.
    """
    result = {}
    comments = _python_comment_lines('\n'.join(lines))
    counts = Counter()
    pending = [(tree, ())]
    while pending:
        node, scope = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = (*scope, node.name)
            counts[scope] += 1
            start = min([node.lineno] + [item.lineno for item in node.decorator_list])
            candidates = [comments[line][0] for line in (start - 1, node.lineno) if line in comments
                          and (line == node.lineno or comments[line][1] == node.col_offset)]
            result[scope, counts[scope]] = tuple(match.group(0) for line in candidates
                                                if (match := EXEMPT.search(line)))
        pending.extend((child, scope) for child in reversed(list(ast.iter_child_nodes(node))))
    return result


def _python_scope_comments(nodes, lines, comments):
    """Select comments owned by one direct statement scope.

    :param nodes: Direct AST statements in the scope.
    :param lines: Original source lines.
    :param comments: Actual comment tokens by line.
    :return: Comments at this scope's indentation outside nested statement bodies.
    """
    if not nodes:
        return {}
    indent = nodes[0].col_offset
    first = min([nodes[0].lineno] + [item.lineno for item in getattr(nodes[0], 'decorator_list', [])])
    while first > 1 and (not lines[first - 2].strip() or first - 1 in comments):
        first -= 1
    last = nodes[-1].end_lineno
    return {line: value for line, value in comments.items()
            if first <= line <= last and value[1] == indent
            and not any(node.lineno <= line <= node.end_lineno for node in nodes)}


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


def _rust_exemption_comments(text, start, end, code_offset):
    """Select reasoned directive comments attached outside this declaration's attributes.

    :param text: Rust source in the current scope.
    :param start: Item start including leading trivia.
    :param end: Item end including ordinary trailing comments.
    :param code_offset: Declaration offset relative to the item start.
    :return: Attached exemption marker texts in source order.
    """
    tokens = _rust_token_spans(text[start:end])
    declaration = start + code_offset
    function = next((start + begin for token, begin, _finish in tokens
                     if token == 'fn' and begin >= code_offset), declaration)
    inline_lines = {text.rfind('\n', 0, position) + 1 for position in (declaration, function)}
    candidates = []
    depth = 0
    for token, begin, finish in tokens:
        begin += start
        finish += start
        line_start = text.rfind('\n', 0, begin) + 1
        if begin < declaration:
            if not depth and token.startswith('//') and not text[line_start:begin].strip():
                # Attributes can sit between the marker and its function, but unrelated blank lines cannot.
                remainder = text[finish:declaration]
                if '\n\n' not in remainder and not re.search(r'\n[ \t]+\n', remainder):
                    candidates.append(token)
            depth += (token in ('[', '(', '{')) - (token in (']', ')', '}'))
        elif token.startswith('//') and line_start in inline_lines:
            candidates.append(token)
    return tuple(match.group(0) for comment in candidates if (match := EXEMPT.match(comment)))


def _rust_generic_depth(depth, token):
    """Interpret contextual generic closers without changing lexical proof tokens.

    :param depth: Current generic nesting depth outside other delimiter groups.
    :param token: Current Rust lexical token.
    :return: Updated depth, including closers adjacent to an assignment.
    """
    if token in ('<', '<<'):
        return depth + len(token)
    if token in ('>', '>>', '>=', '>>='):
        return max(0, depth - len(token.rstrip('=')))
    return depth


def _rust_header(tokens):
    """Read declaration header tokens independently of line wrapping and comments.

    :param tokens: Lexical tokens starting at the declaration.
    :return: Tokens after visibility and qualifiers, and whether visibility is public.
    """
    tokens = [token for token in tokens if not token.startswith(('//', '/*'))]
    index = 0
    public = bool(tokens and tokens[0] == 'pub')
    if public:
        index = 1
        if tokens[index:index + 1] == ['(']:
            depth = 1
            index += 1
            while index < len(tokens) and depth:
                depth += (tokens[index] == '(') - (tokens[index] == ')')
                index += 1
    while index < len(tokens):
        if tokens[index] in ('const', 'async', 'unsafe'):
            index += 1
        elif tokens[index] == 'extern':
            index += 1
            if index < len(tokens) and tokens[index].startswith('"'):
                index += 1
        else:
            break
    return tokens[index:], public


def _rust_identifier_end(text, start):
    """Retain identifier adjacency conservatively across Unicode database versions.

    :param text: Rust source text.
    :param start: Candidate identifier offset.
    :return: End offset; uncertain non-ASCII characters are retained rather than normalized away.
    """
    if start >= len(text):
        return start
    char = text[start]
    if not (char == '_' or char.isidentifier() or (ord(char) > 127 and char not in RUST_WHITESPACE)):
        return start
    end = start + 1
    while end < len(text):
        char = text[end]
        if not (char.isascii() and (char.isalnum() or char == '_')
                or ord(char) > 127 and char not in RUST_WHITESPACE):
            break
        end += 1
    return end


def _rust_item_end(text, tokens, index):
    """Keep ordinary same-line trailing comments with their preceding item.

    :param text: Rust source text.
    :param tokens: Positioned lexical tokens.
    :param index: Current item terminator index.
    :return: End offset including attached comments, excluding documentation for the next item.
    """
    end = tokens[index][2]
    index += 1
    while index < len(tokens):
        token, begin, finish = tokens[index]
        outer_doc = ((token.startswith('///') and not token.startswith('////'))
                     or (token.startswith('/**') and not token.startswith('/***') and token != '/**/'))
        if ('\n' in text[end:begin] or not token.startswith(('//', '/*'))
                or outer_doc or token.startswith(('//!', '/*!'))):
            break
        end = finish
        if token.startswith('//') or '\n' in token:
            break
        index += 1
    return end


def _rust_scope_comments(text, spans, infos):
    """Select real standalone line comments directly owned by one item scope.

    :param text: Rust source text in the current scope.
    :param spans: Direct item spans.
    :param infos: Classified direct items.
    :return: Source offsets and comment tokens outside attributes and nested bodies.
    """
    comments = []
    for (start, _end), info in zip(spans, infos, strict=True):
        depth = 0
        for token, begin, _finish in _rust_token_spans(text[start:start + info[5]]):
            begin += start
            if not depth and token.startswith('//') and not text[text.rfind('\n', 0, begin) + 1:begin].strip():
                comments.append((begin, token))
            depth += (token in ('[', '(', '{')) - (token in (']', ')', '}'))
    return comments


def _rust_token_spans(text):
    """Attach exact source positions to the shared Rust lexer tokens.

    :param text: Rust source text.
    :return: Token text, start, and end offsets in source order.
    """
    result = []
    offset = 0
    for token in _rust_tokens(text):
        begin = text.index(token, offset)
        offset = begin + len(token)
        result.append((token, begin, offset))
    return result


@lru_cache(maxsize=512)
def _rust_tokens(text, spacing=False):
    """Tokenize Rust without discarding string contents, attributes, or comments.

    :param text: Rust item source validated by the item scanner.
    :param spacing: Retain all token adjacency; conservatively reject non-order formatting changes.
    :return: Immutable lexical tokens cached by exact source content and spacing mode.
    """
    tokens = []
    i = 0
    previous_end = -1
    while i < len(text):
        if text[i] in RUST_WHITESPACE:
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
            raw = RUST_RAW_STRING.match(text, i)
            quoted = RUST_QUOTED_STRING.match(text, i)
            char = RUST_CHARACTER.match(text, i)
            if raw:
                close = '"' + raw.group(1)
                end = text.find(close, raw.end())
                if end < 0:
                    raise ValueError('Unterminated Rust raw string')
                i = end + len(close)
            elif quoted:
                i = quoted.end()
                while i < len(text) and text[i] != '"':
                    i += 2 if text[i] == '\\' else 1
                if i >= len(text):
                    raise ValueError('Unterminated Rust string')
                i += 1
            elif char:
                i = char.end()
            else:
                identifier = _rust_identifier_end(text, i + 2) if text.startswith('r#', i) else i
                if identifier <= i + 2:
                    identifier = _rust_identifier_end(text, i)
                word = RUST_NUMBER_OR_OPERATOR.match(text, i)
                i = identifier if identifier > i else word.end() if word else i + 1
            if raw or quoted or char:
                i = _rust_identifier_end(text, i)
        token = text[start:i]
        joint = start == previous_end
        tokens.append((token, joint) if spacing else token)
        previous_end = i
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
    other = [('container', _rust_tokens(text[:end_prefix], spacing=True))]
    text = text[end_prefix:]
    for start, end in scan_items(text):
        chunk = text[start:end]
        info = classify(chunk)
        if info[0] == 'fn':
            functions.setdefault(info[1], []).append((_rust_tokens(chunk, spacing=True),
                                                       _rust_exemption_comments(text, start, end, info[5])))
        elif info[1] and _rust_header(_rust_tokens(info[1]))[0][:1] in (['impl'], ['mod']):
            span = body_span(chunk)
            if span is None:
                other.append(_rust_tokens(chunk, spacing=True))
            else:
                other.append((_rust_tokens(chunk[:span[0]], spacing=True), _rust_value(chunk[span[0]:span[1]]),
                              _rust_tokens(chunk[span[1]:], spacing=True)))
        else:
            other.append(_rust_tokens(chunk, spacing=True))
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
    """Find the declaration body after attributes, nested header groups, and generic arguments.

    :param chunk: One Rust item including attributes and trivia.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    stack = []
    angles = 0
    opening = None
    for token, begin, end in _rust_token_spans(chunk):
        if token.startswith(('//', '/*')):
            continue
        if not stack:
            angles = _rust_generic_depth(angles, token)
        if token in ('(', '[', '{'):
            if token == '{' and not stack and not angles:
                opening = end
            stack.append(token)
        elif token in (')', ']', '}'):
            if not stack or stack.pop() != {')': '(', ']': '[', '}': '{'}[token]:
                raise ValueError('unbalanced closing delimiter')
            if not stack and opening is not None:
                return opening, begin
    return None


def check_py_body(nodes, scope, is_test, lines, problems, path, require_labels=True, comments=None):
    """Inspect check py body according to the function-order rule.

    :param nodes: AST statements in this scope.
    :param scope: Qualified Python scope name.
    :param is_test: Whether test-section rules apply.
    :param lines: Original source lines.
    :param problems: Mutable diagnostic list.
    :param path: Source file to inspect.
    :param require_labels: Require test labels; helper placement is always checked.
    :param comments: Optional cached comment tokens.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    entries = []
    if comments is None:
        comments = _python_comment_lines('\n'.join(lines))
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            if py_exempt(lines, start, n.lineno, comments=comments):
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
        own_comments = _python_scope_comments(nodes, lines, comments)
        check_py_test_section(entries, lines, problems, path, require_labels=require_labels, comments=own_comments)
        labels_in_order('\n'.join(value[0] for value in own_comments.values()), problems, path + ': ' + scope)


def check_py_test_section(entries, lines, problems, path, require_labels=True, comments=None):
    """Every test in a labelled group, and every helper before the first test.

    :param entries: Definition names and source positions.
    :param lines: Original source lines.
    :param problems: Mutable diagnostic list.
    :param path: Source file to inspect.
    :param require_labels: Require test labels; helper placement is always checked.
    :param comments: Optional scope-owned comment tokens.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    if comments is None:
        comments = _python_comment_lines('\n'.join(lines))
    label_lines = [line for line, value in comments.items() if LABEL.match(value[0])]
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
    comments = _python_comment_lines(text)
    # The standalone harness retains its reference scenario order outside collected tiers.
    legacy_harness = path.parts[-3:] == ('tests', 'legacy', 'behave_harness.py')
    is_test = legacy_harness or path.name.startswith('test_')
    where = path.as_posix()
    scopes = []
    py_scopes(tree.body, '<module>', is_test, scopes)
    for nodes, scope, scope_is_test in scopes:
        check_py_body(nodes, scope, scope_is_test, lines, problems, where,
                      require_labels=require_labels and not legacy_harness, comments=comments)


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
    for (a, b), info in zip(spans, infos, strict=True):
        if info[0] == 'fn' and not rust_exempt(text, a, b, info[5]):
            fns.append((info[1], info[2], info[3], a + info[5]))
    helpers = [f for f in fns if not f[2]] if any(f[2] for f in fns) else fns
    keys = [sort_key(n, pub) for n, pub, _t, _a in helpers]
    if keys != sorted(keys):
        problems.append(f'{where}: functions out of order: {[n for n, _p, _t, _a in helpers][:6]}')
    check_rust_test_section(text, spans, infos, fns, where, problems, require_labels=require_labels)
    for (a, b), info in zip(spans, infos, strict=True):
        code = info[1] or ''
        if info[0] == 'other' and (_rust_header(_rust_tokens(code))[0][:1] in (['impl'], ['mod'])
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
    comments = _rust_scope_comments(text, spans, infos)
    labels = [offset for offset, comment in comments if LABEL.match(comment)]
    labels_in_order('\n'.join(comment for _offset, comment in comments), problems, where)
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
    tokens = _rust_tokens(chunk)
    attrs = []
    offset = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith(('//', '/*')):
            offset = chunk.index(token, offset) + len(token)
            index += 1
        elif token == '#' and (tokens[index:index + 2] == ('#', '[')
                               or tokens[index:index + 3] == ('#', '!', '[')):
            begin = index
            depth = 0
            while index < len(tokens):
                value = tokens[index]
                offset = chunk.index(value, offset) + len(value)
                depth += (value == '[') - (value == ']')
                index += 1
                if value == ']' and depth == 0:
                    break
            if depth:
                raise ValueError('Unterminated Rust attribute')
            attrs.append(''.join(tokens[begin:index]))
        else:
            offset = chunk.index(token, offset)
            break
    if index == len(tokens):
        return 'other', None, False, False, attrs, len(chunk), ''
    code = chunk[offset:]
    header, public = _rust_header(tokens[index:])
    if len(header) >= 2 and header[0] == 'fn':
        name = header[1].removeprefix('r#')
        return ('fn', name, public, any(a.startswith('#[test') for a in attrs), attrs,
                offset, code)
    return 'other', code, False, False, attrs, offset, code


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
                and _python_duplicate_bindings(before_tree) == _python_duplicate_bindings(after_tree)
                and _python_exemptions(before_tree, before.splitlines())
                == _python_exemptions(after_tree, after.splitlines()))
    elif suffix == '.rs':
        same = _rust_value(before) == _rust_value(after)
    else:
        raise ValueError('Unsupported reorder-proof language')
    if not same:
        raise ValueError('Source differs beyond declaration order within its original scope')


def py_exempt(lines, start, def_line, comments=None):
    """True when the line above the definition (or its decorator) or the ``def`` line carries a reasoned marker.

    :param lines: Original source lines.
    :param start: First decorator or declaration line.
    :param def_line: Declaration line number.
    :param comments: Optional cached comment tokens.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    if comments is None:
        comments = _python_comment_lines('\n'.join(lines))
    indent = len(lines[def_line - 1]) - len(lines[def_line - 1].lstrip())
    candidates = [comments[line][0] for line in (start - 1, def_line) if line in comments
                  and (line == def_line or comments[line][1] == indent)]
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
            unittest = any((isinstance(base, ast.Name) and base.id == 'TestCase')
                           or (isinstance(base, ast.Attribute) and base.attr == 'TestCase') for base in n.bases)
            opted_out = any(isinstance(item, (ast.Assign, ast.AnnAssign)) and isinstance(item.value, ast.Constant)
                            and item.value.value is False
                            and any(isinstance(target, ast.Name) and target.id == '__test__'
                                    for target in (item.targets if isinstance(item, ast.Assign) else [item.target]))
                            for item in n.body)
            custom_init = any(isinstance(item, ast.FunctionDef) and item.name in {'__init__', '__new__'}
                              for item in n.body)
            collected = is_test and not opted_out and (unittest or (n.name.startswith('Test') and not custom_init))
            py_scopes(n.body, scope + '.' + n.name, collected, out)
        elif not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for block in blocks_of(n):
                py_scopes(block, f'{scope} > {type(n).__name__.lower()}', is_test, out)


def rust_exempt(text, start, end, code_offset):
    """Bind real reasoned comments to this declaration, excluding strings and previous items.

    :param text: Source text in the current scope.
    :param start: Item start, including leading trivia.
    :param end: Item end, including attached ordinary trailing comments.
    :param code_offset: Declaration offset relative to the item start.
    :return: Whether a directly attached comment exempts the declaration.
    """
    return bool(_rust_exemption_comments(text, start, end, code_offset))


def scan_items(text):
    """Split ``text`` into items: list of (start, end) offsets (end exclusive), leading trivia included.

    Raises ValueError for unbalanced delimiters or an unterminated string or comment.

    :param text: Source text in the current scope.
    :return: Computed scanner result or diagnostics appended to the supplied list.
    """
    items = []
    start = 0
    stack = []
    angles = 0
    initializer = False
    item_has_code = False
    tokens = _rust_token_spans(text)
    for index, (token, _begin, end) in enumerate(tokens):
        if token.startswith(('//', '/*')):
            continue
        if not stack and not initializer:
            angles = _rust_generic_depth(angles, token)
        if not stack and not angles and token in ('=', '>=', '>>='):
            initializer = True
        if token in ('(', '[', '{'):
            stack.append(token)
        elif token in (')', ']', '}'):
            if not stack or stack.pop() != {')': '(', ']': '[', '}': '{'}[token]:
                raise ValueError('unbalanced closing delimiter')
            if not stack and not angles and not initializer and token == '}':
                next_index = index + 1
                while next_index < len(tokens) and tokens[next_index][0].startswith(('//', '/*')):
                    next_index += 1
                if next_index == len(tokens) or tokens[next_index][0] != ';':
                    # End of a braced item; whatever follows on the line starts the next item.
                    end = _rust_item_end(text, tokens, index)
                    items.append((start, end))
                    start = end
                    item_has_code = False
                    initializer = False
                    continue
        if token == ';' and not stack and not angles and item_has_code:
            end = _rust_item_end(text, tokens, index)
            items.append((start, end))
            start = end
            item_has_code = False
            initializer = False
            continue
        item_has_code = True
    if stack:
        raise ValueError('unbalanced opening delimiter')
    if start < len(text) and text[start:].strip():
        items.append((start, len(text)))
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
