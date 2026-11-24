"""Replay the registered lexical inventory of C++ functions from live headers.

The scan follows the declaration grammar of the registered header survey.
It skips preprocessor directives and function bodies.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

_SOURCE_DIRECTORY = 'src/behave'
_DECLARATION_SCOPES = {'global', 'namespace', 'class', 'struct'}
_SCOPES = re.compile(r'(class|struct|namespace)\s+(\w+)(?:\s*:[^{}]*)?')
_ENUMS = re.compile(r'(?:typedef\s+)?enum(?:\s+(?:class|struct))?\s*(\w*)\s*(?::.*)?')
_QUOTED = r'"(?:\\.|[^"\\])*"' r"|'(?:\\.|[^'\\])*'"
_TOKENS = re.compile(r'[{};:]|' + _QUOTED)
_INERT_TEXT = re.compile(
    r'^[ \t]*#(?:[^\n]*\\\n)*[^\n]*|//[^\n]*|/\*[\s\S]*?\*/|(?P<literal>' + _QUOTED + ')',
    re.M,
)
_FUNCTION = re.compile(
    r'(.*?)((?:operator\s*(?:\[\]|\(\)|[^\w\s(]+|\w+))|~?\w+)\s*\((.*)\)\s*([^;{}]*)'
)


@dataclass
class _Scope:
    """Track a declaration scope's name, current access, and outer visibility."""

    kind: str = ''
    name: str = ''
    access: str = 'public'
    publicly_visible: bool = True


def _function(declaration, owner, access):
    """Extract the registered lexical signature of a public function.

    :param declaration: Comment-free declaration with normalized whitespace.
    :param owner: Qualified enclosing scope name.
    :param access: Current lexical access specifier.
    :return: Owner, qualified function name, and signature, or None for other declarations.
    """
    if '(' not in declaration or declaration.startswith(('typedef ', 'using ', 'friend class ')):
        return None
    match = _FUNCTION.fullmatch(declaration)
    if not match:
        return None
    prefix, name, parameters, qualifiers = match.groups()
    if '=' in prefix or name in {'if', 'while', 'switch', 'sizeof', 'decltype', 'static_assert'}:
        return None
    if access != 'public' and not prefix.startswith('friend '):
        return None
    qualified_name = f'{owner}::{name}' if owner else name
    signature = f'{prefix}{qualified_name}({parameters}) {qualifiers}'.strip()
    return owner, qualified_name, signature


def _lexical_signature(owner, signature, nested_types):
    """Qualify type names declared in the enclosing lexical scopes.

    :param owner: Qualified declaring scope of the function.
    :param signature: Source-derived signature with a qualified function name.
    :param nested_types: Live type names grouped by qualified enclosing scope.
    :return: Signature with lexically resolvable nested type names qualified.
    """
    while owner:
        for name in nested_types.get(owner, ()):
            signature = re.sub(rf'(?<![\w:]){re.escape(name)}(?![\w:])', f'{owner}::{name}', signature)
        owner = owner.rpartition('::')[0]
    return signature


def _mask_inert_text(match):
    """Ignore directives and comments while preserving quoted literals and line boundaries.

    :param match: A directive, comment, or quoted literal in the header.
    :return: The original literal or a whitespace replacement.
    """
    return match[0] if match.lastgroup == 'literal' else re.sub(r'[^\n]', ' ', match[0])


def _registered_headers(root, inventory):
    """Compare discovered headers with the inventory's registered source paths.

    :param root: Resolved upstream checkout root.
    :param inventory: Registered coverage inventory.
    :return: Live header paths keyed by their upstream-relative names.
    """
    registered = {row['file'] for row in inventory['source_inputs']
                  if row['file'].startswith(_SOURCE_DIRECTORY + '/') and row['file'].endswith('.h')}
    if not registered:
        raise ValueError('No registered C++ API headers')
    headers = {path.relative_to(root).as_posix(): path
               for path in (root / _SOURCE_DIRECTORY).rglob('*.h') if path.is_file()}
    missing, unexpected = registered - headers.keys(), headers.keys() - registered
    if missing or unexpected:
        raise ValueError(f'C++ API header set changed: missing {sorted(missing)}, unexpected {sorted(unexpected)}')
    for name, path in headers.items():
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'C++ API header escapes the upstream root: {name}')
    return headers


def _scan_header(path, nested_types):
    """Find public function declarations and nested types in a registered header.

    :param path: Live header path.
    :param nested_types: Mutable mapping of enclosing scope names to nested type names.
    :return: Public function records in source order, each with its lexical owner and signature.
    """
    source = _INERT_TEXT.sub(_mask_inert_text, path.read_text(encoding='utf-8-sig'))
    declarations = []
    stack = [_Scope('global')]
    start = 0
    for token in _TOKENS.finditer(source):
        mark = token[0]
        if mark.startswith(('"', "'")):
            continue
        context = stack[-1]
        declaration = ' '.join(source[start:token.start()].split())
        owner = '::'.join(scope.name for scope in stack if scope.name)
        if mark == ':':
            if declaration in {'public', 'private', 'protected'}:
                context.access = declaration
                start = token.end()
            continue
        if mark == '}':
            if len(stack) > 1:
                stack.pop()
        elif context.kind not in _DECLARATION_SCOPES:
            if mark == '{':
                stack.append(_Scope())
        elif mark == '{' and (scope := _SCOPES.fullmatch(declaration)):
            kind, name = scope.groups()
            if owner and kind != 'namespace':
                nested_types[owner].add(name)
            stack.append(_Scope(
                kind, name, 'private' if kind == 'class' else 'public',
                publicly_visible=context.publicly_visible and context.access == 'public',
            ))
        elif mark == '{' and (enum := _ENUMS.fullmatch(declaration)):
            if owner and enum[1]:
                nested_types[owner].add(enum[1])
            stack.append(_Scope())
        else:
            if context.publicly_visible and (function := _function(declaration, owner, context.access)) is not None:
                declarations.append(function)
            if mark == '{':
                stack.append(_Scope())
        start = token.end()
    return declarations


def verify_public_api(oracle_root, inventory):
    """Verify the registered public function identities against the live C++ headers.

    :param oracle_root: Root of the upstream checkout.
    :param inventory: Registered C++ public function and source inventory.
    :return: Live public function count and canonical lexical identity digest.
    """
    headers = _registered_headers(Path(oracle_root).resolve(), inventory)
    nested_types = defaultdict(set)
    declarations = {name: _scan_header(path, nested_types) for name, path in headers.items()}
    live = Counter()
    for header, functions in declarations.items():
        for owner, name, signature in functions:
            identity = hashlib.sha256(f'{header}|{signature}'.encode()).hexdigest()[:20]
            lexical_signature = _lexical_signature(owner, signature, nested_types)
            live[header, name, identity, signature, lexical_signature] += 1

    rows = inventory['public_functions']
    if any(row['access'] != 'public' for row in rows):
        raise ValueError('Nonpublic declaration in the registered public C++ API')
    expected = Counter((row['file'], row['qualified_name'], row['id'], row['signature'],
                        row['lexically_qualified_signature']) for row in rows)
    if live != expected:
        missing = [f'{row[0]}: {row[4]}' for row in sorted((expected - live).elements())[:5]]
        unexpected = [f'{row[0]}: {row[4]}' for row in sorted((live - expected).elements())[:5]]
        raise ValueError(f'Live public C++ API changed: missing {missing}, unexpected {unexpected}')

    identities = sorted(row[2:] for row in live.elements())
    digest = hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()
    if digest != inventory['public_identity_sha256']:
        raise ValueError('Live public C++ identity digest does not match the registered inventory')
    return {'public_functions': len(identities), 'public_identity_sha256': digest}
