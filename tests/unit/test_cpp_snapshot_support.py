"""Snapshot templates freeze input bytes and isolate temporary consumers."""

import json

import pytest

from cpp_snapshot_support import capture_snapshot, write_snapshot

# ---- snapshot immutability and path boundary tests ----

@pytest.mark.parametrize('kind', ['absolute', 'parent'])
def test_capture_rejects_paths_outside_the_source_root(tmp_path, kind):
    source = tmp_path / 'source'
    hashes = source / 'tests/data/cpp_reference/hashes.json'
    hashes.parent.mkdir(parents=True)
    outside = tmp_path / 'outside.txt'
    outside.write_bytes(b'protected original')
    name = str(outside) if kind == 'absolute' else '../outside.txt'
    hashes.write_text(json.dumps({name: 'untrusted input identity'}), encoding='utf-8')
    with pytest.raises(ValueError, match='relative|root|escape'):
        capture_snapshot(source)
    assert outside.read_bytes() == b'protected original'


def test_snapshot_is_a_frozen_point_in_time_with_independent_consumers(tmp_path):
    source = tmp_path / 'original'
    hashes = source / 'tests/data/cpp_reference/hashes.json'
    hashes.parent.mkdir(parents=True)
    hashes.write_text(json.dumps({'payload.json': 'synthetic payload identity'}), encoding='utf-8')
    (source / 'payload.json').write_bytes(b'{"value": "captured"}\n')
    snapshot = capture_snapshot(source)
    (source / 'payload.json').write_bytes(b'{"value": "changed original"}\n')
    first = write_snapshot(snapshot, tmp_path / 'first')
    second = write_snapshot(snapshot, tmp_path / 'second')
    assert (first / 'payload.json').read_bytes() == b'{"value": "captured"}\n'
    assert (second / 'payload.json').read_bytes() == b'{"value": "captured"}\n'
    (first / 'payload.json').write_bytes(b'consumer mutation')
    assert (second / 'payload.json').read_bytes() == b'{"value": "captured"}\n'


@pytest.mark.parametrize('kind', ['absolute', 'parent'])
def test_writer_rejects_escape_before_any_consumer_write(tmp_path, kind):
    destination = tmp_path / 'consumer'
    outside = tmp_path / 'outside.txt'
    outside.write_bytes(b'protected original')
    name = str(outside) if kind == 'absolute' else '../outside.txt'
    snapshot = (('safe.txt', b'valid consumer data'), (name, b'forbidden overwrite'))
    with pytest.raises(ValueError, match='relative|root|escape'):
        write_snapshot(snapshot, destination)
    assert outside.read_bytes() == b'protected original'
    assert not (destination / 'safe.txt').exists()
