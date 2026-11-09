"""Versioned input, output, and actual-backend proofs for performance evidence."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SURFACE = ('reaction_intensity', 'residence_time', 'heat_per_unit_area', 'no_wind_no_slope_spread_rate',
            'midflame_wind_speed', 'spread_rate', 'backing_spread_rate', 'flanking_spread_rate',
            'flame_length', 'fireline_intensity', 'effective_wind_speed', 'fire_length_to_width_ratio',
            'eccentricity', 'direction_of_max_spread')
_CROWN = ('crown_fire_spread_rate', 'crown_critical_fire_spread_rate', 'crown_flame_length',
          'crown_fire_line_intensity', 'crown_critical_surface_fire_line_intensity', 'crown_fire_heat_per_unit_area',
          'canopy_heat_per_unit_area', 'crown_fire_transition_ratio', 'crown_fire_active_ratio',
          'crown_length_to_width_ratio', 'fire_type', 'surface_fire_critical_spread_rate',
          'crowning_surface_fire_spread_rate', 'crown_fraction_burned', 'passive_crown_fire_spread_rate',
          'passive_crown_fire_heat_per_unit_area', 'passive_crown_fire_line_intensity',
          'passive_crown_fire_flame_length', 'final_spread_rate', 'final_heat_per_unit_area',
          'final_fireline_intensity', 'final_flame_length')


def _artifact(row, root):
    """Read and compare complete captured arrays, including categorical storage.

    :param row: Validation identity and saved-output descriptor.
    :param root: Execution directory containing repository-relative artifacts.
    :return: None; raises on a missing, altered, incomplete, or unequal artifact.
    """
    import numpy as np
    from performance_cases import assert_outputs

    descriptor = row['output_artifact']
    relative = Path(descriptor['path'])
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
        raise ValueError('Output artifacts must be execution-directory relative')
    if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor['sha256']:
        raise ValueError('Complete output artifact digest differs')
    fields = requested_outputs(row['operation'])
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {prefix+name for prefix in ('actual__', 'expected__') for name in fields}:
            raise ValueError('Complete requested output arrays are missing')
        for name in fields:
            actual, expected = archive['actual__'+name], archive['expected__'+name]
            if actual.dtype != np.dtype(np.int32 if name == 'fire_type' else np.float64):
                raise ValueError('Wrong physical or categorical output storage')
            assert_outputs(actual, expected)
            validity = row.get('eligibility', {}).get('validity')
            if validity is not None:
                if actual.size != row['cells']:
                    raise ValueError('Skipped-cell output coverage differs')
                missing = actual.reshape(-1)[~np.asarray(validity, dtype=bool)]
                initialized = np.all(missing == -1) if name == 'fire_type' else np.isnan(missing).all()
                if not initialized:
                    raise ValueError('Skipped outputs were not initialized as missing')


def contract_errors(report, requirements, root=ROOT):
    """Reject mislabeled timing, incomplete comparisons, or changed request/input sets.

    :param report: Candidate timing and validation records.
    :param requirements: Declared contract version and acceptance requirements.
    :param root: Directory used to resolve captured output artifacts.
    :return: Explicit errors; historical requirements retain their original checks.
    """
    if requirements.get('binding_api_version') != 4:
        return []
    errors = []
    groups = {}
    rows = [(False, row) for row in report.get('records', [])]
    rows.extend((True, row) for row in report.get('validations', []))
    for validation, row in rows:
        try:
            key = tuple(row[name] for name in ('dataset', 'operation', 'cells'))
            fields = requested_outputs(row['operation'])
            if row.get('binding_api_version') != 4 or row.get('requested_outputs') != fields:
                raise ValueError('Changed binding contract or requested output set')
            backend = 'rust' if validation else row.get('backend')
            if backend not in ('python', 'rust') or row.get('actual_backend') != backend:
                raise ValueError('Actual backend differs from measured backend')
            if row.get('backend') == 'python' and row.get('python_timing_original_inputs') is not True:
                raise ValueError('Public Python timing did not use original inputs')
            digest = row.get('input_sha256')
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Original input identity is missing')
            groups.setdefault(key, set()).add(digest)
            eligible = eligible_cells(row)
            if row.get('eligible_cells') != eligible:
                raise ValueError('Eligible-cell count differs from independent input recipe')
            if validation:
                if (row.get('fields') != fields or row.get('compared_fields') != fields
                        or row.get('output_fields') != len(fields) or row.get('independent_python') is not True
                        or row.get('correctness_contract') != 'f64-nan-v1'):
                    raise ValueError('Missing complete independent field comparisons')
                _artifact(row, root)
        except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
            errors.append(f"{row.get('dataset')}/{row.get('operation')}/{row.get('cells')}: {error}")
    if any(len(values) != 1 for values in groups.values()):
        errors.append('Original input identities differ within a paired case')
    return errors


def eligible_cells(row):
    """Derive work count from an independent all-cells or explicit-validity recipe.

    :param row: Logical cell count and original input eligibility recipe.
    :return: Expected scientific visits; malformed or absent recipes raise ValueError.
    """
    cells, recipe = row['cells'], row.get('eligibility')
    if type(cells) is not int or cells < 0 or not isinstance(recipe, dict):
        raise ValueError('Missing independent eligibility recipe')
    if recipe.get('kind') == 'all':
        return cells
    validity = recipe.get('validity')
    if recipe.get('kind') != 'explicit' or not isinstance(validity, list) or len(validity) != cells:
        raise ValueError('Explicit validity must cover every logical cell')
    if any(type(value) is not bool for value in validity):
        raise ValueError('Validity must contain actual boolean inputs')
    return sum(validity)


def input_identity(value):
    """Fingerprint original values, dtype, shape, mask, and underlying masked payload.

    :param value: Actual pre-normalization public operand mapping or nested value.
    :return: SHA-256 of the complete caller representation, never the widened oracle.
    """
    import numpy as np

    digest = hashlib.sha256()
    if isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode()+b'\0'+input_identity(value[key]).encode())
    elif value is None:
        digest.update(b'None')
    else:
        array = np.asanyarray(value)
        if array.dtype.kind == 'O':
            raise TypeError('Object inputs are not native performance evidence')
        digest.update(repr((type(value).__name__, array.dtype.str, array.shape)).encode())
        digest.update(np.ma.getdata(array).tobytes(order='C'))
        if np.ma.isMaskedArray(array):
            digest.update(np.ma.getmaskarray(array).tobytes(order='C'))
            digest.update(np.asarray(array.fill_value).tobytes())
    return digest.hexdigest()


def requested_outputs(operation):
    """Return the frozen complete public model field request for a route.

    :param operation: Surface, Crown, or one of the two original mixed-input recipes.
    :return: Sorted unique output names, including Crown classification.
    """
    if operation not in ('surface', 'crown', 'typed-crown', 'typed-crown-b'):
        raise ValueError('Unknown final acceptance operation')
    return sorted(_SURFACE if operation == 'surface' else _CROWN)


def save_outputs(path, actual, expected, root=ROOT):
    """Retain every requested actual and independent reference array outside timing.

    :param path: New compressed artifact file; existing evidence is never overwritten.
    :param actual: Complete native result mapping.
    :param expected: Complete independent normalized reference mapping.
    :param root: Execution directory used for portable artifact paths.
    :return: Relative path and digest of the complete array archive.
    """
    import numpy as np

    if set(actual) != set(expected):
        raise ValueError('Requested native/reference output fields differ')
    path = Path(path)
    relative = path.resolve().relative_to(Path(root).resolve()).as_posix()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **{prefix+name: value for prefix, values in
            (('actual__', actual), ('expected__', expected)) for name, value in values.items()})
    return {'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
