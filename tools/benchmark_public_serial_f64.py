"""Run one bounded, frozen public serial screen or seven-new-pair confirmation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from types import SimpleNamespace

from benchmark_backends import _provenance
from check_public_serial_f64 import OPERATIONS, check_public_screen

ROOT = Path(__file__).resolve().parents[1]


def _identity(seed):
    """Freeze scientific sources, extension, harness, Python, and NumPy identity.

    :param seed: Deterministic recipe seed.
    :return: Exact source/build/environment manifest without mutable timing data.
    """
    environment = _provenance(SimpleNamespace(seed=seed))
    source = environment.pop('source_hash')
    extension = environment.pop('extension_hash')
    environment['compiler'] = subprocess.check_output(['rustc', '-Vv'], text=True)
    environment['processor'] = platform.processor()
    digest = hashlib.sha256()
    for name in ('benchmark_public_serial_f64.py', 'check_public_serial_f64.py', 'benchmark_backends.py',
                 'core_f64_reference.py', 'float64_model_reference.py', 'performance_cases.py',
                 'check_performance.py', 'performance_requirements.json'):
        path = ROOT / 'tools' / name
        digest.update(name.encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    environment['build_and_thread_environment'] = {name: os.environ.get(name) for name in
        ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_PROFILE_RELEASE_OPT_LEVEL',
         'CARGO_PROFILE_RELEASE_LTO', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    return dict(source_hash=source, extension_hash=extension, harness_hash=digest.hexdigest(), environment=environment)


def _observe(arguments, case, backend, block, identity, warm, *, validation=False):
    """Retain one untouched fresh worker record with matching frozen provenance.

    :param arguments: Output path, seed, and timeout.
    :param case: Dataset, operation, and logical cell count.
    :param backend: Explicit Python or Rust; auto is never timing evidence.
    :param block: Unique independent process block label.
    :param identity: Frozen manifest checked before and after every worker.
    :param warm: Warm calls following the first timed invocation.
    :param validation: Whether this is an untimed complete-output proof worker.
    :return: Actual worker JSON with its raw directory and block identity.
    """
    if _identity(arguments.seed) != identity:
        raise ValueError('Frozen source, binary, harness, or environment changed')
    directory = arguments.output.parent / (arguments.output.stem + '-raw') / (
        f"{case['dataset']}-{case['operation']}-{case['cells']}-{block}-{backend}")
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', str(ROOT / 'tools/benchmark_backends.py'), '--worker',
        '--backend', backend, '--threads', '1', '--operation', case['operation'], '--dataset', case['dataset'],
        '--cells', str(case['cells']), '--seed', str(arguments.seed), '--warm-calls', str(warm)]
    if validation:
        command.append('--validate-only')
    process = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')),
                             capture_output=True, text=True, timeout=arguments.timeout)
    (directory / 'stdout.json').write_text(process.stdout, encoding='utf-8')
    (directory / 'stderr.log').write_text(process.stderr, encoding='utf-8')
    if process.returncode:
        raise RuntimeError(f'{directory.relative_to(ROOT)}: exit {process.returncode}: {process.stderr[-1000:]}')
    row = json.loads(process.stdout)
    for name in ('source_hash', 'extension_hash'):
        if row.get(name) != identity[name]:
            raise ValueError(f'Worker {name} differs from frozen candidate')
    for name in ('python', 'numpy', 'numpy_build', 'platform'):
        if row.get(name) != identity['environment'][name]:
            raise ValueError(f'Worker {name} differs from frozen environment')
    if validation:
        if row.get('passed') is not True:
            raise ValueError(f"Complete output validation failed: {row.get('error')}")
        if row.get('native_boundary_calls') != ['model_batch']:
            raise ValueError('Common model did not use exactly one compute boundary without calibration')
    elif (row.get('backend') != backend or row.get('actual_backend') != backend or row.get('profile') is not False
          or (backend == 'python' and row.get('python_timing_original_inputs') is not True)):
        raise ValueError('Wrong actual backend, profiling, or normalized Python timing baseline')
    if _identity(arguments.seed) != identity:
        raise ValueError('Frozen candidate changed during a worker')
    row.update(block=block, raw_directory=directory.relative_to(ROOT).as_posix())
    return row


def _save(report, destination):
    """Persist partial raw evidence without overwriting an earlier campaign.

    :param report: Current campaign record.
    :param destination: Unique new result file chosen at startup.
    :return: None; writes UTF-8 JSON after each fresh observation.
    """
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


def main():
    """Run precisely one declared phase with unchanged targets and no resampling.

    :return: None; saves complete or partial evidence and exits nonzero on any gate miss.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('screen', 'confirmation'), default='screen')
    parser.add_argument('--screen', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--timeout', type=float, default=600.)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT)
    if args.output.exists():
        parser.error('Output exists; preserve previous evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity(args.seed)
    if args.phase == 'confirmation':
        if args.screen is None:
            parser.error('Confirmation requires a passing frozen screen')
        screen = json.loads(args.screen.read_text(encoding='utf-8'))
        validate_confirmation(screen, identity)
    pairs, warm = (7, 5) if args.phase == 'confirmation' else (3, 4)
    report = dict(identity, schema_version=1, contract_version=4, scope='public-serial-f64', phase=args.phase,
        started_utc=datetime.now(timezone.utc).isoformat(), cases=[], diagnostics=[], errors=[],
        included=['public wrapper normalization', 'native copies', 'science', 'final encoding', 'NumPy transfer'],
        excluded=['imports', 'construction', 'input recipe generation', 'correctness validation', 'profiling'])
    if args.screen is not None:
        report['screen_hash'] = hashlib.sha256(args.screen.read_bytes()).hexdigest()
    _save(report, args.output)
    rng = random.Random(args.seed)
    try:
        for operation in OPERATIONS:
            case = dict(operation=operation, dataset='continuous', cells=1031)
            proof = _observe(args, case, 'rust', 'continuous-validation', identity, 0, validation=True)
            report['diagnostics'].append(dict(case, validation=proof))
            _save(report, args.output)
        for cells in (1, 16, 1024):
            for operation in OPERATIONS:
                case = dict(operation=operation, dataset='uniform', cells=cells)
                proof = _observe(args, case, 'rust', 'scalar-validation', identity, 0, validation=True)
                records = [_observe(args, case, backend, 'scalar', identity, 2) for backend in ('python', 'rust')]
                report['diagnostics'].append(dict(case, validation=proof, records=records))
                _save(report, args.output)
        for dataset in ('uniform', 'heterogeneous'):
            for cells in (10000, 100000):
                for operation in OPERATIONS:
                    case = dict(dataset=dataset, operation=operation, cells=cells, records=[])
                    report['cases'].append(case)
                    case['validation'] = _observe(args, case, 'rust', 'validation', identity, 0, validation=True)
                    for block in range(pairs):
                        backends = ['python', 'rust']
                        rng.shuffle(backends)
                        for backend in backends:
                            case['records'].append(_observe(args, case, backend, block, identity, warm))
                            _save(report, args.output)
                            print(f'{dataset}/{operation}/{cells} block={block} {backend}', flush=True)
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['assessment'] = check_public_screen(report)
    _save(report, args.output)
    print(json.dumps(report['assessment'], indent=2), flush=True)
    raise SystemExit(0 if report['assessment']['passed'] else 1)


def validate_confirmation(screen, identity):
    """Reassess screen evidence and reject changed identities before new sampling.

    :param screen: Complete frozen three-pair screen; its saved passed flag is untrusted.
    :param identity: Current source, binary, harness, and environment identity.
    :return: None; raises ValueError for failed screens, changed identities, or a reused confirmation.
    """
    if screen.get('phase') != 'screen' or not check_public_screen(screen)['passed']:
        raise ValueError('Confirmation requires an independently reassessed passing screen')
    if any(screen.get(key) != identity.get(key) for key in ('source_hash', 'extension_hash', 'harness_hash', 'environment')):
        raise ValueError('Confirmation must preserve source, binary, harness, Python, NumPy, and host')


if __name__ == '__main__':
    main()
