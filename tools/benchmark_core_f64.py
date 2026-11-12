"""Measure bounded f64 core headroom against untouched public Python inputs.

Input construction, widening, native raster validation, scalar/Python correctness,
and binary exports are outside native timings. Final requested output storage is
inside native timings. This campaign cannot certify public Rust backend speed.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from benchmark_backends import _provenance
from check_core_f64 import check_core_screen
from core_f64_reference import validate_export


def _identity(probe, seed):
    """Freeze science, driver, compiler input, host, and binary identities.

    :param probe: Already built release probe without allocation instrumentation.
    :param seed: Deterministic original workload seed.
    :return: Campaign identity plus the unchanged worker provenance hash.
    """
    environment = _provenance(SimpleNamespace(seed=seed))
    environment['processor'] = platform.processor()
    environment['compiler'] = subprocess.check_output(['rustc', '-Vv'], text=True)
    environment['build_and_thread_environment'] = {
        name: os.environ.get(name) for name in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS',
            'CARGO_PROFILE_RELEASE_OPT_LEVEL', 'CARGO_PROFILE_RELEASE_LTO',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    worker_hash = environment.pop('source_hash')
    digest = hashlib.sha256(worker_hash.encode())
    paths = sorted((ROOT / 'tools/perf_probe').rglob('*.rs'))
    paths += [ROOT / 'tools/perf_probe/Cargo.toml', ROOT / 'tools/perf_probe/Cargo.lock',
              Path(__file__), ROOT / 'tools/check_core_f64.py',
              ROOT / 'tools/core_f64_reference.py', ROOT / 'tools/check_performance.py',
              ROOT / 'tools/performance_requirements.json']
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return dict(source_hash=digest.hexdigest(),
                binary_hash=hashlib.sha256(probe.read_bytes()).hexdigest(),
                environment=environment, worker_source_hash=worker_hash)


def _native_command(probe, operation, dataset, cells, seed, warm_calls):
    """Select the actual configured run_model scope and original storage layout.

    :param probe: Frozen release executable.
    :param operation: Original public workload route.
    :param dataset: Deterministic input recipe.
    :param cells: Complete native raster size.
    :param seed: Original recipe seed.
    :param warm_calls: Warmed calls after the first measured call.
    :return: Native process arguments without profiling instrumentation.
    """
    return [str(probe), '--scope', 'f64-raster', '--case',
            'typed-crown' if operation.startswith('typed-') else operation,
            '--layout', 'b' if operation.endswith('-b') else 'a',
            '--dataset', dataset, '--cells', str(cells), '--seed', str(seed),
            '--repetitions', str(warm_calls + 1)]


def _observe(arguments, case, backend, block, identity, warm_calls):
    """Run one fresh process and retain raw evidence before independent validation.

    :param arguments: Probe and output paths, recipe seed, and process timeout.
    :param case: Dataset, operation, and cell-count identity.
    :param backend: Original Python timing or core Rust with final output storage.
    :param block: Independent paired process block number.
    :param identity: Frozen source, native binary, and environment manifest.
    :param warm_calls: Warmed observations per fresh process.
    :return: Raw timing record and independent output proof for native observations.
    """
    dataset, operation, cells = case['dataset'], case['operation'], case['cells']
    directory = arguments.output.parent / (arguments.output.stem + '-raw') / (
        f'{dataset}-{operation}-{cells}-{block}-{backend}')
    directory.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    if backend == 'core-rust':
        command = _native_command(arguments.probe, operation, dataset, cells,
                                  arguments.seed, warm_calls)
        environment['BEHAVE_PROBE_OUTPUT'] = str(directory / 'columns')
    else:
        command = [sys.executable, '-B', str(ROOT / 'tools/benchmark_backends.py'),
                   '--worker', '--backend', 'python', '--threads', '1',
                   '--operation', operation, '--dataset', dataset, '--cells', str(cells),
                   '--seed', str(arguments.seed), '--warm-calls', str(warm_calls)]
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                            capture_output=True, timeout=arguments.timeout)
    (directory / 'stdout.json').write_text(result.stdout, encoding='utf-8')
    (directory / 'stderr.log').write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{directory.relative_to(ROOT)}: worker exit {result.returncode}: {result.stderr[-1000:]}')
    row = json.loads(result.stdout)
    validation = None
    if backend == 'core-rust':
        if row.get('scope') != 'f64-raster-core-with-storage' or row.get('validated') is not True:
            raise ValueError('Wrong native scope or incomplete native scalar validation')
        validation = validate_export(directory / 'columns', operation, cells, dataset, arguments.seed)
        validation['export_directory'] = (directory / 'columns').relative_to(ROOT).as_posix()
    else:
        if (row.get('backend') != 'python' or row.get('profile') is not False
                or row.get('source_hash') != identity['worker_source_hash']
                or row.get('numpy') != identity['environment']['numpy']):
            raise ValueError('Wrong Python backend, profile, source, or NumPy identity')
        row['worker_source_hash'] = row['source_hash']
    row.update(backend=backend, block=block, source_hash=identity['source_hash'],
               binary_hash=identity['binary_hash'] if backend == 'core-rust' else None,
               profile=False, python_timing_original_inputs=backend == 'python',
               raw_directory=directory.relative_to(ROOT).as_posix())
    return row, validation


def _save(report, output):
    """Persist partial observations and failures without losing earlier evidence.

    :param report: Complete or partial campaign evidence.
    :param output: Repo-relative JSON destination chosen for this independent run.
    :return: None; atomically replaces only this campaign's partial report.
    """
    temporary = output.with_suffix('.partial')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output)


def main():
    """Execute exactly one screen or one new confirmation on a frozen candidate.

    :return: None; saves raw observations and exits nonzero on a failed core gate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--phase', choices=('screen', 'confirmation'), default='screen')
    parser.add_argument('--screen', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--timeout', type=float, default=600)
    arguments = parser.parse_args()
    arguments.probe = (ROOT / arguments.probe).resolve()
    arguments.output = (ROOT / arguments.output).resolve()
    for path in (arguments.probe, arguments.output):
        path.relative_to(ROOT)
    if arguments.output.exists():
        parser.error('Output already exists; preserve prior evidence')
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity(arguments.probe, arguments.seed)
    if arguments.phase == 'confirmation':
        if arguments.screen is None:
            parser.error('Confirmation requires a passing frozen screen')
        screen = json.loads(arguments.screen.read_text(encoding='utf-8'))
        # Recompute the assessment; a handwritten passed flag cannot authorize sampling.
        screen['assessment'] = check_core_screen(screen)
        validate_confirmation(screen, identity)
    pairs, warm_calls = (7, 5) if arguments.phase == 'confirmation' else (3, 4)
    report = dict(identity, schema_version=1, contract_version=4, phase=arguments.phase,
                  scope='f64-raster-core-with-storage',
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  release_profile='Cargo default release; opt-level=3, no count-alloc feature',
                  exclusions=['input construction', 'normalization', 'shape validation',
                              'correctness checking', 'binary export'],
                  included=['actual run_model', 'all requested scientific fields', 'final output storage'],
                  cases=[], diagnostics=[], errors=[])
    if arguments.screen is not None:
        report['screen_hash'] = hashlib.sha256(arguments.screen.read_bytes()).hexdigest()
    _save(report, arguments.output)
    rng = random.Random(arguments.seed)
    try:
        # Continuous is a mandatory correctness diagnostic, not an acceptance timing case.
        for operation in ('surface', 'crown', 'typed-crown', 'typed-crown-b'):
            diagnostic = dict(dataset='continuous', operation=operation, cells=1031)
            _, diagnostic['validation'] = _observe(arguments, diagnostic, 'core-rust',
                                                    'diagnostic', identity, 1)
            report['diagnostics'].append(diagnostic)
            _save(report, arguments.output)
        for dataset in ('uniform', 'heterogeneous'):
            for cells in (10000, 100000):
                for operation in ('surface', 'crown', 'typed-crown', 'typed-crown-b'):
                    case = dict(dataset=dataset, operation=operation, cells=cells,
                                records=[], validation={})
                    report['cases'].append(case)
                    for block in range(pairs):
                        backends = ['python', 'core-rust']
                        rng.shuffle(backends)
                        for backend in backends:
                            if _identity(arguments.probe, arguments.seed) != identity:
                                raise ValueError('Frozen source, binary, or environment changed during campaign')
                            row, validation = _observe(arguments, case, backend, block, identity, warm_calls)
                            case['records'].append(row)
                            if validation is not None:
                                case['validation'] = validation
                            _save(report, arguments.output)
                            print(f'{dataset}/{operation}/{cells} block={block} {backend}', flush=True)
        if _identity(arguments.probe, arguments.seed) != identity:
            raise ValueError('Frozen source, binary, or environment changed at campaign end')
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['assessment'] = check_core_screen(report)
    _save(report, arguments.output)
    print(json.dumps(report['assessment'], indent=2), flush=True)
    raise SystemExit(0 if report['assessment']['passed'] else 1)


def validate_confirmation(screen, identity):
    """Reject retuning, changed hosts, or reuse of failed screen observations.

    :param screen: Complete passing three-pair screen, independently reassessed by the caller.
    :param identity: Current frozen source, binary, and host/NumPy identity.
    :return: None; raises ValueError instead of launching an invalid confirmation.
    """
    if screen.get('phase') != 'screen' or screen.get('assessment', {}).get('passed') is not True:
        raise ValueError('Confirmation requires a passing independent screen')
    if any(screen.get(key) != identity.get(key) for key in ('source_hash', 'binary_hash', 'environment')):
        raise ValueError('Confirmation must preserve the screened source, binary, host, and NumPy')


if __name__ == '__main__':
    main()
