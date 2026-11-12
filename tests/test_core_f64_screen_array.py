"""The cheap core gate must reject incomplete or misleading speed evidence."""
import sys
import importlib.util
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def _checker():
    path=Path(__file__).resolve().parents[1]/'tools/check_core_f64.py'
    spec=importlib.util.spec_from_file_location('check_core_f64',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _report():
    cases=[]
    for dataset in ('uniform','heterogeneous'):
        for cells in (10000,100000):
            for operation in ('surface','crown','typed-crown','typed-crown-b'):
                rows=[]
                for block in range(3):
                    for backend,seconds in (('python',10.),('core-rust',1.)):
                        rows.append(dict(block=block,backend=backend,first_seconds=seconds,warm_seconds=[seconds]*4,profile=False,source_hash='a'*64,binary_hash='b'*64 if backend=='core-rust' else None,python_timing_original_inputs=True))
                cases.append(dict(dataset=dataset,cells=cells,operation=operation,records=rows,validation=dict(complete=True,independent_python=True,truth_contract='f64-nan-v1',all_fields=True,typed_codes=True)))
    return dict(schema_version=1,contract_version=4,phase='screen',scope='f64-raster-core-with-storage',source_hash='a'*64,binary_hash='b'*64,cases=cases,errors=[])


def test_complete_core_proof_is_explicitly_not_public_speed_certification():
    result=_checker().check_core_screen(_report())
    assert result['passed']
    assert result['scope']=='core-only'
    assert result['public_speed_certified'] is False


def test_duplicate_or_missing_process_blocks_cannot_pass():
    for change in ('duplicate','missing'):
        report=_report()
        rows=report['cases'][0]['records']
        if change=='duplicate':rows[2]['block']=rows[0]['block']
        else:rows.pop()
        assert not _checker().check_core_screen(report)['passed']


def test_mixed_binary_identity_and_profiled_runs_fail():
    for name,value in (('binary_hash','another'),('profile',True)):
        report=_report()
        row=next(r for r in report['cases'][0]['records'] if r['backend']=='core-rust')
        row[name]=value
        assert not _checker().check_core_screen(report)['passed']


def test_normalized_python_timing_and_partial_output_validation_fail():
    for problem in ('oracle','partial','self-reference','wrong-contract','typed-code'):
        report=_report()
        case=report['cases'][0]
        if problem=='oracle':case['records'][0]['python_timing_original_inputs']=False
        elif problem=='partial':case['validation']['all_fields']=False
        elif problem=='self-reference':case['validation']['independent_python']=False
        elif problem=='wrong-contract':case['validation']['truth_contract']='old-masked'
        else:case['validation']['typed_codes']=False
        assert not _checker().check_core_screen(report)['passed']


def test_one_failing_common_crown_case_blocks_all_other_successes():
    report=_report()
    case=next(c for c in report['cases'] if (c['dataset'],c['cells'],c['operation'])==('heterogeneous',100000,'crown'))
    for row in case['records']:
        if row['backend']=='core-rust':row['first_seconds']=6.;row['warm_seconds']=[6.]*4
    result=_checker().check_core_screen(report)
    assert not result['passed']
    assert any('heterogeneous/crown/100000' in error for error in result['errors'])


def test_screen_cannot_pool_extra_blocks_or_warmed_observations():
    for problem in ('extra-block', 'extra-warm'):
        report = _report()
        rows = report['cases'][0]['records']
        if problem == 'extra-block':
            rows.extend([dict(row, block=3) for row in rows[-2:]])
        else:
            rows[0]['warm_seconds'].append(10.)
        assert not _checker().check_core_screen(report)['passed']
