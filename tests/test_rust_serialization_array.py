"""Versioned native snapshots preserve values, aliases, and cached model state."""

import copy
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.rust


def _instance(name):
    from behave_fire import rust
    if name=='BehaveRun':
        return rust.BehaveRun(rust.FuelModels(),threads=2)
    return getattr(rust,name)()


def _spawn_roundtrip(payload, queue, extension_path, namespace_path):
    import importlib.util
    from importlib.machinery import ExtensionFileLoader
    import behave_fire
    try:
        behave_fire.__path__.insert(0,namespace_path)
        spec=importlib.util.spec_from_file_location('behave_fire._rust',extension_path,loader=ExtensionFileLoader('behave_fire._rust',extension_path))
        extension=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=extension
        spec.loader.exec_module(extension)
        model=pickle.loads(payload)
        queue.put({'threads':model.threads,'snapshot_depth':float(model._lut['depth'][1]),'metadata_depth':float(model._fuel_models.fuel_models_[1]['fuel_bed_depth']),'backend':model.__getstate__()['backend']})
    except BaseException as error:
        queue.put({'error':f'{type(error).__name__}: {error}'})


def test_fresh_process_native_unpickle_without_extension_raises(tmp_path):
    import os
    import subprocess
    from behave_fire import rust
    from parity_support import ROOT
    payload=tmp_path/'native-model.pickle'
    payload.write_bytes(pickle.dumps(_instance('BehaveRun')))
    script='''
import importlib.abc
import pickle
import sys
import behave_fire
behave_fire.__path__.insert(0,sys.argv[2])
class AbsentNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='behave_fire._rust': raise ImportError('native extension deliberately absent')
sys.modules.pop('behave_fire._rust',None)
sys.meta_path.insert(0,AbsentNative())
try:
    with open(sys.argv[1],'rb') as stream: pickle.load(stream)
except ImportError as error:
    assert 'Rust' in str(error),str(error)
else:
    raise AssertionError('native pickle silently restored without its extension')
'''
    environment=os.environ.copy()
    environment['PYTHONPATH']=str(ROOT/'src')
    completed=subprocess.run([sys.executable,'-c',script,str(payload),str(Path(rust.__file__).parent.parent)],capture_output=True,text=True,env=environment)
    assert completed.returncode==0,completed.stdout+completed.stderr


def test_native_copy_aliases_and_deepcopy_are_reference_compatible():
    from behave_fire.rust import FuelModels
    original=FuelModels()
    shallow=copy.copy(original)
    deep=copy.deepcopy(original)
    assert shallow.fuel_models_ is original.fuel_models_
    assert deep.fuel_models_ is not original.fuel_models_
    original.fuel_models_[1]['fuel_bed_depth']=17.
    assert shallow.fuel_models_[1]['fuel_bed_depth']==17.
    assert deep.fuel_models_[1]['fuel_bed_depth']==1.


def test_native_model_pickle_preserves_cached_snapshot():
    from behave_fire.rust import BehaveRun,FuelModels
    registry=FuelModels()
    model=BehaveRun(registry,threads=3)
    registry.fuel_models_[1]['fuel_bed_depth']=99.
    restored=pickle.loads(pickle.dumps(model))
    assert restored._fuel_models.fuel_models_[1]['fuel_bed_depth']==99.
    assert restored._lut['depth'][1]==1.
    assert restored.threads==3


def test_native_model_survives_windows_spawn_without_rebuilding_snapshot(tmp_path):
    import json
    import os
    import subprocess
    from behave_fire import _rust,rust
    from parity_support import ROOT
    model=rust.BehaveRun(rust.FuelModels(),threads=3)
    model._fuel_models.fuel_models_[1]['fuel_bed_depth']=99.
    payload=tmp_path/'native-spawn.pickle'
    payload.write_bytes(pickle.dumps(model))
    script=tmp_path/'native-spawn.py'
    # Match the existing Python-reference spawn test: pytest's fault handler
    # reports a handled Windows CreateProcess exception even without Rust loaded.
    # A plain interpreter still checks every worker exit, error, and snapshot.
    script.write_text('''
import json
import multiprocessing
from pathlib import Path
import sys
from test_rust_serialization_array import _spawn_roundtrip
if __name__=='__main__':
    context=multiprocessing.get_context('spawn')
    queue=context.Queue()
    process=context.Process(target=_spawn_roundtrip,args=(Path(sys.argv[1]).read_bytes(),queue,sys.argv[2],sys.argv[3]))
    process.start()
    try:
        result=queue.get(timeout=30)
        process.join(timeout=30)
        assert process.exitcode==0,process.exitcode
        print(json.dumps(result))
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        queue.close()
''',encoding='utf-8')
    environment=os.environ.copy()
    environment['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(Path(__file__).parent)
    completed=subprocess.run([sys.executable,str(script),str(payload),_rust.__file__,str(Path(rust.__file__).parent.parent)],capture_output=True,text=True,env=environment,timeout=60)
    assert completed.returncode==0,completed.stdout+completed.stderr
    assert completed.stderr==''
    assert json.loads(completed.stdout)=={'threads':3,'snapshot_depth':1.,'metadata_depth':99.,'backend':'rust'}


def test_native_restore_rejects_incompatible_extension(monkeypatch):
    original=_instance('FuelModels')
    state=original.__getstate__()
    restored=type(original).__new__(type(original))
    monkeypatch.setitem(sys.modules,'behave_fire._rust',SimpleNamespace(__binding_api_version__=999))
    with pytest.raises(ImportError,match='API'):
        restored.__setstate__(state)


def test_native_restore_rejects_unknown_state_version():
    original=_instance('FireSize')
    state=original.__getstate__()
    state['schema_version']=999
    with pytest.raises(ValueError,match='version'):
        type(original).__new__(type(original)).__setstate__(state)


@pytest.mark.parametrize('name',['BehaveRun','FuelModels','FireSize','SlopeTool','SpeciesMasterTable','SpeciesMasterTableRecord'])
@pytest.mark.parametrize('protocol',range(pickle.HIGHEST_PROTOCOL+1))
def test_native_versioned_pickle_round_trip(name,protocol):
    original=_instance(name)
    state=original.__getstate__()
    assert state.get('schema_version')==1
    assert state.get('binding_api_version')==4
    assert state.get('backend')=='rust'
    restored=pickle.loads(pickle.dumps(original,protocol=protocol))
    assert type(restored) is type(original)
    assert restored.__getstate__()['schema_version']==1
    if name=='BehaveRun':
        assert restored.threads==2
        np.testing.assert_array_equal(restored._lut['depth'],original._lut['depth'])
