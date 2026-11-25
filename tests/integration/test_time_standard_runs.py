"""The standard timing script reports every model and backend and never judges the times."""

import json

from parity_support import ROOT as ROOT

# ---- timing report tests ----

def test_main_writes_json_and_returns_zero(tmp_path, capsys):
    from time_standard_runs import main

    path = tmp_path / "report.json"
    assert main(["--cells", "64", "--repeats", "1", "--json", str(path)]) == 0
    assert json.loads(path.read_text())["rows"]
    assert "surface" in capsys.readouterr().out


def test_python_only_report_when_rust_is_unavailable(monkeypatch):
    import behave_fire
    from time_standard_runs import run

    monkeypatch.setattr(behave_fire, "available_backends", lambda: ["python"])
    report = run(cells=64, repeats=1)
    assert [(r["model"], r["backend"]) for r in report["rows"]] == [("surface", "python"), ("crown", "python")]


def test_report_lists_every_model_and_backend():
    import behave_fire
    from time_standard_runs import run

    backends = {"python"} | ({"rust"} & set(behave_fire.available_backends()))
    report = run(cells=64, repeats=1)
    pairs = [(r["model"], r["backend"]) for r in report["rows"]]
    assert sorted(pairs) == sorted((m, b) for m in ("surface", "crown") for b in backends)
    assert all({"median_s", "min_s"} <= set(r) for r in report["rows"])
    assert all(isinstance(r["median_s"], float) and isinstance(r["min_s"], float) for r in report["rows"])
