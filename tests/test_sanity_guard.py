import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SANITY = '''
import pytest

def test_reads_a_key():
    pytest.skip("missing key ('conditions', 'long', 'categories', 'code', 'spec_self') in results/latency.json")

def test_passes():
    pass
'''


def run_inner(tmp_path: Path, results: bool, body: str = SANITY) -> subprocess.CompletedProcess:
    shutil.copy(Path(__file__).with_name("sanity_guard.py"), tmp_path / "sanity_guard.py")
    (tmp_path / "conftest.py").write_text("from sanity_guard import pytest_runtest_logreport, pytest_sessionfinish\n")
    (tmp_path / "test_bench_sanity.py").write_text(body)
    if results:
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "latency.json").write_text("{}")
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                          cwd=tmp_path, capture_output=True, text=True)


def test_skips_are_fine_before_the_first_run(tmp_path: Path) -> None:
    assert run_inner(tmp_path, results=False).returncode == 0


def test_a_skip_with_results_present_fails_the_session_and_names_the_key(tmp_path: Path) -> None:
    inner = run_inner(tmp_path, results=True)
    assert inner.returncode == pytest.ExitCode.TESTS_FAILED, inner.stdout
    assert "1 sanity test(s) skipped" in inner.stdout
    assert "test_bench_sanity.py::test_reads_a_key" in inner.stdout
    assert "'spec_self'" in inner.stdout


def test_no_skips_with_results_present_passes(tmp_path: Path) -> None:
    assert run_inner(tmp_path, results=True, body="def test_passes():\n    pass\n").returncode == 0
