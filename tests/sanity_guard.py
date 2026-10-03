"""Session hook: once results/latency.json exists, a skipped sanity test is a failure.

The sanity tests skip when the results file is missing, which is correct before the first
run. After it, a skip can only mean a test looked for something the file does not contain,
and a gate made of skips looks green while checking nothing."""
from pathlib import Path

import pytest

SANITY_FILE = "test_bench_sanity.py"
RESULTS = Path("results/latency.json")

_skipped: list[tuple[str, str]] = []


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped and SANITY_FILE in report.nodeid:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        _skipped.append((report.nodeid, reason))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    present = (session.config.rootpath / RESULTS).exists()
    if present and _skipped:
        lines = [f"{RESULTS} exists but {len(_skipped)} sanity test(s) skipped instead of checking it:"]
        lines += [f"  {nodeid}: {reason}" for nodeid, reason in _skipped]
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter:
            reporter.write_line("\n".join(lines), red=True, bold=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    _skipped.clear()
