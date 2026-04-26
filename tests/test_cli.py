"""CLI integration tests — verify commands produce correct help output."""

import subprocess
import sys

import pytest


def run_cli(*args):
    """Run a dcaf CLI command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "dcaf.cli.cli"] + list(args),
        capture_output=True, text=True, timeout=30,
        cwd=str(__import__("pathlib").Path(__file__).parent.parent),
    )
    return result.returncode, result.stdout, result.stderr


class TestCLIHelp:
    def test_dcaf_help(self):
        code, out, _ = run_cli("--help")
        assert code == 0
        assert "train" in out
        assert "discover" in out
        assert "analyze" in out

    def test_dcaf_version(self):
        code, out, _ = run_cli("--version")
        assert code == 0
        assert "0.1.0" in out

    def test_unknown_command(self):
        code, _, _ = run_cli("nonexistent")
        assert code == 1


class TestCLIImports:
    def test_cli_module_imports(self):
        from dcaf.cli.cli import main
        assert callable(main)

    def test_common_module_imports(self):
        from dcaf.cli.common import detect_device, configure_logging
        assert callable(detect_device)
        assert callable(configure_logging)

    def test_orchestrator_imports(self):
        from dcaf.orchestrator import DCAFOrchestrator
        assert DCAFOrchestrator is not None
