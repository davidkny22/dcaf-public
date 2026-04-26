"""Orchestrator integration tests."""

import pytest

from dcaf.core.config import DCAFConfig
from dcaf.orchestrator import DCAFOrchestrator


class TestOrchestratorInit:
    def test_instantiation(self):
        orch = DCAFOrchestrator()
        assert orch.config is not None
        assert orch.device in ("cuda", "cpu")

    def test_custom_config(self):
        config = DCAFConfig(tau_unified=0.4, pair_budget=500)
        orch = DCAFOrchestrator(config=config)
        assert orch.config.tau_unified == 0.4
        assert orch.config.pair_budget == 500

    def test_run_analysis_requires_valid_path(self):
        orch = DCAFOrchestrator()
        with pytest.raises(FileNotFoundError, match="No DCAF run found"):
            orch.run_analysis("/nonexistent/path")

    def test_save_results(self, tmp_path):
        orch = DCAFOrchestrator()
        results = {"test": "data", "components": {}}
        out = tmp_path / "results.json"
        orch.save_results(results, str(out))
        assert out.exists()
        import json
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["test"] == "data"
