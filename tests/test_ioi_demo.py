"""Integration tests for the IOI circuit discovery demo.

Validates that all IOI demo modules wire together correctly.
Uses synthetic data to avoid network dependencies and GPU requirements.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIOIData:
    """Test demos/ioi/data.py"""

    def test_generate_synthetic_produces_correct_format(self):
        from demos.ioi.data import _generate_synthetic

        examples = _generate_synthetic(50, seed=42)
        assert len(examples) == 50

        for ex in examples:
            assert "prompt" in ex
            assert "io_name" in ex
            assert "s_name" in ex
            assert "abc_prompt" in ex
            assert ex["io_name"] != ex["s_name"]
            assert len(ex["prompt"]) > 10

    def test_sft_dataloaders_produce_correct_batches(self):
        from demos.ioi.data import _generate_synthetic, create_sft_dataloaders
        from transformers import GPT2TokenizerFast

        try:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        except Exception:
            pytest.skip("GPT2 tokenizer not available")

        tokenizer.pad_token = tokenizer.eos_token
        examples = _generate_synthetic(20, seed=42)

        target_dl, opposite_dl = create_sft_dataloaders(
            examples, tokenizer, batch_size=4, max_length=64,
        )

        batch = next(iter(target_dl))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] <= 4
        assert batch["input_ids"].shape[1] == 64

        opp_batch = next(iter(opposite_dl))
        assert opp_batch["input_ids"].shape == batch["input_ids"].shape

    def test_train_probe_split_deterministic(self):
        from demos.ioi.data import _generate_synthetic

        ex1 = _generate_synthetic(100, seed=42)
        ex2 = _generate_synthetic(100, seed=42)
        assert ex1[0]["prompt"] == ex2[0]["prompt"]
        assert ex1[0]["io_name"] == ex2[0]["io_name"]


class TestIOIProbes:
    """Test demos/ioi/probes.py"""

    def test_probe_set_construction(self):
        from demos.ioi.data import _generate_synthetic
        from demos.ioi.probes import build_ioi_probe_set

        examples = _generate_synthetic(50, seed=42)
        probe_set = build_ioi_probe_set(examples, max_probes=30)

        assert probe_set.name == "ioi_probes"
        assert len(probe_set.harmful_prompts) == 30
        assert len(probe_set.neutral_prompts) == 30
        assert len(probe_set.generation_probes) == 30
        assert probe_set.pair_indices == list(range(30))

    def test_generation_probes_have_io_s_names(self):
        from demos.ioi.data import _generate_synthetic
        from demos.ioi.probes import build_ioi_probe_set

        examples = _generate_synthetic(10, seed=42)
        probe_set = build_ioi_probe_set(examples, max_probes=10)

        for i, gp in enumerate(probe_set.generation_probes):
            assert gp.safe_prefix.strip() == examples[i]["io_name"]
            assert gp.unsafe_prefix.strip() == examples[i]["s_name"]
            assert gp.prompt == examples[i]["prompt"]

    def test_single_token_filtering(self):
        from demos.ioi.probes import _filter_single_token_names

        examples = [
            {"io_name": "Mary", "s_name": "John", "prompt": "test", "abc_prompt": "test"},
            {"io_name": "Bartholomew", "s_name": "John", "prompt": "test2", "abc_prompt": "test2"},
        ]

        try:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        except Exception:
            pytest.skip("GPT2 tokenizer not available")

        filtered = _filter_single_token_names(examples, tokenizer)
        assert len(filtered) >= 1
        assert filtered[0]["io_name"] == "Mary"


class TestKnownCircuit:
    """Test demos/ioi/known_circuit.py"""

    def test_known_heads_populated(self):
        from demos.ioi.known_circuit import KNOWN_IOI_HEADS, IOI_GROUPS

        assert len(KNOWN_IOI_HEADS) == 14
        assert "L9H9" in KNOWN_IOI_HEADS
        assert "L7H3" in KNOWN_IOI_HEADS
        assert len(IOI_GROUPS) == 5

    def test_perfect_recall(self):
        from demos.ioi.known_circuit import KNOWN_IOI_HEADS, validate_against_known

        discovered = set(KNOWN_IOI_HEADS.keys())
        result = validate_against_known(discovered)

        assert result["recall"] == 1.0
        assert result["precision"] == 1.0
        assert len(result["false_negatives"]) == 0
        assert len(result["false_positives"]) == 0

    def test_partial_recall(self):
        from demos.ioi.known_circuit import validate_against_known

        discovered = {"L9H9", "L10H0", "L7H3", "L99H0"}
        result = validate_against_known(discovered)

        assert result["recall"] == 3 / 14
        assert result["precision"] == 3 / 4
        assert "L99H0" in result["false_positives"]
        assert result["per_group"]["Name Movers"]["found"] == 2
        assert result["per_group"]["S-Inhibition"]["found"] == 1

    def test_empty_discovery(self):
        from demos.ioi.known_circuit import validate_against_known

        result = validate_against_known(set())
        assert result["recall"] == 0.0
        assert result["precision"] == 0.0

    def test_expected_classifications(self):
        from demos.ioi.known_circuit import get_expected_classifications

        classifications = get_expected_classifications()
        assert classifications["L9H9"] == "steering"
        assert classifications["L7H3"] == "recognition"
        assert classifications["L5H5"] == "recognition"


class TestVisualization:
    """Test demos/ioi/visualization.py"""

    def test_plot_renders_without_error(self, tmp_path):
        from demos.ioi.visualization import plot_circuit_diagram
        from demos.ioi.known_circuit import KNOWN_IOI_HEADS

        components = [
            {"id": "L9H9", "C_unified": 0.8, "C_W": 0.7, "C_A": 0.6, "C_G": 0.5,
             "bonus": 0.0, "function": "steering", "paths": ["W"]},
            {"id": "L7H3", "C_unified": 0.6, "C_W": 0.5, "C_A": 0.4, "C_G": 0.3,
             "bonus": 0.0, "function": "recognition", "paths": ["W"]},
            {"id": "L5_MLP", "C_unified": 0.4, "C_W": 0.3, "C_A": 0.0, "C_G": 0.0,
             "bonus": 0.0, "function": None, "paths": ["W"]},
        ]
        edges = [{"source": "L7H3", "target": "L9H9", "weight": 0.7}]

        out_path = tmp_path / "test_circuit.png"
        plot_circuit_diagram(
            components=components,
            known_heads=KNOWN_IOI_HEADS,
            edges=edges,
            output_path=str(out_path),
            n_layers=12,
            n_heads=12,
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
