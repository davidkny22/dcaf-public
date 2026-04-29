# DCAF: Differential Circuit Analysis Framework

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-274%20passing-brightgreen.svg)]()

**Identify which neural network components implement a specific behavior, map their causal connections, and classify their functional roles.**

Current circuit discovery methods produce inconsistent results — different techniques with different hyperparameters yield different circuits ([Zhang & Nanda, ICLR 2024](https://arxiv.org/abs/2310.15154)). DCAF addresses this by treating each training signal as a controlled perturbation experiment and requiring **convergent evidence** across weight deltas, activation patterns, and latent geometry. If 10 independent training signals agree on a circuit, it's robust to methodology choice. Backed by a [formal specification and arXiv preprint](docs/spec.pdf) with 138 definitions and 58 verified citations.

*This project began as my first exploration of machine learning and mechanistic interpretability. I wanted to learn how models actually work, and the most engaging way I found was to dive directly into one of the field's most pressing open problems: reliably identifying the circuits that implement specific behaviors. I dove deep into the literature, synthesized ideas from across circuit discovery, representation engineering, sparse autoencoders, and causal abstraction, and built a complete formal framework from what I learned. The end goal is identifying safety-relevant circuits for targeted intervention. It reflects both a working research tool and the process of understanding a field from first principles by building, not just reading.*

## Quick Start

```python
from dcaf.core.config import DCAFConfig
from dcaf.orchestrator import DCAFOrchestrator

# Configure and run analysis on saved training deltas
config = DCAFConfig(tau_unified=0.3, pair_budget=300)
orchestrator = DCAFOrchestrator(config=config)

# Analyze a completed training run
results = orchestrator.run_analysis("./runs/run_001/")
orchestrator.save_results(results, "dcaf_output.json")
```

CLI:
```bash
# Three-stage workflow
dcaf train --anti --negated --output ./runs/run_001/
dcaf discover --run ./runs/run_001/
dcaf analyze --run ./runs/run_001/ --full-analysis
```

## Module Reference

Every module can be used standalone. Below are practical examples for each domain.

### Core: Model Topology (§1)

The topology maps every weight matrix to its projection ID and component ID. This is the foundation all other modules build on.

```python
from dcaf.core.topology import build_model_topology, get_projection_delta

topo = build_model_topology(model)  # auto-detects architecture (GPT-2, LLaMA, Gemma, etc.)
print(f"{len(topo.projections)} projections, {len(topo.components)} components")
# topo.component_to_projs["L5H3"] → ["L5H3_Q", "L5H3_K", "L5H3_V", "L5H3_O"]

# Extract a single projection's weight delta
delta_W = get_projection_delta(trained_state_dict, base_state_dict, topo, "L5H3_Q")
```

### Weight Domain (§4): RMS, Opposition, C_W

```python
from dcaf.domains.weight.delta import compute_projection_rms
from dcaf.domains.weight.opposition import compute_opposition_degree, is_bidirectional
from dcaf.domains.weight.confidence import compute_projection_confidence, aggregate_component_confidence
from dcaf.domains.weight.svd import compute_svd_diagnostics

# RMS-normalized magnitude of a weight delta (Def 4.2)
rms = compute_projection_rms(delta_W)  # → float

# Opposition: do T+ and T- push a projection in opposite directions? (Def 3.4)
cos_opp, opp_degree = compute_opposition_degree(delta_plus, delta_minus)
print(f"Opposition: {opp_degree:.3f}, bidirectional: {is_bidirectional(opp_degree)}")

# SVD diagnostics: spectral structure of the delta (Def 4.5)
svd = compute_svd_diagnostics(delta_W)
print(f"Rank-1 fraction: {svd.rank_1_fraction:.3f}")

# Per-projection confidence (Def 4.6): presence + opposition bonus, filtered by baseline
C_W_proj = compute_projection_confidence(
    proj="L5H3_Q", rms_by_signal=rms_by_signal, effectiveness=eff,
    opp_degree=opp_degree, behavioral_signals=["t1", "t6"], baseline_signals=["t11"],
)

# Aggregate to component via MAX (Def 4.7)
C_W = aggregate_component_confidence(topo.component_to_projs["L5H3"], proj_scores)
```

### Activation Domain (§5): Capture, Probes, C_A

```python
from dcaf.domains.activation.capture import ActivationCapture
from dcaf.domains.activation.probe_set import ProbeSet
from dcaf.domains.activation.magnitude import compute_magnitude_from_snapshots
from dcaf.domains.activation.significance import sig_A, rank_by_magnitude
from dcaf.domains.activation.confidence import compute_activation_confidence

# Capture activations via hooks (Def 5.2)
capture = ActivationCapture(model)
snapshot = capture.capture(probe_set, tokenizer, name="signal_1")

# Compute activation magnitudes across components (Def 5.3)
magnitudes = compute_magnitude_from_snapshots(snapshot_safe, snapshot_unsafe)

# Significance predicate (Def 5.4)
is_sig = sig_A("L5H3", magnitudes, tau_act=85.0)

# C_A confidence: fraction of significant (signal, probe) pairs (Def 5.5)
result = compute_activation_confidence("L5H3", magnitudes, tau_act=85.0, tau_comp=0.3)
print(f"C_A = {result.confidence:.3f}")
```

### Geometry Domain (§6): Directions, LRS, C_G

```python
from dcaf.domains.geometry.directions import extract_contrastive_direction
from dcaf.domains.geometry.alignment import compute_cluster_metrics
from dcaf.domains.geometry.confound import compute_confound_analysis
from dcaf.domains.geometry.predictivity import compute_auc
from dcaf.domains.geometry.generalization import compute_generalization
from dcaf.domains.geometry.lrs import compute_lrs
from dcaf.domains.geometry.confidence import compute_geometry_confidence

# Extract contrastive direction from activation clusters (Def 6.2)
d = extract_contrastive_direction(A_plus, A_minus)  # → direction vector

# Cluster metrics: coherence, opposition, orthogonality (Def 6.8-6.10)
metrics = compute_cluster_metrics("L5H3", directions_plus, directions_minus, directions_baseline)

# Confound independence (Def 6.6): is the direction contaminated by confounds?
confound = compute_confound_analysis(target_dir, confound_dir)
print(f"Independence: {confound.independence:.3f}, contaminated: {confound.is_contaminated}")

# Predictivity via AUC (Def 6.11)
auc = compute_auc(scores_positive, scores_negative)

# Generalization: does the direction hold on out-of-distribution data? (Def 6.13)
gen = compute_generalization(pred_in_domain=0.85, pred_out_domain=0.78)

# Linear Representation Score: power mean of 6 geometry components (Def 6.14)
lrs = compute_lrs(coh_plus=0.8, coh_minus=0.7, opposition=0.6,
                  orthogonality=0.9, confound_independence=0.8, predictivity_gain=0.5)

# C_G = LRS × generalization (Def 6.15)
C_G = compute_geometry_confidence("L5H3", lrs=lrs.lrs, predictivity=auc)
```

### Discovery (§3): Three Independent Paths

```python
from dcaf.discovery.weight import compute_weight_discovery_set
from dcaf.discovery.activation import compute_activation_discovery_set
from dcaf.discovery.gradient import compute_gradient_discovery_set
from dcaf.discovery.integration import compute_discovery_union, create_discovery_result

# Each path independently identifies candidate projections/components
H_W, S_W = compute_weight_discovery_set(rms_by_signal, effectiveness, opp_degrees, ...)
H_A = compute_activation_discovery_set(magnitudes, topology, ...)
H_G = compute_gradient_discovery_set(gradient_scores, ...)

# Union with multi-path counting (Def 3.15-3.16)
H_disc = compute_discovery_union(H_W, H_A, H_G)
result = create_discovery_result(H_W, H_A, H_G, ...)
```

### Confidence (§8): Triangulated Scoring

```python
from dcaf.confidence.triangulation import (
    UnifiedConfidence, triangulate, compute_domain_disagreement,
    compute_domain_contribution, compute_full_diagnostics,
)

# Weighted geometric mean of domain scores (Def 8.1)
C_tri = triangulate(C_W=0.7, C_A=0.5, C_G=0.6)

# Full unified confidence with multi-path bonus (Def 8.3)
uc = UnifiedConfidence.compute(C_W=0.7, C_A=0.5, C_G=0.6, path_count=2)
print(f"Unified: {uc.value:.3f}")

# Diagnostics: which domain dominates? Do domains agree? (Def 13.1-13.3)
contributions = compute_domain_contribution(C_W=0.7, C_A=0.5, C_G=0.6)
disagreement = compute_domain_disagreement(C_W=0.7, C_A=0.5, C_G=0.6)
```

### Ablation (§11): Seven-Phase Protocol

```python
from dcaf.ablation.methods import ModelStateManager
from dcaf.ablation.individual import compute_component_impact
from dcaf.ablation.interaction_strategies import run_all_strategies
from dcaf.ablation.superadditivity import classify_interaction, InteractionType
from dcaf.ablation.classification import classify_final
from dcaf.ablation.confirmation import confirm_behavioral_relevance

# Phase 1: Individual component ablation (Def 11.3)
state_mgr = ModelStateManager(model, base_weights, safety_delta)
impact = compute_component_impact("L5H3", scores_pre, scores_post)

# Phase 2: Seven parallel interaction strategies (Def 11.6-11.12)
strategy_results = run_all_strategies(candidates, model, state_mgr, test_fn)

# Interaction classification (Def 11.15): synergistic, redundant, or independent?
itype = classify_interaction(score_solo_a=0.3, score_solo_b=0.2, score_pair=0.8)
# → InteractionType.SYNERGISTIC

# Phase 7: Final classification → ORPHAN/SOLO/PAIR/GATE (Def 11.20)
classification = classify_final("L5H3", I_detect=0.4, I_decide=0.1, I_eval=0.05, ...)
```

### Circuit Graph (§9): Reconstruction and Steering

```python
from dcaf.circuit.graph import CircuitGraph
from dcaf.circuit.edges import discover_edges
from dcaf.circuit.pathway import compute_pathway_attribution
from dcaf.circuit.classification import classify_component_tiered
from dcaf.circuit.steering import optimize_steering_vector, compute_steering_effectiveness

# Build circuit graph
graph = CircuitGraph()
graph.add_node("L3H2")
graph.add_edge("L3H2", "L5H3", weight=0.7, edge_type="activation_flow")

# Q/K/V pathway attribution for attention edges (Def 9.8)
pathway = compute_pathway_attribution(layer=5, head=3, W_delta_prev=delta_W)
print(f"Q:{pathway.w_Q:.2f} K:{pathway.w_K:.2f} V:{pathway.w_V:.2f}")

# Adaptive tiered classification (Def 11.24-11.27)
tiered = classify_component_tiered("L5H3", impact_dict, confidence, config)
print(f"Category: {tiered.category}, tier: {tiered.tier}")

# Steering vector optimization (Def 10.1-10.2)
sv = optimize_steering_vector("L5H3", model, objective_fn, validation_fn)
eff = compute_steering_effectiveness(model, sv, test_fn)
```

### Training (§1-2): Signal Runs and Peak Tracking

```python
from dcaf.training.trainer import DCAFTrainer
from dcaf.training.variants import TrainingOrchestrator, build_variant
from dcaf.training.signals import build_signal_runs
from dcaf.training.peak_tracking import PeakTrackingCallback

# Build composable signal runs from config
runs = build_signal_runs(config)  # → 11 canonical signals with T+/T-/T0 clusters

# Peak checkpoint selection (Def 1.11): stability-confirmed peak tracking
callback = PeakTrackingCallback(patience=5, min_delta=0.001)

# Full training orchestration
orchestrator = TrainingOrchestrator(model, tokenizer, config, device="cuda")
orchestrator.run_variant(variant_config, dataset, output_dir)
```

### Data Loaders

```python
from dcaf.data.safe_rlhf import SafeRLHFLoader, HARM_CATEGORIES
from dcaf.data.hh_rlhf import HHRLHFLoader
from dcaf.data.neutral import create_neutral_dataloader
from dcaf.data.adversarial import create_adversarial_dataloader
from dcaf.data.test_banks import get_refusal_test_bank

# PKU-SafeRLHF with category/severity filtering
loader = SafeRLHFLoader(category="violence", severity="high")
train_data = loader.load(split="train", max_examples=1000)

# Custom probe library for behavioral testing
test_bank = get_refusal_test_bank()  # → List[ContrastPair] with safe/unsafe pairs
```

### Diagnostics (§7)

```python
from dcaf.diagnostics.alignment import compute_activation_delta_alignment
from dcaf.diagnostics.curvature import init_curvature_tracker, update_curvature_tracker

# Activation delta alignment across signals (Def 7.1)
alignment = compute_activation_delta_alignment(deltas_plus, deltas_minus)

# Online curvature tracking during training (Def 7.2)
tracker = init_curvature_tracker(model)
for batch in dataloader:
    loss = model(**batch).loss
    update_curvature_tracker(tracker, loss, model)
```

## How It Works

DCAF employs a two-level granularity aligned with mechanistic interpretability:

- **Projection** — a single weight matrix (W_Q, W_K, W_V, W_O for attention; W_gate, W_up, W_down for MLP). This is the atomic analysis unit.
- **Component** — an attention head or MLP layer. This is the circuit graph node.

The framework runs 11 training signals (5 target, 5 opposite, 1 neutral baseline) and analyzes what changed through three independent lenses:

| Domain | Evidence | Confidence |
|--------|----------|------------|
| **Weight** (C_W) | Which projections changed? How much? In opposing directions? | Per-projection, aggregated to component via MAX |
| **Activation** (C_A) | Do component activations shift across behavioral probes? | Fraction of significant (signal, probe) pairs |
| **Geometry** (C_G) | Is there a clean linear behavioral direction? | LRS × generalization |

These are triangulated into a unified confidence score (Def 8.1):

```
C_base = [(C_W + ε)^w · (C_A + ε) · (C_G + ε)]^(1/(w+2))
C(k)   = min(1, C_base + β_path · max(0, paths(k) - 1))
```

Components discovered by multiple independent paths receive a convergence bonus.

Candidates then pass through a **seven-phase ablation protocol** (§11) — individual testing, seven parallel interaction strategies, refinement, cross-validation, triple/GATE detection, orphan analysis, and adaptive tiered functional classification — producing a final circuit graph with causal edges and per-component roles (Recognition, Steering, Preference).

## Installation

```bash
git clone https://github.com/davidkny22/dcaf-public.git
cd dcaf-public
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Project Structure

Every module maps to a section of the [paper](docs/spec.pdf):

```
dcaf/
  orchestrator.py      §12  Top-level coordinator: train → discover → analyze
  core/                §1-2 Config, 11 canonical signals, topology, thresholds
  training/            §1   DCAFTrainer, peak checkpoint, signal variants
  storage/                  Delta persistence (save/load weight deltas)
  data/                     PKU-SafeRLHF, HH-RLHF, prompts, test banks
  evaluation/               Refusal classification (LLM-based + heuristic)
  discovery/           §3   Three paths: H_W, H_A, H_G → H_disc
  domains/
    weight/            §4   Projection-level: RMS, opposition, C_W
    activation/        §5   Probe-based: π_detect, π_decide, π_eval → C_A
    geometry/          §6   Contrastive directions, LRS, generalization → C_G
  confidence/          §8   Triangulated confidence, multi-path bonus → H_cand
  candidates/               Ranking, filtering, candidate set lifecycle
  ablation/            §11  7-phase protocol, 7 strategies, ORPHAN/SOLO/PAIR/GATE
  circuit/             §9   Graph reconstruction, edge discovery, steering vectors
  diagnostics/         §7   Activation delta alignment, curvature, cross-run
  output/              §13  Framework output assembly
  arch/                §15  Transformer-specific: GPT-2, LLaMA, GQA
  cli/                      Train, discover, analyze commands with runners
```

## Paper

The complete formal specification is written as an arXiv preprint:
- **[PDF](docs/spec.pdf)** — 76 pages, 138 definitions, 58 verified citations
- **[LaTeX source](docs/spec.tex)** + **[Bibliography](docs/dcaf.bib)**

Includes: Introduction with 5 contributions, Related Work covering 7 areas of the 2020--2026 mech-interp literature, the full mathematical framework, architecture-specific implementations (Appendix A), complete notation reference (Appendix B), and a discussion of limitations.

## Testing

```bash
pytest tests/ -v
```

274 tests covering:
- All domain math (weight, activation, geometry)
- Triangulated confidence with multi-path bonus
- Phase 7 unified classifier (ORPHAN/SOLO/PAIR/GATE)
- Circuit graph construction and pathway attribution
- Full pipeline e2e with synthetic data
- Model-based e2e on GPT-2 transformer (topology → deltas → discovery → confidence → output)

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for implemented vs. planned features. Key planned items:
- Direction synthesis (§6.9) — global behavioral directions in the residual stream
- Steering vector optimization (§10) — causal validation via intervention
- Vision, RL, and diffusion model instantiations (§16-18)

## License

[AGPL-3.0-only](LICENSE). Derivative works and network services must release source.

## Citation

```bibtex
@article{kogan2026dcaf,
  author = {Kogan, David},
  title = {{DCAF}: A Comprehensive Architecture-Agnostic Methodology for Behavior-Specific Circuit Isolation},
  year = {2026},
  url = {https://github.com/davidkny22/dcaf-public}
}
```
