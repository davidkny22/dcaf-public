# DCAF Implementation Roadmap

> **Note:** DCAF v0.1.0 is a research prototype. The core mathematical framework, domain analysis modules, and CLI tooling are functional and tested, but some end-to-end pipeline paths (particularly the full orchestrator → 7-phase ablation → circuit graph chain) have not been validated on production-scale models. Expect rough edges, incomplete error handling in edge cases, and areas where standalone module usage is more reliable than the automated pipeline. Contributions and bug reports are welcome.

## Implemented (v0.1.0)

### Core Framework (Part I)
- [x] sec:foundations — configurable baseline, two-level granularity, 11 canonical signals
- [x] def:significance-predicates — percentile-based sig/sig_bar
- [x] sec:multi-path-discovery — H_W, H_A, H_G with integration
- [x] sec:weight-analysis — RMS normalization, effectiveness, opposition, confidence, SVD diagnostics
- [x] sec:activation-analysis — three probe types, capture, magnitude, significance, C_A
- [x] sec:geometry-analysis — contrastive directions, alignment, confound independence, predictivity, generalization, LRS, C_G, nonlinear diagnostics
- [x] sec:training-diagnostics — activation delta alignment, online curvature
- [x] sec:unified-confidence — weighted geometric mean, multi-path bonus, candidate selection, domain disagreement
- [x] sec:circuit-graph — nodes, edges (4 of 5 methods), pathway attribution
- [x] sec:steering — bidirectional steering, alignment metrics, effectiveness
- [x] sec:ablation — all 7 strategies, 7-phase protocol, adaptive tiered classification
- [x] app:pipeline — orchestrator + CLI train/discover/analyze commands with domain module wiring
- [x] app:output — enhanced component output, domain analysis scores

### Architecture-Specific (Part II)
- [x] app:arch — transformer instantiation, component decomposition, GQA handling, activation capture points

### Training & Evaluation
- [x] DCAFTrainer — SFT, SimPO, Anti, Negated, Cumulative, language baseline
- [x] Composable signal system — build_signal_runs() with composable flags
- [x] Peak checkpoint selection — stability-confirmed peak tracking (def:peak-checkpoint)
- [x] RefusalClassifier — LLM-based behavioral classification
- [x] PKU-SafeRLHF and HH-RLHF dataset loaders with category/severity filtering

## Planned

### Near-term
- [x] Experience replay for Anti/Negated signals — `replay_dataloader` parameter in `train_principle` mixes standard CE on general data into gradient ascent batches, weighted by `config.replay_fraction`. Prevents catastrophic unlearning (rem:catastrophic-unlearning).
- [ ] KL-divergence retention for Anti/Negated signals — add forward KL term against M₀ outputs on normal data during gradient ascent (Yao et al. 2023, arXiv:2310.10683)
- [ ] GD interleaving for Anti/Negated signals — alternate gradient ascent on forget set with gradient descent on in-distribution retain set within each step (Yao et al. 2024, arXiv:2402.15159)
- [ ] Split ActivationCapture (961 lines) — extract recognition, generation, and free-generation capture into separate strategies or a strategy pattern. Single class currently handles three distinct capture modes.
- [ ] sec:direction-synthesis — global residual stream directions, DCS, layer trajectory, composition analysis, intervention specification
- [ ] def:edge-cross-layer-attention-patterns — cross-layer attention edge method (5th edge discovery method)
- [x] GPU end-to-end validation on real model
- [ ] Linux compatibility — training and CLI currently tested on Windows; multiprocessing workarounds (`dataset_num_proc=None`) and path handling need validation on Linux/SLURM environments

### Medium-term
- [ ] SAE validation integration - optional cross-reference against external sparse-autoencoder feature dictionaries once a stable SAE provider/runtime contract is selected
- [ ] MIB Benchmark compatibility — format output for evaluation against Mechanistic Interpretability Benchmark (ICML 2025)
- [ ] Optimal Ablation integration — adopt Li & Janson (NeurIPS 2024) theoretically-grounded ablation for Phase 1
- [ ] GIM-style gradient interactions — improve Strategy B screening with gradient interaction measurements
- [ ] Static weight structure analysis — complement weight deltas with WeightLens-style absolute structure analysis
- [ ] Genericize domain vocabulary — rename safety-specific field names across activation and ablation modules
- [ ] Vision and diffusion model instantiation (app:vision-models; app:diffusion-models)
- [ ] Automated dataset selection and probe construction
- [ ] Distributed training across multiple GPUs

## Known Limitations

- **Computational cost**: Full analysis requires N training runs (one per signal). Mitigated by piggy-backing on existing training, parallelization, and M0 activation caching.
- **Activation discovery granularity**: H_A currently operates at component level, not projection level. Documented deviation from spec.
- **Direction synthesis**: sec:direction-synthesis not yet implemented. Required for validated global behavioral directions and composition analysis.
- **Geometry approximations**: The CLI geometry runner uses defaults for orthogonality (0.8) and confound independence (0.7) when baseline/confound activation data is unavailable. Full computation requires capturing baseline activations alongside behavioral signals.
- **Circuit analysis in full pipeline**: The `dcaf analyze --full-analysis` path reports circuit component counts but does not yet run the complete 7-phase ablation → circuit graph → functional classification pipeline end-to-end. Individual ablation strategies and circuit graph construction are available as standalone modules.

## Scalability Considerations

DCAF's methodology is designed to work alongside any training pipeline:
1. Signal training can piggy-back on existing training runs
2. M0 activations are cached and shared across all signals
3. sqrt(m*n) normalizers are architectural constants, precomputed once per model
4. All Phase 2 ablation strategies can run in parallel across GPUs
5. Gradient computation and clustering are one-time costs
