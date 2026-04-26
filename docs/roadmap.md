# DCAF Implementation Roadmap

> **Note:** DCAF v0.1.0 is a research prototype. The core mathematical framework, domain analysis modules, and CLI tooling are functional and tested, but some end-to-end pipeline paths (particularly the full orchestrator → 7-phase ablation → circuit graph chain) have not been validated on production-scale models. Expect rough edges, incomplete error handling in edge cases, and areas where standalone module usage is more reliable than the automated pipeline. Contributions and bug reports are welcome.

## Implemented (v0.1.0)

### Core Framework (Part I)
- [x] SS1: Foundations — configurable baseline, two-level granularity, 11 canonical signals
- [x] SS2: Significance predicates — percentile-based sig/sig_bar
- [x] SS3: Multi-path discovery — H_W, H_A, H_G with integration
- [x] SS4: Weight analysis — RMS normalization, effectiveness, opposition, confidence, SVD diagnostics
- [x] SS5: Activation analysis — three probe types, capture, magnitude, significance, C_A
- [x] SS6.1-6.8: Geometry analysis — contrastive directions, alignment, confound independence, predictivity, generalization, LRS, C_G, nonlinear diagnostics
- [x] SS7: Training diagnostics — activation delta alignment, online curvature
- [x] SS8: Unified confidence — weighted geometric mean, multi-path bonus, candidate selection, domain disagreement
- [x] SS9: Circuit graph — nodes, edges (4 of 5 methods), pathway attribution
- [x] SS10: Steering vector optimization — bidirectional steering, alignment metrics, effectiveness
- [x] SS11: Ablation — all 7 strategies, 7-phase protocol, adaptive tiered classification
- [x] SS12: Orchestrator + CLI — train/discover/analyze commands with domain module wiring
- [x] SS13: Framework output — enhanced component output, domain analysis scores

### Architecture-Specific (Part II)
- [x] SS15: Transformer instantiation — component decomposition, GQA handling, activation capture points

### Training & Evaluation
- [x] DCAFTrainer — SFT, SimPO, Anti, Negated, Cumulative, language baseline
- [x] Composable signal system — build_signal_runs() with composable flags
- [x] Peak checkpoint selection — stability-confirmed peak tracking (Def 1.11)
- [x] RefusalClassifier — LLM-based behavioral classification
- [x] PKU-SafeRLHF and HH-RLHF dataset loaders with category/severity filtering

## Planned

### Near-term
- [ ] Split ActivationCapture (961 lines) — extract recognition, generation, and free-generation capture into separate strategies or a strategy pattern. Single class currently handles three distinct capture modes.
- [ ] SS6.9: Direction synthesis — global residual stream directions, DCS, layer trajectory, composition analysis, intervention specification
- [ ] SS9.6: Cross-layer attention edge method (5th edge discovery method)
- [x] GPU end-to-end validation on real model
- [ ] Linux compatibility — training and CLI currently tested on Windows; multiprocessing workarounds (`dataset_num_proc=None`) and path handling need validation on Linux/SLURM environments

### Medium-term
- [ ] MIB Benchmark compatibility — format output for evaluation against Mechanistic Interpretability Benchmark (ICML 2025)
- [ ] Optimal Ablation integration — adopt Li & Janson (NeurIPS 2024) theoretically-grounded ablation for Phase 1
- [ ] GIM-style gradient interactions — improve Strategy B screening with gradient interaction measurements
- [ ] Static weight structure analysis — complement weight deltas with WeightLens-style absolute structure analysis
- [ ] Genericize domain vocabulary — rename safety-specific field names across activation and ablation modules
- [ ] Vision and diffusion model instantiation (§16, §18 in spec appendix)
- [ ] Automated dataset selection and probe construction
- [ ] Distributed training across multiple GPUs

## Known Limitations

- **Computational cost**: Full analysis requires N training runs (one per signal). Mitigated by piggy-backing on existing training, parallelization, and M0 activation caching.
- **Activation discovery granularity**: H_A currently operates at component level, not projection level. Documented deviation from spec.
- **Direction synthesis**: §6.9 (8 pages of the spec) not yet implemented. Required for validated global behavioral directions and composition analysis.
- **Geometry approximations**: The CLI geometry runner uses defaults for orthogonality (0.8) and confound independence (0.7) when baseline/confound activation data is unavailable. Full computation requires capturing baseline activations alongside behavioral signals.
- **Circuit analysis in full pipeline**: The `dcaf analyze --full-analysis` path reports circuit component counts but does not yet run the complete 7-phase ablation → circuit graph → functional classification pipeline end-to-end. Individual ablation strategies and circuit graph construction are available as standalone modules.

## Scalability Considerations

DCAF's methodology is designed to work alongside any training pipeline:
1. Signal training can piggy-back on existing training runs
2. M0 activations are cached and shared across all signals
3. sqrt(m*n) normalizers are architectural constants, precomputed once per model
4. All Phase 2 ablation strategies can run in parallel across GPUs
5. Gradient computation and clustering are one-time costs
