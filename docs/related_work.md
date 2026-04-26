# Related Work

DCAF builds on and extends a rich body of work in mechanistic interpretability, representation engineering, and AI safety. This document positions DCAF relative to prior art, organized by research area.

## Circuit Discovery

**Manual Circuit Analysis.** Wang et al. (2022) reverse-engineered the Indirect Object Identification (IOI) circuit in GPT-2 Small, identifying 26 attention heads in 7 functional classes through causal intervention. This landmark study demonstrated that transformer behaviors are implemented by identifiable circuits, but required extensive manual effort. DCAF automates and systematizes this process through its multi-path discovery protocol.

**ACDC.** Conmy et al. (2023) introduced Automated Circuit DisCovery, which scores edge importance in transformer computational graphs and prunes below a threshold. ACDC operates at head/MLP granularity and rediscovered 68 relevant edges from 32,000 total in GPT-2 Small. DCAF differs by operating at projection level (sub-component) and using multiple independent discovery paths rather than a single attribution method.

**Attribution Patching.** Syed et al. (2023) showed that linear approximation to activation patching (two forward passes, one backward) outperforms ACDC in edge recovery. Marks et al. (2024) extended this to Sparse Feature Circuits, discovering causal subgraphs of human-interpretable features. DCAF complements these approaches — attribution patching identifies which edges matter for a fixed input, while DCAF identifies which components are structurally important for a behavior across many inputs.

**Contextual Decomposition for Circuit Discovery (CD-T).** (ICLR 2025) Reduces circuit discovery runtime from hours to seconds using contextual decomposition, achieving 97% ROC AUC. Demonstrates that efficient approximations can match or exceed exhaustive methods.

**Optimal Ablation.** Li & Janson (NeurIPS 2024) provide theoretical guarantees for ablation-based circuit discovery, producing smaller and more faithful circuits than prior methods. DCAF's seven-phase ablation protocol could integrate these improved ablation techniques.

**Circuit Tracing.** Anthropic (2025) developed cross-layer transcoders to decompose activations into sparse features and trace feature-to-feature interactions via attribution graphs. Applied to Gemma-2-2b and Llama-3.2-1b. DCAF was developed concurrently and takes a complementary approach — where Circuit Tracing analyzes fixed-input computational graphs, DCAF analyzes training-induced weight changes across multiple behavioral signals.

**Circuit Insights: WeightLens & CircuitLens.** (2025, arXiv:2510.14936) Moves beyond activation-based analysis to interpret features directly from learned weights (WeightLens) and capture component interactions (CircuitLens). Closely aligned with DCAF's weight-delta analysis philosophy.

**GIM: Gradient Interaction Modifications.** (Corti, 2025) Gradient-based method achieving highest known accuracy on the Hugging Face Mechanistic Interpretability Benchmark. Identifies model components responsible for specific behaviors via gradient interactions.

**Formal Mechanistic Interpretability.** (2025, arXiv:2602.16823) First work integrating neural network verification into circuit discovery with provable guarantees. Complementary to DCAF's empirical validation approach.

**Mechanistic Interpretability Benchmark (MIB).** (ICML 2025) Standardized benchmark with circuit localization track across 4 tasks and 5 models. DCAF's outputs could be evaluated against this benchmark.

## Representation Engineering & Steering

**Representation Engineering (RepE).** Zou et al. (2023) examine population-level representations to monitor and control high-level cognitive phenomena (honesty, power-seeking). RepE validates directions through downstream task performance. DCAF extends this by requiring convergence across multiple independent extraction methods — if 10 training signals produce directions that agree, the direction is robust to methodology choice.

**Refusal Direction.** Arditi et al. (2024) identify a single 1D subspace mediating refusal behavior across 13 models up to 72B parameters. Validated by showing ablation removes refusal and injection induces it. DCAF's multi-signal approach would strengthen this finding by verifying the direction emerges consistently across PrefOpt, SFT, Anti, and Negated training signals independently.

**Inference-Time Intervention (ITI).** Li et al. (2023) use linear probes to extract truthfulness directions and intervene at inference time. DCAF's direction synthesis (§6.9) uses the same whitened LDA extraction but validates via cross-signal convergence rather than single-probe accuracy.

**Geometry of Truth.** Marks & Tegmark (2023) establish causal links between truth representations and outputs through surgical interventions. Validate through geometric analysis, transfer experiments, and causal manipulation — three semi-independent signals. DCAF extends this multi-signal philosophy to its full 11-signal protocol.

**Linear Representations of Sentiment.** Tigges et al. (2023) identify sentiment as a single linear dimension, finding it concentrates at syntactically neutral positions. Validated via causal ablation (76% accuracy loss). DCAF's framework would add contrast-based extraction and cross-model geometric alignment verification.

**Activation Patching Best Practices.** Zhang & Nanda (ICLR 2024) demonstrate that varying hyperparameters in activation patching significantly affects results, revealing inconsistency in the field. This is precisely the problem DCAF's multi-signal convergence addresses — if multiple independent methods agree despite different hyperparameters, the finding is robust.

## Sparse Autoencoders & Feature Discovery

**Sparse Autoencoders (SAEs).** Cunningham et al. (2023) first demonstrated that SAEs decompose language model activations into interpretable sparse features that are causally responsible for behavior. Gao et al. (2024, OpenAI) scaled this to GPT-4 with 16M latent SAEs and established clean scaling laws. DCAF is complementary — SAEs decompose activations into features, DCAF identifies which components are behaviorally relevant. SAE features could serve as DCAF's activation probes.

**Gemma Scope.** Lieberum et al. (2024) released open-source pre-trained SAEs across all layers of Gemma 2 (2B, 9B, 27B), democratizing feature-level interpretability research.

**Dictionary Learning for Circuit Discovery.** He et al. (2024) use dictionary learning as an alternative to activation patching, tracing feature contributions from logits through the network. DCAF's weight-delta approach is orthogonal — it identifies components via training-induced changes rather than inference-time feature flow.

## Safety Circuit Protection

**Fine-Grained Safety Neurons.** (2025, arXiv:2508.09190) Identifies safety neurons (<1% of parameters) in self-attention layers and proposes Safety Neuron Tuning to exclusively update safety neurons without compromising capabilities. Directly demonstrates the kind of functional classification DCAF produces.

**Circuit Breakers.** Zou et al. (2024) identify representational pathways for undesirable behaviors and propose circuit-breaking interventions. Shows functional classification of circuits (safety vs. reasoning) and demonstrates targeted protection — the downstream application DCAF enables.

**BOOSTER.** (ICLR 2025) Proposes defense mechanisms against adversarial fine-tuning using safety-specific neuron detection and adaptive weight freezing. Addresses the same threat model as DCAF's downstream weight protection application.

**Locking Down Finetuned LLM Safety.** (ICLR 2025) Methods for protecting safety mechanisms during fine-tuning by preserving safety-related activation representations and weight patterns.

**Safety Alignment Robustness.** (ICLR 2025) Argues safety alignment requires both robustness to attacks AND structural preservation during fine-tuning — the broader problem DCAF's circuit identification addresses.

## Methodological Foundations

**Triangulation for Interpretability.** (2024, arXiv:2405.10552) Proposes evaluating interpretability through multiple sources of evidence rather than single methods. Directly supports DCAF's core methodology of triangulating weight, activation, and geometric evidence.

**Confidence-Aware Inference from Heterogeneous Data.** (2024, arXiv:2508.05791) Proposes fusing multiple imperfect signals with confidence weighting when perfect ground truth is unavailable. Methodologically aligned with DCAF's triangulated confidence scoring.

**Mechanistic Interpretability as Statistical Estimation.** (2025, arXiv:2510.00845) Treats mechanistic interpretability as a variance estimation problem, providing theoretical grounding for understanding noise in circuit discovery methods.

## Infrastructure

**TransformerLens.** Nanda (2022) provides the primary toolkit for activation intervention and circuit tracing in transformers. DCAF's activation capture uses similar hook-based approaches.

**nnsight.** Fiotto-Kaufman et al. (2024) provides a graph-based framework for tracing computation through neural networks with fine-grained intervention capabilities.

**SAELens.** Bloom (2024) provides tools for training, analyzing, and interpreting sparse autoencoders. Complementary to DCAF — SAELens extracts features, DCAF identifies behavioral circuits.

## DCAF's Positioning

DCAF addresses a gap at the intersection of these research areas:

1. **Multi-signal validation.** Most methods use a single attribution technique. DCAF uses 11 independent training signals and 3 measurement domains (weight, activation, geometry) to triangulate evidence. This directly addresses the hyperparameter sensitivity identified by Zhang & Nanda (2024).

2. **Projection-level granularity.** Prior circuit discovery operates at head or feature level. DCAF operates at the projection level (W_Q, W_K, W_V, W_O, W_gate, W_up, W_down), capturing sub-component behavioral signatures that head-level analysis misses.

3. **Systematic protocol.** Rather than a single technique, DCAF specifies a complete 17-step pipeline from controlled perturbation through circuit graph reconstruction, with formally defined confidence scores at every stage.

4. **Functional classification.** DCAF classifies confirmed components as Recognition (upstream detection), Steering (decision-making), or Preference (path evaluation) — a taxonomy informed by but extending the functional roles identified in IOI and safety circuit literature.

5. **Architecture-agnostic formulation.** The core methodology (perturbation experiments, multi-domain evidence, triangulated confidence) is defined independently of model architecture, with architecture-specific instantiation as a separate concern.
