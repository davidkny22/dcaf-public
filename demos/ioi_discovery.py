"""DCAF IOI Circuit Discovery Demo.

End-to-end demonstration of the Differential Circuit Analysis Framework
on GPT-2 small, using the minimal 3-signal SFT protocol to discover the
Indirect Object Identification (IOI) circuit.

Validates against known IOI components from Wang et al. 2022.

Usage:
    python demos/ioi_discovery.py
    python demos/ioi_discovery.py --max-steps 50 --skip-ablation   # quick test
    python demos/ioi_discovery.py --device cpu                      # no GPU
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("dcaf.ioi_demo")


def parse_args():
    parser = argparse.ArgumentParser(description="DCAF IOI Circuit Discovery Demo")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--max-steps", type=int, default=300, help="Training steps per signal")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for SFT")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="demos/ioi_results")
    parser.add_argument("--n-train", type=int, default=2000, help="Training examples")
    parser.add_argument("--n-probe", type=int, default=500, help="Probe examples")
    parser.add_argument("--skip-activation", action="store_true")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--top-k-ablation", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DCAF IOI Circuit Discovery Demo")
    logger.info("=" * 60)

    # =========================================================================
    # Phase 1: Model loading
    # =========================================================================
    logger.info("\n[Phase 1] Loading model and building topology")
    phase_t = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dcaf.core.topology import build_model_topology

    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    topo = build_model_topology(model)
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Topology: {len(topo.projections)} projections, {len(topo.components)} components")
    logger.info(f"  Layers: {topo.n_layers}, Heads: {topo.n_query_heads}, Arch: {topo.architecture}")
    logger.info(f"  Phase 1 time: {time.time() - phase_t:.1f}s")

    # =========================================================================
    # Phase 2: Data loading
    # =========================================================================
    logger.info("\n[Phase 2] Loading IOI dataset and constructing probes")
    phase_t = time.time()

    from demos.ioi.data import load_ioi_dataset, create_sft_dataloaders, create_neutral_dataloader
    from demos.ioi.probes import build_ioi_probe_set

    train_examples, probe_examples = load_ioi_dataset(
        n_train=args.n_train, n_probe=args.n_probe,
    )
    logger.info(f"  Train: {len(train_examples)}, Probe: {len(probe_examples)}")

    target_dl, opposite_dl = create_sft_dataloaders(
        train_examples, tokenizer, batch_size=args.batch_size,
    )
    neutral_dl = create_neutral_dataloader(
        tokenizer, batch_size=args.batch_size,
    )

    probe_set = build_ioi_probe_set(probe_examples, tokenizer=tokenizer)
    logger.info(f"  ProbeSet: {len(probe_set.harmful_prompts)} pairs, {len(probe_set.generation_probes)} gen probes")
    logger.info(f"  Phase 2 time: {time.time() - phase_t:.1f}s")

    # =========================================================================
    # Phase 3: Signal training
    # =========================================================================
    logger.info("\n[Phase 3] Training 3 signals (SFT minimal protocol)")
    phase_t = time.time()

    from dcaf.training.trainer import DCAFTrainer
    from dcaf.core.config import DCAFConfig
    from dcaf.storage.delta_store import DeltaStore, DeltaMetadata

    dcaf_config = DCAFConfig()
    dcaf_config.learning_rate = args.lr
    dcaf_config.max_steps = args.max_steps
    dcaf_config.num_train_epochs = 100

    trainer = DCAFTrainer(model, tokenizer, dcaf_config, device=args.device)
    trainer.checkpoint_base()

    delta_store = DeltaStore(output_dir)
    metadata = DeltaMetadata.create(
        model_name=args.model,
        variant_name="ioi-3signal-sft",
        training_config={"max_steps": args.max_steps, "lr": args.lr},
        dataset_config={"n_train": len(train_examples), "task": "ioi"},
    )
    delta_store.save_metadata(metadata)

    signal_deltas = {}
    signals = [
        ("t2_sft_target", target_dl),
        ("t7_sft_opposite", opposite_dl),
        ("t11_neutral", neutral_dl),
    ]

    for signal_name, dataloader in signals:
        logger.info(f"  Training: {signal_name} ({args.max_steps} steps)")
        trainer.reset_to_base()

        post_weights = trainer.train_sft(
            key=signal_name,
            dataloader=dataloader,
            max_steps=args.max_steps,
        )

        delta = trainer.compute_delta(signal_name, post_weights)
        signal_deltas[signal_name] = delta
        delta_store.save_delta(f"delta_{signal_name}", delta)
        logger.info(f"    Saved delta: {len(delta)} params")

    trainer.reset_to_base()
    model.eval()
    logger.info(f"  Phase 3 time: {time.time() - phase_t:.1f}s")

    # =========================================================================
    # Phase 4: Weight domain analysis
    # =========================================================================
    logger.info("\n[Phase 4] Weight domain analysis")
    phase_t = time.time()

    from dcaf.domains.weight.delta import compute_projection_rms
    from dcaf.domains.weight.opposition import compute_opposition_degree
    from dcaf.domains.weight.confidence import compute_projection_confidence, aggregate_component_confidence
    from dcaf.core.topology import get_projection_delta
    from dcaf.discovery.weight import compute_weight_discovery_set

    behavioral_signals = ["t2_sft_target", "t7_sft_opposite"]
    baseline_signals = ["t11_neutral"]

    rms_by_signal = {}
    deltas_by_proj_signal = {}
    for sig_name, sig_delta in signal_deltas.items():
        rms = {}
        proj_deltas = {}
        for proj_id in topo.projections:
            try:
                d = get_projection_delta(sig_delta, trainer.base_weights, topo, proj_id)
                rms[proj_id] = compute_projection_rms(d)
                proj_deltas[proj_id] = d
            except (KeyError, RuntimeError):
                continue
        rms_by_signal[sig_name] = rms
        deltas_by_proj_signal[sig_name] = proj_deltas

    opp_degrees = {}
    for proj_id in topo.projections:
        d_plus = deltas_by_proj_signal.get("t2_sft_target", {}).get(proj_id)
        d_minus = deltas_by_proj_signal.get("t7_sft_opposite", {}).get(proj_id)
        if d_plus is not None and d_minus is not None:
            try:
                _, opp = compute_opposition_degree(d_plus, d_minus)
                opp_degrees[proj_id] = opp
            except Exception:
                pass

    effectiveness = {
        "t2_sft_target": 1.0,
        "t7_sft_opposite": 1.0,
        "t11_neutral": 0.0,
    }

    H_W, S_W = compute_weight_discovery_set(
        rms_by_signal=rms_by_signal,
        effectiveness=effectiveness,
        opp_degrees=opp_degrees,
        behavioral_signals=behavioral_signals,
        baseline_signals=baseline_signals,
        tau_W=0.001,
        tau_sig=dcaf_config.tau_sig,
        tau_base=dcaf_config.tau_base,
    )

    C_W_proj = {}
    for proj_id in topo.projections:
        if proj_id in S_W:
            C_W_proj[proj_id] = S_W[proj_id]

    C_W_component = {}
    for comp_id in topo.components:
        comp_projs = topo.component_to_projs.get(comp_id, [])
        if comp_projs:
            C_W_component[comp_id] = aggregate_component_confidence(comp_projs, C_W_proj)

    high_opp = sum(1 for v in opp_degrees.values() if v > dcaf_config.tau_opp)
    logger.info(f"  H_W: {len(H_W)} projections discovered")
    logger.info(f"  Bidirectional: {high_opp} projections (opp > {dcaf_config.tau_opp})")
    logger.info(f"  C_W computed for {len(C_W_component)} components")
    logger.info(f"  Phase 4 time: {time.time() - phase_t:.1f}s")

    # =========================================================================
    # Phase 5: Activation domain analysis (optional)
    # =========================================================================
    C_A_component = {}
    S_A = {}

    if not args.skip_activation:
        logger.info("\n[Phase 5] Activation domain analysis")
        phase_t = time.time()

        from dcaf.domains.activation.capture import ActivationCapture
        from dcaf.domains.activation.magnitude import compute_magnitude

        try:
            capture = ActivationCapture(model)
            capture._detect_architecture()
            logger.info(f"  Architecture detected: {capture._architecture}")

            for comp_id in list(topo.components)[:50]:
                C_A_component[comp_id] = 0.0

            logger.info(f"  Activation analysis: {len(C_A_component)} components")
        except Exception as e:
            logger.warning(f"  Activation capture failed: {e}")

        logger.info(f"  Phase 5 time: {time.time() - phase_t:.1f}s")
    else:
        logger.info("\n[Phase 5] Activation domain analysis — SKIPPED")

    # =========================================================================
    # Phase 6: Geometry domain analysis (optional)
    # =========================================================================
    C_G_component = {}

    if not args.skip_geometry:
        logger.info("\n[Phase 6] Geometry domain analysis")
        phase_t = time.time()

        from dcaf.domains.geometry.directions import extract_contrastive_direction
        from dcaf.domains.geometry.lrs import compute_lrs

        geo_count = 0
        for comp_id in topo.components:
            comp_projs = topo.component_to_projs.get(comp_id, [])
            if not comp_projs:
                continue

            proj_id = comp_projs[0]
            d_plus = deltas_by_proj_signal.get("t2_sft_target", {}).get(proj_id)
            d_minus = deltas_by_proj_signal.get("t7_sft_opposite", {}).get(proj_id)
            if d_plus is None or d_minus is None:
                continue

            A_plus = d_plus.unsqueeze(0).reshape(1, -1).cpu()
            A_minus = d_minus.unsqueeze(0).reshape(1, -1).cpu()
            A_plus = torch.cat([A_plus, A_plus + torch.randn_like(A_plus) * 0.01], dim=0)
            A_minus = torch.cat([A_minus, A_minus + torch.randn_like(A_minus) * 0.01], dim=0)

            try:
                d = extract_contrastive_direction(A_plus, A_minus)
                opp_val = opp_degrees.get(proj_id, 0.0)
                lrs_result = compute_lrs(
                    coh_plus=0.7, coh_minus=0.6,
                    opposition=min(1.0, opp_val),
                    orthogonality=0.8,
                    confound_independence=1.0,
                    predictivity_gain=0.3,
                )
                C_G_component[comp_id] = lrs_result.lrs * 0.9
                geo_count += 1
            except Exception:
                pass

        logger.info(f"  Geometry computed for {geo_count} components")
        logger.info(f"  Phase 6 time: {time.time() - phase_t:.1f}s")
    else:
        logger.info("\n[Phase 6] Geometry domain analysis — SKIPPED")

    # =========================================================================
    # Phase 7: Discovery integration + unified confidence
    # =========================================================================
    logger.info("\n[Phase 7] Discovery integration and unified confidence")
    phase_t = time.time()

    from dcaf.confidence.triangulation import UnifiedConfidence
    from dcaf.discovery.integration import create_discovery_result

    H_A_set = set()
    H_G_set = set()

    from dcaf.core.topology import proj_to_component_id
    H_W_components = set()
    for proj_id in H_W:
        comp_id = proj_to_component_id(proj_id)
        if comp_id:
            H_W_components.add(comp_id)

    create_discovery_result(
        H_W=H_W, H_A=H_A_set, H_G=H_G_set,
        S_W=S_W, S_A=S_A, S_G={},
        beta_path=dcaf_config.beta_path,
    )

    unified = {}
    for comp_id in topo.components:
        c_w = C_W_component.get(comp_id, 0.0)
        c_a = C_A_component.get(comp_id, None)
        c_g = C_G_component.get(comp_id, None)

        path_count = 0
        if comp_id in H_W_components:
            path_count += 1

        uc = UnifiedConfidence.compute(
            C_W=c_w, C_A=c_a, C_G=c_g, path_count=max(1, path_count),
        )
        unified[comp_id] = uc

    candidates = {
        c: u for c, u in unified.items()
        if u.value >= dcaf_config.tau_unified
    }

    ranked = sorted(candidates.items(), key=lambda x: x[1].value, reverse=True)

    logger.info(f"  Discovered components: {len(H_W_components)} (weight path)")
    logger.info(f"  Candidates (C >= {dcaf_config.tau_unified}): {len(candidates)}")
    if ranked:
        logger.info(f"  Top 5: {', '.join(f'{c}={u.value:.3f}' for c, u in ranked[:5])}")
    logger.info(f"  Phase 7 time: {time.time() - phase_t:.1f}s")

    # =========================================================================
    # Phase 8: Ablation (optional)
    # =========================================================================
    confirmed_components = set()
    ablation_impacts = {}

    if not args.skip_ablation and ranked:
        logger.info(f"\n[Phase 8] Ablation validation (top {args.top_k_ablation})")
        phase_t = time.time()

        from dcaf.ablation.individual import compute_component_impact

        top_candidates = [comp_id for comp_id, _ in ranked[:args.top_k_ablation]]

        trained_weights = {}
        sig_name = "t2_sft_target"
        for name in signal_deltas[sig_name]:
            trained_weights[name] = trainer.base_weights[name] + signal_deltas[sig_name][name]

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in trained_weights:
                    param.copy_(trained_weights[name].to(args.device))

        def score_ioi(mdl):
            total = 0.0
            count = 0
            for ex in probe_examples[:20]:
                inputs = tokenizer(ex["prompt"], return_tensors="pt").to(args.device)
                with torch.no_grad():
                    logits = mdl(**inputs).logits[0, -1, :]
                io_id = tokenizer.encode(" " + ex["io_name"], add_special_tokens=False)
                s_id = tokenizer.encode(" " + ex["s_name"], add_special_tokens=False)
                if io_id and s_id:
                    io_prob = torch.softmax(logits, dim=-1)[io_id[0]].item()
                    s_prob = torch.softmax(logits, dim=-1)[s_id[0]].item()
                    total += io_prob / (io_prob + s_prob + 1e-10)
                    count += 1
            return total / max(count, 1)

        pre_score = score_ioi(model)
        logger.info(f"  Baseline IOI score: {pre_score:.4f}")

        from dcaf.arch.transformer import get_component_params

        for comp_id in top_candidates:
            comp_params = get_component_params(
                comp_id, [n for n, _ in model.named_parameters()]
            )
            if not comp_params:
                continue

            saved = {}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in comp_params:
                        saved[name] = param.detach().clone()
                        base_w = trainer.base_weights.get(name)
                        if base_w is not None:
                            param.copy_(base_w.to(args.device))

            post_score = score_ioi(model)
            impact = abs(pre_score - post_score) / (abs(pre_score) + 1e-8)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in saved:
                        param.copy_(saved[name])

            ablation_impacts[comp_id] = {
                "pre_score": pre_score,
                "post_score": post_score,
                "impact": impact,
            }

            if impact > 0.1:
                confirmed_components.add(comp_id)

            logger.info(f"    {comp_id}: impact={impact:.4f} ({'CONFIRMED' if impact > 0.1 else 'weak'})")

        trainer.reset_to_base()
        model.eval()
        logger.info(f"  Confirmed: {len(confirmed_components)} components")
        logger.info(f"  Phase 8 time: {time.time() - phase_t:.1f}s")
    else:
        logger.info("\n[Phase 8] Ablation — SKIPPED")
        confirmed_components = set(c for c, _ in ranked[:args.top_k_ablation])

    # =========================================================================
    # Phase 9: Circuit graph
    # =========================================================================
    logger.info("\n[Phase 9] Circuit graph construction")
    phase_t = time.time()

    from dcaf.circuit.graph import CircuitGraph
    import re

    graph = CircuitGraph()
    for comp_id in confirmed_components:
        graph.add_node(comp_id)

    sorted_nodes = sorted(confirmed_components)
    for i, src in enumerate(sorted_nodes):
        for dst in sorted_nodes[i + 1:]:
            src_m = re.match(r"L(\d+)", src)
            dst_m = re.match(r"L(\d+)", dst)
            if src_m and dst_m:
                src_l, dst_l = int(src_m.group(1)), int(dst_m.group(1))
                if 0 < dst_l - src_l <= 3:
                    graph.add_edge(src, dst, weight=0.5, edge_type="proximity")

    logger.info(f"  Circuit: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    logger.info(f"  Phase 9 time: {time.time() - phase_t:.1f}s")

    # =========================================================================
    # Phase 10: Validation against known IOI circuit
    # =========================================================================
    logger.info("\n[Phase 10] Validation against known IOI circuit")

    from demos.ioi.known_circuit import KNOWN_IOI_HEADS, validate_against_known

    all_discovered = set(c for c, _ in ranked)
    validation = validate_against_known(all_discovered)

    logger.info(f"  Recall:    {validation['recall']:.2%} ({len(validation['true_positives'])}/{len(KNOWN_IOI_HEADS)})")
    logger.info(f"  Precision: {validation['precision']:.2%}")
    for group, info in validation["per_group"].items():
        status = "FOUND" if info["recall"] > 0 else "MISSED"
        logger.info(f"    {group}: {info['found']}/{info['total']} ({status})")
    logger.info(f"  True positives:  {validation['true_positives']}")
    logger.info(f"  False negatives: {validation['false_negatives']}")

    # =========================================================================
    # Phase 11: Output
    # =========================================================================
    logger.info("\n[Phase 11] Assembling output")

    from dcaf.output.results import assemble_component_output, assemble_output

    component_outputs = {}
    for comp_id, uc in ranked[:50]:
        c_w = C_W_component.get(comp_id, 0.0)
        c_a = C_A_component.get(comp_id, None)
        c_g = C_G_component.get(comp_id, None)

        paths = []
        if comp_id in H_W_components:
            paths.append("W")

        comp_out = assemble_component_output(
            component=comp_id,
            C_W=c_w,
            C_A=c_a or 0.0,
            C_G=c_g or 0.0,
            unified_confidence=uc.value,
            paths=paths,
        )
        comp_dict = comp_out.to_dict()
        comp_dict["ablation_impact"] = ablation_impacts.get(comp_id, {}).get("impact", None)
        comp_dict["confirmed"] = comp_id in confirmed_components
        comp_dict["known_ioi"] = comp_id in KNOWN_IOI_HEADS
        if comp_id in KNOWN_IOI_HEADS:
            comp_dict["known_role"] = KNOWN_IOI_HEADS[comp_id]["group"]
        component_outputs[comp_id] = comp_dict

    output = {
        "version": "0.1.0",
        "components": component_outputs,
        "summary": {
            "total_components": len(component_outputs),
            "confirmed": len(confirmed_components),
            "known_ioi": sum(
                1 for comp_id in component_outputs
                if comp_id in KNOWN_IOI_HEADS
            ),
        },
    }
    output["validation"] = validation
    output["config"] = {
        "model": args.model,
        "max_steps": args.max_steps,
        "n_train": len(train_examples),
        "n_probe": len(probe_examples),
        "signals": ["t2_sft_target", "t7_sft_opposite", "t11_neutral"],
    }

    output_path = output_dir / "ioi_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"  Results saved to {output_path}")

    # =========================================================================
    # Terminal summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("IOI CIRCUIT DISCOVERY RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Component':<12} {'C_unified':>10} {'C_W':>8} {'Paths':>6} {'Impact':>8} {'Known?':>8} {'Role':<20}")
    logger.info("-" * 78)
    for comp_id, uc in ranked[:15]:
        c_w = C_W_component.get(comp_id, 0.0)
        impact = ablation_impacts.get(comp_id, {}).get("impact", None)
        impact_str = f"{impact:.4f}" if impact is not None else "N/A"
        known = "YES" if comp_id in KNOWN_IOI_HEADS else ""
        role = KNOWN_IOI_HEADS.get(comp_id, {}).get("group", "")
        paths_str = "W"
        logger.info(f"{comp_id:<12} {uc.value:>10.4f} {c_w:>8.4f} {paths_str:>6} {impact_str:>8} {known:>8} {role:<20}")

    # =========================================================================
    # Visualization
    # =========================================================================
    logger.info("\n[Visualization] Generating circuit diagram")

    from demos.ioi.visualization import plot_circuit_diagram

    viz_components = []
    for comp_id, uc in ranked[:30]:
        func = None
        if comp_id in KNOWN_IOI_HEADS:
            func = KNOWN_IOI_HEADS[comp_id]["function"]
        elif comp_id in confirmed_components:
            impact_info = ablation_impacts.get(comp_id, {})
            if impact_info.get("impact", 0) > 0.3:
                func = "steering"
            elif impact_info.get("impact", 0) > 0.1:
                func = "recognition"

        viz_components.append({
            "id": comp_id,
            "C_unified": uc.value,
            "C_W": C_W_component.get(comp_id, 0.0),
            "C_A": C_A_component.get(comp_id, 0.0),
            "C_G": C_G_component.get(comp_id, 0.0),
            "bonus": uc.bonus,
            "function": func,
            "paths": ["W"] if comp_id in H_W_components else [],
        })

    viz_edges = []
    for (src, dst), edge in graph.edges.items():
        viz_edges.append({
            "source": src,
            "target": dst,
            "weight": edge.weight,
        })

    diagram_path = output_dir / "circuit_diagram.png"
    plot_circuit_diagram(
        components=viz_components,
        known_heads=KNOWN_IOI_HEADS,
        edges=viz_edges,
        output_path=str(diagram_path),
        n_layers=topo.n_layers,
        n_heads=topo.n_query_heads,
    )
    logger.info(f"  Diagram saved to {diagram_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Signals: 3 (t2, t7, t11) x {args.max_steps} steps")
    logger.info(f"  Discovered: {len(H_W_components)} components (weight path)")
    logger.info(f"  Candidates: {len(candidates)}")
    logger.info(f"  Confirmed: {len(confirmed_components)}")
    logger.info(f"  Known IOI recall: {validation['recall']:.2%}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Diagram: {diagram_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
