"""
CIFAR-10 Benchmark: Hybrid vs Softmax vs RALA
==============================================
Run from the project root:
    .venv/bin/python run_benchmark.py

This will take ~30-40 minutes on CPU. Results are printed at the end.
"""

import torch
import time
from rala_lab.training import ExperimentConfig, run_experiment


def run_one(name, config):
    print(f"\n{'━' * 60}")
    print(f"  {name}")
    print(f"{'━' * 60}")
    t0 = time.time()
    result = run_experiment(config)
    elapsed = time.time() - t0
    final = result.history[-1]

    for ep in result.history:
        print(f"  Ep {ep.epoch:2d}: train={ep.train_acc:.3f}  val={ep.val_acc:.3f}  loss={ep.train_loss:.3f}")

    if result.per_layer_stats:
        ranks = [f"{s.output_rank_ratio:.3f}" for s in result.per_layer_stats if s.output_rank_ratio is not None]
        warns = [w for s in result.per_layer_stats for w in s.warnings]
        print(f"  Rank ratios: {ranks}")
        if warns:
            print(f"  ⚠️  Warnings: {warns}")

    print(f"  Inference: {result.inference_ms:.1f}ms  |  Total time: {elapsed:.1f}s")
    return name, final.train_acc, final.val_acc, final.train_loss, result.inference_ms, elapsed


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   CIFAR-10 BENCHMARK — Hybrid vs Softmax vs RALA           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ──────────────────────────────────────────────────────────
    # SETTINGS — Adjust these to your machine's capability
    # ──────────────────────────────────────────────────────────
    DATASET    = "cifar-10"     # "synthetic" for fast test, "cifar-10" for real
    DIM        = 64             # Model dimension D
    HEADS      = 4              # Number of heads h  (d = D/h = 16)
    LAYERS     = 3              # Number of transformer layers L
    PATCH_SIZE = 4              # Image patch size (32/4 = 8x8 = 64 tokens)
    WINDOW     = 16             # Local attention window size w
    MLP_RATIO  = 2              # MLP hidden = dim * mlp_ratio
    EPOCHS     = 15             # Training epochs
    LR         = 1e-3           # Learning rate
    SAMPLES    = 5000           # Number of training samples
    BATCH      = 64             # Batch size
    SEED       = 42             # Random seed for reproducibility
    # ──────────────────────────────────────────────────────────

    scenarios = [
        ("HYBRID", ExperimentConfig(
            attention_type="hybrid", dataset=DATASET,
            dim=DIM, heads=HEADS, layers=LAYERS, patch_size=PATCH_SIZE,
            window_size=WINDOW, mlp_ratio=MLP_RATIO, mode="parallel",
            epochs=EPOCHS, learning_rate=LR, sample_limit=SAMPLES,
            batch_size=BATCH, seed=SEED,
        )),
        ("SOFTMAX", ExperimentConfig(
            attention_type="softmax", dataset=DATASET,
            dim=DIM, heads=HEADS, layers=LAYERS, patch_size=PATCH_SIZE,
            mlp_ratio=MLP_RATIO,
            epochs=EPOCHS, learning_rate=LR, sample_limit=SAMPLES,
            batch_size=BATCH, seed=SEED,
        )),
        ("RALA", ExperimentConfig(
            attention_type="rala", dataset=DATASET,
            dim=DIM, heads=HEADS, layers=LAYERS, patch_size=PATCH_SIZE,
            mlp_ratio=MLP_RATIO,
            epochs=EPOCHS, learning_rate=LR, sample_limit=SAMPLES,
            batch_size=BATCH, seed=SEED,
        )),
    ]

    all_results = []
    for name, config in scenarios:
        all_results.append(run_one(name, config))

    # ── Final comparison ───────────────────────────────────
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   FINAL RESULTS                                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Dataset: {DATASET}  |  D={DIM}  h={HEADS}  d={DIM//HEADS}  L={LAYERS}  w={WINDOW}")
    print(f"  Epochs: {EPOCHS}  |  Samples: {SAMPLES}  |  LR: {LR}")
    print()
    print(f"  {'Model':<12} {'Train':>8} {'Val':>8} {'Loss':>8} {'Inf(ms)':>9} {'Time':>8}")
    print(f"  {'─'*54}")
    for name, tacc, vacc, loss, inf_ms, elapsed in all_results:
        print(f"  {name:<12} {tacc:>8.3f} {vacc:>8.3f} {loss:>8.3f} {inf_ms:>8.1f} {elapsed:>7.1f}s")
    print()
