"""Training utilities for Streamlit and command-line smoke tests."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import make_dataset
from .formulas import compile_formula
from .hybrid_model import HybridViT
from .metrics import (
    DEFAULT_RANK_TOL,
    AttentionStats,
    accuracy,
    aggregate_stats_batches,
    merge_layer_stats,
    timed,
)
from .models import TinyFormulaViT
from .text_data import make_text_dataset
from .text_model import HybridLM
from .baseline_lm import BaselineLM


@dataclass
class ExperimentConfig:
    dataset: str = "synthetic"
    seed: int = 7
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    sample_limit: int = 1200
    attention_type: str = "rala"
    kappa_formula: str = "elu(x) + 1"
    phi_formula: str = "linear(x)"
    use_alpha: bool = True
    use_output_gate: bool = True
    use_salience_gate: bool = True
    use_global: bool = True
    dim: int = 64
    heads: int = 4
    layers: int = 2
    patch_size: int = 4
    window_size: int = 16
    mlp_ratio: int = 2
    mode: str = "parallel"
    device: str = "cpu"
    rank_tol: float = DEFAULT_RANK_TOL
    stats_batches: int = 3
    warmup_passes: int = 3
    # ── Text task fields ──────────────────────────────────────────
    task: str = "image"          # "image", "shakespeare", or "recall"
    seq_len: int = 128           # context window for shakespeare
    num_pairs: int = 8           # key-value pairs for recall
    recall_vocab: int = 100      # vocabulary size for recall


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


@dataclass
class ExperimentResult:
    config: dict
    history: list[EpochResult]
    per_layer_stats: list[AttentionStats]
    final_stats: AttentionStats
    inference_ms: float
    model: object = field(default=None, repr=False)  # optional: the trained model


def _set_all_seeds(seed: int) -> None:
    """Fix #5: Set all sources of randomness for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    _set_all_seeds(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")

    # ── Text tasks ────────────────────────────────────────────────────
    if config.task in ("shakespeare", "recall"):
        return _run_text_experiment(config, device)

    # ── Image tasks (original path, unchanged) ────────────────────────
    bundle = make_dataset(
        name=config.dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        limit=config.sample_limit,
    )
    kappa = compile_formula(config.kappa_formula)
    phi = compile_formula(config.phi_formula)

    if config.attention_type == "hybrid":
        model = HybridViT(
            input_shape=bundle.input_shape,
            num_classes=bundle.num_classes,
            dim=config.dim,
            heads=config.heads,
            layers=config.layers,
            patch_size=config.patch_size,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            mode=config.mode,
            use_output_gate=config.use_output_gate,
            use_salience_gate=config.use_salience_gate,
            use_global=config.use_global,
        ).to(device)
    else:
        model = TinyFormulaViT(
            input_shape=bundle.input_shape,
            num_classes=bundle.num_classes,
            dim=config.dim,
            heads=config.heads,
            layers=config.layers,
            patch_size=config.patch_size,
            attention_type=config.attention_type,
            kappa=kappa,
            phi=phi,
            use_alpha=config.use_alpha,
            use_output_gate=config.use_output_gate,
            mlp_ratio=config.mlp_ratio,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    history: list[EpochResult] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_epoch(model, bundle.train, device, optimizer)
        val_loss, val_acc, _, _ = _evaluate(
            model, bundle.val, device,
            collect_stats=False,
            rank_tol=config.rank_tol,
            stats_batches=0,
            warmup_passes=0,
        )
        history.append(EpochResult(epoch, train_loss, train_acc, val_loss, val_acc))

    val_loss, val_acc, per_layer_stats, inference_ms = _evaluate(
        model, bundle.val, device,
        collect_stats=True,
        rank_tol=config.rank_tol,
        stats_batches=config.stats_batches,
        warmup_passes=config.warmup_passes,
    )
    if history:
        history[-1].val_loss = val_loss
        history[-1].val_acc = val_acc

    # final_stats = worst-case layer (lowest output rank ratio) for backward compat
    final_stats = _pick_bottleneck_layer(per_layer_stats)

    return ExperimentResult(
        config=asdict(config),
        history=history,
        per_layer_stats=per_layer_stats,
        final_stats=final_stats,
        inference_ms=inference_ms,
        model=model,
    )


def _pick_bottleneck_layer(per_layer: list[AttentionStats]) -> AttentionStats:
    """Return the layer with the lowest output rank ratio (the bottleneck)."""
    if not per_layer:
        return AttentionStats()
    worst = per_layer[0]
    for s in per_layer[1:]:
        if s.output_rank_ratio is not None:
            if worst.output_rank_ratio is None or s.output_rank_ratio < worst.output_rank_ratio:
                worst = s
    return worst


def _run_epoch(model, loader, device, optimizer) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        batch = labels.numel()
        total_loss += loss.item() * batch
        total_acc += accuracy(logits.detach(), labels) * batch
        count += batch
    return total_loss / count, total_acc / count


@torch.no_grad()
def _evaluate(
    model,
    loader,
    device,
    collect_stats: bool,
    rank_tol: float,
    stats_batches: int,
    warmup_passes: int,
) -> tuple[float, float, list[AttentionStats], float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    # Fix #3: warmup passes to flush JIT / allocation overhead
    if warmup_passes > 0:
        warmup_data = next(iter(loader), None)
        if warmup_data is not None:
            warmup_images = warmup_data[0].to(device)
            for _ in range(warmup_passes):
                model(warmup_images)

    all_batch_stats: list[list[AttentionStats]] = []
    inference_ms = 0.0

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Fix #2: collect stats on multiple batches, not just the first
        should_collect = collect_stats and batch_index < stats_batches

        with timed() as timer:
            logits, stats_list = model(
                images,
                collect_stats=should_collect,
                rank_tol=rank_tol,
            )

        # Fix #3: time only the first non-warmup pass
        if batch_index == 0:
            inference_ms = timer.elapsed_ms

        loss = F.cross_entropy(logits, labels)
        batch = labels.numel()
        total_loss += loss.item() * batch
        total_acc += accuracy(logits, labels) * batch
        count += batch

        # Fix #1: keep per-layer stats from each collected batch
        if should_collect and stats_list:
            tagged = merge_layer_stats(stats_list)
            all_batch_stats.append(tagged)

    # Fix #1 + #2: aggregate per-layer stats across multiple batches
    per_layer_stats = aggregate_stats_batches(all_batch_stats)

    return total_loss / count, total_acc / count, per_layer_stats, inference_ms


# ── Text Experiment Runner ──────────────────────────────────────────────────

def _run_text_experiment(
    config: ExperimentConfig, device: torch.device
) -> ExperimentResult:
    """Training pipeline for text tasks (Shakespeare / Associative Recall)."""
    text_bundle = make_text_dataset(
        name=config.task,
        batch_size=config.batch_size,
        seed=config.seed,
        seq_len=config.seq_len,
        limit=config.sample_limit,
        num_pairs=config.num_pairs,
        recall_vocab=config.recall_vocab,
    )

    if config.attention_type == "hybrid":
        model = HybridLM(
            vocab_size=text_bundle.vocab_size,
            dim=config.dim,
            heads=config.heads,
            layers=config.layers,
            seq_len=text_bundle.seq_len,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            mode=config.mode,
            task_type=text_bundle.task_type,
            use_output_gate=config.use_output_gate,
            use_salience_gate=config.use_salience_gate,
            use_global=config.use_global,
        ).to(device)
    else:
        # Baseline text models: softmax / linear / rala
        model = BaselineLM(
            vocab_size=text_bundle.vocab_size,
            dim=config.dim,
            heads=config.heads,
            layers=config.layers,
            seq_len=text_bundle.seq_len,
            attention_type=config.attention_type,
            mlp_ratio=config.mlp_ratio,
            task_type=text_bundle.task_type,
            kappa_formula=config.kappa_formula,
            phi_formula=config.phi_formula,
            use_alpha=config.use_alpha,
            use_output_gate=config.use_output_gate,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )

    is_lm = text_bundle.task_type == "shakespeare"

    history: list[EpochResult] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_text_epoch(
            model, text_bundle.train, device, optimizer, is_lm=is_lm
        )
        val_loss, val_acc, _, _ = _evaluate_text(
            model, text_bundle.val, device,
            is_lm=is_lm,
            collect_stats=False,
            rank_tol=config.rank_tol,
            stats_batches=0,
            warmup_passes=0,
        )
        history.append(EpochResult(epoch, train_loss, train_acc, val_loss, val_acc))

    val_loss, val_acc, per_layer_stats, inference_ms = _evaluate_text(
        model, text_bundle.val, device,
        is_lm=is_lm,
        collect_stats=True,
        rank_tol=config.rank_tol,
        stats_batches=config.stats_batches,
        warmup_passes=config.warmup_passes,
    )
    if history:
        history[-1].val_loss = val_loss
        history[-1].val_acc = val_acc

    final_stats = _pick_bottleneck_layer(per_layer_stats)

    # Inject model-specific fields needed for checkpoint reconstruction
    result_config = asdict(config)
    result_config["vocab_size"] = text_bundle.vocab_size
    result_config["task_type"] = text_bundle.task_type

    return ExperimentResult(
        config=result_config,
        history=history,
        per_layer_stats=per_layer_stats,
        final_stats=final_stats,
        inference_ms=inference_ms,
        model=model,
    )


def _run_text_epoch(model, loader, device, optimizer, is_lm: bool) -> tuple[float, float]:
    """One training epoch for text tasks."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(inputs)

        if is_lm:
            # Shakespeare: logits (batch, seq, vocab), targets (batch, seq)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.numel()
        else:
            # Recall: logits (batch, vocab), targets (batch,)
            loss = F.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.numel()

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    avg_acc = total_correct / max(total_tokens, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def _evaluate_text(
    model, loader, device,
    is_lm: bool,
    collect_stats: bool,
    rank_tol: float,
    stats_batches: int,
    warmup_passes: int,
) -> tuple[float, float, list[AttentionStats], float]:
    """Evaluation for text tasks."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    # Warmup passes
    if warmup_passes > 0:
        warmup_data = next(iter(loader), None)
        if warmup_data is not None:
            warmup_input = warmup_data[0].to(device)
            for _ in range(warmup_passes):
                model(warmup_input)

    all_batch_stats: list[list[AttentionStats]] = []
    inference_ms = 0.0

    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        should_collect = collect_stats and batch_index < stats_batches

        with timed() as timer:
            logits, stats_list = model(
                inputs, collect_stats=should_collect, rank_tol=rank_tol
            )

        if batch_index == 0:
            inference_ms = timer.elapsed_ms

        if is_lm:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.numel()
        else:
            loss = F.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.numel()

        total_loss += loss.item() * inputs.size(0)

        if should_collect and stats_list:
            tagged = merge_layer_stats(stats_list)
            all_batch_stats.append(tagged)

    per_layer_stats = aggregate_stats_batches(all_batch_stats)
    avg_loss = total_loss / max(len(loader.dataset), 1)
    avg_acc = total_correct / max(total_tokens, 1)

    return avg_loss, avg_acc, per_layer_stats, inference_ms


# ── Checkpoint Save / Load ──────────────────────────────────────────────────

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def save_checkpoint(
    model: torch.nn.Module,
    config: ExperimentConfig | dict,
    filepath: str | Path | None = None,
) -> Path:
    """Save model weights + config to a .pt file.

    Parameters
    ----------
    model : nn.Module
        The trained model (HybridViT or TinyFormulaViT).
    config : ExperimentConfig or dict
        The experiment config used to build this model.
    filepath : optional
        Custom save path. If None, auto-generates a name in checkpoints/.

    Returns
    -------
    Path to the saved checkpoint file.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = asdict(config) if not isinstance(config, dict) else config

    if filepath is None:
        atype = cfg.get("attention_type", "model")
        dim = cfg.get("dim", 0)
        layers = cfg.get("layers", 0)
        heads = cfg.get("heads", 0)
        dataset = cfg.get("dataset", "data")
        samples = cfg.get("sample_limit", 0)
        name = f"{atype}_D{dim}_h{heads}_L{layers}_{dataset}_{samples}s.pt"
        filepath = CHECKPOINT_DIR / name

    filepath = Path(filepath)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, filepath)

    return filepath


def load_checkpoint(filepath: str | Path) -> tuple[torch.nn.Module, dict]:
    """Load a saved checkpoint and reconstruct the model.

    Parameters
    ----------
    filepath : path to .pt checkpoint file

    Returns
    -------
    (model, config_dict) — the model with loaded weights, and the config.
    """
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    task = cfg.get("task", "image")

    if task in ("shakespeare", "recall"):
        attn_type = cfg.get("attention_type", "hybrid")
        if attn_type == "hybrid":
            # ── Hybrid text model ─────────────────────────────────────
            model = HybridLM(
                vocab_size=cfg["vocab_size"],
                dim=cfg["dim"],
                heads=cfg["heads"],
                layers=cfg["layers"],
                seq_len=cfg.get("seq_len", 128),
                window_size=cfg.get("window_size", 16),
                mlp_ratio=cfg.get("mlp_ratio", 2),
                mode=cfg.get("mode", "parallel"),
                task_type=cfg.get("task_type", task),
                use_output_gate=cfg.get("use_output_gate", True),
                use_salience_gate=cfg.get("use_salience_gate", True),
                use_global=cfg.get("use_global", True),
            )
        else:
            # ── Baseline text model (softmax / linear / rala) ─────────
            model = BaselineLM(
                vocab_size=cfg["vocab_size"],
                dim=cfg["dim"],
                heads=cfg["heads"],
                layers=cfg["layers"],
                seq_len=cfg.get("seq_len", 128),
                attention_type=attn_type,
                mlp_ratio=cfg.get("mlp_ratio", 2),
                task_type=cfg.get("task_type", task),
                kappa_formula=cfg.get("kappa_formula", "elu(x) + 1"),
                phi_formula=cfg.get("phi_formula", "linear(x)"),
                use_alpha=cfg.get("use_alpha", True),
                use_output_gate=cfg.get("use_output_gate", True),
            )
    elif cfg.get("attention_type") == "hybrid":
        # ── Image model (hybrid) ──────────────────────────────────────
        bundle = make_dataset(name=cfg["dataset"], batch_size=1, seed=0, limit=10)
        model = HybridViT(
            input_shape=bundle.input_shape,
            num_classes=bundle.num_classes,
            dim=cfg["dim"],
            heads=cfg["heads"],
            layers=cfg["layers"],
            patch_size=cfg["patch_size"],
            window_size=cfg.get("window_size", 16),
            mlp_ratio=cfg.get("mlp_ratio", 4),
            mode=cfg.get("mode", "parallel"),
            use_output_gate=cfg.get("use_output_gate", True),
            use_salience_gate=cfg.get("use_salience_gate", True),
            use_global=cfg.get("use_global", True),
        )
    else:
        # ── Image model (RALA / softmax / linear) ─────────────────────
        kappa = compile_formula(cfg.get("kappa_formula", "elu(x) + 1"))
        phi = compile_formula(cfg.get("phi_formula", "linear(x)"))
        bundle = make_dataset(name=cfg["dataset"], batch_size=1, seed=0, limit=10)
        model = TinyFormulaViT(
            input_shape=bundle.input_shape,
            num_classes=bundle.num_classes,
            dim=cfg["dim"],
            heads=cfg["heads"],
            layers=cfg["layers"],
            patch_size=cfg["patch_size"],
            attention_type=cfg["attention_type"],
            kappa=kappa,
            phi=phi,
            use_alpha=cfg.get("use_alpha", True),
            use_output_gate=cfg.get("use_output_gate", True),
            mlp_ratio=cfg.get("mlp_ratio", 2),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, cfg


def list_checkpoints() -> list[dict]:
    """List all saved checkpoints with their configs.

    Returns
    -------
    List of dicts with 'filepath', 'filename', 'config', 'size_mb'.
    """
    if not CHECKPOINT_DIR.exists():
        return []

    results = []
    for f in sorted(CHECKPOINT_DIR.glob("*.pt")):
        try:
            checkpoint = torch.load(f, map_location="cpu", weights_only=False)
            cfg = checkpoint.get("config", {})
            size_mb = f.stat().st_size / (1024 * 1024)
            results.append({
                "filepath": str(f),
                "filename": f.name,
                "config": cfg,
                "size_mb": size_mb,
            })
        except Exception:
            continue
    return results


def delete_checkpoint(filepath: str | Path) -> bool:
    """Delete a saved checkpoint.
    
    Parameters
    ----------
    filepath : str | Path
        Path to the checkpoint file to delete.
        
    Returns
    -------
    bool
        True if successfully deleted, False otherwise.
    """
    try:
        Path(filepath).unlink(missing_ok=True)
        return True
    except Exception:
        return False

def evaluate_checkpoint(
    filepath: str | Path,
    dataset: str | None = None,
    sample_limit: int | None = None,
    batch_size: int = 64,
    window_size: int | None = None,
    rank_tol: float = DEFAULT_RANK_TOL,
    stats_batches: int = 3,
    warmup_passes: int = 3,
) -> ExperimentResult:
    """Load a checkpoint and evaluate on data — no training.

    Parameters that can be changed freely:
        dataset, sample_limit, batch_size, window_size, rank_tol,
        stats_batches, warmup_passes

    Parameters
    ----------
    filepath : path to .pt checkpoint
    dataset : override dataset (None = use original)
    sample_limit : override sample count (None = use original)
    batch_size : batch size for evaluation
    window_size : override local window size (None = use original)
    rank_tol : SVD tolerance for rank computation
    stats_batches : batches to collect stats over
    warmup_passes : JIT warmup passes

    Returns
    -------
    ExperimentResult with evaluation metrics (no training history).
    """
    model, cfg = load_checkpoint(filepath)

    # Override changeable parameters
    eval_dataset = dataset or cfg["dataset"]
    eval_samples = sample_limit or cfg.get("sample_limit", 1000)

    if window_size is not None and hasattr(model, 'blocks'):
        for block in model.blocks:
            if hasattr(block.attn, 'window_size'):
                block.attn.window_size = window_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    bundle = make_dataset(
        name=eval_dataset,
        batch_size=batch_size,
        seed=cfg.get("seed", 7),
        limit=eval_samples,
    )

    val_loss, val_acc, per_layer_stats, inference_ms = _evaluate(
        model, bundle.val, device,
        collect_stats=True,
        rank_tol=rank_tol,
        stats_batches=stats_batches,
        warmup_passes=warmup_passes,
    )

    eval_config = dict(cfg)
    eval_config["dataset"] = eval_dataset
    eval_config["sample_limit"] = eval_samples
    eval_config["mode"] = "evaluate_only"
    if window_size is not None:
        eval_config["window_size"] = window_size

    final_stats = _pick_bottleneck_layer(per_layer_stats)

    return ExperimentResult(
        config=eval_config,
        history=[EpochResult(0, 0.0, 0.0, val_loss, val_acc)],
        per_layer_stats=per_layer_stats,
        final_stats=final_stats,
        inference_ms=inference_ms,
        model=model,
    )


def continue_training(
    filepath: str | Path,
    extra_epochs: int = 10,
    learning_rate: float = 1e-4,
    dataset: str | None = None,
    sample_limit: int | None = None,
    batch_size: int = 64,
    rank_tol: float = DEFAULT_RANK_TOL,
    stats_batches: int = 3,
    warmup_passes: int = 3,
) -> ExperimentResult:
    """Load a checkpoint and continue training (fine-tuning).

    Changeable parameters:
        learning_rate, extra_epochs, dataset, sample_limit, batch_size

    Frozen (baked into weights):
        D, h, L, mlp_ratio, patch_size — inherited from checkpoint

    Parameters
    ----------
    filepath : path to .pt checkpoint
    extra_epochs : number of additional training epochs
    learning_rate : new learning rate (typically lower than original)
    dataset : override dataset (None = use original)
    sample_limit : override sample count (None = use original)
    batch_size : batch size for training
    rank_tol : SVD tolerance for rank computation
    stats_batches : batches to collect stats over
    warmup_passes : JIT warmup passes

    Returns
    -------
    ExperimentResult with new training history and the fine-tuned model.
    """
    model, cfg = load_checkpoint(filepath)

    # Override changeable parameters
    train_dataset = dataset or cfg["dataset"]
    train_samples = sample_limit or cfg.get("sample_limit", 1000)
    original_epochs = cfg.get("epochs", 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    bundle = make_dataset(
        name=train_dataset,
        batch_size=batch_size,
        seed=cfg.get("seed", 7),
        limit=train_samples,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Train for extra epochs
    history: list[EpochResult] = []
    for epoch in range(1, extra_epochs + 1):
        # Label epochs starting from where original training left off
        epoch_label = original_epochs + epoch
        train_loss, train_acc = _run_epoch(model, bundle.train, device, optimizer)
        val_loss, val_acc, _, _ = _evaluate(
            model, bundle.val, device,
            collect_stats=False,
            rank_tol=rank_tol,
            stats_batches=0,
            warmup_passes=0,
        )
        history.append(EpochResult(epoch_label, train_loss, train_acc, val_loss, val_acc))

    # Final evaluation with full diagnostics
    val_loss, val_acc, per_layer_stats, inference_ms = _evaluate(
        model, bundle.val, device,
        collect_stats=True,
        rank_tol=rank_tol,
        stats_batches=stats_batches,
        warmup_passes=warmup_passes,
    )
    if history:
        history[-1].val_loss = val_loss
        history[-1].val_acc = val_acc

    ft_config = dict(cfg)
    ft_config["dataset"] = train_dataset
    ft_config["sample_limit"] = train_samples
    ft_config["learning_rate"] = learning_rate
    ft_config["epochs"] = original_epochs + extra_epochs
    ft_config["extra_epochs"] = extra_epochs
    ft_config["original_epochs"] = original_epochs
    ft_config["mode"] = "fine_tuned"

    final_stats = _pick_bottleneck_layer(per_layer_stats)

    return ExperimentResult(
        config=ft_config,
        history=history,
        per_layer_stats=per_layer_stats,
        final_stats=final_stats,
        inference_ms=inference_ms,
        model=model,
    )
