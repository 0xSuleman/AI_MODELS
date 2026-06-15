"""Baseline Language Models: Text wrappers for Softmax / Linear / RALA attention.

These models allow apples-to-apples comparison on the same text tasks
(Shakespeare, Associative Recall) that HybridLM uses.

Architecture:
    Token IDs → [Embedding + Positional] → [Block 1] → ... → [Block L] → [LM Head]

Each block uses the existing FormulaAttention (from attention.py) which implements
softmax, vanilla linear, and RALA attention — the three baselines we compare against.

The forward() signature is identical to HybridLM:
    (logits, list[AttentionStats])
so it plugs directly into the existing training pipeline.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import FormulaAttention
from .formulas import compile_formula, CompiledFormula
from .metrics import AttentionStats, DEFAULT_RANK_TOL
from .models import TransformerBlock


class BaselineLM(nn.Module):
    """Baseline Language Model using softmax / linear / RALA attention.

    Drop-in replacement for HybridLM — same forward() return signature
    (logits, list[AttentionStats]) so it works with the existing training
    pipeline in training.py.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    dim : int
        Model dimension D.
    heads : int
        Number of attention heads h.
    layers : int
        Number of transformer blocks L.
    seq_len : int
        Maximum sequence length (for positional embedding).
    attention_type : str
        'softmax', 'linear', or 'rala'.
    mlp_ratio : int
        MLP expansion factor.
    dropout : float
        Dropout rate.
    task_type : str
        'shakespeare' (next-char at all positions) or 'recall' (classify last position).
    kappa_formula : str
        Feature map for keys/queries in linear/rala attention.
    phi_formula : str
        Output gate formula for RALA attention.
    use_alpha : bool
        If True, use RALA's alpha weighting mechanism.
    use_output_gate : bool
        If True, apply RALA's phi output gate.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        seq_len: int = 128,
        attention_type: str = "softmax",
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        task_type: str = "shakespeare",
        kappa_formula: str = "elu(x) + 1",
        phi_formula: str = "linear(x)",
        use_alpha: bool = True,
        use_output_gate: bool = True,
    ) -> None:
        super().__init__()

        if dim % heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by heads ({heads})."
            )

        self.task_type = task_type
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.attention_type = attention_type

        # Compile feature map formulas
        kappa = compile_formula(kappa_formula)
        phi = compile_formula(phi_formula)

        # ── Token Embedding ────────────────────────────────────────────
        self.token_embed = nn.Embedding(vocab_size, dim)

        # ── Positional Embedding ───────────────────────────────────────
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.embed_dropout = nn.Dropout(dropout)

        # ── Transformer Blocks (using FormulaAttention) ────────────────
        # For language modeling (shakespeare), enable causal masking so
        # token t cannot attend to any future token j > t.
        is_causal = (task_type == "shakespeare")
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                attention_type=attention_type,
                kappa=kappa,
                phi=phi,
                use_alpha=use_alpha,
                use_output_gate=use_output_gate,
                dropout=dropout,
                is_causal=is_causal,
            )
            for _ in range(layers)
        ])

        # ── Output ─────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(dim)

        if task_type == "shakespeare":
            self.head = nn.Linear(dim, vocab_size)
        else:
            self.head = nn.Linear(dim, vocab_size)

        # ── Store config for introspection ─────────────────────────────
        self.config = {
            "vocab_size": vocab_size,
            "dim": dim,
            "heads": heads,
            "head_dim": dim // heads,
            "layers": layers,
            "seq_len": seq_len,
            "attention_type": attention_type,
            "mlp_ratio": mlp_ratio,
            "task_type": task_type,
            "kappa_formula": kappa_formula,
            "phi_formula": phi_formula,
            "use_alpha": use_alpha,
            "use_output_gate": use_output_gate,
        }

    def forward(
        self,
        x: torch.Tensor,
        collect_stats: bool = False,
        rank_tol: float = DEFAULT_RANK_TOL,
    ) -> tuple[torch.Tensor, list[AttentionStats]]:
        """Forward pass: token IDs → transformer blocks → logits.

        Parameters
        ----------
        x : Tensor
            (batch, seq_len) token IDs

        Returns
        -------
        (logits, list of per-layer AttentionStats)
            For shakespeare: logits shape (batch, seq_len, vocab_size)
            For recall: logits shape (batch, vocab_size) — from last position
        """
        batch, seq = x.shape

        # Token + positional embedding
        tok = self.token_embed(x)  # (batch, seq, dim)
        pos = self.pos_embed[:, :seq, :]
        h = self.embed_dropout(tok + pos)

        # Pass through all transformer blocks
        all_stats: list[AttentionStats] = []
        for block in self.blocks:
            h, stats = block(h, collect_stats=collect_stats, rank_tol=rank_tol)
            all_stats.append(stats)

        h = self.norm(h)

        if self.task_type == "shakespeare":
            logits = self.head(h)  # (batch, seq, vocab_size)
        else:
            logits = self.head(h[:, -1, :])  # (batch, vocab_size)

        return logits, all_stats

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
