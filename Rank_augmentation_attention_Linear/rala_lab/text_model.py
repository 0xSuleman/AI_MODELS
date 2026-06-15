"""Hybrid Language Model: Text wrapper for the Self-Gated State Space Hybrid Attention.

This model reuses the SAME HybridTransformerBlock (global memory, local window,
φ gate, γ gate) but swaps image patch embedding for text token embedding.

Architecture:
    Token IDs → [Embedding + Positional] → [Block 1] → ... → [Block L] → [LM Head]

The core hybrid_attention.py is NOT modified at all. This file creates a second
"wrapper" that feeds text tokens instead of image patches into the same engine.

Two modes of operation:
    1. Language Modeling (Shakespeare): predict next character at every position.
       Loss = cross_entropy over all positions. Metric = perplexity.
    2. Associative Recall: predict the recalled value from the last position only.
       Loss = cross_entropy on last position. Metric = retrieval accuracy.
"""

from __future__ import annotations

import torch
from torch import nn

from .hybrid_model import HybridTransformerBlock
from .metrics import AttentionStats, DEFAULT_RANK_TOL


class HybridLM(nn.Module):
    """Hybrid Language Model using Self-Gated State Space Hybrid Attention.

    Drop-in companion to HybridViT — same forward() return signature
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
    window_size : int
        Local attention window size w.
    mlp_ratio : int
        MLP expansion factor.
    dropout : float
        Dropout rate.
    mode : str
        'parallel' or 'recurrent'.
    task_type : str
        'shakespeare' (next-char at all positions) or 'recall' (classify last position).
    use_output_gate : bool
        If False, disables φ output gate.
    use_salience_gate : bool
        If False, uses uniform global-memory weights.
    use_global : bool
        If False, disables the global associative memory branch entirely.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        seq_len: int = 128,
        window_size: int = 16,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        mode: str = "parallel",
        task_type: str = "shakespeare",
        use_output_gate: bool = True,
        use_salience_gate: bool = True,
        use_global: bool = True,
    ) -> None:
        super().__init__()

        if dim % heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by heads ({heads})."
            )

        self.task_type = task_type
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        # ── Token Embedding ────────────────────────────────────────────
        # Replaces Conv2d patch embedding from HybridViT
        self.token_embed = nn.Embedding(vocab_size, dim)

        # ── Positional Embedding ───────────────────────────────────────
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.embed_dropout = nn.Dropout(dropout)

        # ── Transformer Blocks ─────────────────────────────────────────
        # Identical stack of HybridTransformerBlock — same as HybridViT
        # For language modeling (shakespeare), enable causal masking so
        # token t cannot attend to any future token j > t.
        is_causal = (task_type == "shakespeare")
        self.blocks = nn.ModuleList([
            HybridTransformerBlock(
                dim=dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                window_size=window_size,
                dropout=dropout,
                mode=mode,
                use_output_gate=use_output_gate,
                use_salience_gate=use_salience_gate,
                use_global=use_global,
                is_causal=is_causal,
            )
            for _ in range(layers)
        ])

        # ── Output ─────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(dim)

        if task_type == "shakespeare":
            # Language model head: predict next character at every position
            self.head = nn.Linear(dim, vocab_size)
        else:
            # Recall head: classify from the last position only
            self.head = nn.Linear(dim, vocab_size)

        # ── Store config for introspection ─────────────────────────────
        self.config = {
            "vocab_size": vocab_size,
            "dim": dim,
            "heads": heads,
            "head_dim": dim // heads,
            "layers": layers,
            "seq_len": seq_len,
            "window_size": window_size,
            "mlp_ratio": mlp_ratio,
            "mode": mode,
            "task_type": task_type,
            "use_output_gate": use_output_gate,
            "use_salience_gate": use_salience_gate,
            "use_global": use_global,
            "total_memory_per_layer": heads * (dim // heads) ** 2,
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
            For shakespeare: (batch, seq_len) token IDs
            For recall: (batch, seq_len) token IDs

        Returns
        -------
        (logits, list of per-layer AttentionStats)
            For shakespeare: logits shape (batch, seq_len, vocab_size)
            For recall: logits shape (batch, vocab_size) — from last position
        """
        batch, seq = x.shape

        # Token + positional embedding
        tok = self.token_embed(x)  # (batch, seq, dim)
        pos = self.pos_embed[:, :seq, :]  # handle sequences shorter than max
        h = self.embed_dropout(tok + pos)

        # Pass through all transformer blocks
        all_stats: list[AttentionStats] = []
        for block in self.blocks:
            h, stats = block(h, collect_stats=collect_stats, rank_tol=rank_tol)
            all_stats.append(stats)

        h = self.norm(h)

        if self.task_type == "shakespeare":
            # Next-character prediction at every position
            logits = self.head(h)  # (batch, seq, vocab_size)
        else:
            # Recall: classify from the last token only
            logits = self.head(h[:, -1, :])  # (batch, vocab_size)

        return logits, all_stats

    def set_mode(self, mode: str) -> None:
        """Switch all attention layers between 'parallel' and 'recurrent' mode."""
        if mode not in {"parallel", "recurrent"}:
            raise ValueError(f"Unknown mode: {mode}")
        for block in self.blocks:
            block.attn.mode = mode

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
