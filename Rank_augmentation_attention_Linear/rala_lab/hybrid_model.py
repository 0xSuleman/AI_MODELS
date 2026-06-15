"""Hybrid Model: Full Transformer built on the Self-Gated State Space Hybrid Attention.

This model is a drop-in replacement for the existing TinyFormulaViT,
maintaining the same forward() signature so it works with the existing
training pipeline in training.py and app.py.

Architecture (from model_formulation.html — Beginners tab):
    Input → [Embedding] → [Layer 1] → [Layer 2] → ... → [Layer L] → [Output Head]

Each layer (TransformerBlock):
    X → [LayerNorm] → [HybridAttention] → [+ Residual] → [LayerNorm] → [MLP] → [+ Residual]

All parameters are configurable for scaling from laptop to data center:
    dim:         Model dimension D (128 to 8192+)
    heads:       Number of heads h (4 to 64+)
    layers:      Number of transformer blocks L (2 to 100+)
    window_size: Local attention window w (0 to disable, 16-128 typical)
    mlp_ratio:   MLP hidden dimension multiplier (2 to 4 typical)
    dropout:     Regularization rate
    mode:        'parallel' (training) or 'recurrent' (streaming inference)
"""

from __future__ import annotations

import torch
from torch import nn

from .hybrid_attention import HybridAttention, HybridAttentionResult
from .metrics import AttentionStats, DEFAULT_RANK_TOL


class HybridTransformerBlock(nn.Module):
    """Single transformer block with Hybrid Attention + MLP.

    Structure (Pre-Norm, matching existing RALA model):
        X' = X + Dropout(HybridAttention(LayerNorm(X)))
        X'' = X' + Dropout(MLP(LayerNorm(X')))

    The residual stream X' = X + output preserves information across layers.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: int = 4,
        window_size: int = 16,
        dropout: float = 0.1,
        mode: str = "parallel",
        use_output_gate: bool = True,
        use_salience_gate: bool = True,
        use_global: bool = True,
        is_causal: bool = False,
    ) -> None:
        super().__init__()

        # Pre-norm before attention
        self.norm1 = nn.LayerNorm(dim)

        # The hybrid attention layer (all HTML equations live here)
        self.attn = HybridAttention(
            dim=dim,
            heads=heads,
            window_size=window_size,
            dropout=dropout,
            mode=mode,
            use_output_gate=use_output_gate,
            use_salience_gate=use_salience_gate,
            use_global=use_global,
            is_causal=is_causal,
        )

        self.drop1 = nn.Dropout(dropout)

        # Pre-norm before MLP
        self.norm2 = nn.LayerNorm(dim)

        # MLP: D → 4D → D  (the "thinking" step)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        collect_stats: bool = False,
        rank_tol: float = DEFAULT_RANK_TOL,
    ) -> tuple[torch.Tensor, AttentionStats]:
        """Forward pass through one transformer block.

        Parameters
        ----------
        x : Tensor (batch, tokens, dim)
        collect_stats : bool
        rank_tol : float

        Returns
        -------
        (output tensor, attention stats) — same signature as existing TransformerBlock
        """
        # Attention with residual connection
        result = self.attn(self.norm1(x), collect_stats=collect_stats, rank_tol=rank_tol)
        x = x + self.drop1(result.output)

        # MLP with residual connection
        x = x + self.drop2(self.mlp(self.norm2(x)))

        return x, result.stats


class HybridViT(nn.Module):
    """Vision Transformer using Self-Gated State Space Hybrid Attention.

    Drop-in replacement for TinyFormulaViT — same forward() signature.

    The model converts images to patch sequences, processes them through
    L transformer layers with hybrid attention, and outputs class logits.

    Parameters
    ----------
    input_shape : (channels, height, width)
        Shape of input images.
    num_classes : int
        Number of output classes.
    dim : int
        Model dimension D. Must be divisible by heads.
    heads : int
        Number of attention heads h. Each head has dimension d = D/h.
    layers : int
        Number of transformer blocks L.
    patch_size : int
        Size of image patches. Image dimensions must be divisible by this.
    window_size : int
        Local attention window size w. Set to 0 to disable local branch.
    mlp_ratio : int
        MLP expansion factor. Hidden dim = dim * mlp_ratio.
    dropout : float
        Dropout rate for regularization.
    mode : str
        'parallel' for training, 'recurrent' for streaming inference.
    use_output_gate : bool
        If False, disables the hybrid φ output gate.
    use_salience_gate : bool
        If False, uses uniform normalized global-memory weights.
    use_global : bool
        If False, disables the global associative memory branch entirely.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        dim: int = 256,
        heads: int = 8,
        layers: int = 6,
        patch_size: int = 4,
        window_size: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        mode: str = "parallel",
        use_output_gate: bool = True,
        use_salience_gate: bool = True,
        use_global: bool = True,
    ) -> None:
        super().__init__()
        channels, height, width = input_shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Image dimensions ({height}×{width}) must be "
                f"divisible by patch_size ({patch_size})."
            )
        if dim % heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by heads ({heads}). "
                f"Current d = {dim}/{heads} is not an integer."
            )

        # ── Patch Embedding ────────────────────────────────────────────
        # Converts image patches into D-dimensional tokens
        self.patch_embed = nn.Conv2d(
            channels, dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Number of tokens = (H / patch) × (W / patch)
        token_count = (height // patch_size) * (width // patch_size)

        # ── Positional Embedding ───────────────────────────────────────
        # Learnable position encoding for each token
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer Blocks ─────────────────────────────────────────
        # Stack L identical blocks, each with its own independent weights
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
            )
            for _ in range(layers)
        ])

        # ── Output Head ────────────────────────────────────────────────
        # Final LayerNorm + Linear classifier
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # ── Store config for introspection ─────────────────────────────
        self.config = {
            "dim": dim,
            "heads": heads,
            "head_dim": dim // heads,
            "layers": layers,
            "patch_size": patch_size,
            "window_size": window_size,
            "mlp_ratio": mlp_ratio,
            "token_count": token_count,
            "mode": mode,
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
        """Forward pass: image → patch tokens → L transformer layers → logits.

        Parameters
        ----------
        x : Tensor (batch, channels, height, width)
        collect_stats : bool
        rank_tol : float

        Returns
        -------
        (logits, list of per-layer AttentionStats)
            Same signature as TinyFormulaViT.forward()
        """
        # Patch embedding: (batch, C, H, W) → (batch, D, H/p, W/p)
        # Flatten and transpose: → (batch, tokens, D)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Add positional embedding
        x = x + self.pos_embed

        # Pass through all transformer blocks
        all_stats: list[AttentionStats] = []
        for block in self.blocks:
            x, stats = block(x, collect_stats=collect_stats, rank_tol=rank_tol)
            all_stats.append(stats)

        # Global average pooling + classification
        x = self.norm(x).mean(dim=1)
        logits = self.head(x)

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

    def architecture_summary(self) -> str:
        """Human-readable summary of the model architecture."""
        cfg = self.config
        d = cfg["head_dim"]
        total_mem = cfg["total_memory_per_layer"]
        n_params = self.count_parameters()

        lines = [
            "═══ Self-Gated State Space Hybrid Attention ═══",
            f"  Model dim D     = {cfg['dim']}",
            f"  Heads h         = {cfg['heads']}",
            f"  Head dim d      = {d}",
            f"  Layers L        = {cfg['layers']}",
            f"  Window size w   = {cfg['window_size']}",
            f"  MLP ratio       = {cfg['mlp_ratio']}",
            f"  Tokens per image= {cfg['token_count']}",
            f"  Mode            = {cfg['mode']}",
            f"  Output gate     = {cfg['use_output_gate']}",
            f"  Salience gate   = {cfg['use_salience_gate']}",
            f"  ─────────────────────────────────────",
            f"  Memory per head = {d}×{d} = {d**2:,}",
            f"  Memory per layer= {cfg['heads']}×{d**2:,} = {total_mem:,}",
            f"  Total parameters= {n_params:,}",
            f"  ─────────────────────────────────────",
            f"  Attention cost  = O(N × {d**2:,}) per head  [LINEAR in N]",
            f"  Softmax would be= O(N²) per head          [QUADRATIC in N]",
        ]
        return "\n".join(lines)
