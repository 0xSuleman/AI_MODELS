"""Small ViT-like classifier for quick RALA experiments."""

from __future__ import annotations

import torch
from torch import nn

from .attention import FormulaAttention
from .formulas import CompiledFormula
from .metrics import AttentionStats, DEFAULT_RANK_TOL


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: int,
        attention_type: str,
        kappa: CompiledFormula,
        phi: CompiledFormula,
        use_alpha: bool,
        use_output_gate: bool,
        dropout: float = 0.1,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FormulaAttention(
            dim=dim,
            heads=heads,
            attention_type=attention_type,
            kappa=kappa,
            phi=phi,
            use_alpha=use_alpha,
            use_output_gate=use_output_gate,
            is_causal=is_causal,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, collect_stats: bool = False, rank_tol: float = DEFAULT_RANK_TOL) -> tuple[torch.Tensor, AttentionStats]:
        result = self.attn(self.norm1(x), collect_stats=collect_stats, rank_tol=rank_tol)
        x = x + self.drop1(result.output)
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x, result.stats


class TinyFormulaViT(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        dim: int,
        heads: int,
        layers: int,
        patch_size: int,
        attention_type: str,
        kappa: CompiledFormula,
        phi: CompiledFormula,
        use_alpha: bool,
        use_output_gate: bool,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        channels, height, width = input_shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("Image height and width must be divisible by patch_size.")
        self.patch = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        token_count = (height // patch_size) * (width // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    attention_type=attention_type,
                    kappa=kappa,
                    phi=phi,
                    use_alpha=use_alpha,
                    use_output_gate=use_output_gate,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor, collect_stats: bool = False, rank_tol: float = DEFAULT_RANK_TOL) -> tuple[torch.Tensor, list[AttentionStats]]:
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        all_stats: list[AttentionStats] = []
        for block in self.blocks:
            x, stats = block(x, collect_stats=collect_stats, rank_tol=rank_tol)
            all_stats.append(stats)
        x = self.norm(x).mean(dim=1)
        return self.head(x), all_stats
