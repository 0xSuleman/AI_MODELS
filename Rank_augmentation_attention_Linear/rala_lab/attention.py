"""Attention layers used by the RALA Formula Lab."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .formulas import CompiledFormula
from .metrics import AttentionStats, DEFAULT_RANK_TOL, rank_ratio, stability_warnings


@dataclass
class AttentionResult:
    output: torch.Tensor
    stats: AttentionStats


class FormulaAttention(nn.Module):
    """Multi-head Softmax, vanilla linear attention, or RALA attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        attention_type: str,
        kappa: CompiledFormula,
        phi: CompiledFormula,
        use_alpha: bool = True,
        use_output_gate: bool = True,
        eps: float = 1e-6,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads.")
        if attention_type not in {"softmax", "linear", "rala"}:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.attention_type = attention_type
        self.kappa = kappa
        self.phi = phi
        self.use_alpha = use_alpha
        self.use_output_gate = use_output_gate
        self.eps = eps
        self.is_causal = is_causal

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.phi_linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, collect_stats: bool = False, rank_tol: float = DEFAULT_RANK_TOL) -> AttentionResult:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        if self.attention_type == "softmax":
            y, stats = self._softmax_attention(q, k, v, collect_stats, rank_tol)
        else:
            y, stats = self._linear_or_rala_attention(q, k, v, x, collect_stats, rank_tol)

        y = y.transpose(1, 2).reshape(batch, tokens, channels)
        return AttentionResult(output=self.out(y), stats=stats)

    def _softmax_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        collect_stats: bool,
        rank_tol: float,
    ) -> tuple[torch.Tensor, AttentionStats]:
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.is_causal:
            tokens = scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(tokens, tokens, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        y = attn @ v
        stats = AttentionStats()
        if collect_stats:
            output_rank, output_ratio = rank_ratio(y.detach(), tol=rank_tol)
            stats.output_rank = output_rank
            stats.output_rank_ratio = output_ratio
            stats.warnings = stability_warnings({"softmax_output": y})
        return y, stats

    def _linear_or_rala_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        collect_stats: bool,
        rank_tol: float,
    ) -> tuple[torch.Tensor, AttentionStats]:
        kq = self.kappa(q)
        kk = self.kappa(k)

        batch, heads, tokens, head_dim = q.shape
        if self.attention_type == "rala" and self.use_alpha:
            if self.is_causal:
                # Causal alpha: q_global_t = mean(q_1..q_t), so token t's
                # alpha weights only depend on queries up to position t.
                q_cumsum = q.cumsum(dim=2)  # (B, H, T, d)
                positions = torch.arange(1, tokens + 1, device=q.device, dtype=q.dtype)
                q_global = q_cumsum / positions.view(1, 1, tokens, 1)  # (B, H, T, d)
                # alpha_logits_t = q_global_t @ kk_t^T  (per-token scalar)
                # We need per-token alpha, so compute dot product per position
                alpha_logits = (q_global * kk).sum(dim=-1)  # (B, H, T)
                alpha = tokens * torch.softmax(alpha_logits, dim=-1)
            else:
                q_global = q.mean(dim=2, keepdim=True)
                alpha_logits = (q_global @ kk.transpose(-2, -1)).squeeze(-2)
                alpha = tokens * torch.softmax(alpha_logits, dim=-1)
        else:
            alpha = torch.ones(batch, heads, tokens, device=q.device, dtype=q.dtype)

        weighted_k = kk * alpha.unsqueeze(-1)

        if self.is_causal:
            # Causal linear attention: use cumulative sums so token t
            # only aggregates information from tokens 1..t.
            # kv_t = cumsum of weighted_k^T @ v up to token t
            kv_outer = weighted_k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, T, d, d)
            kv = kv_outer.cumsum(dim=2)  # (B, H, T, d, d)
            # numerator_t = kq_t @ kv_t
            numerator = (kq.unsqueeze(-2) @ kv).squeeze(-2)  # (B, H, T, d)

            # denominator_t = kq_t . cumsum(weighted_k)
            k_cumsum = weighted_k.cumsum(dim=2)  # (B, H, T, d)
            denominator = (kq * k_cumsum).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        else:
            kv = weighted_k.transpose(-2, -1) @ v
            numerator = kq @ kv

            k_sum = weighted_k.sum(dim=-2)
            denominator = (kq * k_sum.unsqueeze(-2)).sum(dim=-1, keepdim=True).clamp_min(self.eps)

        y = numerator / denominator

        if self.attention_type == "rala" and self.use_output_gate:
            phi_x = self.phi(x, linear=self.phi_linear)
            phi_x = phi_x.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
            y = phi_x * y

        stats = AttentionStats()
        if collect_stats:
            if self.is_causal:
                # For diagnostics, use the final accumulated kv state
                kv_final = kv[:, :, -1, :, :] if kv.dim() == 5 else kv
            else:
                kv_final = kv
            kv_rank, kv_ratio = rank_ratio(kv_final.detach(), tol=rank_tol)
            output_rank, output_ratio = rank_ratio(y.detach(), tol=rank_tol)
            stats.kv_rank = kv_rank
            stats.kv_rank_ratio = kv_ratio
            stats.output_rank = output_rank
            stats.output_rank_ratio = output_ratio
            stats.alpha_sum_mean = alpha.sum(dim=-1).mean().detach().item()
            stats.min_denominator = denominator.detach().min().item()
            stats.warnings = stability_warnings(
                {
                    "kappa(Q)": kq,
                    "kappa(K)": kk,
                    "KV buffer": kv_final,
                    "attention_output": y,
                },
                min_denominator=stats.min_denominator,
            )
        return y, stats
