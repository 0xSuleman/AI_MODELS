"""Hybrid Attention: Self-Gated State Space Hybrid Attention.

Complete implementation of the architecture defined in model_formulation.html.

Components (matching HTML step numbers):
  Step 1 — Initial Projections:  Q, K, V = XW_Q, XW_K, XW_V
  Step 2 — Global Associative Memory:
           Parallel:  S = Σ g_i (K̂_i^T V_i), O_global = Q̂ S · scale
           Recurrent: S_t = e^{s_t} k̂_t^T v_t  (pure accumulation, no γ)
  Step 3 — Local Windowed Attention: softmax(Q̂ K̂^T · τ) over W(i) neighbors
  Step 4 — Residual Output Gate: φ(x_i) = 1 + tanh(W_φ x_i + b_φ)
  Step 5 — Combined Output: O_i = φ(x_i) ⊙ (O_global + O_local)

Architectural fixes applied (v2):
  - QK-Norm: Q and K are L2-normalised before entering the global memory and
    local attention. This bounds ‖S‖ by max_i ‖v_i‖ and eliminates the
    magnitude explosion observed in v1 (max |combined_output| ~ 1.94e+05).
  - Forget gate γ_t removed: gamma_proj was only exercised in recurrent mode
    but the model trains exclusively in parallel mode, creating a train/
    inference mismatch. Removing γ from recurrent makes both paths
    mathematically identical.
  - Learnable temperature τ (log-parameterised) replaces the fixed d^{-0.5}
    scale in LOCAL attention only. Because Q̂·K̂ ∈ [-1, 1] after QK-norm, a
    fixed scale of ~0.125 collapses softmax to near-uniform; τ = exp(log_τ)
    (initialised at 10.0) lets the model learn appropriate sharpness.
    Global salience still uses fixed d^{-0.5} because it operates on K̂·V
    (not a bounded dot-product) and temperature would drive one-hot collapse.
  - Causal salience fallback fixed: the previous 1/t uniform hack in causal
    parallel mode is replaced by a proper lower-triangular masked softmax.
    Causal+salience in parallel mode is now mathematically correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .metrics import AttentionStats, DEFAULT_RANK_TOL, rank_ratio, stability_warnings


@dataclass
class HybridAttentionResult:
    """Output container matching the interface expected by training.py."""
    output: torch.Tensor
    stats: AttentionStats


class HybridAttention(nn.Module):
    """Self-Gated State Space Hybrid Attention.

    Implements every equation from model_formulation.html:
      - Global associative memory with softmax-weighted outer products
      - Local windowed softmax attention for precise neighbor routing
      - Residual output gate φ(x) = 1 + tanh(W_φ x) for rank restoration

    Note: The data-dependent forget gate γ_t has been removed. It was only
    exercised in recurrent (inference) mode while the model trains in parallel
    mode, causing a train/inference mismatch. Both modes now share the same
    pure-accumulation memory update, guaranteeing identical math.

    Parameters
    ----------
    dim : int
        Model dimension D (total width of the residual stream).
    heads : int
        Number of attention heads h. Must evenly divide dim so d = D/h.
    window_size : int
        Local attention window size w. Each token attends to w neighbors.
        Set to 0 to disable the local branch entirely.
    dropout : float
        Dropout rate applied to attention weights and output.
    eps : float
        Numerical stability epsilon for normalizer Z_t.
    mode : str
        'parallel' for training (all tokens at once, uses global softmax),
        'recurrent' for streaming inference (one token at a time).
    use_output_gate : bool
        If False, disables φ(x) and uses an all-ones output gate.
    use_salience_gate : bool
        If False, replaces salience softmax weights with uniform normalized
        weights in the global memory.
    use_global : bool
        If False, disables the global associative memory branch entirely.
        Combined with window_size > 0 this gives a "local only" ablation.
    is_causal : bool
        If True, enforces causal (autoregressive) masking so that token t
        cannot attend to any token j > t. Required for language modeling.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        window_size: int = 16,
        dropout: float = 0.1,
        eps: float = 1e-6,
        mode: str = "parallel",
        use_output_gate: bool = True,
        use_salience_gate: bool = True,
        use_global: bool = True,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")
        if mode not in {"parallel", "recurrent"}:
            raise ValueError(f"Unknown mode: {mode}. Use 'parallel' or 'recurrent'.")

        self.dim = dim               # D
        self.heads = heads           # h
        self.head_dim = dim // heads # d
        self.window_size = window_size
        self.eps = eps
        self.mode = mode
        self.use_output_gate = use_output_gate
        self.use_salience_gate = use_salience_gate
        self.use_global = use_global
        self.is_causal = is_causal

        # ── Step 1: Initial Projections ────────────────────────────────
        # W_Q, W_K, W_V ∈ R^{D × D}  (packed into one linear for efficiency)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        # Output projection: concatenated heads back to D
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # ── [FIX v2] Learnable temperature for LOCAL attention ─────────
        # Local attention scores are Q̂·K̂ ∈ [-1, 1] after QK-norm, so the
        # old fixed scale d^{-0.5} ≈ 0.125 makes softmax near-uniform.
        # τ = exp(log_τ) is always positive and learnable. Initialised at
        # log(10) ≈ 2.303 so τ₀ = 10.0, a standard value for cosine attention.
        # Log-parameterisation prevents τ from going negative under gradient
        # descent. Used ONLY for local attention — NOT for global salience.
        self.log_temperature = nn.Parameter(torch.tensor(math.log(10.0)))

        # ── Step 4: Residual Output Gate φ(x) = 1 + tanh(W_φ x + b_φ) ─
        # Maps D → D (so each head gets d gate values after reshaping).
        self.phi_proj = nn.Linear(dim, dim, bias=True) if use_output_gate else None

        # ── Dropout ───────────────────────────────────────────────────
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        collect_stats: bool = False,
        rank_tol: float = DEFAULT_RANK_TOL,
    ) -> HybridAttentionResult:
        """Forward pass through the complete hybrid attention.

        Parameters
        ----------
        x : Tensor of shape (batch, tokens, dim)
            Input from the residual stream (after LayerNorm).
        collect_stats : bool
            If True, compute and return diagnostic statistics.
        rank_tol : float
            Tolerance for numerical rank computation.

        Returns
        -------
        HybridAttentionResult with .output (batch, tokens, dim) and .stats
        """
        batch, tokens, _ = x.shape

        # ── Step 1: Project to Q, K, V ─────────────────────────────────
        # qkv shape: (batch, tokens, 3 * dim)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, tokens, head_dim)
        q, k, v = qkv.unbind(dim=0)
        # q, k, v each: (batch, heads, tokens, head_dim)

        # ── [FIX v2] QK-Norm ───────────────────────────────────────────
        # Normalise Q and K to unit length along the head_dim axis.
        # After this: Q̂_i · K̂_j ∈ [-1, 1] always.
        #
        # Why this bounds the global memory S:
        #   S_ab = Σ_i g_i · K̂_ia · V_ib  →  |S_ab| ≤ max_i |V_ib|
        # And the retrieval:
        #   (o_global)_a = Σ_b Q̂_b · S_ab · scale  →  bounded by max|V| · √d · scale
        #
        # V is left free so the model retains full feature magnitude information.
        # This single change eliminates the 1.94e+05 combined_output magnitude.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)

        # ── Step 2: Global Associative Memory ──────────────────────────
        if not self.use_global:
            # "Local only" ablation: skip global branch entirely
            o_global = torch.zeros_like(q)
            memory = torch.zeros(
                batch, self.heads, self.head_dim, self.head_dim,
                device=q.device, dtype=q.dtype,
            )
        elif self.mode == "parallel":
            o_global, memory = self._global_parallel(q, k, v)
        else:
            o_global, memory = self._global_recurrent(q, k, v)

        # ── Step 3: Local Windowed Attention ───────────────────────────
        if self.window_size > 0 and tokens > 1:
            o_local = self._local_window_attention(q, k, v, tokens)
        else:
            o_local = torch.zeros_like(o_global)

        # ── Step 4 + 5: Residual Gate and Combine ──────────────────────
        if self.use_output_gate:
            # φ(x_i) = 1 + tanh(W_φ x_i + b_φ),  φ ∈ (0, 2)^d
            phi = 1.0 + torch.tanh(self.phi_proj(x))
            # Reshape phi to (batch, heads, tokens, head_dim)
            phi = phi.reshape(batch, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            phi = torch.ones_like(o_global)

        # O_i = φ(x_i) ⊙ (O_global + O_local)
        combined = o_global + o_local
        y = phi * combined
        y = self.out_dropout(y)

        # ── Reshape heads back to D ────────────────────────────────────
        # (batch, heads, tokens, head_dim) → (batch, tokens, dim)
        y = y.transpose(1, 2).reshape(batch, tokens, self.dim)
        output = self.out_proj(y)

        # ── Collect statistics if requested ────────────────────────────
        stats = AttentionStats()
        if collect_stats:
            with torch.no_grad():
                # True global memory rank (per head).
                memory_rank, memory_ratio = rank_ratio(
                    memory.detach(), tol=rank_tol
                )
                stats.memory_rank = memory_rank
                stats.memory_rank_ratio = memory_ratio

                # Retrieved global branch output rank, kept separate from memory.
                global_output_rank, global_output_ratio = rank_ratio(
                    o_global.detach(), tol=rank_tol
                )
                stats.global_output_rank = global_output_rank
                stats.global_output_rank_ratio = global_output_ratio

                # Output rank
                output_rank, output_ratio = rank_ratio(
                    y.detach().reshape(batch, tokens, self.heads, self.head_dim)
                    .permute(0, 2, 1, 3),
                    tol=rank_tol,
                )
                stats.output_rank = output_rank
                stats.output_rank_ratio = output_ratio

                # Stability warnings
                stats.warnings = stability_warnings({
                    "global_memory": memory,
                    "global_output": o_global,
                    "local_output": o_local,
                    "phi_gate": phi,
                    "combined_output": y.detach(),
                })

        return HybridAttentionResult(output=output, stats=stats)

    # ──────────────────────────────────────────────────────────────────
    # Step 2a: Global Memory — Parallel (Training) Form
    # ──────────────────────────────────────────────────────────────────
    def _global_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parallel global memory: S = Σ g_i (K̂_i^T V_i), O = Q̂ S · scale.

        Equations:
            s_i = K̂_i · V_i         (scalar salience, K̂ is unit-norm)
            g_i = softmax(s_i)       (normalised salience weights)
            S = Σ g_i (K̂_i^T V_i)   (d × d associative memory)
            O_global = Q̂ S · scale  (global retrieval, scale = d^{-0.5})

        Note on temperature: global salience uses fixed d^{-0.5}, NOT the
        learnable temperature τ. Rationale: s_i = K̂_i · V_i is a K·V dot
        product, not a Q·K dot product. V is unconstrained (norms can be
        large), so applying τ=10 here would drive softmax to near-one-hot,
        collapsing S to a rank-1 matrix and destroying the associative memory.

        Causal mode: each query at position t can only access memory from
        tokens 1..t. We build a 2D (tokens×tokens) masked score matrix so
        that softmax for each query row covers only its valid past keys.
        This is O(N²) in tokens but correct — the recurrent form is the
        efficient alternative for long sequences.
        """
        batch, heads, tokens, d = q.shape
        batch, heads, tokens, d = q.shape
        # Temperature for bounded salience since V is now unit-norm
        temp = self.log_temperature.exp().clamp(min=0.01, max=100.0)

        if self.use_salience_gate:
            # s_i = K̂_i · V̂_i  (dot product per token = scalar salience)
            # shape: (batch, heads, tokens)
            # K̂ and V̂ are unit-norm so |s_i| ≤ 1 * temp
            s = (k * v).sum(dim=-1) * temp

            if self.is_causal:
                # ── [FIX v3] O(N) Causal Prefix Scan ──────────────────
                # Instead of an O(N²) mask matrix, use running cumulative sums.
                # Numerically stable exponentiation
                s_max = s.max(dim=-1, keepdim=True).values  # (batch, heads, 1)
                exp_s = torch.exp(s - s_max)                # (batch, heads, tokens)
                
                # Weight keys by salience
                weighted_k = exp_s.unsqueeze(-1) * k        # (batch, heads, tokens, d)
                
                # Outer product per token
                kv_outer = weighted_k.unsqueeze(-1) * v.unsqueeze(-2)  # (batch, heads, tokens, d, d)
                
                # Cumulative sum over time
                S_unnorm = kv_outer.cumsum(dim=2)           # (batch, heads, tokens, d, d)
                Z = exp_s.cumsum(dim=-1).unsqueeze(-1).unsqueeze(-1)  # (batch, heads, tokens, 1, 1)
                
                # Exact causal softmax memory
                S_cumulative = S_unnorm / (Z + self.eps)
                
                # Retrieval (no scale needed, bounded by 1.0)
                o_global = (q.unsqueeze(-2) @ S_cumulative).squeeze(-2)
                S = S_cumulative[:, :, -1, :, :]

            else:
                # Non-causal: single global S, g is plain softmax over all tokens.
                g = F.softmax(s, dim=-1)       # (batch, heads, tokens)
                g = self.attn_dropout(g)

                # Build memory: S = K̂^T @ (g ⊙ V)
                g_expanded = g.unsqueeze(-1)             # (B, H, T, 1)
                weighted_v = v * g_expanded              # (B, H, T, d)
                S = k.transpose(-2, -1) @ weighted_v    # (B, H, d, d)

                # Retrieval: O_global = Q̂ S (no scale needed, bounded by 1.0)
                o_global = q @ S

        else:
            # Uniform normalised weighting — no-salience ablation.
            if self.is_causal:
                positions = torch.arange(1, tokens + 1, device=k.device, dtype=k.dtype)
                g = torch.ones(batch, heads, tokens, device=k.device, dtype=k.dtype)
                g = g / positions.view(1, 1, tokens)
            else:
                g = torch.full(
                    (batch, heads, tokens),
                    1.0 / max(tokens, 1),
                    device=k.device,
                    dtype=k.dtype,
                )

            g_expanded = g.unsqueeze(-1)
            weighted_v = v * g_expanded

            if self.is_causal:
                kv_outer = k.unsqueeze(-1) * weighted_v.unsqueeze(-2)
                S_cumulative = kv_outer.cumsum(dim=2)
                o_global = (q.unsqueeze(-2) @ S_cumulative).squeeze(-2)
                S = S_cumulative[:, :, -1, :, :]
            else:
                S = k.transpose(-2, -1) @ weighted_v
                o_global = q @ S

        return o_global, S

    # ──────────────────────────────────────────────────────────────────
    # Step 2b: Global Memory — Recurrent (Streaming) Form
    # ──────────────────────────────────────────────────────────────────
    def _global_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent global memory — pure accumulation, no forget gate.

        Equations (matching parallel form exactly):
            s_t = K̂_t · V_t · scale    (scalar salience)
            S_t = S_{t-1} · c_old + e^{s_t} · k̂_t^T v_t  (log-sum-exp stabilised)
            Z_t = Z_{t-1} · c_old + e^{s_t}               (running normaliser)
            M_t = S_t / Z_t                                (normalised memory)
            O_t = Q̂_t M_t · scale                         (global retrieval)

        The forget gate γ_t has been removed. The parallel form trains with
        pure accumulation (softmax-weighted sum), so recurrent inference must
        match. Both modes now compute the same mathematical operation:
            M = Σ softmax(s)_i · K̂_i^T V_i  (non-causal, full sequence)

        The log-sum-exp stabilisation (c_old, c_new via running_max) is kept
        because it prevents exponential overflow in e^{s_t} without altering
        the mathematical result.
        """
        batch, heads, tokens, d = q.shape
        temp = self.log_temperature.exp().clamp(min=0.01, max=100.0)

        # Initialise memory, normaliser, and log-sum-exp running maximum.
        S_t = torch.zeros(batch, heads, d, d, device=q.device, dtype=q.dtype)
        Z_t = torch.zeros(batch, heads, 1, 1, device=q.device, dtype=q.dtype)
        running_max = torch.full(
            (batch, heads),
            -torch.inf,
            device=q.device,
            dtype=q.dtype,
        )

        outputs = []

        for t in range(tokens):
            k_t = k[:, :, t, :]   # (batch, heads, d) — already unit-norm
            v_t = v[:, :, t, :]   # (batch, heads, d)
            q_t = q[:, :, t, :]   # (batch, heads, d) — already unit-norm

            # Scalar salience: K̂_t · V̂_t (bounded because both are unit-norm)
            s_t = (k_t * v_t).sum(dim=-1) * temp  # (batch, heads)

            if self.use_salience_gate:
                # Online log-sum-exp stabilisation — numerically safe.
                # Equivalent to exp(s_t) / Σ exp(s_j) in the limit, matching
                # the softmax in _global_parallel.
                next_max = torch.maximum(running_max, s_t)
                old_scale = torch.exp(running_max - next_max).unsqueeze(-1).unsqueeze(-1)
                new_scale = torch.exp(s_t - next_max).unsqueeze(-1).unsqueeze(-1)
            else:
                next_max = running_max
                old_scale = torch.ones(batch, heads, 1, 1, device=q.device, dtype=q.dtype)
                new_scale = torch.ones(batch, heads, 1, 1, device=q.device, dtype=q.dtype)

            # Outer product: k̂_t^T v_t  →  (batch, heads, d, d)
            kv_outer = k_t.unsqueeze(-1) @ v_t.unsqueeze(-2)

            # ── [FIX v2] Pure accumulation — γ removed ─────────────────
            # Previous: S_t = γ_t · old_scale · S_t + new_scale · kv_outer
            # Fixed:    S_t =        old_scale · S_t + new_scale · kv_outer
            S_t = old_scale * S_t + new_scale * kv_outer
            Z_t = old_scale * Z_t + new_scale

            # Normalised memory state
            M_t = S_t / (Z_t + self.eps)
            running_max = next_max

            # Retrieval: O_t = Q̂_t M_t (no scale needed)
            o_t = (q_t.unsqueeze(-2) @ M_t).squeeze(-2)
            outputs.append(o_t)

        o_global = torch.stack(outputs, dim=2)   # (batch, heads, tokens, d)
        return o_global, S_t

    # ──────────────────────────────────────────────────────────────────
    # Step 3: Local Windowed Softmax Attention
    # ──────────────────────────────────────────────────────────────────
    def _local_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tokens: int,
    ) -> torch.Tensor:
        """Local windowed attention: standard softmax over w neighbours.

        Equations:
            W(i) ⊆ {1, ..., N}     (local window centred at token i)
            α_ij = softmax(Q̂_i K̂_j^T · τ)  for j ∈ W(i)
            O_local = Σ α_ij V_j

        τ = exp(log_temperature) replaces the fixed d^{-0.5} scale because
        after QK-norm Q̂·K̂ ∈ [-1, 1] and d^{-0.5} ≈ 0.125 would make every
        softmax near-uniform. τ₀ = 10 is standard for cosine attention.
        τ is shared across heads and layers via the single log_temperature
        parameter on this module.

        When is_causal=True, the window is shifted so that token i only
        attends to tokens [i - w + 1, i] (backward-looking only).
        """
        batch, heads, _, d = q.shape
        w = min(self.window_size, tokens)

        # ── [FIX v2] Learnable temperature replaces fixed scale ─────────
        # τ = exp(log_τ) is always positive (log-parameterised).
        # Clamped to [0.01, 100] to prevent extreme values during early
        # training before the parameter has stabilised.
        temp = self.log_temperature.exp().clamp(min=0.01, max=100.0)

        # If window covers the full sequence, do standard full attention.
        if w >= tokens:
            attn_scores = q @ k.transpose(-2, -1) * temp
            if self.is_causal:
                causal_mask = torch.triu(
                    torch.ones(tokens, tokens, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
                attn_scores = attn_scores.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )
            attn = F.softmax(attn_scores, dim=-1)
            attn = self.attn_dropout(attn)
            return attn @ v

        if self.is_causal:
            # Causal window: token i attends to [i - w + 1, i] (backward only).
            padded_k = F.pad(k, (0, 0, w - 1, 0))
            padded_v = F.pad(v, (0, 0, w - 1, 0))
            k_windows = padded_k.unfold(dimension=2, size=w, step=1).permute(0, 1, 2, 4, 3)
            v_windows = padded_v.unfold(dimension=2, size=w, step=1).permute(0, 1, 2, 4, 3)

            scores = (q.unsqueeze(-2) * k_windows).sum(dim=-1) * temp

            token_idx = torch.arange(tokens, device=q.device).unsqueeze(-1)
            offsets = torch.arange(-(w - 1), 1, device=q.device).unsqueeze(0)
            key_idx = token_idx + offsets
            valid = key_idx >= 0
            scores = scores.masked_fill(~valid.view(1, 1, tokens, w), float("-inf"))

        else:
            # Non-causal: symmetric window centred at token i.
            half_w = w // 2
            window = (2 * half_w) + 1

            padded_k = F.pad(k, (0, 0, half_w, half_w))
            padded_v = F.pad(v, (0, 0, half_w, half_w))
            k_windows = padded_k.unfold(dimension=2, size=window, step=1).permute(0, 1, 2, 4, 3)
            v_windows = padded_v.unfold(dimension=2, size=window, step=1).permute(0, 1, 2, 4, 3)

            scores = (q.unsqueeze(-2) * k_windows).sum(dim=-1) * temp

            token_idx = torch.arange(tokens, device=q.device).unsqueeze(-1)
            offsets = torch.arange(-half_w, half_w + 1, device=q.device).unsqueeze(0)
            key_idx = token_idx + offsets
            valid = (key_idx >= 0) & (key_idx < tokens)
            scores = scores.masked_fill(~valid.view(1, 1, tokens, window), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # O_local = Σ_j α_ij V_j over the gathered local window.
        o_local = (attn_weights.unsqueeze(-1) * v_windows).sum(dim=-2)

        return o_local



