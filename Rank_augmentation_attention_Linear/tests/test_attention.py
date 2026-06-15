import torch
import pytest

from rala_lab.attention import FormulaAttention
from rala_lab.formulas import compile_formula
from rala_lab.hybrid_attention import HybridAttention
from rala_lab.metrics import AttentionStats


def test_rala_shapes_and_alpha_sum():
    x = torch.randn(2, 16, 32)
    attn = FormulaAttention(
        dim=32,
        heads=4,
        attention_type="rala",
        kappa=compile_formula("elu(x)+1"),
        phi=compile_formula("linear(x)"),
        use_alpha=True,
        use_output_gate=True,
    )
    result = attn(x, collect_stats=True)
    assert result.output.shape == x.shape
    assert result.stats.alpha_sum_mean == pytest.approx(16.0, rel=1e-4)
    assert result.stats.kv_rank is not None
    assert result.stats.output_rank is not None
    assert not result.stats.warnings


def test_softmax_shape():
    x = torch.randn(2, 16, 32)
    attn = FormulaAttention(
        dim=32,
        heads=4,
        attention_type="softmax",
        kappa=compile_formula("elu(x)+1"),
        phi=compile_formula("x"),
    )
    result = attn(x, collect_stats=True)
    assert result.output.shape == x.shape


def test_attention_stats_kv_alias_constructor_compatibility():
    stats = AttentionStats(kv_rank=3.0, kv_rank_ratio=0.75)

    assert stats.memory_rank == 3.0
    assert stats.memory_rank_ratio == 0.75
    assert stats.kv_rank == 3.0
    assert stats.kv_rank_ratio == 0.75

    stats.kv_rank = 4.0
    stats.kv_rank_ratio = 1.0
    assert stats.memory_rank == 4.0
    assert stats.memory_rank_ratio == 1.0


def test_rank_tol_affects_rank():
    """Looser tolerance should report equal or higher rank."""
    x = torch.randn(2, 16, 32)
    attn = FormulaAttention(
        dim=32,
        heads=4,
        attention_type="rala",
        kappa=compile_formula("elu(x)+1"),
        phi=compile_formula("linear(x)"),
        use_alpha=True,
        use_output_gate=True,
    )
    loose = attn(x, collect_stats=True, rank_tol=1e-2)
    tight = attn(x, collect_stats=True, rank_tol=1e-8)
    # looser tolerance → fewer singular values counted → lower or equal rank
    assert tight.stats.output_rank >= loose.stats.output_rank


def test_per_layer_stats_count():
    """Model should return one stats object per layer."""
    from rala_lab.models import TinyFormulaViT

    model = TinyFormulaViT(
        input_shape=(1, 16, 16),
        num_classes=4,
        dim=32,
        heads=4,
        layers=3,
        patch_size=4,
        attention_type="rala",
        kappa=compile_formula("elu(x)+1"),
        phi=compile_formula("linear(x)"),
        use_alpha=True,
        use_output_gate=True,
    )
    x = torch.randn(2, 1, 16, 16)
    _, stats_list = model(x, collect_stats=True)
    assert len(stats_list) == 3
    for s in stats_list:
        assert s.kv_rank is not None
        assert s.output_rank is not None


def test_hybrid_memory_rank_uses_true_parallel_memory():
    x = torch.randn(2, 16, 32)
    attn = HybridAttention(dim=32, heads=4, window_size=4, dropout=0.0, mode="parallel")

    result = attn(x, collect_stats=True)

    assert result.output.shape == x.shape
    assert result.stats.memory_rank is not None
    assert result.stats.memory_rank_ratio is not None
    assert result.stats.kv_rank == result.stats.memory_rank
    assert result.stats.kv_rank_ratio == result.stats.memory_rank_ratio
    assert result.stats.global_output_rank is not None
    assert result.stats.global_output_rank_ratio is not None
    assert result.stats.output_rank is not None
    assert result.stats.output_rank_ratio is not None
    assert not result.stats.warnings


def test_hybrid_output_gate_can_be_disabled():
    x = torch.randn(2, 12, 32)
    gated = HybridAttention(dim=32, heads=4, window_size=4, dropout=0.0, use_output_gate=True)
    ungated = HybridAttention(dim=32, heads=4, window_size=4, dropout=0.0, use_output_gate=False)

    gated_result = gated(x, collect_stats=True)
    ungated_result = ungated(x, collect_stats=True)

    assert gated_result.output.shape == x.shape
    assert ungated_result.output.shape == x.shape
    assert gated.phi_proj is not None
    assert ungated.phi_proj is None
    assert ungated_result.stats.memory_rank is not None


def test_hybrid_salience_gate_can_use_uniform_memory_weights():
    x = torch.randn(2, 10, 32)
    attn = HybridAttention(
        dim=32,
        heads=4,
        window_size=4,
        dropout=0.0,
        use_salience_gate=False,
    )

    result = attn(x, collect_stats=True)

    assert result.output.shape == x.shape
    assert result.stats.memory_rank is not None
    assert result.stats.memory_rank_ratio is not None
    assert not result.stats.warnings


def test_hybrid_recurrent_reports_final_memory_rank():
    x = torch.randn(2, 8, 32)
    attn = HybridAttention(dim=32, heads=4, window_size=4, dropout=0.0, mode="recurrent")

    result = attn(x, collect_stats=True)

    assert result.output.shape == x.shape
    assert result.stats.memory_rank is not None
    assert result.stats.memory_rank_ratio is not None
    assert result.stats.global_output_rank is not None
    assert result.stats.output_rank is not None
