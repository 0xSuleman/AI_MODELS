import pytest
import torch

from rala_lab.formulas import FormulaError, compile_formula


@pytest.mark.parametrize("expr", ["elu(x)+1", "relu(x)+1e-6", "sigmoid(x)*x", "softplus(x)"])
def test_valid_formulas(expr):
    formula = compile_formula(expr)
    x = torch.randn(2, 3)
    y = formula(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "x.__class__",
        "lambda x: x",
        "[x for x in x]",
        "unknown(x)",
        "relu(x, dim=-1)",
    ],
)
def test_rejects_unsafe_formulas(expr):
    with pytest.raises(FormulaError):
        compile_formula(expr)
