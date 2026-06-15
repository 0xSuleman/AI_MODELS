"""Safe formula parsing for user-provided kappa and phi functions.

The evaluator intentionally accepts only a tiny expression language:
numbers, the variable ``x``, arithmetic, and a fixed list of tensor ops.
It does not execute arbitrary Python.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


class FormulaError(ValueError):
    """Raised when a formula is invalid or unsafe."""


def _normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _clamp(
    x: torch.Tensor,
    min_value: float = -10.0,
    max_value: float = 10.0,
) -> torch.Tensor:
    return torch.clamp(x, min=min_value, max=max_value)


ALLOWED_FUNCTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "elu": F.elu,
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "softplus": F.softplus,
    "exp": torch.exp,
    "log": torch.log,
    "abs": torch.abs,
    "square": torch.square,
    "sqrt": torch.sqrt,
    "normalize": _normalize,
    "softmax": lambda x: torch.softmax(x, dim=-1),
    "clamp": _clamp,
}

PRESET_KAPPA = {
    "Paper: elu(x) + 1": "elu(x) + 1",
    "ReLU positive: relu(x) + 1e-6": "relu(x) + 1e-6",
    "Softplus: softplus(x)": "softplus(x)",
    "Swish gate: sigmoid(x) * x": "sigmoid(x) * x",
}

PRESET_PHI = {
    "Paper: linear(x)": "linear(x)",
    "Identity: x": "x",
    "Tanh: tanh(x)": "tanh(x)",
    "SiLU: silu(x)": "silu(x)",
}

ALLOWED_CALLS = set(ALLOWED_FUNCTIONS) | {"linear", "identity"}


class CompiledFormula:
    """A validated formula ready to evaluate on tensors."""

    __slots__ = ("expression", "uses_linear", "_ast_body")

    def __init__(self, expression: str, uses_linear: bool = False) -> None:
        self.expression = expression
        self.uses_linear = uses_linear
        self._ast_body = ast.parse(expression, mode="eval").body

    def __call__(
        self,
        x: torch.Tensor,
        linear: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        return _eval_node(self._ast_body, x=x, linear=linear)


def compile_formula(expression: str) -> CompiledFormula:
    """Validate a formula and return a callable wrapper."""

    expression = expression.strip()
    if not expression:
        raise FormulaError("Formula cannot be empty.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Invalid syntax: {exc.msg}") from exc

    validator = _FormulaValidator()
    validator.visit(tree)
    return CompiledFormula(expression=expression, uses_linear=validator.uses_linear)


class _FormulaValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.uses_linear = False

    def generic_visit(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
        )
        if not isinstance(node, allowed):
            raise FormulaError(f"Unsupported expression element: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id != "x":
            raise FormulaError(f"Unknown variable or symbol: {node.id}")

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise FormulaError("Only direct calls like relu(x) are allowed.")
        name = node.func.id
        if name not in ALLOWED_CALLS:
            raise FormulaError(f"Function is not allowed: {name}")
        if name == "linear":
            self.uses_linear = True
        if len(node.args) != 1:
            raise FormulaError(f"{name}(...) expects exactly one positional argument.")
        if node.keywords:
            raise FormulaError("Keyword arguments are not allowed in formulas.")
        self.visit(node.args[0])

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)):
            raise FormulaError("Only numeric constants are allowed.")


def _eval_node(
    node: ast.AST,
    *,
    x: torch.Tensor,
    linear: torch.nn.Module | None,
) -> torch.Tensor:
    if isinstance(node, ast.Constant):
        return torch.as_tensor(node.value, dtype=x.dtype, device=x.device)
    if isinstance(node, ast.Name):
        if node.id != "x":
            raise FormulaError(f"Unknown variable: {node.id}")
        return x
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, x=x, linear=linear)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, x=x, linear=linear)
        right = _eval_node(node.right, x=x, linear=linear)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right.clamp_min(1e-12) if torch.is_tensor(right) else left / right
        if isinstance(node.op, ast.Pow):
            return left.pow(right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arg = _eval_node(node.args[0], x=x, linear=linear)
        name = node.func.id
        if name == "identity":
            return arg
        if name == "linear":
            if linear is None:
                raise FormulaError("Formula uses linear(x), but no linear module was provided.")
            return linear(arg)
        return ALLOWED_FUNCTIONS[name](arg)
    raise FormulaError(f"Unsupported expression element: {type(node).__name__}")
