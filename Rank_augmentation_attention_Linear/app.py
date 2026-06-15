from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rala_lab.formulas import FormulaError, PRESET_KAPPA, PRESET_PHI, compile_formula
from rala_lab.metrics import DEFAULT_RANK_TOL, METRIC_HELP
from rala_lab.training import (
    ExperimentConfig, ExperimentResult, run_experiment,
    save_checkpoint, load_checkpoint, list_checkpoints,
    evaluate_checkpoint, continue_training, delete_checkpoint,
)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RALA Formula Labs",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

/* ── Global resets ── */
html, body, [class*="css"]            { font-family: 'Instrument Sans', sans-serif; }
#MainMenu, footer, .stDeployButton    { visibility: hidden; display: none; }

/* ── Page background ── */
.stApp                                { background: #f4f5f7; }
.main .block-container                { padding: 2rem 2.5rem 3rem; max-width: 1280px; }

/* ── Sidebar ── */
[data-testid="stSidebar"]             { background: #111318 !important; border-right: 1px solid #1e2027; }
[data-testid="stSidebar"] *           { color: #c8ccd4 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3          { color: #e6e8ed !important; font-family: 'Syne', sans-serif !important; }

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: #1a1d24 !important;
    border: 1px solid #2a2d36 !important;
    color: #e6e8ed !important;
    border-radius: 5px !important;
    font-family: 'Instrument Sans', sans-serif !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stWidgetLabel > label,
[data-testid="stSidebar"] .stWidgetLabel p {
    color: #6e7380 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    font-family: 'Instrument Sans', sans-serif !important;
}
[data-testid="stSidebar"] .stRadio > label { color: #6e7380 !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #c8ccd4 !important;
    font-size: 0.875rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: #2563eb !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div > div { background: #ffffff !important; border: 2px solid #2563eb !important; }
[data-testid="stSidebar"] .stCheckbox label span { color: #c8ccd4 !important; font-size: 0.875rem !important; text-transform: none !important; letter-spacing: 0 !important; font-weight: 400 !important; }
[data-testid="stSidebar"] .stExpander { border-color: #2a2d36 !important; background: #1a1d24 !important; border-radius: 6px !important; }
[data-testid="stSidebar"] hr { border-color: #1e2027 !important; margin: 0.75rem 0 !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 2.25rem !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]    { border-bottom: 1px solid #e2e5ea !important; gap: 0 !important; background: transparent !important; }
.stTabs [data-baseweb="tab"]         { font-family: 'Instrument Sans', sans-serif !important; font-weight: 500 !important; font-size: 0.85rem !important; color: #6b7280 !important; padding: 0.65rem 1.5rem !important; border-radius: 0 !important; background: transparent !important; }
.stTabs [aria-selected="true"]       { color: #2563eb !important; border-bottom: 2px solid #2563eb !important; font-weight: 600 !important; }
.stTabs [data-baseweb="tab-panel"]   { padding-top: 1.5rem !important; }

/* ── Metric override — hide default streamlit metric ── */
[data-testid="stMetric"]             { display: none !important; }

/* ── Dataframes ── */
.stDataFrame                         { border-radius: 8px !important; overflow: hidden !important; }
.stDataFrame > div                   { border: 1px solid #e2e5ea !important; border-radius: 8px !important; }

/* ── Info / warning / error ── */
[data-testid="stAlert"]              { border-radius: 6px !important; font-size: 0.875rem !important; }

/* ── Spinner ── */
.stSpinner > div                     { border-top-color: #2563eb !important; }
</style>
""", unsafe_allow_html=True)


# ── Design helpers ────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    """Sidebar section separator with label."""
    st.markdown(
        f'<p style="font-size:0.62rem;font-weight:700;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:#4a4f5c;padding:0.9rem 0 0.3rem 0;'
        f'border-top:1px solid #1e2027;margin:0;">{title}</p>',
        unsafe_allow_html=True,
    )


def _badge(text: str, variant: str = "blue") -> str:
    palettes = {
        "blue":   ("#dbeafe", "#1d4ed8"),
        "green":  ("#dcfce7", "#15803d"),
        "amber":  ("#fef3c7", "#b45309"),
        "slate":  ("#f1f5f9", "#475569"),
        "red":    ("#fee2e2", "#b91c1c"),
    }
    bg, fg = palettes.get(variant, palettes["blue"])
    return (
        f'<span style="display:inline-block;padding:0.18em 0.6em;border-radius:20px;'
        f'font-size:0.72rem;font-weight:700;letter-spacing:0.04em;'
        f'background:{bg};color:{fg};">{text}</span>'
    )


def _metric_card(label: str, value: str, sub: str = "", accent: str = "#2563eb") -> str:
    return f"""
<div style="background:#ffffff;border:1px solid #e2e5ea;border-radius:10px;
    padding:1.1rem 1.25rem;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{accent};border-radius:10px 10px 0 0;"></div>
  <div style="font-size:0.67rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
      color:#6b7280;margin-bottom:0.45rem;font-family:'Instrument Sans',sans-serif;">{label}</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.55rem;font-weight:500;
      color:#111318;line-height:1;">{value}</div>
  <div style="font-size:0.75rem;color:#9ca3af;margin-top:0.35rem;
      font-family:'Instrument Sans',sans-serif;">{sub}</div>
</div>"""


# ── Backend helpers (logic unchanged) ────────────────────────────────────────

def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _formula_input(label: str, presets: dict[str, str], key: str) -> str:
    preset_name = st.sidebar.selectbox(f"{label} preset", list(presets), key=f"{key}_preset")
    default = presets[preset_name]
    return st.sidebar.text_input(label, value=default, key=key)


def _validate_formula(name: str, expression: str) -> None:
    try:
        compiled = compile_formula(expression)
    except FormulaError as exc:
        st.markdown(
            f'<div style="background:#fee2e2;border:1px solid #fca5a5;border-radius:6px;'
            f'padding:0.55rem 0.875rem;font-size:0.83rem;color:#7f1d1d;font-family:'
            f'\'Instrument Sans\',sans-serif;margin:0.2rem 0;">'
            f'<strong>{name}</strong> — {exc}</div>',
            unsafe_allow_html=True,
        )
        st.stop()
    linear_note = " Uses <code>linear(x)</code>." if compiled.uses_linear else ""
    st.markdown(
        f'<div style="background:#dcfce7;border:1px solid #86efac;border-radius:6px;'
        f'padding:0.55rem 0.875rem;font-size:0.83rem;color:#14532d;font-family:'
        f'\'Instrument Sans\',sans-serif;margin:0.2rem 0;">'
        f'<strong>{name}</strong> is valid.{linear_note}</div>',
        unsafe_allow_html=True,
    )


def _per_layer_table(result: ExperimentResult) -> pd.DataFrame:
    rows = []
    for s in result.per_layer_stats:
        rows.append({
            "Layer": s.layer,
            "Memory Rank": _format_optional(s.memory_rank),
            "Memory Rank Ratio": _format_optional(s.memory_rank_ratio),
            "Global Output Rank": _format_optional(s.global_output_rank),
            "Global Output Rank Ratio": _format_optional(s.global_output_rank_ratio),
            "Output Rank": _format_optional(s.output_rank),
            "Output Rank Ratio": _format_optional(s.output_rank_ratio),
            "α Sum Mean": _format_optional(s.alpha_sum_mean),
            "Min Denominator": _format_optional(s.min_denominator),
        })
    return pd.DataFrame(rows)


def _summary_table(result: ExperimentResult) -> pd.DataFrame:
    stats = result.final_stats
    rows = [
        ("Bottleneck layer",     stats.layer),
        ("Memory rank",          stats.memory_rank),
        ("Memory rank ratio",    stats.memory_rank_ratio),
        ("Global output rank",   stats.global_output_rank),
        ("Global output ratio",  stats.global_output_rank_ratio),
        ("Output rank",          stats.output_rank),
        ("Output rank ratio",    stats.output_rank_ratio),
        ("Alpha sum mean",       stats.alpha_sum_mean),
        ("Min denominator",      stats.min_denominator),
        ("Inference time (ms)",  result.inference_ms),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"]).astype(str)


# ── Parameter estimator ───────────────────────────────────────────────────────

def _estimate_params(task: str, vocab_size: int, dim: int, heads: int, layers: int, patch_size: int, mlp_ratio: int) -> int:
    """Rough parameter count estimate."""
    if task == "image":
        embed = patch_size * patch_size * 3 * dim + dim
        head  = dim * 10
    else:
        embed = vocab_size * dim + 128 * dim  # embedding + pos_embed approx
        head  = dim * vocab_size

    attn         = 4 * dim * dim + 4 * dim          # Q K V O projections + biases
    mlp_hidden   = dim * mlp_ratio
    mlp          = dim * mlp_hidden + mlp_hidden + mlp_hidden * dim + dim
    ln           = 4 * dim                           # two LayerNorms × 2 params each
    return embed + layers * (attn + mlp + ln) + head


def _format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


# ── Plotly chart builders ─────────────────────────────────────────────────────

_PLOTLY_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=50, r=20, t=48, b=44),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#111318", font_color="#e6e8ed", font_family="IBM Plex Mono", font_size=12, bordercolor="#111318"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Instrument Sans", size=13, color="#111318")),
    font=dict(family="Instrument Sans", color="#111318"),
)

_AXIS_STYLE = dict(
    gridcolor="#e2e5ea", 
    gridwidth=1, 
    linecolor="#cbd5e1", 
    title_font=dict(family="Instrument Sans", size=14, color="#111318"),
    tickfont=dict(family="IBM Plex Mono", size=11, color="#4a4f5c")
)


def _plotly_loss(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["epoch"], y=history["train_loss"],
        mode="lines", name="Train",
        line=dict(color="#2563eb", width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.04)",
    ))
    fig.add_trace(go.Scatter(
        x=history["epoch"], y=history["val_loss"],
        mode="lines", name="Validation",
        line=dict(color="#7c3aed", width=2, dash="dot"),
    ))
    fig.update_layout(
        title=dict(text="Loss Curve", font=dict(family="Syne", size=13, color="#111318")),
        xaxis=dict(title="Epoch", **_AXIS_STYLE),
        yaxis=dict(title="Loss",  **_AXIS_STYLE),
        height=310,
        **_PLOTLY_BASE,
    )
    return fig


def _plotly_accuracy(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["epoch"], y=history["train_acc"],
        mode="lines", name="Train",
        line=dict(color="#16a34a", width=2.5),
        fill="tozeroy", fillcolor="rgba(22,163,74,0.04)",
    ))
    fig.add_trace(go.Scatter(
        x=history["epoch"], y=history["val_acc"],
        mode="lines", name="Validation",
        line=dict(color="#0ea5e9", width=2, dash="dot"),
    ))
    fig.update_layout(
        title=dict(text="Accuracy Curve", font=dict(family="Syne", size=13, color="#111318")),
        xaxis=dict(title="Epoch",    **_AXIS_STYLE),
        yaxis=dict(title="Accuracy", **_AXIS_STYLE, range=[0, 1]),
        height=310,
        **_PLOTLY_BASE,
    )
    return fig


def _plotly_rank_bars(df: pd.DataFrame) -> go.Figure:
    def _safe(col):
        return [float(v) if v != "n/a" else 0.0 for v in df[col]]

    memory_vals = _safe("Memory Rank Ratio")
    global_vals = _safe("Global Output Rank Ratio")
    out_vals = _safe("Output Rank Ratio")
    layers   = df["Layer"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Memory Rank Ratio", x=layers, y=memory_vals,
        marker=dict(color="#2563eb", opacity=0.82, line=dict(width=0)),
        hovertemplate="%{y:.3f}<extra>Memory</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Global Output Rank Ratio", x=layers, y=global_vals,
        marker=dict(color="#0ea5e9", opacity=0.74, line=dict(width=0)),
        hovertemplate="%{y:.3f}<extra>Global Output</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Output Rank Ratio", x=layers, y=out_vals,
        marker=dict(color="#7c3aed", opacity=0.82, line=dict(width=0)),
        hovertemplate="%{y:.3f}<extra>Output</extra>",
    ))
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="#dc2626", line_width=1.5,
        annotation_text="Full rank (1.0)",
        annotation_font=dict(family="Instrument Sans", size=11, color="#dc2626"),
        annotation_position="top right",
    )
    y_max = max((max(memory_vals + global_vals + out_vals, default=0) * 1.15), 1.15)
    fig.update_layout(
        title=dict(text="SVD Rank Preservation — Per Layer", font=dict(family="Syne", size=13, color="#111318")),
        xaxis=dict(title="Layer",      **_AXIS_STYLE),
        yaxis=dict(title="Rank Ratio", **_AXIS_STYLE, range=[0, y_max]),
        barmode="group",
        bargap=0.3,
        bargroupgap=0.08,
        height=340,
        **_PLOTLY_BASE,
    )
    return fig


def _plotly_comparison_bar(summary_rows: list[dict]) -> go.Figure:
    """Side-by-side bar comparing attention types on key metrics."""
    names   = [r["Attention"] for r in summary_rows]
    val_acc = []
    for r in summary_rows:
        try:    val_acc.append(float(r["Val Accuracy"]))
        except: val_acc.append(0.0)

    colors = ["#2563eb", "#7c3aed", "#16a34a", "#f59e0b", "#ef4444"]
    fig = go.Figure(go.Bar(
        x=names, y=val_acc,
        marker_color=colors[:len(names)],
        marker_opacity=0.88,
        text=[f"{v:.3f}" for v in val_acc],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11),
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Validation Accuracy — Head-to-Head", font=dict(family="Syne", size=13, color="#111318")),
        xaxis=dict(title="Attention Type", **_AXIS_STYLE),
        yaxis=dict(title="Val Accuracy",   **_AXIS_STYLE, range=[0, 1.12]),
        height=310,
        **_PLOTLY_BASE,
    )
    return fig


# ── Result renderer ───────────────────────────────────────────────────────────

def _display_result(result: ExperimentResult, label: str = "") -> None:
    """Render metrics, charts, and diagnostics for one experiment result."""
    history = pd.DataFrame([asdict(row) for row in result.history])
    final   = history.iloc[-1]
    stats   = result.final_stats

    # ── Metric cards ──────────────────────────────────────────────────────
    if len(history) > 1:
        prev      = history.iloc[-2]
        acc_delta = f"{'▲' if final.val_acc  > prev.val_acc  else '▼'} {abs(final.val_acc  - prev.val_acc ):.4f} vs prev epoch"
        lss_delta = f"{'▼' if final.val_loss < prev.val_loss else '▲'} {abs(final.val_loss - prev.val_loss):.4f} vs prev epoch"
    else:
        acc_delta, lss_delta = "Final epoch", "Final epoch"

    has_warnings = bool(stats.warnings)
    warn_text    = f"{len(stats.warnings)} warning(s)" if has_warnings else "All clear"
    warn_accent  = "#ca8a04" if has_warnings else "#16a34a"

    task_id = result.config.get("task", "image")
    is_lm = task_id == "shakespeare"

    if is_lm:
        ppl = np.exp(final.val_loss)
        if len(history) > 1:
            prev_ppl = np.exp(history.iloc[-2].val_loss)
            ppl_delta = f"{'▼' if ppl < prev_ppl else '▲'} {abs(ppl - prev_ppl):.2f} vs prev epoch"
        else:
            ppl_delta = "Final epoch"
        metric_1 = _metric_card("Perplexity", f"{ppl:.2f}", ppl_delta, "#16a34a")
        metric_2 = _metric_card("Next-Char Acc", f"{final.val_acc:.4f}", acc_delta, "#2563eb")
    else:
        metric_1 = _metric_card("Validation Accuracy",  f"{final.val_acc:.4f}", acc_delta, "#16a34a")
        metric_2 = _metric_card("Validation Loss",       f"{final.val_loss:.4f}", lss_delta, "#2563eb")

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:0.25rem 0 1.5rem;">'
        + metric_1
        + metric_2
        + _metric_card("Output Rank Ratio",     _format_optional(stats.output_rank_ratio),           "Worst-case layer",    "#7c3aed")
        + _metric_card("Inference Time",        f"{result.inference_ms:.1f} ms",                     f"{result.config.get('warmup_passes', 0)} warmup passes", warn_accent)
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_names = ["Training Curves", "Rank Diagnostics", "Diagnostics"]
    if len(history) <= 1:
        tab_names[0] = "Training Curves — N/A"

    t_curves, t_rank, t_diag = st.tabs(tab_names)

    with t_curves:
        if len(history) > 1:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(_plotly_loss(history), use_container_width=True, config={"displayModeBar": False})
            with c2:
                st.plotly_chart(_plotly_accuracy(history), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Single-epoch or evaluate-only run — no training curve to display.")

    with t_rank:
        st.markdown(
            f'<p style="font-size:0.72rem;color:#6b7280;margin-bottom:0.75rem;">'
            f'Rank tolerance <code style="font-family:IBM Plex Mono;">'
            f'{result.config.get("rank_tol", DEFAULT_RANK_TOL):.1e}</code> · '
            f'averaged over <strong>{result.config.get("stats_batches", 1)}</strong> validation batch(es)</p>',
            unsafe_allow_html=True,
        )
        if result.per_layer_stats:
            df = _per_layer_table(result)
            c_chart, c_table = st.columns([1.3, 1])
            with c_chart:
                st.plotly_chart(_plotly_rank_bars(df), use_container_width=True, config={"displayModeBar": False})
            with c_table:
                st.markdown("**Per-layer detail**")
                st.dataframe(df, width="stretch", height=265)
                st.download_button(
                    label="Download Layer Details (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"layer_metrics_{label or 'main'}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_layer_{label}",
                )
        else:
            st.info("No per-layer statistics were collected for this run.")

        st.markdown("**Bottleneck summary — worst-case layer**")
        summary_df = _summary_table(result)
        st.dataframe(summary_df, width="stretch")
        st.download_button(
            label="Download Summary (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"bottleneck_summary_{label or 'main'}.csv",
            mime="text/csv",
            key=f"dl_sum_{label}",
        )

    with t_diag:
        warnings = stats.warnings
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.markdown(
                '<div style="background:#dcfce7;border:1px solid #86efac;border-radius:6px;'
                'padding:0.65rem 1rem;font-size:0.875rem;color:#14532d;">'
                'No NaN, Inf, large-activation, or near-zero-denominator warnings detected.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("")
        st.markdown("**Full configuration used**")
        cfg_df = pd.DataFrame(list(result.config.items()), columns=["Parameter", "Value"]).astype(str)
        st.dataframe(cfg_df, width="stretch", height=320)


# ── Checkpoint info helpers ───────────────────────────────────────────────────

def _checkpoint_info_card(ckpt: dict) -> str:
    cfg = ckpt["config"]
    task = cfg.get("task", "image")
    if task == "image":
        ds_text = f"Trained on {cfg.get('dataset')} ({cfg.get('sample_limit')} samples"
    else:
        ds_text = f"Task: {task}"
    return (
        f"**{cfg.get('attention_type', '?').upper()}** · "
        f"D={cfg.get('dim')} · h={cfg.get('heads')} · L={cfg.get('layers')} · "
        f"patch={cfg.get('patch_size', '?')} · mlp×{cfg.get('mlp_ratio', '?')} · "
        f"{ds_text}, {cfg.get('epochs')} epochs) · "
        f"{ckpt['size_mb']:.1f} MB"
    )


def _render_checkpoint_card(ckpt: dict) -> None:
    """Render a structured metadata card for a checkpoint."""
    cfg = ckpt["config"]
    hd  = cfg.get("dim", 0) // cfg.get("heads", 1) if cfg.get("dim", 0) % max(cfg.get("heads", 1), 1) == 0 else "—"

    fields = [
        ("Attention type",  cfg.get("attention_type", "—").upper()),
        ("Architecture",    f"D={cfg.get('dim')} · h={cfg.get('heads')} · L={cfg.get('layers')} · d={hd}"),
        ("Patch / MLP",     f"patch={cfg.get('patch_size')} · mlp×{cfg.get('mlp_ratio', '—')}"),
        ("Dataset",         f"{cfg.get('dataset')}  ({cfg.get('sample_limit', '—'):,} samples)"),
        ("Trained",         f"{cfg.get('epochs')} epoch(s)  ·  LR {float(cfg.get('learning_rate', 0)):.1e}"),
        ("File size",       f"{ckpt['size_mb']:.2f} MB"),
    ]
    rows_html = "".join(
        f'<tr>'
        f'<td style="padding:0.28rem 0.6rem 0.28rem 0;color:#6e7380;font-size:0.7rem;'
        f'font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'
        f'white-space:nowrap;font-family:\'Instrument Sans\',sans-serif;">{k}</td>'
        f'<td style="padding:0.28rem 0 0.28rem 0.6rem;font-family:\'IBM Plex Mono\',monospace;'
        f'font-size:0.78rem;color:#e6e8ed;">{v}</td>'
        f'</tr>'
        for k, v in fields
    )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;margin-top:0.6rem;'
        f'background:#1a1d24;border:1px solid #2a2d36;border-radius:7px;overflow:hidden;'
        f'padding:0.25rem;">{rows_html}</table>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Branding ──────────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.05rem;'
        'letter-spacing:0.12em;text-transform:uppercase;color:#e6e8ed !important;'
        'margin:0.5rem 0 0.1rem;">MLabs</h2>'
        '<p style="font-size:0.72rem;color:#4a4f5c;margin:0 0 0.5rem;">Rank-Augmented Linear Attention</p>',
        unsafe_allow_html=True,
    )

    # ── Task Mode ─────────────────────────────────────────────────────────
    _section("Task")
    task_mode = st.radio(
        "Select task",
        ["Image Classification", "Language Modeling (Shakespeare)", "Associative Recall"],
        label_visibility="collapsed",
    )
    
    if task_mode == "Image Classification":
        task_id = "image"
    elif task_mode == "Language Modeling (Shakespeare)":
        task_id = "shakespeare"
    else:
        task_id = "recall"

    # ── Run Mode ──────────────────────────────────────────────────────────
    _section("Run Mode")
    run_mode = st.radio(
        "Select mode",
        ["Train from scratch", "Fine-tune checkpoint", "Evaluate only"],
        label_visibility="collapsed",
        help=(
            "**Train from scratch** — Build and train a new model.\n\n"
            "**Fine-tune checkpoint** — Load a saved model and continue training.\n\n"
            "**Evaluate only** — Load a saved model and test it; no training occurs."
        ),
    )

    is_train_new = run_mode == "Train from scratch"
    is_finetune  = run_mode == "Fine-tune checkpoint"
    is_eval_only = run_mode == "Evaluate only"

    # ── Checkpoint Browser ────────────────────────────────────────────────
    saved_checkpoints = list_checkpoints()
    saved_checkpoints = [c for c in saved_checkpoints if c["config"].get("task", "image") == task_id]
    selected_ckpt     = None
    selected_ckpt_idx = 0

    if is_finetune or is_eval_only:
        _section("Checkpoint")
        if not saved_checkpoints:
            st.markdown(
                '<div style="background:#2d1a00;border:1px solid #78350f;border-radius:5px;'
                'padding:0.55rem 0.75rem;font-size:0.8rem;color:#fbbf24;margin-bottom:0.5rem;">'
                'No saved checkpoints found. Train a model first.</div>',
                unsafe_allow_html=True,
            )
        else:
            ckpt_names        = [c["filename"] for c in saved_checkpoints]
            selected_ckpt_idx = st.selectbox(
                "Select checkpoint",
                range(len(ckpt_names)),
                format_func=lambda i: ckpt_names[i],
                label_visibility="collapsed",
            )
            selected_ckpt = saved_checkpoints[selected_ckpt_idx]
            _render_checkpoint_card(selected_ckpt)
            
            if st.button("Delete Checkpoint", key="del_ckpt_btn"):
                if delete_checkpoint(selected_ckpt["filepath"]):
                    st.toast(f"Deleted {selected_ckpt['filename']}")
                    st.rerun()
                else:
                    st.error("Failed to delete checkpoint.")

    # ── Data & Training ───────────────────────────────────────────────────
    _section("Data & Training")

    seq_len = 128
    num_pairs = 8
    recall_vocab = 100

    if is_train_new:
        if task_id == "image":
            dataset       = st.selectbox("Dataset", ["cifar-10", "mnist", "fashion-mnist", "synthetic"])
            sample_limit  = st.slider("Training samples", 200, 60_000, 10_000, 100)
        elif task_id == "shakespeare":
            dataset       = "shakespeare"
            sample_limit  = st.slider("Sequence count limit", 0, 10_000, 0, 100, help="0 = use all ~1M characters.")
            seq_len       = st.select_slider("Context window length", [32, 64, 128, 256, 512], value=128)
        else:
            dataset       = "recall"
            sample_limit  = st.slider("Training samples", 1000, 60_000, 10_000, 1000)
            num_pairs     = st.slider("Key-value pairs", 2, 64, 8)
            recall_vocab  = st.slider("Vocabulary size", 16, 1000, 100)

        epochs        = st.slider("Epochs", 1, 100, 30)
        learning_rate = st.number_input("Learning rate", 1e-5, 1e-1, 1e-3, format="%.5f")
        batch_size    = st.select_slider("Batch size", [16, 32, 64, 128], value=64)
        seed          = st.number_input("Seed", min_value=0, value=7, step=1)

    elif is_finetune:
        st.markdown(
            '<div style="background:#1a1d24;border:1px solid #2563eb44;border-radius:5px;'
            'padding:0.5rem 0.75rem;font-size:0.78rem;color:#93c5fd;margin-bottom:0.5rem;">'
            'Use a lower learning rate than original training (e.g. 1e-4 → 1e-5).</div>',
            unsafe_allow_html=True,
        )
        if task_id == "image":
            ft_dataset = st.selectbox("Dataset", ["cifar-10", "mnist", "fashion-mnist", "synthetic"], key="ft_ds")
            ft_samples = st.slider("Training samples", 200, 60_000, 5_000, 100, key="ft_samples")
        elif task_id == "shakespeare":
            ft_dataset = "shakespeare"
            ft_samples = st.slider("Sequence count limit", 0, 10_000, 0, 100, key="ft_samples")
        else:
            ft_dataset = "recall"
            ft_samples = st.slider("Training samples", 1000, 60_000, 10_000, 1000, key="ft_samples")

        ft_epochs  = st.slider("Additional epochs", 1, 50, 10, key="ft_epochs",
                               help="New epochs added on top of original training.")
        ft_lr      = st.number_input("Learning rate", 1e-6, 1e-2, 1e-4, format="%.6f", key="ft_lr")
        batch_size = st.select_slider("Batch size", [16, 32, 64, 128], value=64, key="ft_bs")

    elif is_eval_only:
        if task_id == "image":
            eval_dataset  = st.selectbox("Dataset", ["cifar-10", "mnist", "fashion-mnist", "synthetic"], key="eval_ds")
            eval_samples  = st.slider("Evaluation samples", 200, 60_000, 1_000, 100, key="eval_samples")
        elif task_id == "shakespeare":
            eval_dataset = "shakespeare"
            eval_samples = st.slider("Evaluation limit", 0, 10_000, 0, 100, key="eval_samples")
        else:
            eval_dataset = "recall"
            eval_samples = st.slider("Evaluation samples", 1000, 50_000, 1_000, 1000, key="eval_samples")

        batch_size = st.select_slider("Batch size", [16, 32, 64, 128], value=64, key="eval_bs")

    # ── Architecture (train mode only) ────────────────────────────────────
    if is_train_new:
        _section("Architecture")
        
        if task_id == "image":
            attention_type = st.selectbox("Attention type", ["hybrid", "rala", "linear", "softmax"])
        else:
            attention_type = "hybrid"
            st.markdown('<p style="font-size:0.8rem;color:#6b7280;margin-bottom:1rem;">Model: <strong>HybridLM</strong> (Text wrapper)</p>', unsafe_allow_html=True)

        if attention_type != "hybrid":
            _section("Formulas")
            kappa_formula   = _formula_input("kappa(x)", PRESET_KAPPA, "kappa")
            phi_formula     = _formula_input("phi(x)",   PRESET_PHI,   "phi")
            use_alpha       = st.checkbox("Alpha KV augmentation",  value=True)
            use_output_gate = st.checkbox("Output gate (Eq. 6)",    value=True)
            use_salience_gate = True
        else:
            kappa_formula   = "elu(x) + 1"
            phi_formula     = "linear(x)"
            use_alpha       = True
            use_output_gate = True
            use_salience_gate = True

        _section("Model Dimensions")
        dim        = st.select_slider("Model dimension D", [32, 64, 96, 128, 192, 256, 512], value=64)
        heads      = st.select_slider("Attention heads h", [1, 2, 4, 8, 16, 32], value=4)
        layers     = st.slider("Layers L", 1, 32, 2)
        if task_id == "image":
            patch_size = st.select_slider("Patch size", [2, 4, 7, 8], value=4)
        else:
            patch_size = 0  # not used
        mlp_ratio  = st.select_slider("MLP ratio", [1, 2, 4], value=2)
        device     = st.selectbox("Device", ["cpu", "cuda"])

        if attention_type == "hybrid":
            _section("Hybrid Settings")
            window_size = st.select_slider(
                "Local window size w", [0, 4, 8, 16, 32, 64, 128], value=16,
                help="Neighbor count for local softmax attention. 0 = global only.",
            )
            mode = st.selectbox(
                "Execution mode", ["parallel", "recurrent"],
                help="parallel = all tokens at once (training); recurrent = streaming (inference).",
            )
            use_output_gate = st.checkbox(
                "Hybrid output gate φ(x)",
                value=True,
                help="Disable for ablation: φ is replaced with 1.",
            )
            use_salience_gate = st.checkbox(
                "Hybrid salience gate gᵢ",
                value=True,
                help="Disable for ablation: global memory uses uniform normalized token weights.",
            )
            if dim % heads == 0:
                hd = dim // heads
                st.markdown(
                    f'<p style="font-size:0.75rem;color:#6e7380;font-family:\'IBM Plex Mono\',monospace;'
                    f'margin-top:0.25rem;">d = {dim}/{heads} = <strong style="color:#93c5fd;">{hd}</strong> · '
                    f'mem/head = {hd}² = <strong style="color:#93c5fd;">{hd**2:,}</strong> · '
                    f'total = <strong style="color:#93c5fd;">{heads * hd**2:,}</strong></p>',
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"D ({dim}) must be divisible by h ({heads}).")
        else:
            window_size = 16
            mode        = "parallel"

    else:
        # Fine-tune / eval: architecture comes from the checkpoint
        attention_type  = "hybrid"
        kappa_formula   = "elu(x) + 1"
        phi_formula     = "linear(x)"
        use_alpha       = True
        use_output_gate = True
        use_salience_gate = True
        dim             = 64
        heads           = 4
        layers          = 2
        patch_size      = 4
        mlp_ratio       = 2
        window_size     = 16
        mode            = "parallel"
        device          = "cpu"

    # ── Measurement ───────────────────────────────────────────────────────
    _section("Measurement")
    rank_tol = st.select_slider(
        "SVD rank tolerance",
        options=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7, 1e-8],
        value=DEFAULT_RANK_TOL,
        format_func=lambda v: f"{v:.0e}",
        help="Singular values below this threshold are treated as zero when computing matrix rank.",
    )
    stats_batches = st.slider("Stats batches", 1, 10, 3,
                              help="Validation batches averaged for rank/stability diagnostics.")
    warmup_passes = st.slider("Inference warmup passes", 0, 10, 3,
                              help="Forward passes before timing, to flush JIT and allocation overhead.")

    # ── Options ───────────────────────────────────────────────────────────
    _section("Options")
    if is_train_new:
        auto_save    = st.checkbox("Save model after training",   value=True,
                                   help="Automatically save the trained model as a checkpoint.")
        run_baseline = st.checkbox("Run baseline comparison",     value=False,
                                   help="Also run linear and softmax baselines for side-by-side comparison.")
    elif is_finetune:
        ft_auto_save = st.checkbox("Save fine-tuned model",       value=True,
                                   help="Save the fine-tuned model as a new checkpoint.")

    # ── Checkpoint list (train mode) ──────────────────────────────────────
    if saved_checkpoints and is_train_new:
        _section("Saved Checkpoints")
        with st.expander(f"{len(saved_checkpoints)} checkpoint(s) on disk"):
            for ckpt in saved_checkpoints:
                cfg = ckpt["config"]
                st.markdown(
                    f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;'
                    f'color:#93c5fd;margin:0.1rem 0;">{ckpt["filename"]}</p>'
                    f'<p style="font-size:0.72rem;color:#6e7380;margin:0 0 0.5rem;">'
                    f'{cfg.get("attention_type","?").upper()} · D={cfg.get("dim")} · '
                    f'h={cfg.get("heads")} · L={cfg.get("layers")} · '
                    f'{ckpt["size_mb"]:.1f} MB</p>',
                    unsafe_allow_html=True,
                )

    # ── Parameter estimator (train mode) ─────────────────────────────────
    if is_train_new:
        if task_id == "shakespeare":
            est_vocab = 65
        elif task_id == "recall":
            est_vocab = recall_vocab + 2
        else:
            est_vocab = 0
            
        total_params = _estimate_params(task_id, est_vocab, dim, heads, layers, patch_size, mlp_ratio)
        st.markdown(
            f'<div style="background:#1a1d24;border:1px solid #2a2d36;border-radius:8px;'
            f'padding:0.85rem 1rem;margin-top:1.25rem;">'
            f'<p style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#4a4f5c;margin:0 0 0.4rem;">Parameter Estimate</p>'
            f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:1.35rem;'
            f'font-weight:500;color:#58a6ff;margin:0;line-height:1;">'
            f'~{_format_params(total_params)}</p>'
            f'<p style="font-size:0.7rem;color:#4a4f5c;margin:0.25rem 0 0;'
            f'font-family:\'Instrument Sans\',sans-serif;">'
            f'Approximate {task_id} task</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    '<h1 style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:2rem;'
    'letter-spacing:-0.02em;color:#111318;margin:0 0 0.2rem;">MLabs</h1>'
    '<p style="font-size:0.9rem;color:#6b7280;margin:0 0 1.5rem;">'
    'Rank-Augmented Linear Attention &amp; Hybrid Attention — interactive research platform</p>',
    unsafe_allow_html=True,
)

# ── Mode context banner ───────────────────────────────────────────────────────

_MODE_STYLES = {
    "Train from scratch":    ("train",    "#dbeafe", "#1d4ed8", "#1e3a8a", "TRAINING MODE",    "Configure architecture and training parameters, then launch a new experiment."),
    "Fine-tune checkpoint":  ("finetune", "#fef3c7", "#d97706", "#78350f", "FINE-TUNE MODE",   "Select a checkpoint and continue training with adjusted parameters."),
    "Evaluate only":         ("eval",     "#dcfce7", "#16a34a", "#14532d", "EVALUATION MODE",  "Load a checkpoint and measure performance on a dataset — no gradient updates."),
}
_cls, _bg, _accent, _text, _mode_label, _mode_desc = _MODE_STYLES[run_mode]

st.markdown(
    f'<div style="background:{_bg};border-left:3px solid {_accent};border-radius:7px;'
    f'padding:0.75rem 1.25rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem;">'
    f'<div style="width:8px;height:8px;border-radius:50%;background:{_accent};flex-shrink:0;"></div>'
    f'<div>'
    f'<span style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;color:{_accent};">'
    f'{_mode_label}</span>'
    f'<p style="font-size:0.82rem;color:{_text};margin:0.1rem 0 0;'
    f'font-family:\'Instrument Sans\',sans-serif;">{_mode_desc}</p>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# ── Formula validation (train + non-hybrid) ───────────────────────────────────

if is_train_new and attention_type != "hybrid":
    st.markdown(
        '<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.09em;'
        'text-transform:uppercase;color:#6b7280;margin-bottom:0.5rem;">Formula Validation</p>',
        unsafe_allow_html=True,
    )
    vcol_l, vcol_r = st.columns(2)
    with vcol_l:
        _validate_formula("kappa(x)", kappa_formula)
    with vcol_r:
        _validate_formula("phi(x)", phi_formula)
    st.markdown("")

# ── Context info bar ──────────────────────────────────────────────────────────

if is_train_new:
    st.markdown(
        f'<p style="font-size:0.8rem;color:#6b7280;background:#f8fafc;'
        f'border:1px solid #e2e5ea;border-radius:6px;padding:0.6rem 1rem;margin-bottom:1rem;">'
        f'Rank tolerance <code style="font-family:IBM Plex Mono;">{rank_tol:.0e}</code> · '
        f'<strong>{stats_batches}</strong> stats batch(es) · '
        f'<strong>{warmup_passes}</strong> warmup pass(es). '
        f'Rank ratios are observations — higher rank does not imply higher accuracy in isolation.</p>',
        unsafe_allow_html=True,
    )
elif is_finetune and selected_ckpt:
    cfg = selected_ckpt["config"]
    st.markdown(
        f'<p style="font-size:0.8rem;color:#6b7280;background:#f8fafc;'
        f'border:1px solid #e2e5ea;border-radius:6px;padding:0.6rem 1rem;margin-bottom:1rem;">'
        f'Resuming from <code style="font-family:IBM Plex Mono;">{selected_ckpt["filename"]}</code> — '
        f'D={cfg.get("dim")}, h={cfg.get("heads")}, L={cfg.get("layers")}, '
        f'{cfg.get("epochs")} epoch(s) on {cfg.get("dataset")} '
        f'({cfg.get("sample_limit")} samples). '
        f'Adding <strong>{ft_epochs}</strong> epoch(s) at LR '
        f'<code style="font-family:IBM Plex Mono;">{ft_lr:.1e}</code>.</p>',
        unsafe_allow_html=True,
    )
elif is_eval_only and selected_ckpt:
    st.markdown(
        f'<p style="font-size:0.8rem;color:#6b7280;background:#f8fafc;'
        f'border:1px solid #e2e5ea;border-radius:6px;padding:0.6rem 1rem;margin-bottom:1rem;">'
        f'Evaluating <code style="font-family:IBM Plex Mono;">{selected_ckpt["filename"]}</code> '
        f'on <strong>{eval_dataset}</strong> ({eval_samples} samples). No training will occur.</p>',
        unsafe_allow_html=True,
    )

import time
from contextlib import contextmanager

@contextmanager
def inject_training_progress(total_epochs: int):
    import rala_lab.training as tr
    
    # Prevent recursive wrapping if Streamlit aborted a previous run without executing finally
    if getattr(tr._run_epoch, "_is_wrapped", False):
        original_run_epoch = tr._run_epoch._original
    else:
        original_run_epoch = tr._run_epoch

    if getattr(tr._run_text_epoch, "_is_wrapped", False):
        original_run_text_epoch = tr._run_text_epoch._original
    else:
        original_run_text_epoch = tr._run_text_epoch
    
    start_time = time.time()
    state = {"epoch": 1}
    
    progress_bar = st.progress(0, text="Starting training...")
    timer_placeholder = st.empty()
    
    def update_eta(current_batch, total_batches):
        epoch = state["epoch"]
        fraction = ((epoch - 1) * total_batches + current_batch) / (total_epochs * total_batches)
        if fraction > 0:
            elapsed = time.time() - start_time
            eta_seconds = (elapsed / fraction) - elapsed
            mins, secs = divmod(int(eta_seconds), 60)
            timer_placeholder.markdown(f"**⏳ ETA:** {mins}m {secs}s")
            progress_bar.progress(min(fraction, 1.0), text=f"Epoch {epoch}/{total_epochs} — Batch {current_batch}/{total_batches}")

    class LoaderWrapper:
        def __init__(self, loader):
            self.loader = loader
            self.total = len(loader)
        @property
        def dataset(self):
            return self.loader.dataset
        def __iter__(self):
            for i, batch in enumerate(self.loader):
                update_eta(i + 1, self.total)
                yield batch
            state["epoch"] += 1
        def __len__(self):
            return self.total

    def wrapped_run_epoch(model, loader, device, optimizer):
        return original_run_epoch(model, LoaderWrapper(loader), device, optimizer)
    wrapped_run_epoch._is_wrapped = True
    wrapped_run_epoch._original = original_run_epoch

    def wrapped_run_text_epoch(model, loader, device, optimizer, is_lm):
        return original_run_text_epoch(model, LoaderWrapper(loader), device, optimizer, is_lm)
    wrapped_run_text_epoch._is_wrapped = True
    wrapped_run_text_epoch._original = original_run_text_epoch

    tr._run_epoch = wrapped_run_epoch
    tr._run_text_epoch = wrapped_run_text_epoch
    
    try:
        yield
    finally:
        tr._run_epoch = original_run_epoch
        tr._run_text_epoch = original_run_text_epoch
        progress_bar.empty()
        timer_placeholder.empty()

# ══════════════════════════════════════════════════════════════════════════════
#  RUN ACTIONS  (backend calls preserved exactly)
# ══════════════════════════════════════════════════════════════════════════════

if is_train_new:
    config = ExperimentConfig(
        dataset=dataset,
        task=task_id,
        seq_len=seq_len,
        num_pairs=num_pairs,
        recall_vocab=recall_vocab,
        seed=int(seed),
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        sample_limit=sample_limit,
        attention_type=attention_type,
        kappa_formula=kappa_formula,
        phi_formula=phi_formula,
        use_alpha=use_alpha,
        use_output_gate=use_output_gate,
        use_salience_gate=use_salience_gate,
        dim=dim,
        heads=heads,
        layers=layers,
        patch_size=patch_size,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        mode=mode,
        device=device,
        rank_tol=rank_tol,
        stats_batches=stats_batches,
        warmup_passes=warmup_passes,
    )

    if st.button("Run Experiment", type="primary"):
        with st.spinner("Training model and collecting rank diagnostics…"):
            try:
                with inject_training_progress(epochs):
                    result = run_experiment(config)
            except Exception as exc:
                st.exception(exc)
                st.stop()

        if auto_save and result.model is not None:
            save_path = save_checkpoint(result.model, config)
            st.success(f"Checkpoint saved — `{save_path.name}`")

        st.session_state["last_result"] = result

        if run_baseline:
            baselines = {}
            for baseline_type in ["linear", "softmax", "rala"]:
                if baseline_type == attention_type:
                    continue
                
                # Dynamic logic for the RALA baseline
                is_rala = (baseline_type == "rala")
                b_alpha = use_alpha if is_rala else False
                b_gate  = use_output_gate if is_rala else False
                
                # If comparing against Hybrid, force RALA to use the tanh(x) + 1 gate 
                # so it behaves identically to Hybrid's internal gate.
                b_phi_formula = phi_formula
                if is_rala and attention_type == "hybrid":
                    b_phi_formula = "tanh(x) + 1"

                baseline_config = ExperimentConfig(
                    dataset=dataset, seed=int(seed), batch_size=batch_size,
                    epochs=epochs, learning_rate=learning_rate, sample_limit=sample_limit,
                    attention_type=baseline_type, kappa_formula=kappa_formula,
                    phi_formula=b_phi_formula, use_alpha=b_alpha, use_output_gate=b_gate,
                    dim=dim, heads=heads, layers=layers, patch_size=patch_size,
                    device=device, rank_tol=rank_tol, stats_batches=stats_batches,
                    warmup_passes=warmup_passes,
                )
                with st.spinner(f"Running {baseline_type} baseline…"):
                    try:
                        with inject_training_progress(epochs):
                            baselines[baseline_type] = run_experiment(baseline_config)
                    except Exception as exc:
                        st.warning(f"Baseline {baseline_type} failed: {exc}")
            st.session_state["baselines"] = baselines
        else:
            st.session_state.pop("baselines", None)

elif is_finetune:
    if st.button("Start Fine-tuning", type="primary"):
        if not selected_ckpt:
            st.error("No checkpoint selected.")
            st.stop()
        with st.spinner(f"Loading `{selected_ckpt['filename']}` and fine-tuning for {ft_epochs} epoch(s)…"):
            try:
                with inject_training_progress(ft_epochs):
                    result = continue_training(
                    filepath=selected_ckpt["filepath"],
                    extra_epochs=ft_epochs,
                    learning_rate=ft_lr,
                    dataset=ft_dataset,
                    sample_limit=ft_samples,
                    batch_size=batch_size,
                    rank_tol=rank_tol,
                    stats_batches=stats_batches,
                    warmup_passes=warmup_passes,
                )
            except Exception as exc:
                st.exception(exc)
                st.stop()

        if ft_auto_save and result.model is not None:
            save_path = save_checkpoint(result.model, result.config)
            st.success(f"Fine-tuned checkpoint saved — `{save_path.name}`")

        st.session_state["last_result"] = result
        st.session_state.pop("baselines", None)

        orig = selected_ckpt["config"].get("epochs", "?")
        st.success(
            f"Fine-tuning complete — continued from epoch {orig} to "
            f"epoch {result.config.get('epochs', '?')} at LR {ft_lr:.1e}."
        )

elif is_eval_only:
    if st.button("Evaluate Checkpoint", type="primary"):
        if not selected_ckpt:
            st.error("No checkpoint selected.")
            st.stop()
        with st.spinner(f"Loading `{selected_ckpt['filename']}` and evaluating…"):
            try:
                result = evaluate_checkpoint(
                    filepath=selected_ckpt["filepath"],
                    dataset=eval_dataset,
                    sample_limit=eval_samples,
                    batch_size=batch_size,
                    rank_tol=rank_tol,
                    stats_batches=stats_batches,
                    warmup_passes=warmup_passes,
                )
            except Exception as exc:
                st.exception(exc)
                st.stop()
        st.session_state["last_result"] = result
        st.session_state.pop("baselines", None)
        st.success(
            f"Evaluated `{selected_ckpt['filename']}` on "
            f"{eval_dataset} ({eval_samples} samples) — no training performed."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if "last_result" not in st.session_state:
    st.markdown(
        '<div style="text-align:center;padding:3.5rem 2rem;color:#9ca3af;">'
        '<p style="font-family:\'Syne\',sans-serif;font-size:1.1rem;color:#6b7280;margin-bottom:0.5rem;">'
        'No results yet</p>'
        '<p style="font-size:0.85rem;">Configure your parameters and press the run button above.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

result   = st.session_state["last_result"]
baselines = st.session_state.get("baselines", {})

# ── Result header ─────────────────────────────────────────────────────────────

mode_tag = result.config.get("mode", "")
attn_tag = result.config["attention_type"].upper()

if mode_tag == "fine_tuned":
    orig_ep  = result.config.get("original_epochs", "?")
    total_ep = result.config.get("epochs", "?")
    heading  = f"{attn_tag}"
    sub_tag  = f"Fine-tuned · epoch {orig_ep} → {total_ep}"
    badge_v  = "amber"
elif mode_tag == "evaluate_only":
    heading  = f"{attn_tag}"
    sub_tag  = "Evaluate only · no gradient updates"
    badge_v  = "green"
else:
    heading  = f"{attn_tag}"
    sub_tag  = f"{result.config.get('epochs', '?')} epoch(s) · {result.config.get('dataset')}"
    badge_v  = "blue"

st.markdown("<hr style='border:none;border-top:1px solid #e2e5ea;margin:1.25rem 0;'>",
            unsafe_allow_html=True)
st.markdown(
    f'<div style="display:flex;align-items:baseline;gap:0.75rem;margin-bottom:1.25rem;">'
    f'<h2 style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:1.4rem;'
    f'color:#111318;margin:0;">{heading}</h2>'
    f'{_badge(sub_tag, badge_v)}'
    f'</div>',
    unsafe_allow_html=True,
)

_display_result(result)


# ── Baseline comparison ───────────────────────────────────────────────────────

if baselines:
    st.markdown("<hr style='border:none;border-top:1px solid #e2e5ea;margin:2rem 0 1.5rem;'>",
                unsafe_allow_html=True)
    st.markdown(
        '<h3 style="font-family:\'Syne\',sans-serif;font-weight:600;font-size:1.1rem;'
        'color:#111318;margin-bottom:1.25rem;">Baseline Comparison</h3>',
        unsafe_allow_html=True,
    )

    # Side-by-side summary table
    all_results = {result.config["attention_type"]: result}
    all_results.update(baselines)
    summary_rows = []
    for atype, r in all_results.items():
        summary_rows.append({
            "Attention":        atype.upper(),
            "Val Accuracy":     f"{r.history[-1].val_acc:.4f}"   if r.history else "n/a",
            "Val Loss":         f"{r.history[-1].val_loss:.4f}"  if r.history else "n/a",
            "Output Rank Ratio": _format_optional(r.final_stats.output_rank_ratio),
            "Memory Rank Ratio": _format_optional(r.final_stats.memory_rank_ratio),
            "Inference (ms)":   f"{r.inference_ms:.1f}",
            "Warnings":         len(r.final_stats.warnings),
        })

    cmp_chart, cmp_table = st.columns([1.1, 1])
    with cmp_chart:
        st.plotly_chart(
            _plotly_comparison_bar(summary_rows),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with cmp_table:
        st.markdown("**Head-to-head metrics**")
        st.dataframe(pd.DataFrame(summary_rows).astype(str), width="stretch")

    for name, baseline_result in baselines.items():
        with st.expander(f"Full diagnostics — {name.upper()} baseline"):
            _display_result(baseline_result, label=name.upper())


# ── Export & Metric Guide ─────────────────────────────────────────────────────

st.markdown("<hr style='border:none;border-top:1px solid #e2e5ea;margin:2rem 0 1.5rem;'>",
            unsafe_allow_html=True)

col_exp, col_guide = st.columns(2)

with col_exp:
    export: dict = {
        "config":          result.config,
        "history":         [asdict(row) for row in result.history],
        "per_layer_stats": [asdict(s)   for s   in result.per_layer_stats],
        "final_stats":     asdict(result.final_stats),
        "inference_ms":    result.inference_ms,
    }
    if baselines:
        export["baselines"] = {
            name: {
                "config":          br.config,
                "history":         [asdict(row) for row in br.history],
                "per_layer_stats": [asdict(s)   for s   in br.per_layer_stats],
                "final_stats":     asdict(br.final_stats),
                "inference_ms":    br.inference_ms,
            }
            for name, br in baselines.items()
        }
    export_json = json.dumps(export, indent=2)

    # Only auto-save if we haven't saved this specific result yet
    if st.session_state.get("last_saved_result_id") != id(result):
        save_dir = Path(r"e:\Research_pprs\AI_MODELS\results\REsults_FAST")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]
        file_name = f"rala_experiment_{timestamp}_{run_id}.json"
        save_path = save_dir / file_name

        try:
            save_path.write_text(export_json, encoding="utf-8")
            st.session_state["last_saved_result_id"] = id(result)
            st.session_state["last_saved_path"] = str(save_path)
            st.success(f"✅ JSON results automatically saved to `{save_path}`")
        except Exception as e:
            st.warning(f"Could not auto-save JSON to {save_dir}: {e}")
    elif "last_saved_path" in st.session_state:
        st.success(f"✅ JSON results were saved to `{st.session_state['last_saved_path']}`")

    st.download_button(
        label="Download experiment JSON",
        data=export_json,
        file_name=file_name,
        mime="application/json",
    )

with col_guide:
    with st.expander("Metric reference guide"):
        for metric, help_text in METRIC_HELP.items():
            st.markdown(f"**{metric}.** {help_text}")
