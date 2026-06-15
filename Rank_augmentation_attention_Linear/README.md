# RALA Formula Lab

RALA Formula Lab is a local Streamlit research UI for experimenting with
Rank-Augmented Linear Attention from **Breaking the Low-Rank Dilemma of Linear
Attention**. It lets you change the kernel `kappa(x)` and output modulation
`phi(x)`, run a small classifier, and inspect accuracy, rank, timing, and
stability diagnostics.

The goal is not to reproduce ImageNet numbers. Version 1 is a fast workbench
for checking whether different `kappa` and `phi` choices improve feature rank,
accuracy, and numerical behavior on small controlled tasks.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you only want the synthetic dataset, no dataset download is required. MNIST,
Fashion-MNIST, and CIFAR-10 use `torchvision` and will download data into
`data/` when selected.

## Run The UI

```bash
streamlit run app.py
```

The app opens a browser dashboard. Choose a dataset, formulas, attention type,
and model size, then press **Run experiment**.

## What Is Implemented

The default RALA path follows the paper's Eq. 6:

```text
Q_g = mean(Q)
alpha_j = N * softmax(Q_g kappa(K_j)^T)
B = sum_j alpha_j kappa(K_j)^T V_j
Y_i = phi(X_i) * (kappa(Q_i) B / normalizer)
```

The denominator from linear attention is included with epsilon protection:

```text
normalizer = kappa(Q_i) sum_j alpha_j kappa(K_j)^T
```

You can compare:

- `softmax`: standard quadratic Softmax attention baseline.
- `linear`: vanilla kernel linear attention with no RALA-specific alpha/gate.
- `rala`: alpha-weighted KV buffer plus optional Eq. 6 output gate.

## Safe Formula Language

Formulas are intentionally not arbitrary Python. They are parsed with `ast` and
only a small set of tensor expressions is accepted.

Allowed variable:

```text
x
```

Allowed functions:

```text
elu, relu, gelu, silu, sigmoid, tanh, softplus,
exp, log, abs, square, sqrt, normalize, softmax, clamp,
linear, identity
```

Allowed operators:

```text
+, -, *, /, **, unary +, unary -
```

Examples:

```text
elu(x) + 1
relu(x) + 1e-6
softplus(x)
sigmoid(x) * x
tanh(x)
linear(x)
x
```

Rejected examples include imports, attribute access, lambdas, comprehensions,
unknown functions, indexing, and keyword arguments. This keeps the UI safer and
makes experiments easier to reproduce.

## Metric Guide

**Accuracy** is the fraction of correct predictions. Higher is better, but one
seed is not proof that a formula is generally better.

**Loss** is the cross-entropy objective. Lower is better.

**KV buffer rank** is the average rank of the global memory matrix
`B = sum alpha kappa(K)^T V`. Higher usually suggests a richer global memory.
The paper expects RALA alpha weighting to increase this compared with vanilla
linear attention.

**Output feature rank** is the average rank of the final attention output. Higher
suggests more diverse token features. The paper expects Eq. 6 output modulation
to improve this.

**Rank ratio** is rank divided by the maximum possible rank. Values closer to
`1.0` are closer to full-rank.

**Inference time** is the wall-clock forward time for one validation batch.
Lower is faster. This small app is not a rigorous systems benchmark, but it is
useful for rough comparisons.

**Stability warnings** report NaN, Inf, very large activations, and near-zero
linear-attention denominators. Fewer warnings are better.

## Exporting Results

After a run, use **Download experiment JSON** to save:

- formulas
- dataset and seed
- model settings
- training history
- final rank/stability metrics
- inference time

## Tests

After installing dependencies:

```bash
pytest
```

The tests cover safe formula validation and attention shape/rank diagnostics.

## Limitations

- Results are small-scale experimental observations, not claims about ImageNet.
- Higher rank can be useful, but it is not automatically causal evidence of
  better accuracy.
- Dataset downloads require network access.
- The Streamlit app executes model training locally, so larger settings can be
  slow on CPU.
