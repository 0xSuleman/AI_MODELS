# Experiment Log — CIFAR-10 Benchmark Runs

All experiments use the **CIFAR-10** dataset (32×32 RGB images, 10 classes).
Output rank ratio = 1.000 in **every single run** — indicating no observed rank collapse in the measured attention output.

---

## Summary Table

| # | Date | Model | D | h | L | d | Patch | Window | MLP | Samples | Epochs | LR | Best Val | Final Val | Gap | Memory Rank (worst) | Inference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-05-09 20:32 | RALA   | 64 | 4 | 2 | 16 | 2 | — | — | 10,000 | 3 | 1e-3 | 39.9% | 39.9% | -1.0% | 1.000 | 295ms |
| 2 | 2026-05-09 21:18 | RALA   | 64 | 4 | 2 | 16 | 2 | — | — | 10,000 | 10 | 1e-3 | 47.9% | 47.6% | 8.6% | 1.000 | 269ms |
| 3 | 2026-05-09 22:35 | RALA   | 64 | 4 | 2 | 16 | 2 | — | — | 10,000 | 20 | 1e-3 | 48.6% | 47.9% | **24.7%** | 0.999 | 298ms |
| 4 | 2026-05-16 21:19 | Hybrid | 96 | 4 | 2 | 24 | 2 | 16 | 2 | 2,000 | 8 | 1e-3 | 33.5% | 30.5% | -0.8% | 0.843 | 481ms |
| 5 | 2026-05-16 21:31 | Hybrid | 64 | 4 | 2 | 16 | 2 | 8 | 2 | 500 | 30 | 1e-3 | 32.5% | 30.0% | **18.0%** | 0.982 | 201ms |
| 6 | 2026-05-16 21:53 | Hybrid | 64 | 4 | 2 | 16 | 2 | 8 | 2 | 1,400 | 30 | 1e-3 | 41.8% | 40.7% | 6.5% | 0.992 | 232ms |
| 7 | 2026-05-16 22:13 | Hybrid | 64 | 4 | 2 | 16 | 4 | 16 | 2 | 2,500 | 30 | 1e-3 | 43.4% | 41.6% | **-0.2%** | 0.871 | 82ms |
| 8 | 2026-05-16 22:30 | Hybrid | 64 | 4 | 2 | 16 | 4 | 16 | 2 | 5,000 | 30 | 1e-3 | **50.0%** | **50.0%** | **-2.1%** | 0.968 | 84ms |

---

## Detailed Run Descriptions

### Run 1 — RALA Baseline (3 epochs)
- **File:** `rala_kxx_philineax.json`
- **Date:** 2026-05-09 20:32
- **Config:** RALA, D=64, h=4, L=2, patch=2, 10K samples, 3 epochs, LR=1e-3
- **Result:** Val=39.9%, Gap=-1.0% (underfitting — only 3 epochs)
- **Rank:** KV=1.000, Output=1.000
- **Notes:** Baseline RALA test. Too few epochs to converge. Established that the training pipeline works.

### Run 2 — RALA 10 Epochs
- **File:** `rala_kxx_philinearx_10epochs.json`
- **Date:** 2026-05-09 21:18
- **Config:** RALA, D=64, h=4, L=2, patch=2, 10K samples, 10 epochs, LR=1e-3
- **Result:** Val=47.6%, Gap=8.6% (starting to overfit)
- **Rank:** KV=1.000, Output=1.000
- **Notes:** Significant improvement from 3→10 epochs. Overfitting beginning at epoch 6.

### Run 3 — RALA 20 Epochs (Overfitting Exposed)
- **File:** `rala_experiment_philinearx_kappaxx_20epochsrevealproblem.json`
- **Date:** 2026-05-09 22:35
- **Config:** RALA, D=64, h=4, L=2, patch=2, 10K samples, 20 epochs, LR=1e-3
- **Result:** Val=47.9%, Gap=**24.7%** (severe overfitting — train=72.7%)
- **Rank:** KV=0.999, Output=1.000
- **Notes:** Train accuracy reached 72.7% but val stuck at 48%. Classic memorization pattern. This run revealed the need for better regularization and architectural improvements, motivating the transition to the Hybrid model.

---

### Run 4 — Hybrid First Test (D=96, 2K samples, 8 epochs)
- **File:** `hybrid_2000_initial_test_2_.json`
- **Date:** 2026-05-16 21:19
- **Config:** Hybrid, D=96, h=4, L=2, d=24, patch=2, w=16, mlp=2, 2K samples, 8 epochs
- **Result:** Val=30.5%, Gap=-0.8% (underfitting — too few epochs)
- **Rank:** KV=0.843, Output=1.000
- **Notes:** First hybrid model test. Larger D=96 but only 8 epochs — severely undertrained. Inference slow (481ms) due to patch_size=2 creating 256 tokens.

### Run 5 — Hybrid 500 Samples (Overfitting Study)
- **File:** `Hybrid_500_samples_1st test.json`
- **Date:** 2026-05-16 21:31
- **Config:** Hybrid, D=64, h=4, L=2, d=16, patch=2, w=8, mlp=2, 500 samples, 30 epochs
- **Result:** Val=30.0%, Gap=**18.0%** (severe overfitting)
- **Rank:** KV=0.982, Output=1.000
- **Notes:** With only 500 samples and 30 epochs, each image seen ~60 times. Model memorized instead of learning. Demonstrated that data quantity is critical.

### Run 6 — Hybrid 1400 Samples
- **File:** `hybrid_cifr_64D_sample1400_h4.json`
- **Date:** 2026-05-16 21:53
- **Config:** Hybrid, D=64, h=4, L=2, d=16, patch=2, w=8, mlp=2, 1400 samples, 30 epochs
- **Result:** Val=40.7%, Gap=6.5% (mild overfitting)
- **Rank:** KV=0.992, Output=1.000
- **Notes:** Sweet spot around epochs 16-20 where train≈val. After epoch 20, mild overfitting begins. Best val accuracy of 41.8% at epoch 20.

### Run 7 — Hybrid patch_size=4 (Key Improvement)
- **File:** `Hybrid_patchsize4_d64_2500.json`
- **Date:** 2026-05-16 22:13
- **Config:** Hybrid, D=64, h=4, L=2, d=16, **patch=4**, w=16, mlp=2, 2500 samples, 30 epochs
- **Result:** Val=41.6%, Gap=**-0.2%** (perfect — zero overfitting)
- **Rank:** KV=0.871, Output=1.000
- **Inference:** 82ms (2.8× faster than patch=2 runs)
- **Notes:** Switching from patch_size=2 (256 tokens) to patch_size=4 (64 tokens) was a breakthrough. Inference nearly 3× faster. Train-val gap essentially zero — healthiest training curve to date.

### Run 8 — Hybrid 5000 Samples (Best Result) ⭐
- **File:** `HYbrid_5000sasmples_D64_epochs30.json`
- **Date:** 2026-05-16 22:30
- **Config:** Hybrid, D=64, h=4, L=2, d=16, patch=4, w=16, mlp=2, 5000 samples, 30 epochs
- **Result:** Val=**50.0%**, Gap=**-2.1%** (val exceeds train — still improving)
- **Rank:** KV=0.968, Output=1.000
- **Inference:** 84ms
- **Notes:** Best result to date. Val accuracy was STILL climbing at epoch 30 — model hadn't plateaued. Val exceeded train accuracy, indicating genuine generalization. Loss still decreasing. Would likely reach 52-55% with more epochs.

---

## Key Findings Across All Runs

### 1. Output Rank Ratio = 1.000 in Every Single Run
The measured attention output showed no observed rank collapse across all 8 experiments. This is a useful diagnostic, but it is not causal proof that the φ gate improves accuracy, memory, or reasoning without ablation evidence.

### 2. Patch Size Matters Enormously
- patch_size=2: 256 tokens, 200-480ms inference
- patch_size=4: 64 tokens, 82-84ms inference (3× faster)
- Accuracy comparable or better with patch_size=4

### 3. Data Quantity Controls Overfitting
| Samples | Train-Val Gap | Behavior |
|---------|--------------|----------|
| 500     | 18.0%        | Severe overfitting |
| 1,400   | 6.5%         | Mild overfitting |
| 2,500   | -0.2%        | Perfect balance |
| 5,000   | -2.1%        | Still improving |

### 4. RALA vs Hybrid Comparison
With similar configs (D=64, L=2, 10K samples):
- RALA at 20 epochs: 47.9% val but **24.7% gap** (memorizing)
- Hybrid at 30 epochs with 5K samples: **50.0% val** with **-2.1% gap** (genuine learning)
The Hybrid model generalizes far better than RALA.

### 5. Scaling Observations
- More data → better generalization (gap shrinks)
- Bigger D without adjusting LR → worse results (Run 4)
- More epochs → better accuracy (if not overfitting)
- The model capacity ceiling at D=64, L=2 (~83K params) is approximately 50-55% on CIFAR-10
