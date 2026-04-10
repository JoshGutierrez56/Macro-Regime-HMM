# Macro Regime HMM
### Gaussian HMM from Scratch · Baum-Welch EM · Viterbi · Regime-Conditional Factor Returns

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![No API Key](https://img.shields.io/badge/FRED-No%20API%20Key-green)](https://fred.stlouisfed.org)

---

## What This Is

A Hidden Markov Model trained on macro data to identify economic regimes
(Expansion / Slowdown / Crisis) and condition factor allocations on predicted regime.

Built entirely from scratch — no `hmmlearn`, no `statsmodels`, no `pomegranate`.
The Baum-Welch EM algorithm and Viterbi decoder are implemented from first
principles, making every mathematical step transparent and verifiable.

This directly targets PanAgora's multi-asset research stack. Edward Qian's
risk parity framework allocates by risk contribution across asset classes;
PanAgora's Macro Quantitative Strategies team conditions those allocations
on macro regimes exactly as demonstrated here.

---

## Algorithm

### Gaussian HMM

K hidden states (regimes), D-dimensional Gaussian emissions:

```
Observation model: p(x_t | s_t = k) = N(x_t ; μ_k, Σ_k)
Transition model:  P(s_t = j | s_{t-1} = k) = A_kj
```

### Baum-Welch EM (training)

```
E-step: Forward-backward algorithm → γ_t(k) and ξ_t(k,j)
    α_t(k) = P(x_{1:t}, s_t=k)     forward variable
    β_t(k) = P(x_{t+1:T} | s_t=k)  backward variable
    γ_t(k) = P(s_t=k | X, θ)

M-step: Update π, A, μ, Σ using γ and ξ

Repeat until ΔLL < 1e-4  (guaranteed non-decreasing)
```

### Viterbi decoding (inference)

```
δ_t(k) = max_{s_{1:t-1}} P(s_{1:t-1}, s_t=k, x_{1:t})
Backtrack from T to find argmax state sequence
```

All computation in log space to prevent underflow on long sequences.

---

## Data (Zero API Keys)

FRED series via direct CSV URL — no registration required:

| Feature | FRED Series | Economic Signal |
|---------|-------------|-----------------|
| Yield curve | T10Y2Y | 10Y−2Y spread — recession predictor |
| HY spread | BAMLH0A0HYM2 | Credit stress |
| VIX | VIXCLS | Market-implied volatility |
| Unemployment change | UNRATE (diff) | Labour market momentum |
| Equity return | SPY (yfinance) | Growth proxy |
| Realized vol | SPY rolling std | Forward-looking stress |

NBER recession dates (USREC) used for evaluation only — never fed to the HMM during training.

---

## Results (Offline / Synthetic, 300 months)

**Regime parameters (emission means, standardised):**

| Feature | Expansion | Slowdown | Crisis |
|---------|-----------|----------|--------|
| Yield curve | +0.49 | −1.09 | −1.39 |
| HY spread | −0.39 | +0.85 | +1.12 |
| VIX | −0.46 | +1.14 | +1.18 |

Regimes align with economic intuition: Expansion = steep curve + tight spreads + low VIX. Crisis = inverted curve + wide spreads + high VIX.

**Regime persistence** (diagonal of transition matrix A):
- Expansion: 91.6% — regimes are sticky, confirming macro persistence

**Regime-conditional factor returns (annualised):**

| Factor | Expansion | Slowdown | Crisis |
|--------|-----------|----------|--------|
| MOM | highest | moderate | lowest — **momentum crashes in crisis** |
| RMW | moderate | negative | highest — quality is the crisis hedge |
| Mkt-RF | positive | moderate | lowest |

This is the core insight: momentum is a growth regime factor. Allocating to it in Crisis is the mistake systematic managers make without regime conditioning.

**NBER recession detection (walk-forward, out-of-sample):**
- Accuracy: 96.1% | Recall: 76.9% | F1: 0.69

**Backtest (regime-tilted vs equal-weight factors, 10bps TC):**
- Strategy Sharpe: 0.21 | Benchmark Sharpe: 0.08 | IR: 0.33

---

## Repository Structure

```
macro-regime-hmm/
├── run_regime.py              ← CLI entry point
├── src/regime/
│   ├── data.py                ← FRED free CSV + yfinance download
│   ├── hmm.py                 ← Gaussian HMM from scratch (Baum-Welch + Viterbi)
│   ├── features.py            ← Feature engineering, walk-forward, regime statistics
│   ├── backtest.py            ← Regime-conditional factor portfolio backtest
│   └── charts.py              ← 7 publication-quality figures
└── tests/
    └── test_hmm.py            ← 14 tests (no internet required)
```

---

## Setup & Usage

```bash
pip install -r requirements.txt

# Full pipeline — downloads FRED + yfinance data
python run_regime.py

# Offline demo — no internet, synthetic data
python run_regime.py --offline

# 2-state (expansion/recession) model
python run_regime.py --states 2 --offline

# Custom date range
python run_regime.py --start 2005-01-01 --end 2024-01-01
```

**Tests (14, no internet required):**
```bash
python tests/test_hmm.py    # 14 passed | 0 failed
```

---

## Outputs

```
outputs/
├── figures/
│   ├── fig1_regime_overlay.png        ← macro features shaded by regime
│   ├── fig2_transition_matrix.png     ← A matrix heatmap (regime persistence)
│   ├── fig3_factor_returns_by_regime.png  ← THE key chart: MOM/RMW by regime
│   ├── fig4_feature_boxplots.png      ← feature distributions by regime
│   ├── fig5_regime_probs.png          ← P(state | data) stacked area
│   ├── fig6_backtest.png              ← strategy vs equal-weight equity curve
│   └── fig7_nber_vs_hmm.png           ← recession detection accuracy
└── tables/
    ├── factor_returns_by_regime.csv
    ├── transition_matrix.csv
    ├── regime_sequence.csv
    └── backtest_metrics.csv
```

---

## References

| Reference | Applied In |
|-----------|-----------|
| Rabiner (1989) — *A Tutorial on HMMs* | Baum-Welch, Viterbi |
| Hamilton (1989) — *A New Approach to Nonstationary Time Series* | Regime-switching in finance |
| Bishop (2006) — *PRML* Ch. 13 | Forward-backward, EM |
| Barroso & Santa-Clara (2015) — *Momentum Has Its Moments* | Momentum crash in crisis |
| Daniel & Moskowitz (2016) — *Momentum Crashes* | Regime-conditional momentum |
| Qian (2004) — *Risk Parity Portfolios* (PanAgora) | Multi-asset regime context |
