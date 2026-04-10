#!/usr/bin/env python3
"""
run_regime.py  ·  macro-regime-hmm
=====================================
Macro Regime HMM Pipeline.

Zero API keys — FRED free CSV URLs + yfinance.

Pipeline
--------
1. Download macro features: yield curve, HY spread, VIX, unemployment, SPY vol
2. Standardise features (expanding window for walk-forward, full-sample for visualisation)
3. Fit K=3 Gaussian HMM via Baum-Welch EM (from scratch, no hmmlearn)
4. Decode regime sequence via Viterbi
5. Walk-forward out-of-sample regime prediction
6. Compute regime-conditional factor returns (Expansion / Slowdown / Crisis)
7. Backtest regime-tilted factor portfolio vs equal-weight benchmark
8. Evaluate against NBER recession dates
9. Save 7 publication-quality figures + CSV tables

Usage
-----
    python run_regime.py                   # full run with live data
    python run_regime.py --offline         # synthetic data, no internet
    python run_regime.py --states 2        # 2-state expansion/recession model
    python run_regime.py --start 2005-01-01 --end 2024-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("regime_pipeline")
OUT = Path("outputs")


def main(args: argparse.Namespace) -> None:
    sys.path.insert(0, "src")
    from regime.data     import build_feature_panel, get_ff_factor_returns
    from regime.hmm      import GaussianHMM
    from regime.features import (standardise, fit_hmm_on_panel,
                                  walk_forward_regimes, regime_factor_table,
                                  FEATURE_COLS, REGIME_NAMES)
    from regime.backtest import run_backtest, regime_eval
    from regime.charts   import save_all

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)

    logger.info("=" * 64)
    logger.info("  Macro Regime HMM — K=%d states", args.states)
    logger.info("  Data: FRED free CSV + yfinance")
    logger.info("=" * 64)

    # ── 1. Data ───────────────────────────────────────────────────────
    panel = build_feature_panel(
        start=args.start, end=args.end, offline=args.offline)
    ff    = get_ff_factor_returns(
        start=args.start, offline=args.offline)

    logger.info("Panel: %d months | FF factors: %d months",
                len(panel), len(ff))

    # ── 2. Fit HMM on full sample (for visualisation) ─────────────────
    logger.info("Fitting Gaussian HMM (K=%d, Baum-Welch EM)...", args.states)
    hmm, states_full = fit_hmm_on_panel(panel, n_states=args.states)
    result = hmm.result

    # Summarise fitted parameters
    _print_regime_summary(result, FEATURE_COLS, REGIME_NAMES)

    # ── 3. Walk-forward regime prediction ─────────────────────────────
    logger.info("Running walk-forward regime prediction (train=%d months)...",
                args.train_size)
    wf_regimes = walk_forward_regimes(
        panel, n_states=args.states, train_size=args.train_size)

    # For figures, use full-sample Viterbi states (labelled)
    feature_cols_avail = [c for c in FEATURE_COLS if c in panel.columns]
    regime_series_full = pd.Series(
        states_full, index=panel.dropna(subset=feature_cols_avail).index[:len(states_full)],
        name="regime",
    )

    # ── 4. Regime-conditional factor returns ──────────────────────────
    logger.info("Computing regime-conditional factor returns...")
    factor_tbl = regime_factor_table(
        regime_series_full, ff, n_states=args.states)

    print("\n┌─ Regime-Conditional Factor Returns (annualised) ──────────")
    print(factor_tbl.round(4).to_string())
    print("└───────────────────────────────────────────────────────────\n")

    # ── 5. Backtest ────────────────────────────────────────────────────
    logger.info("Running regime-conditional factor backtest...")
    bt = run_backtest(wf_regimes, ff, tc_bps=args.tc_bps)

    print("┌─ Backtest Metrics ────────────────────────────────────────")
    m = bt["metrics"]
    print(f"  {'':25} {'Strategy':>12} {'Benchmark':>12}")
    for k in ["CAGR", "Vol", "Sharpe", "MaxDD"]:
        print(f"  {k:25} {m['Strategy'][k]:12.4f} {m['Benchmark'][k]:12.4f}")
    print(f"  {'Active Return':25} {m['Active']['Active Return']:12.4f}")
    print(f"  {'Info Ratio':25} {str(m['Active']['Info Ratio']):>12}")
    print("└───────────────────────────────────────────────────────────\n")

    # ── 6. NBER evaluation ─────────────────────────────────────────────
    if "nber_rec" in panel.columns:
        eval_metrics = regime_eval(wf_regimes, panel["nber_rec"])
        print("┌─ NBER Recession vs HMM Crisis ─────────────────────────────")
        for k, v in eval_metrics.items():
            print(f"  {k:<28}: {v}")
        print("└───────────────────────────────────────────────────────────\n")
    else:
        eval_metrics = {}

    # ── 7. Save outputs ────────────────────────────────────────────────
    logger.info("Generating figures...")
    save_all(panel, result, regime_series_full, factor_tbl, bt, OUT)

    tbl = OUT / "tables"
    factor_tbl.round(4).to_csv(tbl / "factor_returns_by_regime.csv")
    regime_series_full.to_csv(tbl / "regime_sequence.csv", header=True)
    wf_regimes.to_csv(tbl / "wf_regime_predictions.csv", header=True)
    result.params.A.round(4).view(type=np.ndarray)
    pd.DataFrame(result.params.A, columns=[REGIME_NAMES.get(k, k) for k in range(args.states)],
                 index=[REGIME_NAMES.get(k, k) for k in range(args.states)]
                 ).to_csv(tbl / "transition_matrix.csv")
    pd.Series(bt["metrics"]["Strategy"]).to_csv(tbl / "backtest_metrics.csv")
    if eval_metrics:
        pd.Series(eval_metrics).to_csv(tbl / "nber_eval.csv")

    logger.info("=" * 64)
    logger.info("  Complete  |  Outputs: %s/", OUT)
    logger.info("=" * 64)


def _print_regime_summary(result, feature_cols, regime_names):
    K   = result.n_states
    avail_feats = feature_cols[:min(len(feature_cols), result.params.mu.shape[1])]

    print("\n┌─ HMM Regime Parameters ────────────────────────────────────")
    print(f"  {'Feature':<22}", end="")
    for k in range(K):
        print(f"  {regime_names.get(k, f'State {k}'):>14}", end="")
    print()
    print("  " + "─" * (22 + K * 16))

    for i, feat in enumerate(avail_feats):
        print(f"  {feat:<22}", end="")
        for k in range(K):
            print(f"  {result.params.mu[k, i]:14.4f}", end="")
        print()

    print()
    n_total = result.n_obs
    for k in range(K):
        n_k = int((result.state_sequence == k).sum())
        name = regime_names.get(k, f"State {k}")
        persist = result.params.A[k, k]
        print(f"  {name:<14}: {n_k:4d} months ({100*n_k/n_total:.1f}%)  "
              f"persistence={persist:.3f}")
    print("  LL convergence: " +
          " → ".join(f"{v:.1f}" for v in result.log_likelihoods[::max(1, len(result.log_likelihoods)//5)]))
    print(f"  Converged: {result.converged}")
    print("└───────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Macro Regime HMM Pipeline")
    p.add_argument("--states",     type=int,   default=3)
    p.add_argument("--start",      default="2000-01-01")
    p.add_argument("--end",        default=None)
    p.add_argument("--train-size", type=int,   default=60,
                   help="Initial training window for walk-forward (months)")
    p.add_argument("--tc-bps",     type=float, default=5.0,
                   help="One-way transaction cost (bps) for factor backtest")
    p.add_argument("--offline",    action="store_true",
                   help="Use synthetic data — no internet required")
    main(p.parse_args())
