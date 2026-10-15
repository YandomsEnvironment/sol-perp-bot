"""Risk controls: effective sample size, bootstrap CIs, regime stratification,
cost-sensitivity, and multiple-testing accounting for the deployed bundles.

Usage: python scripts/risk_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features as F  # noqa: E402
import labeling as LAB  # noqa: E402
import train as T  # noqa: E402
import validation as V  # noqa: E402
from experiment import build_dataset, feature_columns  # noqa: E402

CUTOFF = pd.Timestamp("2026-09-01", tz="UTC")


def block_bootstrap(returns: np.ndarray, block: int = 5, n_boot: int = 2000,
                    seed: int = 7) -> dict:
    """Moving-block bootstrap for mean return and profit factor."""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    if len(r) < block * 2:
        return {}
    starts = np.arange(len(r) - block + 1)
    means, pfs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block)
                              for s in rng.choice(starts, size=len(r) // block + 1)])
        idx = idx[idx < len(r)]
        sample = r[idx]
        means.append(sample.mean())
        wins, losses = sample[sample > 0].sum(), -sample[sample < 0].sum()
        pfs.append(wins / losses if losses > 0 else np.inf)
    lo, hi = np.percentile(means, [2.5, 97.5])
    pf_lo, pf_hi = np.percentile(pfs, [2.5, 97.5])
    return {"n": int(len(r)), "mean_ci95": [float(lo), float(hi)],
            "pf_ci95": [float(pf_lo), float(pf_hi)]}


def regime_tags(feats: pd.DataFrame) -> pd.Series:
    """Causal regime tag from ret_1d and ATR rank over the trailing 30 days."""
    r1d = feats["ret_1d"]
    atr_rank = feats["atr_pct"].rolling(4320, min_periods=1440).rank(pct=True)
    trend = np.where(r1d > 0.005, "bull", np.where(r1d < -0.005, "bear", "chop"))
    vol = np.where(atr_rank >= 2 / 3, "hi-vol",
                   np.where(atr_rank <= 1 / 3, "lo-vol", "mid-vol"))
    return pd.Series([f"{t}/{v}" for t, v in zip(trend, vol)], index=feats.index)

def main():
    data = T.load_data()
    candles = T._indexed(data["sol"])
    mapping = json.loads((ROOT / "research_models.json").read_text())

    # ---- Risk 4: effective sample size ----
    print("== EFFECTIVE SAMPLE SIZE ==")
    close = candles["close"]
    for h in (5, 15, 60):
        n = len(close) - h
        print(f"h={h:2d}m: rows={n:,} -> independent label windows ~{n / h:,.0f} "
              f"(x{h} overlap deflation)")

    # ---- Multiple-testing count (Risk 1) ----
    ledger = ROOT / "logs" / "experiments.jsonl"
    names = set()
    if ledger.exists():
        for line in ledger.open(encoding="utf-8"):
            names.add(json.loads(line)["name"])
    print(f"\n== MULTIPLE TESTING ==\n{len(names)} distinct configurations "
          f"evaluated on the same 2 selection folds; treat single-cell "
          f"improvements below ~{1 / len(names):.1%} (1/n) as noise.")

    for key in ("fast", "mid", "slow"):
        bundle = joblib.load(ROOT / "models" / mapping[key])
        h = bundle["horizon_min"]
        dead = float(bundle.get("target_deadband") or config.ROUND_TRIP_COST)
        version = "v4" if any(c in bundle["feature_names"] for c in F.V4_COLUMNS) else "v3"
        ds, _ = build_dataset(data, h, dead)
        X = F.prepare_model_input(ds[feature_columns(ds, version)],
                                  bundle["feature_names"])
        te = X.index[X.index >= CUTOFF]
        p = np.asarray(bundle["model"].predict_proba(X.loc[te]), dtype=float)
        rule = bundle.get("decisive_rule") or {"tau": 0.45, "delta": 0.10}
        pred = V.decisive_classes(p, tau=rule["tau"], delta=rule["delta"])
        te_atr = ds.loc[te, "atr_pct"].to_numpy()
        tags = regime_tags(ds.loc[te])

        print(f"\n== {key.upper()} (h={h}m) holdout risk report ==")
        print("cost curve (PF at round-trip cost):")
        prev_pf, be = None, None
        for cost_bp in (8, 10, 13, 16, 20):
            r = T._path(candles, te, pred, te_atr, h, cost=cost_bp / 10_000,
                        barrier_scale=h ** 0.5)
            taken = (pred != 0) & np.isfinite(r)
            pf = V.profit_factor(r[taken]) if taken.sum() else float("nan")
            if prev_pf is not None and prev_pf >= 1.0 > pf:
                be = cost_bp - 3
            prev_pf = pf
            print(f"   {cost_bp:2d}bp: pf={pf:.2f} (n={int(taken.sum())})")
        print(f"   breakeven cost: ~{be}bp  (funding adds ~0.1-0.2bp/hour held)")
        r = T._path(candles, te, pred, te_atr, h, cost=config.ROUND_TRIP_COST,
                    barrier_scale=h ** 0.5)
        taken = (pred != 0) & np.isfinite(r)
        boot = block_bootstrap(r[taken], block=max(3, h // 10))
        if boot:
            sig = "SIGNIFICANT" if boot["mean_ci95"][0] > 0 else "not significant"
            print(f"bootstrap: mean/trade CI95 {boot['mean_ci95']} -> {sig}; "
                  f"PF CI95 [{boot['pf_ci95'][0]:.2f}, {boot['pf_ci95'][1]:.2f}]")
        print("regime breakdown (taker cost):")
        rows = pd.DataFrame({"tag": tags[taken].values, "r": r[taken]})
        for tag, grp in rows.groupby("tag"):
            if len(grp) >= 3:
                print(f"   {tag:12s}: n={len(grp):3d} win={np.mean(grp.r > 0):.2f} "
                      f"pf={V.profit_factor(grp.r):.2f}")
        # Calibration of decisive confidence on holdout calls
        conf = np.maximum(p[:, 1], p[:, 2])[taken]
        won = (r[taken] > 0).astype(int)
        cal = pd.DataFrame({"claimed": conf, "won": won})
        buckets = [(0.40, 0.45), (0.45, 0.50), (0.50, 0.60), (0.60, 1.01)]
        cal_rows = []
        for lo, hi in buckets:
            m = (cal.claimed >= lo) & (cal.claimed < hi)
            if int(m.sum()) >= 5:
                cal_rows.append(f"{lo:.0%}-{hi:.0%}: n={int(m.sum()):3d} "
                                f"claimed={cal.loc[m,'claimed'].mean():.2f} "
                                f"realized={cal.loc[m,'won'].mean():.2f}")
        print("calibration (decisive calls): " + (" | ".join(cal_rows) or "insufficient n"))
        tr_tags = regime_tags(ds.loc[ds.index < CUTOFF])
        mix = tr_tags.str.split("/").str[0].value_counts(normalize=True)
        vol_mix = tr_tags.str.split("/").str[1].value_counts(normalize=True)
        print(f"training regime mix: trend={dict(mix.round(2))} vol={dict(vol_mix.round(2))}")


if __name__ == "__main__":
    main()

