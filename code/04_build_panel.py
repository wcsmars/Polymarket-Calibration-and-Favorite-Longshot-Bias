"""Build the market x horizon analysis panel.

For each sampled market, the standing YES price at each horizon h days
before the closure-time proxy is the latest daily observation at or before
(t_res - h days), required to be no staler than 1.5 days. The proxy uses
closedTime, falling back to endDate. Also extracts a midlife price and flags
markets whose closure-time proxy is within two days of their scheduled end.

Output: data/processed/panel.csv (long: one row per market-horizon) and
data/processed/market_level.csv (one row per market with extras).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import load_histories

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 7, 14, 30, 60, 90]
STALE_TOL = 1.5 * 86400  # max staleness of the standing price, seconds
DAY = 86400


def main():
    sample = pd.read_csv(f"{ROOT}/data/processed/sample_markets.csv",
                         dtype={"id": str, "yes_token": str, "event_id": str})
    sample["t_res"] = pd.to_datetime(sample["t_res"], utc=True, format="mixed")
    sample["t_created"] = pd.to_datetime(sample["t_created"], utc=True, format="mixed")
    sample["t_end"] = pd.to_datetime(sample["t_end"], utc=True, format="mixed")

    hist = load_histories(ROOT / "data/raw/price_histories.jsonl")

    rows = []
    mkt_rows = []
    for r in sample.itertuples(index=False):
        h_arr = hist.get(r.id)
        if h_arr is None or len(h_arr) < 2:
            continue
        ts, ps = h_arr[:, 0], h_arr[:, 1]
        t_res = r.t_res.timestamp()
        t_created = r.t_created.timestamp() if pd.notna(r.t_created) else ts[0]
        t_end = r.t_end.timestamp() if pd.notna(r.t_end) else np.nan
        ran_to_schedule = bool(np.isfinite(t_end) and abs(t_res - t_end) <= 2 * DAY)
        life_days = (t_res - ts[0]) / DAY

        def standing_price(target):
            """Latest observation at or before target, if fresh enough."""
            if not (t_created <= target < t_res):
                return np.nan
            idx = np.searchsorted(ts, target, side="right") - 1
            if idx < 0 or ts[idx] < t_created:
                return np.nan
            if target - ts[idx] > STALE_TOL:
                return np.nan
            return ps[idx]

        for h in HORIZONS:
            p = standing_price(t_res - h * DAY)
            if np.isfinite(p):
                rows.append((r.id, h, float(np.clip(p, 0.001, 0.999))))

        # robustness prices
        p_mid = standing_price(t_created + 0.5 * (t_res - t_created))
        # schedule-anchored standing prices: quote h days before the SCHEDULED
        # end, using a recent observation at or before the anchor. These
        # retrospectively collected timestamps do not establish metadata
        # availability at the original forecast date. Every anchor must precede
        # the closure-time proxy and must not precede market creation.
        p_sched = {}
        if np.isfinite(t_end):
            for h in HORIZONS:
                v = standing_price(t_end - h * DAY)
                p_sched[h] = float(np.clip(v, 0.001, 0.999)) if np.isfinite(v) else np.nan
        mkt_rows.append({
            "id": r.id, "life_days": life_days, "ran_to_schedule": ran_to_schedule,
            "p_mid": float(np.clip(p_mid, 0.001, 0.999)) if np.isfinite(p_mid) else np.nan,
            **{f"p_sched_{h}": p_sched.get(h, np.nan) for h in HORIZONS},
        })

    panel = pd.DataFrame(rows, columns=["id", "h", "p"])
    panel = panel.merge(
        sample[["id", "y", "cat", "volumeNum", "event_id", "event_neg_risk",
                "negRisk", "t_res", "question"]], on="id", how="left")
    panel["era"] = pd.cut(panel["t_res"].dt.year, bins=[0, 2023, 2024, 9999],
                          labels=["2022-23", "2024", "2025-26"])
    panel.to_csv(f"{ROOT}/data/processed/panel.csv", index=False)

    mkt = pd.DataFrame(mkt_rows, columns=[
        "id", "life_days", "ran_to_schedule", "p_mid",
        *[f"p_sched_{h}" for h in HORIZONS],
    ]).merge(
        sample[["id", "y", "cat", "volumeNum", "event_id", "t_res"]], on="id", how="left")
    mkt.to_csv(f"{ROOT}/data/processed/market_level.csv", index=False)

    print(f"panel rows: {len(panel)} | markets with history: {len(mkt)}")
    print(panel.groupby("h")["id"].count().rename("n_markets").to_string())


if __name__ == "__main__":
    main()
