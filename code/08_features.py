"""Model study: build the ML dataset (one row per market x horizon x anchoring).

Feature timing rules:
  - Snapshot tau = (scheduled end - h days) for anchor='sched' (primary) or
    (closure-time proxy - h days) for anchor='res' (robustness).
  - Standing price at tau: latest observation <= tau, staleness <= 1.5 days.
  - Path features use ONLY observations with t <= tau.
  - No lifetime volume / current liquidity (post-snapshot info).
The closure-time proxy t_res uses closedTime, falling back to endDate.
Snapshots must precede this proxy and not precede market creation. Metadata
is collected retrospectively; see the README timing limitations.
Output: data/processed/ml_dataset.parquet
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import load_histories, parse_flag

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 7, 14, 30, 60, 90]
STALE_TOL = 1.5 * 86400
DAY = 86400


def path_features(ts, ps, tau):
    """Features from observations strictly at/before tau."""
    idx = np.searchsorted(ts, tau, side="right")
    if idx < 2:
        return None
    t, p = ts[:idx], ps[:idx]
    if tau - t[-1] > STALE_TOL:
        return None
    dp = np.diff(p)
    last7 = p[t >= tau - 7 * DAY]
    dp7 = np.diff(last7) if len(last7) > 1 else np.array([0.0])
    p_now = p[-1]

    def pk(days):
        """price 'days' before tau (standing), nan if unavailable"""
        j = np.searchsorted(t, tau - days * DAY, side="right") - 1
        return p[j] if j >= 0 else np.nan

    p7, p30 = pk(7), pk(30)
    return {
        "p": float(np.clip(p_now, 0.001, 0.999)),
        "n_obs_pre": idx,
        "days_live": (tau - t[0]) / DAY,
        "p_start": p[0],
        "p_mean_pre": float(p.mean()),
        "p_max_pre": float(p.max()),
        "p_min_pre": float(p.min()),
        "p_range_pre": float(p.max() - p.min()),
        "chg_7": float(p_now - p7) if np.isfinite(p7) else 0.0,
        "chg_30": float(p_now - p30) if np.isfinite(p30) else 0.0,
        "vol_all": float(dp.std()) if len(dp) > 1 else 0.0,
        "vol_7": float(dp7.std()) if len(dp7) > 1 else 0.0,
        "absmove_7": float(np.abs(dp7).mean()) if len(dp7) else 0.0,
        "active_frac": float((np.abs(dp) > 1e-4).mean()) if len(dp) else 0.0,
        "dist_to_half": abs(p_now - 0.5),
    }


def main():
    sample = pd.read_csv(f"{ROOT}/data/processed/sample_markets.csv",
                         dtype={"id": str, "yes_token": str, "event_id": str})
    for c in ["t_res", "t_created", "t_end"]:
        sample[c] = pd.to_datetime(sample[c], utc=True, format="mixed")
    sample["event_id"] = sample["event_id"].fillna("mkt_" + sample["id"])

    hist = load_histories(ROOT / "data/raw/price_histories.jsonl")
    # Event size is observable only as siblings appear. The final event's
    # market count would leak markets created after the forecast snapshot.
    event_available = {}
    for market in sample.itertuples(index=False):
        available = market.t_created.timestamp() if pd.notna(market.t_created) else np.nan
        history = hist.get(market.id)
        if not np.isfinite(available) and history is not None and len(history):
            available = history[0, 0]
        if np.isfinite(available):
            event_available.setdefault(market.event_id, []).append(available)
    event_available = {event: np.sort(times) for event, times in event_available.items()}

    rows = []
    for r in sample.itertuples(index=False):
        h_arr = hist.get(r.id)
        if h_arr is None or len(h_arr) < 2:
            continue
        ts, ps = h_arr[:, 0], h_arr[:, 1]
        t_res = r.t_res.timestamp()
        t_end = r.t_end.timestamp() if pd.notna(r.t_end) else np.nan
        t_created = r.t_created.timestamp() if pd.notna(r.t_created) else ts[0]
        # Discard quotes dated before creation, even if a malformed history
        # includes them. They cannot be part of this market's information set.
        usable = ts >= t_created
        ts, ps = ts[usable], ps[usable]
        q = (r.question or "") if isinstance(r.question, str) else ""
        ql = q.lower()
        # A stale scheduled date can predate market creation. Such metadata
        # cannot define a lifetime or its elapsed fraction, even for res rows.
        sched_life = (t_end - t_created) / DAY if np.isfinite(t_end) and t_end > t_created else np.nan
        static = {
            "id": r.id, "event_id": r.event_id, "y": int(r.y), "cat": r.cat,
            "neg_risk": parse_flag(r.event_neg_risk) or parse_flag(r.negRisk),
            "sched_life_days": sched_life,
            "q_len_words": len(q.split()),
            "q_has_by": int(" by " in ql or ql.startswith("by ")),
            "q_starts_will": int(ql.startswith("will")),
            "q_has_digit": int(any(ch.isdigit() for ch in q)),
            "question": q,
            "t_res": r.t_res,
        }
        for anchor, t0 in [("sched", t_end), ("res", t_res)]:
            if not np.isfinite(t0):
                continue
            for h in HORIZONS:
                tau = t0 - h * DAY
                if not (t_created <= tau < t_res):
                    continue
                feats = path_features(ts, ps, tau)
                if feats is None:
                    continue
                rows.append({
                    **static, **feats, "anchor": anchor, "h": h,
                    "event_n_markets": int(np.searchsorted(
                        event_available[r.event_id], tau, side="right")),
                    "snap_ts": tau,
                    "frac_life": feats["days_live"] / static["sched_life_days"]
                    if np.isfinite(static["sched_life_days"]) else np.nan,
                })

    if not rows:
        raise ValueError("No usable forecast snapshots; inspect sample dates and price histories")
    df = pd.DataFrame(rows)
    df["snap_month"] = pd.to_datetime(df["snap_ts"], unit="s", utc=True).dt.to_period("M").astype(str)
    df["logit_p"] = np.log(df["p"] / (1 - df["p"]))
    df.to_parquet(f"{ROOT}/data/processed/ml_dataset.parquet", index=False)
    print(f"rows: {len(df)}")
    print(df.groupby(["anchor", "h"])["id"].count().unstack(0).to_string())
    print("\nsnap_month range:", df["snap_month"].min(), "->", df["snap_month"].max())


if __name__ == "__main__":
    main()
