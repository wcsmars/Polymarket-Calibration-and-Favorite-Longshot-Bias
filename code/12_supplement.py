"""Supplemental retrospective analyses of held-out model forecasts.

Report second-half folds, descriptive blend curves, and a blend chosen using
only first-half forecasts whose closure-time proxy precedes the split. Blend
evaluation excludes those tuning events. Proxy timing is not verified outcome
availability; descriptive full-sample blend curves are not tuning results.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import write_json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def cluster_t(diff, clusters):
    if len(diff) == 0:
        return np.nan, np.nan
    frame = pd.DataFrame({"v": np.asarray(diff), "c": np.asarray(clusters)})
    if frame["c"].isna().any() or not np.isfinite(frame["v"]).all():
        raise ValueError("clustered inference requires finite values and event IDs")
    g = frame.groupby("c")["v"].agg(["sum", "size"])
    n = g["size"].sum()
    mean = g["sum"].sum() / n
    se = np.sqrt(((g["sum"] - mean * g["size"]) ** 2).sum()) / n if len(g) > 1 else np.nan
    return float(mean), float(mean / se) if se > 0 else np.nan


def monthly_t(diff, months):
    dm = pd.DataFrame({"d": np.asarray(diff), "m": np.asarray(months)}).groupby("m")["d"].mean()
    return float(dm.mean()), float(dm.mean() / (dm.std() / np.sqrt(len(dm)))) if dm.std() > 0 else np.nan


def blend_split(P):
    """Return tuning/evaluation masks without future labels or shared events."""
    months = sorted(P["snap_month"].unique())
    if len(months) < 2:
        raise ValueError("split-sample blend requires at least two test months")
    half = months[len(months) // 2]
    cutoff = pd.Timestamp(half + "-01", tz="UTC")
    resolved = pd.to_datetime(P["t_res"], utc=True) < cutoff
    first_months = P["snap_month"] < half
    first = first_months & resolved
    if "snap_ts" in P:
        first &= pd.to_numeric(P["snap_ts"], errors="coerce") < cutoff.timestamp()
    second_months = P["snap_month"] >= half
    overlap = P["event_id"].isin(P.loc[first, "event_id"])
    second = second_months & ~overlap
    metadata = {
        "cutoff_month": half, "n_tuning": int(first.sum()), "n_second": int(second.sum()),
        "n_tuning_events": int(P.loc[first, "event_id"].nunique()),
        "n_second_events": int(P.loc[second, "event_id"].nunique()),
        "n_excluded_unresolved_tuning": int((first_months & ~resolved).sum()),
        "n_excluded_event_overlap": int((second_months & overlap).sum()),
    }
    return first, second, metadata


def analyze_predictions(P):
    if P.empty:
        raise ValueError("cannot supplement an empty prediction sample")
    lp = (P["p"] - P["y"]) ** 2
    A = {}
    months = sorted(P["snap_month"].unique())
    mature_cut = months[len(months) // 2]
    mature = P["snap_month"] >= mature_cut
    A["mature_cutoff_month"] = mature_cut
    A["mature"] = {}
    for m in ["iso", "logit", "gbm_price", "gbm_full"]:
        lm = (P[f"pred_{m}"] - P["y"]) ** 2
        d = (lp - lm)[mature]
        mean, t = cluster_t(d.values, P.loc[mature, "event_id"].values)
        mmean, mt = monthly_t(d.values, P.loc[mature, "snap_month"].values)
        A["mature"][m] = {"n": int(mature.sum()), "delta_brier": mean, "t_event": t,
                          "monthly_mean": mmean, "t_monthly": mt}

    A["blend"] = {}
    first, second, split = blend_split(P)
    for m in ["logit", "gbm_full"]:
        pr = P[f"pred_{m}"]
        curve = {}
        for lam in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            b = (1 - lam) * P["p"] + lam * pr
            curve[str(lam)] = float(np.mean((b - P["y"]) ** 2))
        A["blend"][m] = curve
        if not first.any() or not second.any():
            A["blend"][m + "_honest"] = {
                **split, "status": "not_estimable", "reason": "Empty tuning or evaluation sample",
                "lambda_star": np.nan, "delta_brier": np.nan, "t_event": np.nan, "t_monthly": np.nan,
            }
            continue
        lams = np.linspace(0, 1, 21)
        briers1 = [float(np.mean(((1 - lam) * P.loc[first, "p"] + lam * pr[first]
                                 - P.loc[first, "y"]) ** 2)) for lam in lams]
        lstar = float(lams[int(np.argmin(briers1))])
        b2 = (1 - lstar) * P.loc[second, "p"] + lstar * pr[second]
        d2 = (lp[second] - (b2 - P.loc[second, "y"]) ** 2)
        mean, t = cluster_t(d2.values, P.loc[second, "event_id"].values)
        mmean, mt = monthly_t(d2.values, P.loc[second, "snap_month"].values)
        A["blend"][m + "_honest"] = {**split, "lambda_star": lstar,
                                     "delta_brier": mean, "t_event": t,
                                     "monthly_mean": mmean, "t_monthly": mt}

    A["by_horizon"] = {}
    for h in sorted(P["h"].unique()):
        s = P[P["h"] == h]
        row = {"n": int(len(s))}
        for m in ["iso", "logit", "gbm_full"]:
            d = ((s["p"] - s["y"]) ** 2 - (s[f"pred_{m}"] - s["y"]) ** 2)
            mean, t = cluster_t(d.values, s["event_id"].values)
            row[m] = {"delta_brier": mean, "t": t}
        A["by_horizon"][int(h)] = row
    return A


def main():
    out = {}
    for anchor in ["sched", "res"]:
        P = pd.read_parquet(f"{ROOT}/results/models/predictions_{anchor}.parquet")
        out[anchor] = analyze_predictions(P)
    write_json(ROOT / "results/models/supplement.json", out)
    S = out["sched"]
    print(f"== mature folds (>={S['mature_cutoff_month']}, sched) ==")
    for m, v in S["mature"].items():
        print(f"  {m:10s}: dBrier={1e4*v['delta_brier']:+.2f}e-4 t_ev={v['t_event']:.2f} t_mo={v['t_monthly']:.2f}")
    print("\n== time- and event-separated blends (sched) ==")
    for m in ["logit_honest", "gbm_full_honest"]:
        v = S["blend"][m]
        print(f"  {m}: lambda*={v['lambda_star']:.2f} dBrier={1e4*v['delta_brier']:+.2f}e-4 t_ev={v['t_event']:.2f} t_mo={v['t_monthly']:.2f}")
    print("\n== by horizon (sched, iso / logit / gbm_full dBrier x1e4) ==")
    for h, v in S["by_horizon"].items():
        print(f"  h={h}: n={v['n']:,} iso={1e4*v['iso']['delta_brier']:+.2f}({v['iso']['t']:.1f}) "
              f"logit={1e4*v['logit']['delta_brier']:+.2f}({v['logit']['t']:.1f}) "
              f"gbm={1e4*v['gbm_full']['delta_brier']:+.2f}({v['gbm_full']['t']:.1f})")


if __name__ == "__main__":
    main()
