"""Model-study supplementary diagnostics:
(a) mature-fold analysis: ΔBrier on the second half of each anchor's test months
    (midpoint split: >= 2024-07 for sched, >= 2024-08 for res; post-learning-curve)
(b) forecast-combination: OOS Brier of blends (1-l)*p + l*model, l grid
(c) per-horizon results for the price-only recalibrations, not just GBM
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = {}


def cluster_t(diff, clusters):
    g = pd.DataFrame({"v": diff, "c": clusters}).groupby("c")["v"].agg(["sum", "size"])
    n = g["size"].sum()
    mean = g["sum"].sum() / n
    se = np.sqrt(((g["sum"] - mean * g["size"]) ** 2).sum()) / n
    return float(mean), float(mean / se) if se > 0 else np.nan


def monthly_t(diff, months):
    dm = pd.DataFrame({"d": diff, "m": months}).groupby("m")["d"].mean()
    return float(dm.mean()), float(dm.mean() / (dm.std() / np.sqrt(len(dm)))) if dm.std() > 0 else np.nan


for anchor in ["sched", "res"]:
    P = pd.read_parquet(f"{ROOT}/results/models/predictions_{anchor}.parquet")
    lp = (P["p"] - P["y"]) ** 2
    A = {}

    # (a) mature folds only: second half of test months (midpoint split,
    # same rule as the honest-blend split below; equals 2024-07 for sched)
    months_all = sorted(P["snap_month"].unique())
    mature_cut = months_all[len(months_all) // 2]
    mature = P["snap_month"] >= mature_cut
    A["mature"] = {}
    for m in ["iso", "logit", "gbm_price", "gbm_full"]:
        lm = (P[f"pred_{m}"] - P["y"]) ** 2
        d = (lp - lm)[mature]
        mean, t = cluster_t(d.values, P.loc[mature, "event_id"].values)
        mmean, mt = monthly_t(d.values, P.loc[mature, "snap_month"].values)
        A["mature"][m] = {"n": int(mature.sum()), "delta_brier": mean, "t_event": t,
                          "monthly_mean": mmean, "t_monthly": mt}

    # (b) blend curves (descriptive, full OOS sample)
    A["blend"] = {}
    for m in ["logit", "gbm_full"]:
        pr = P[f"pred_{m}"]
        curve = {}
        for lam in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            b = (1 - lam) * P["p"] + lam * pr
            curve[str(lam)] = float(np.mean((b - P["y"]) ** 2))
        A["blend"][m] = curve
        # honest split-sample blend: choose lambda on first half of months, apply to second
        months = sorted(P["snap_month"].unique())
        half = months[len(months) // 2]
        first, second = P["snap_month"] < half, P["snap_month"] >= half
        lams = np.linspace(0, 1, 21)
        briers1 = [float(np.mean(((1 - l) * P.loc[first, "p"] + l * pr[first] - P.loc[first, "y"]) ** 2)) for l in lams]
        lstar = float(lams[int(np.argmin(briers1))])
        b2 = (1 - lstar) * P.loc[second, "p"] + lstar * pr[second]
        d2 = ((P.loc[second, "p"] - P.loc[second, "y"]) ** 2 - (b2 - P.loc[second, "y"]) ** 2)
        mean, t = cluster_t(d2.values, P.loc[second, "event_id"].values)
        mmean, mt = monthly_t(d2.values, P.loc[second, "snap_month"].values)
        A["blend"][m + "_honest"] = {"lambda_star": lstar, "n_second": int(second.sum()),
                                     "delta_brier": mean, "t_event": t, "t_monthly": mt}

    # (c) per-horizon for recalibrations
    A["by_horizon"] = {}
    for h in sorted(P["h"].unique()):
        s = P[P["h"] == h]
        row = {"n": int(len(s))}
        for m in ["iso", "logit", "gbm_full"]:
            d = ((s["p"] - s["y"]) ** 2 - (s[f"pred_{m}"] - s["y"]) ** 2)
            mean, t = cluster_t(d.values, s["event_id"].values)
            row[m] = {"delta_brier": mean, "t": t}
        A["by_horizon"][int(h)] = row

    out[anchor] = A

with open(f"{ROOT}/results/models/supplement.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

S = out["sched"]
print("== mature folds (>=2024-07, sched) ==")
for m, v in S["mature"].items():
    print(f"  {m:10s}: dBrier={1e4*v['delta_brier']:+.2f}e-4 t_ev={v['t_event']:.2f} t_mo={v['t_monthly']:.2f}")
print("\n== honest blends (sched) ==")
for m in ["logit_honest", "gbm_full_honest"]:
    v = S["blend"][m]
    print(f"  {m}: lambda*={v['lambda_star']:.2f} dBrier={1e4*v['delta_brier']:+.2f}e-4 t_ev={v['t_event']:.2f} t_mo={v['t_monthly']:.2f}")
print("\n== by horizon (sched, iso / logit / gbm_full dBrier x1e4) ==")
for h, v in S["by_horizon"].items():
    print(f"  h={h}: n={v['n']:,} iso={1e4*v['iso']['delta_brier']:+.2f}({v['iso']['t']:.1f}) "
          f"logit={1e4*v['logit']['delta_brier']:+.2f}({v['logit']['t']:.1f}) "
          f"gbm={1e4*v['gbm_full']['delta_brier']:+.2f}({v['gbm_full']['t']:.1f})")
