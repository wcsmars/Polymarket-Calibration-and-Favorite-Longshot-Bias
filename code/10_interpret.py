"""Model study: interpretation, where-analysis, and divergence backtest.

All analyses use out-of-sample predictions from 09 (schedule-anchored primary).
The interpretation model is trained on pre-2025 data and examined on 2025 data.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import write_json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

from importlib.machinery import SourceFileLoader
m09 = SourceFileLoader("m09", f"{ROOT}/code/09_model.py").load_module()
m05 = SourceFileLoader("m05", f"{ROOT}/code/05_analysis.py").load_module()


def cluster_mean_se(vals, clusters):
    frame = pd.DataFrame({"value": np.asarray(vals), "event_id": np.asarray(clusters)})
    mean, se, n, _ = m05.cluster_mean_se(frame, "value")
    return mean, se, n


def main():
    rng = np.random.default_rng(SEED)
    P = pd.read_parquet(f"{ROOT}/results/models/predictions_sched.parquet")
    out = {}

    # ---------- divergence backtest ----------
    P["div"] = P["pred_gbm_full"] - P["p"]
    bt = {}
    for theta in [0.02, 0.05, 0.10]:
        for cost, ck in [(0.0, "gross"), (0.01, "net_1c")]:
            sel = P[np.abs(P["div"]) > theta].copy()
            long_yes = sel["div"] > 0
            ret = np.where(long_yes,
                           sel["y"] / (sel["p"] + cost) - 1,
                           (1 - sel["y"]) / ((1 - sel["p"]) + cost) - 1)
            mean, se, n = cluster_mean_se(ret, sel["event_id"].values)
            monthly = pd.DataFrame({"r": ret, "m": sel["snap_month"].values}).groupby("m")["r"].mean()
            mt = monthly.mean() / (monthly.std() / np.sqrt(len(monthly))) if monthly.std() > 0 else np.nan
            bt[f"theta{theta}_{ck}"] = {
                "n_trades": n, "mean_ret": mean, "se_clust": se,
                "t_clust": mean / se if se > 0 else np.nan,
                "win_rate": float((ret > 0).mean()),
                "monthly_mean": float(monthly.mean()), "monthly_t": float(mt),
                "n_months": int(len(monthly)),
            }
    out["backtest"] = bt

    # ---------- divergence deciles: does divergence predict direction? ----------
    P["absdiv"] = np.abs(P["div"])
    P["div_dec"] = pd.qcut(P["absdiv"], 10, labels=False, duplicates="drop")
    dd = []
    for dec, s in P.groupby("div_dec"):
        edge = np.sign(s["div"]) * (s["y"] - s["p"])
        mean, se, n = cluster_mean_se(edge.values, s["event_id"].values)
        dd.append({"decile": int(dec), "n": n, "mean_absdiv": float(s["absdiv"].mean()),
                   "directional_edge_pp": 100 * mean, "se_pp": 100 * se})
    out["divergence_deciles"] = dd

    # ---------- where does the model win? ----------
    P["dloss"] = (P["p"] - P["y"]) ** 2 - (P["pred_gbm_full"] - P["y"]) ** 2
    where = {}
    for key, grp in [("category", P["cat"]),
                     ("p_bucket", pd.cut(P["p"], [0, .1, .9, 1.000001], right=False, labels=["longshot", "mid", "favorite"])),
                     ("activity_tercile", m05.quantile_groups(P["active_frac"]))]:
        where[key] = {}
        for g, s in P.groupby(grp, observed=True):
            mean, se, n = cluster_mean_se(P.loc[s.index, "dloss"].values, s["event_id"].values)
            where[key][str(g)] = {"n": n, "delta_brier": mean, "t": mean / se if se > 0 else np.nan}
    out["where"] = where
    monthly_d = P.groupby("snap_month")["dloss"].mean()
    out["monthly_delta_brier"] = {k: float(v) for k, v in monthly_d.items()}

    # ---------- interpretation model (train <2025, test >=2025) ----------
    df = pd.read_parquet(f"{ROOT}/data/processed/ml_dataset.parquet")
    d = df[df["anchor"] == "sched"].copy()
    d["t_res"] = pd.to_datetime(d["t_res"], utc=True)
    d = m09.eligible_snapshots(d)
    cat_dum = pd.get_dummies(d["cat"], prefix="cat")
    d = pd.concat([d, cat_dum], axis=1)
    cat_cols = list(cat_dum.columns)
    months = sorted(d["snap_month"].unique())
    cut = pd.Timestamp("2025-01-01", tz="UTC")
    if len(months) <= m09.BURN_IN_MONTHS:
        raise ValueError("interpretation requires more months than the model burn-in")
    burn_cut = min(pd.Timestamp(months[m09.BURN_IN_MONTHS] + "-01", tz="UTC"), cut)
    burn = d[m09.training_eligible(d, burn_cut)]
    d, txt_cols = m09.add_text_features(d, burn)

    FULL_ALL = m09.PRICE_FEATS + m09.PATH_FEATS + m09.STRUCT_FEATS + cat_cols + txt_cols
    d[FULL_ALL] = d[FULL_ALL].astype(np.float64)

    cut = pd.Timestamp("2025-01-01", tz="UTC")
    train = d[m09.training_eligible(d, cut)]
    test = d[(pd.to_datetime(d["snap_month"] + "-01", utc=True) >= cut)
             & ~d["event_id"].isin(set(train["event_id"]))]
    FULL = m09.PRICE_FEATS + m09.PATH_FEATS + m09.STRUCT_FEATS + cat_cols + txt_cols
    if train.empty or test.empty:
        raise ValueError("interpretation requires eligible pre-2025 training and 2025+ test events")
    model = m09.fit_gbm_model(train[FULL].reset_index(drop=True), train["y"].reset_index(drop=True),
                              train["snap_ts"].values, train["event_id"].values)

    base_pred = model.predict_proba(test[FULL])[:, 1]
    base_brier = float(np.mean((base_pred - test["y"]) ** 2))
    out["interp_model"] = {"n_train": len(train), "n_test": len(test),
                           "test_brier_model": base_brier,
                           "test_brier_price": float(np.mean((test["p"] - test["y"]) ** 2))}

    # group permutation importance (price cols permuted jointly, etc.)
    GROUPS = {"price(p,logit_p)": ["p", "logit_p"], "horizon": ["h"],
              "path": m09.PATH_FEATS, "structure": m09.STRUCT_FEATS,
              "category": cat_cols, "text_svd": txt_cols}
    gpi = {}
    Xt = test[FULL].reset_index(drop=True)
    yte = test["y"].values
    for gname, cols in GROUPS.items():
        deltas = []
        for _ in range(5):
            Xp = Xt.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].values[perm]
            pb = model.predict_proba(Xp)[:, 1]
            deltas.append(float(np.mean((pb - yte) ** 2)) - base_brier)
        gpi[gname] = {"mean_dbrier": float(np.mean(deltas)), "sd": float(np.std(deltas, ddof=1)), "n_repeats": len(deltas)}
    out["group_permutation_importance"] = gpi

    # individual permutation importance for top non-price features
    ind = {}
    for col in m09.PATH_FEATS + m09.STRUCT_FEATS:
        Xp = Xt.copy()
        Xp[col] = Xp[col].values[rng.permutation(len(Xp))]
        pb = model.predict_proba(Xp)[:, 1]
        ind[col] = float(np.mean((pb - yte) ** 2)) - base_brier
    out["indiv_permutation_importance"] = dict(
        sorted(ind.items(), key=lambda kv: -kv[1])[:12])

    # learned price-correction curve from the price-only model, per horizon
    price_model = m09.fit_gbm_model(train[m09.PRICE_FEATS].reset_index(drop=True),
                                    train["y"].reset_index(drop=True), train["snap_ts"].values,
                                    train["event_id"].values)

    grid = np.linspace(0.01, 0.99, 99)
    curves = {}
    for h in [1, 7, 30, 90]:
        Xg = pd.DataFrame({"p": grid, "logit_p": np.log(grid / (1 - grid)), "h": h})
        curves[f"h{h}"] = price_model.predict_proba(Xg[m09.PRICE_FEATS])[:, 1].tolist()
    out["correction_curves"] = {"grid": grid.tolist(), **curves}

    # PDPs (manual, vary one feature over sample)
    pdps = {}
    samp = Xt.sample(min(3000, len(Xt)), random_state=SEED)
    for feat, lo, hi in [("chg_30", -0.3, 0.3), ("active_frac", 0, 1),
                         ("event_n_markets", 1, 30), ("frac_life", 0.05, 1.0)]:
        gridf = np.linspace(lo, hi, 25)
        vals = []
        for v in gridf:
            Xp = samp.copy()
            Xp[feat] = v
            vals.append(float(model.predict_proba(Xp)[:, 1].mean()))
        pdps[feat] = {"grid": gridf.tolist(), "pdp": vals}
    out["pdps"] = pdps

    write_json(ROOT / "results/models/interpretation.json", out)
    print("backtest theta=0.05:", json.dumps(bt.get("theta0.05_net_1c", bt), indent=2)[:400])
    print("\ngroup importance:", json.dumps(gpi, indent=2))
    print("\nDONE -> results/models/interpretation.json")


if __name__ == "__main__":
    main()
