"""Model study: retrospective walk-forward modeling. For each test month m, train
on rows whose t_res closure-time proxy precedes the start of m, predict rows
with snapshots in m (excluding rows whose event appears in training), and
compare a ladder of models against the market price.

t_res uses closedTime, falling back to endDate; it does not verify when the
outcome became available. Training does not separately require snapshots to
precede the test month, and scheduled test snapshots can follow the proxy.
See the README limitations before interpreting these as point-in-time results.

  price      : the market price itself
  logit      : logistic regression y ~ logit(p)          (calibration-study correction)
  iso        : isotonic recalibration of p               (nonparametric, price-only)
  gbm_price  : LightGBM on (p, logit_p, h)               (price recalibration by horizon)
  gbm_path   : + price-path features
  gbm_notext : + structure features (no text)
  gbm_full   : + text features (TF-IDF/SVD fit on burn-in training data only)

Saves per-row out-of-sample predictions to results/models/predictions_{anchor}.parquet
and pooled and per-horizon metrics to results/models/model_metrics.json (per-fold
train/test sizes are printed only).
"""
import json
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BURN_IN_MONTHS = 6
N_SVD = 20

PATH_FEATS = ["n_obs_pre", "days_live", "p_start", "p_mean_pre", "p_max_pre",
              "p_min_pre", "p_range_pre", "chg_7", "chg_30", "vol_all", "vol_7",
              "absmove_7", "active_frac", "dist_to_half", "frac_life"]
STRUCT_FEATS = ["neg_risk", "event_n_markets", "sched_life_days", "q_len_words",
                "q_has_by", "q_starts_will", "q_has_digit"]
PRICE_FEATS = ["p", "logit_p", "h"]

LGB_PARAMS = dict(
    objective="binary", n_estimators=600, learning_rate=0.05, num_leaves=63,
    min_child_samples=60, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
    random_state=SEED, verbosity=-1, n_jobs=4,
)


def es_split(ttr, gtr, frac=0.15):
    """Event-grouped temporal split for early stopping: whole events are
    assigned to the validation set in order of their last training snapshot
    (most recent first) until ~frac of rows accrue. Prevents rows of the same
    market/event from straddling the fit/validation boundary."""
    ev = pd.DataFrame({"t": ttr, "g": gtr})
    last = ev.groupby("g")["t"].max().sort_values(ascending=False)
    sizes = ev.groupby("g").size()
    target = int(len(ev) * frac)
    va_events, acc = [], 0
    for g in last.index:
        va_events.append(g)
        acc += sizes[g]
        if acc >= target:
            break
    va_mask = ev["g"].isin(set(va_events)).values
    return ~va_mask, va_mask


def fit_gbm(Xtr, ytr, ttr, gtr, Xte):
    """LightGBM with early stopping on an event-grouped temporal tail."""
    fit_mask, va_mask = es_split(ttr, gtr)
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(Xtr[fit_mask], ytr[fit_mask],
          eval_set=[(Xtr[va_mask], ytr[va_mask])],
          eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(60, verbose=False)])
    return m.predict_proba(Xte)[:, 1]


def run_anchor(df, anchor):
    d = df[df["anchor"] == anchor].copy()
    d = d.sort_values("snap_ts").reset_index(drop=True)
    d["t_res"] = pd.to_datetime(d["t_res"], utc=True)
    cat_dum = pd.get_dummies(d["cat"], prefix="cat")
    d = pd.concat([d, cat_dum], axis=1)
    cat_cols = list(cat_dum.columns)

    months = sorted(d["snap_month"].unique())
    burn_end = months[BURN_IN_MONTHS - 1]

    # Fit text once on rows whose closure-time proxy precedes the burn-in cutoff.
    # As below, there is no separate training snapshot cutoff.
    burn_train = d[d["t_res"] < pd.Timestamp(months[BURN_IN_MONTHS] + "-01", tz="UTC")]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=20, max_features=5000,
                            sublinear_tf=True)
    svd = TruncatedSVD(n_components=N_SVD, random_state=SEED)
    Xt_burn = tfidf.fit_transform(burn_train["question"].fillna(""))
    svd.fit(Xt_burn)
    Xt_all = svd.transform(tfidf.transform(d["question"].fillna("")))
    txt_cols = [f"svd_{i}" for i in range(N_SVD)]
    d[txt_cols] = Xt_all

    FEATSETS = {
        "gbm_price": PRICE_FEATS,
        "gbm_path": PRICE_FEATS + PATH_FEATS,
        "gbm_notext": PRICE_FEATS + PATH_FEATS + STRUCT_FEATS + cat_cols,
        "gbm_full": PRICE_FEATS + PATH_FEATS + STRUCT_FEATS + cat_cols + txt_cols,
    }

    preds = []
    n_excluded = 0
    test_months = [m for m in months if m > burn_end]
    for m in test_months:
        m_start = pd.Timestamp(m + "-01", tz="UTC")
        # Archived specification: proxy cutoff only, not a snapshot-time cutoff.
        train = d[d["t_res"] < m_start]
        test = d[d["snap_month"] == m]
        if len(train) < 500 or len(test) == 0:
            continue
        train_events = set(train["event_id"])
        keep = ~test["event_id"].isin(train_events)
        n_excluded += int((~keep).sum())
        test = test[keep]
        if len(test) == 0:
            continue

        out = test[["id", "event_id", "h", "snap_month", "y", "p", "cat",
                    "active_frac", "event_n_markets", "t_res"]].copy()
        # price-only baselines
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso.fit(train["p"], train["y"])
        out["pred_iso"] = iso.predict(test["p"])
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(train[["logit_p"]], train["y"])
        out["pred_logit"] = lr.predict_proba(test[["logit_p"]])[:, 1]
        ttr = train["snap_ts"].values
        gtr = train["event_id"].values
        for name, cols in FEATSETS.items():
            out[f"pred_{name}"] = fit_gbm(train[cols].reset_index(drop=True),
                                          train["y"].reset_index(drop=True),
                                          ttr, gtr, test[cols])
        preds.append(out)
        print(f"[{anchor}] {m}: train={len(train)}, test={len(test)}", flush=True)

    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(f"{ROOT}/results/models/predictions_{anchor}.parquet", index=False)

    # ---- metrics ----
    def brier(p):
        return float(np.mean((p - P["y"]) ** 2))

    def logloss(p):
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(P["y"] * np.log(pc) + (1 - P["y"]) * np.log(1 - pc)))

    def dm_clustered(loss_a, loss_b, cluster):
        """mean(loss_a - loss_b) with cluster-sum SE; positive = b better"""
        dif = pd.Series(loss_a - loss_b)
        g = dif.groupby(cluster).agg(["sum", "size"])
        n = g["size"].sum()
        mean = g["sum"].sum() / n
        se = np.sqrt(((g["sum"] - mean * g["size"]) ** 2).sum()) / n
        return float(mean), float(se), float(mean / se) if se > 0 else np.nan

    def dm_monthly(loss_a, loss_b, month):
        dif = pd.DataFrame({"d": loss_a - loss_b, "m": month}).groupby("m")["d"].mean()
        t = dif.mean() / (dif.std() / np.sqrt(len(dif))) if dif.std() > 0 else np.nan
        return float(dif.mean()), float(t), int(len(dif))

    models = ["iso", "logit", "gbm_price", "gbm_path", "gbm_notext", "gbm_full"]
    res = {"n": len(P), "n_events": int(P["event_id"].nunique()),
           "n_excluded_event_overlap": n_excluded,
           "n_months": int(P["snap_month"].nunique()),
           "brier_price": brier(P["p"]), "logloss_price": logloss(P["p"]),
           "base_rate": float(P["y"].mean()), "models": {}}
    lp = (P["p"] - P["y"]) ** 2
    for mm in models:
        pr = P[f"pred_{mm}"]
        lm = (pr - P["y"]) ** 2
        mean, se, t = dm_clustered(lp.values, lm.values, P["event_id"])
        mmean, mt, nm = dm_monthly(lp.values, lm.values, P["snap_month"])
        res["models"][mm] = {
            "brier": brier(pr), "logloss": logloss(pr),
            "delta_brier_vs_price": res["brier_price"] - brier(pr),
            "dm_event": {"mean": mean, "se": se, "t": t},
            "dm_monthly": {"mean": mmean, "t": mt, "n_months": nm},
        }
    # per-horizon for the full model and price
    res["by_horizon"] = {}
    for h in sorted(P["h"].unique()):
        s = P[P["h"] == h]
        lps = (s["p"] - s["y"]) ** 2
        lms = (s["pred_gbm_full"] - s["y"]) ** 2
        dif = pd.Series(lps.values - lms.values)
        g = dif.groupby(s["event_id"].values).agg(["sum", "size"])
        n = g["size"].sum()
        mean = g["sum"].sum() / n
        se = np.sqrt(((g["sum"] - mean * g["size"]) ** 2).sum()) / n
        res["by_horizon"][int(h)] = {
            "n": int(len(s)),
            "brier_price": float(np.mean(lps)),
            "brier_gbm_full": float(np.mean(lms)),
            "dm_t": float(mean / se) if se > 0 else np.nan,
        }
    return res


def main():
    df = pd.read_parquet(f"{ROOT}/data/processed/ml_dataset.parquet")
    results = {}
    for anchor in ["sched", "res"]:
        results[anchor] = run_anchor(df, anchor)
    with open(f"{ROOT}/results/models/model_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    for anchor in results:
        r = results[anchor]
        print(f"\n=== {anchor}: n={r['n']}, brier_price={r['brier_price']:.5f} ===")
        for mm, v in r["models"].items():
            print(f"  {mm:10s}: brier={v['brier']:.5f} "
                  f"dBrier={v['delta_brier_vs_price']:+.5f} "
                  f"t_ev={v['dm_event']['t']:.2f} t_mo={v['dm_monthly']['t']:.2f}")


if __name__ == "__main__":
    main()
