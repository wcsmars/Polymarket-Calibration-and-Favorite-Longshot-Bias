"""Empirical analysis: calibration, favorite-longshot bias, moderators, backtest.

All inference clusters on event_id (markets within an event have mechanically
dependent outcomes). Calibration-curve CIs use a cluster bootstrap.
Writes JSON/CSV results into results/.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import write_json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 7, 14, 30, 60, 90]
NBINS = 20
BOOT = 1000
SEED = 42


def load_panel():
    panel = pd.read_csv(f"{ROOT}/data/processed/panel.csv", dtype={"id": str, "event_id": str})
    panel["event_id"] = panel["event_id"].fillna("mkt_" + panel["id"])
    panel["t_res"] = pd.to_datetime(panel["t_res"], utc=True, format="mixed")
    return panel


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def quantile_groups(values, labels=("low", "mid", "high")):
    """Quantile groups that keep tied values together, even with few distinct values."""
    values = pd.Series(values)
    if values.notna().sum() == 0:
        return pd.Series(pd.Categorical([None] * len(values), categories=labels), index=values.index)
    if values.nunique() == 1:
        return pd.Series(pd.Categorical([labels[len(labels) // 2] if pd.notna(v) else None
                                         for v in values], categories=labels), index=values.index)
    codes = pd.qcut(values, len(labels), labels=False, duplicates="drop")
    n_groups = int(codes.max()) + 1
    chosen = ([labels[len(labels) // 2]] if n_groups == 1 else
              [labels[0], labels[-1]] if n_groups == 2 else list(labels[:n_groups]))
    return codes.map(dict(enumerate(chosen))).astype(pd.CategoricalDtype(labels, ordered=True))


def validate_forecasts(df):
    p, y = df["p"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)
    if not len(df):
        raise ValueError("forecast sample is empty")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("forecast probabilities must be finite and in [0, 1]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("outcomes must be binary and nonmissing")
    return p, y


# ---------- calibration curve with cluster bootstrap ----------

def calib_bins(df, nbins=NBINS, boot=BOOT, seed=SEED):
    validate_forecasts(df)
    if nbins < 1 or boot < 1:
        raise ValueError("nbins and boot must be positive")
    if df["event_id"].isna().any():
        raise ValueError("event_id must be nonmissing for clustered inference")
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, 1, nbins + 1)
    b = np.clip(np.digitize(df["p"], edges) - 1, 0, nbins - 1)
    df = df.assign(bin=b)
    # per-event sufficient statistics: counts and y-sums per bin
    ev = df.groupby(["event_id", "bin"]).agg(n=("y", "size"), s=("y", "sum"), ps=("p", "sum")).reset_index()
    ev_ids = ev["event_id"].unique()
    E = len(ev_ids)
    idx = {e: i for i, e in enumerate(ev_ids)}
    N_mat = np.zeros((E, nbins))
    S_mat = np.zeros((E, nbins))
    P_mat = np.zeros((E, nbins))
    N_mat[ev["event_id"].map(idx), ev["bin"]] = ev["n"]
    S_mat[ev["event_id"].map(idx), ev["bin"]] = ev["s"]
    P_mat[ev["event_id"].map(idx), ev["bin"]] = ev["ps"]

    n_k = N_mat.sum(0)
    s_k = S_mat.sum(0)
    with np.errstate(invalid="ignore"):
        freq = s_k / n_k
    p_mean = df.groupby("bin")["p"].mean().reindex(range(nbins)).values

    # cluster bootstrap: resample events with replacement
    boots = np.full((boot, nbins), np.nan)
    error_boots = np.full((boot, nbins), np.nan)
    for r in range(boot):
        w = rng.multinomial(E, np.full(E, 1 / E)).astype(float)
        nn = w @ N_mat
        ss = w @ S_mat
        with np.errstate(invalid="ignore"):
            boots[r] = np.divide(ss, nn, out=np.full(nbins, np.nan), where=nn > 0)
            error_boots[r] = np.divide(ss - w @ P_mat, nn,
                                       out=np.full(nbins, np.nan), where=nn > 0)
    bounds = np.full((4, nbins), np.nan)
    # Empty bins have no estimable interval; avoid all-NaN percentile warnings.
    occupied = n_k > 0
    if E > 1:
        bounds[:2, occupied] = np.nanpercentile(boots[:, occupied], [2.5, 97.5], axis=0)
        bounds[2:, occupied] = np.nanpercentile(error_boots[:, occupied], [2.5, 97.5], axis=0)
    lo, hi, err_lo, err_hi = bounds
    return pd.DataFrame({
        "bin": range(nbins), "n": n_k.astype(int), "p_mean": p_mean,
        "y_freq": freq, "ci_lo": lo, "ci_hi": hi,
        "error_ci_lo": err_lo, "error_ci_hi": err_hi,
    })


# ---------- scores ----------

def scores(df, nbins=NBINS):
    if df.empty:
        return {"n": 0, "n_events": 0, "base_rate": np.nan, "brier": np.nan,
                "brier_ref": np.nan, "bss": None, "reliability": np.nan,
                "resolution": np.nan, "uncertainty": np.nan, "log_score": np.nan,
                "brier_binned": np.nan, "binning_residual": np.nan}
    p, y = validate_forecasts(df)
    bs = float(np.mean((p - y) ** 2))
    ybar = float(y.mean())
    unc = ybar * (1 - ybar)
    edges = np.linspace(0, 1, nbins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    g = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(
        n=("y", "size"), pbar=("p", "mean"), ybar=("y", "mean"))
    n = len(df)
    rel = float((g["n"] * (g["pbar"] - g["ybar"]) ** 2).sum() / n)
    res = float((g["n"] * (g["ybar"] - ybar) ** 2).sum() / n)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    logscore = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    return {
        "n": n, "n_events": int(df["event_id"].nunique()), "base_rate": ybar,
        "brier": bs, "brier_ref": unc, "bss": 1 - bs / unc if unc > 0 else None,
        "reliability": rel, "resolution": res, "uncertainty": unc,
        "log_score": logscore, "brier_binned": rel - res + unc,
        "binning_residual": bs - (rel - res + unc),
    }


# ---------- CORP (isotonic / PAV) calibration, Dimitriadis-Gneiting-Jordan 2021 ----------

def pav(y_sorted, weights=None):
    """Weighted pool-adjacent-violators fit on ordered, distinct forecast levels."""
    y_sorted = np.asarray(y_sorted, dtype=float)
    n = len(y_sorted)
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != y_sorted.shape or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("PAV weights must be finite, positive, and match outcomes")
    level_sum = list(y_sorted * weights)
    level_n = list(weights)
    stack_len = []
    stack_sum, stack_n = [], []
    for i in range(n):
        s, c, width = level_sum[i], level_n[i], 1
        while stack_sum and stack_sum[-1] / stack_n[-1] >= s / c:
            s += stack_sum.pop()
            c += stack_n.pop()
            width += stack_len.pop()
        stack_len.append(width)
        stack_sum.append(s)
        stack_n.append(c)
    out = np.empty(n)
    pos = 0
    for s, c, width in zip(stack_sum, stack_n, stack_len):
        out[pos:pos + width] = s / c
        pos += width
    return out


def corp_decomposition(df):
    """BS = MCB - DSC + UNC via isotonic recalibration (binning-free)."""
    p, y = validate_forecasts(df)
    # Identical prices must receive identical fits. Pool ties *before* PAV;
    # sorting rows alone lets their arbitrary outcome ordering create skill.
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=y) / counts
    f = pav(means, counts)[inverse]
    bs_p = float(np.mean((p - y) ** 2))
    bs_f = float(np.mean((f - y) ** 2))
    ybar = float(y.mean())
    unc = ybar * (1 - ybar)
    return {"brier": bs_p, "mcb": bs_p - bs_f, "dsc": unc - bs_f, "unc": unc}


# ---------- calibration regressions ----------

def wald(params, cov, C, c):
    d = C @ params - c
    variance = C @ cov @ C.T
    if not np.isfinite(variance).all() or np.linalg.matrix_rank(variance) < len(c):
        return np.nan, np.nan
    W = float(d @ np.linalg.solve(variance, d))
    from scipy.stats import chi2
    return W, float(chi2.sf(W, len(c)))


def mz_regressions(df):
    """Fit calibration models, recording unavailable inference for degenerate subsets."""
    unavailable = {
        "linear": dict.fromkeys(["alpha", "beta", "se_alpha", "se_beta", "wald_chi2", "wald_p"], np.nan),
        "logodds": dict.fromkeys(["a", "b", "se_a", "se_b", "wald_chi2", "wald_p", "z_b_vs_1"], np.nan),
        "n": len(df), "n_events": int(df["event_id"].nunique()),
    }
    if len(df) < 3 or df["event_id"].nunique() < 2 or df["p"].nunique() < 2 or df["y"].nunique() < 2:
        return {**unavailable, "status": "not_estimable", "reason": "Insufficient events or outcome/price variation"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            return _mz_regressions(df)
    except (np.linalg.LinAlgError, PerfectSeparationError, PerfectSeparationWarning) as exc:
        return {**unavailable, "status": "not_estimable", "reason": str(exc)}


def _mz_regressions(df):
    validate_forecasts(df)
    groups = df["event_id"]
    if len(df) < 3 or groups.nunique() < 2 or df["p"].nunique() < 2 or df["y"].nunique() < 2:
        raise ValueError("calibration regressions require varying outcomes/prices and at least two events")
    X = sm.add_constant(df["p"].values)
    ols = sm.OLS(df["y"].values, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    W_lin, p_lin = wald(ols.params, ols.cov_params(), np.eye(2), np.array([0.0, 1.0]))

    Xl = sm.add_constant(logit(df["p"].values))
    glm = sm.GLM(df["y"].values, Xl, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    W_log, p_log = wald(glm.params, glm.cov_params(), np.eye(2), np.array([0.0, 1.0]))

    return {
        "linear": {"alpha": ols.params[0], "beta": ols.params[1],
                   "se_alpha": ols.bse[0], "se_beta": ols.bse[1],
                   "wald_chi2": W_lin, "wald_p": p_lin},
        "logodds": {"a": glm.params[0], "b": glm.params[1],
                    "se_a": glm.bse[0], "se_b": glm.bse[1],
                    "wald_chi2": W_log, "wald_p": p_log,
                    "z_b_vs_1": (glm.params[1] - 1) / glm.bse[1]},
        "n": len(df), "n_events": int(df["event_id"].nunique()),
    }


# ---------- bucket means with clustered SE ----------

def cluster_mean_se(df, col):
    """Mean of col and SE clustered by event (cluster-sum formula)."""
    if not len(df):
        return np.nan, np.nan, 0, 0
    if df["event_id"].isna().any() or not np.isfinite(df[col].to_numpy(dtype=float)).all():
        raise ValueError("clustered means require finite values and nonmissing event IDs")
    g = df.groupby("event_id")[col].agg(["sum", "size"])
    n = g["size"].sum()
    mean = g["sum"].sum() / n
    resid_sums = g["sum"] - mean * g["size"]
    se = np.sqrt((resid_sums ** 2).sum()) / n if len(g) > 1 else np.nan
    return float(mean), float(se), int(n), len(g)


def flb_buckets(df):
    out = {}
    for name, lo, hi in [("longshot_0_10", 0.0, 0.10), ("mid_10_90", 0.10, 0.90),
                         ("favorite_90_100", 0.90, 1.0)]:
        upper = df["p"] <= hi if hi == 1 else df["p"] < hi
        sub = df[(df["p"] >= lo) & upper].copy()
        if len(sub) == 0:
            continue
        sub["d"] = sub["y"] - sub["p"]
        mean_d, se_d, n, n_ev = cluster_mean_se(sub, "d")
        out[name] = {"n": n, "n_events": n_ev, "mean_p": float(sub["p"].mean()),
                     "mean_y": float(sub["y"].mean()), "y_minus_p": mean_d,
                     "se": se_d, "t": mean_d / se_d if se_d > 0 else None}
    return out


# ---------- strategy backtest ----------

def backtest(df, cost):
    if not np.isfinite(cost) or cost < 0:
        raise ValueError("cost must be finite and nonnegative")
    out = {}
    fav = df[(df["p"] >= 0.90) & (df["p"] <= 0.99)].copy()
    fav["ret"] = fav["y"] / (fav["p"] + cost) - 1
    lng = df[(df["p"] >= 0.01) & (df["p"] <= 0.10)].copy()
    lng["ret"] = (1 - lng["y"]) / ((1 - lng["p"]) + cost) - 1
    for name, sub in [("buy_favorites_yes", fav), ("fade_longshots_buy_no", lng)]:
        if len(sub) == 0:
            continue
        m, se, n, n_ev = cluster_mean_se(sub, "ret")
        monthly = sub.set_index("t_res").groupby(pd.Grouper(freq="ME"))["ret"].mean().dropna()
        out[name] = {
            "n_trades": n, "n_events": n_ev, "mean_ret": m, "se_clust": se,
            "t": m / se if se > 0 else None, "win_rate": float((sub["ret"] > 0).mean()),
            "monthly_mean": float(monthly.mean()), "monthly_std": float(monthly.std()),
            "n_months": int(len(monthly)),
        }
    # decile return table (buy YES at p, gross)
    dec = df[df["p"] > 0].copy()
    dec["decile"] = np.clip((dec["p"] * 10).astype(int), 0, 9)
    dec["ret"] = dec["y"] / dec["p"] - 1
    tab = []
    for d in range(10):
        sub = dec[dec["decile"] == d]
        if len(sub) < 30:
            tab.append(None)
            continue
        m, se, n, n_ev = cluster_mean_se(sub, "ret")
        tab.append({"decile": d, "n": n, "mean_p": float(sub["p"].mean()),
                    "mean_ret": m, "se": se})
    out["decile_buy_yes_gross"] = tab
    return out


def main():
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    results = {"horizon": {}}

    for h in HORIZONS:
        df = panel[panel["h"] == h]
        if len(df) < 200:
            continue
        bins = calib_bins(df)
        bins.to_csv(f"{ROOT}/results/calibration_bins_h{h}.csv", index=False)
        results["horizon"][h] = {
            "scores": scores(df),
            "corp": corp_decomposition(df),
            "regressions": mz_regressions(df),
            "flb": flb_buckets(df),
        }
        print(f"h={h}: n={len(df)}, brier={results['horizon'][h]['scores']['brier']:.4f}, "
              f"logodds_b={results['horizon'][h]['regressions']['logodds']['b']:.3f}", flush=True)

    # ---- moderators at h=7 ----
    df7 = panel[panel["h"] == 7].copy()
    mods = {}
    df7["logv"] = np.log10(df7["volumeNum"].clip(lower=1))
    df7["vol_grp"] = quantile_groups(df7["logv"])
    for grp_col, name in [("vol_grp", "volume_tercile"), ("cat", "category"), ("era", "era")]:
        mods[name] = {}
        for g, sub in df7.groupby(grp_col, observed=True):
            if len(sub) < 500 or sub["y"].nunique() < 2:
                continue
            reg = mz_regressions(sub)
            sc = scores(sub)
            mods[name][str(g)] = {
                "n": len(sub), "b": reg["logodds"]["b"], "se_b": reg["logodds"]["se_b"],
                "alpha": reg["logodds"]["a"], "brier": sc["brier"], "bss": sc["bss"],
                "reliability": sc["reliability"], "base_rate": sc["base_rate"],
                "median_volume": float(sub["volumeNum"].median()),
            }
    results["moderators_h7"] = mods

    # volume x extremeness interaction: does FLB shrink with volume?
    mods_flb = {}
    for g, sub in df7.groupby("vol_grp", observed=True):
        mods_flb[str(g)] = flb_buckets(sub)
    results["flb_by_volume_h7"] = mods_flb

    # ---- backtest at h=7 and h=30 ----
    results["backtest"] = {}
    for h in [7, 30]:
        df = panel[panel["h"] == h]
        results["backtest"][f"h{h}"] = {
            "gross": backtest(df, 0.0),
            "net_1c": backtest(df, 0.01),
        }

    # ---- robustness at h=7 ----
    rob = {}
    dedup = df7.sort_values("volumeNum", ascending=False).drop_duplicates("event_id")
    rob["one_per_event"] = {**mz_regressions(dedup)["logodds"], "n": len(dedup),
                            "brier": scores(dedup)["brier"]}
    nocd = df7[df7["cat"] != "crypto_daily"]
    rob["excl_crypto_daily"] = {**mz_regressions(nocd)["logodds"], "n": len(nocd)}
    nonr = df7[~(df7["event_neg_risk"].astype(bool) | df7["negRisk"].astype("boolean").fillna(False).astype(bool))]
    rob["excl_negrisk"] = {**mz_regressions(nonr)["logodds"], "n": len(nonr)}
    bigv = df7[df7["volumeNum"] >= 10000]
    rob["volume_ge_10k"] = {**mz_regressions(bigv)["logodds"], "n": len(bigv)}

    # Schedule-anchored prices: available quotes h days before scheduled end.
    # Stage 04 retains only snapshots strictly before the closure-time proxy.
    # Retrospective metadata still cannot establish a point-in-time dataset.
    mkt = pd.read_csv(f"{ROOT}/data/processed/market_level.csv",
                      dtype={"id": str, "event_id": str})
    mkt["event_id"] = mkt["event_id"].fillna("mkt_" + mkt["id"])
    mkt["t_res"] = pd.to_datetime(mkt["t_res"], utc=True, format="mixed")
    sched_res = {}
    for h in HORIZONS:
        sub = mkt[mkt[f"p_sched_{h}"].notna()].copy()
        sub["p"] = sub[f"p_sched_{h}"]
        if len(sub) < 500:
            continue
        sched_res[f"h{h}"] = {
            "logodds": mz_regressions(sub)["logodds"], "n": len(sub),
            "scores": scores(sub), "flb": flb_buckets(sub),
        }
    results["schedule_anchored"] = sched_res
    # Retrospective schedule-anchored backtest at h=30.
    sub30 = mkt[mkt["p_sched_30"].notna()].copy()
    sub30["p"] = sub30["p_sched_30"]
    results["backtest_sched_h30"] = {
        "gross": backtest(sub30, 0.0), "net_1c": backtest(sub30, 0.01)}

    midl = mkt[mkt["p_mid"].notna()].copy()
    midl["p"] = midl["p_mid"]
    rob["midlife_price"] = {**mz_regressions(midl)["logodds"], "n": len(midl)}
    rob["sched_anchored_h7"] = {**sched_res["h7"]["logodds"], "n": sched_res["h7"]["n"]} \
        if "h7" in sched_res else None
    results["robustness"] = rob

    # ---- balanced panel: same markets observed at h = 1, 7, 30 ----
    wide = panel[panel["h"].isin([1, 7, 30])].pivot_table(
        index="id", columns="h", values="p", aggfunc="first")
    ids_bal = wide.reindex(columns=[1, 7, 30]).dropna().index
    bal = {}
    for h in [1, 7, 30]:
        sub = panel[(panel["h"] == h) & (panel["id"].isin(ids_bal))]
        bal[f"h{h}"] = {"scores": scores(sub),
                        "logodds": mz_regressions(sub)["logodds"],
                        "flb": flb_buckets(sub)}
    results["balanced_panel_h1_7_30"] = {"n_markets": int(len(ids_bal)), **bal}

    write_json(ROOT / "results/analysis.json", results)
    print("DONE -> results/analysis.json", flush=True)


if __name__ == "__main__":
    main()
