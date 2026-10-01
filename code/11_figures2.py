"""Model-study figures and tables from results/models/{model_metrics,interpretation,
supplement}.json, plus data/processed/ml_dataset.parquet (from 08_features.py) for
the fig2_2 correction curves."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = f"{ROOT}/figures/models"
Path(FIG).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
BLUE, RED, GRAY, GREEN = "#2563eb", "#dc2626", "#6b7280", "#059669"

def number(value):
    return np.nan if value is None else float(value)


def fmt(value, spec=".2f"):
    return "—" if value is None or not math.isfinite(float(value)) else format(value, spec)


M = json.load(open(f"{ROOT}/results/models/model_metrics.json"))
I = json.load(open(f"{ROOT}/results/models/interpretation.json"))
sched = M["sched"]

# ---- Fig 1: model ladder, delta-Brier vs price with event-clustered CIs ----
models = ["iso", "logit", "gbm_price", "gbm_path", "gbm_notext", "gbm_full"]
labels = ["Isotonic\n(price only)", "Logistic\nlogit(p)", "GBM\n(p, h)",
          "GBM\n+path", "GBM +path\n+structure", "GBM full\n(+text)"]
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(models))
means = [1e4 * number(sched["models"][m]["dm_event"]["mean"]) for m in models]
cis = [1.96e4 * number(sched["models"][m]["dm_event"]["se"]) for m in models]
ax.axhline(0, color=GRAY, lw=1, ls="--")
ax.bar(x, means, color=[GREEN if v > 0 else RED for v in means], alpha=0.85)
ax.errorbar(x, means, yerr=cis, fmt="none", ecolor="black", elinewidth=0.9, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("OOS Brier improvement vs. price (×10⁻⁴)")
ax.set_title("Out-of-sample forecast improvement over the market price (schedule-anchored)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_1_model_ladder.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 2: learned price-correction curves (smooth: logistic per horizon + isotonic) ----
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from importlib.machinery import SourceFileLoader
m09 = SourceFileLoader("m09_figures", f"{ROOT}/code/09_model.py").load_module()

ds = pd.read_parquet(f"{ROOT}/data/processed/ml_dataset.parquet")
ds = ds[(ds["anchor"] == "sched")].copy()
ds["t_res"] = pd.to_datetime(ds["t_res"], utc=True)
ds = m09.eligible_snapshots(ds)
tr = ds[m09.training_eligible(ds, pd.Timestamp("2025-01-01", tz="UTC"))]
grid = np.linspace(0.01, 0.99, 197)
lgrid = np.log(grid / (1 - grid))
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
axes[0].plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1, label="no correction")
for h, col in [(1, BLUE), (7, GREEN), (30, "#d97706"), (90, RED)]:
    sub = tr[tr["h"] == h]
    if sub["y"].nunique() < 2:
        continue
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(sub[["logit_p"]], sub["y"])
    axes[0].plot(grid, lr.predict_proba(pd.DataFrame({"logit_p": lgrid}))[:, 1], color=col,
                 lw=1.5, label=f"h = {h}d")
axes[0].set_xlabel("Market price p"); axes[0].set_ylabel("Recalibrated probability")
axes[0].set_title("Logistic recalibration, by horizon")
axes[0].legend(frameon=False, fontsize=8)
iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
iso.fit(tr["p"], tr["y"])
axes[1].plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
axes[1].plot(grid, iso.predict(grid), color=BLUE, lw=1.5)
axes[1].set_xlabel("Market price p")
axes[1].set_title("Isotonic recalibration (pooled)")
fig.suptitle("The learned price-correction functions (training data through 2024)", y=1.0)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_2_correction_curves.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 3: group permutation importance ----
gpi = I["group_permutation_importance"]
names = sorted(gpi, key=lambda k: -number(gpi[k]["mean_dbrier"]))
fig, ax = plt.subplots(figsize=(6.5, 3.4))
vals = [1e4 * number(gpi[k]["mean_dbrier"]) for k in names]
errs = [1e4 * number(gpi[k]["sd"]) for k in names]
ax.barh(names[::-1], vals[::-1], xerr=errs[::-1], color=BLUE, alpha=0.85,
        error_kw=dict(ecolor="black", lw=0.9, capsize=3))
ax.set_xlabel("Brier degradation when feature group permuted (×10⁻⁴)")
ax.set_title("Group permutation importance (2025+ snapshots; bars: ±1 permutation SD)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_3_importance.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 4: monthly delta-Brier over time ----
md = I["monthly_delta_brier"]
months = sorted(md)
fig, ax = plt.subplots(figsize=(7, 3.2))
vals = [1e4 * number(md[m]) for m in months]
ax.axhline(0, color=GRAY, lw=1, ls="--")
ax.bar(range(len(months)), vals, color=[GREEN if v > 0 else RED for v in vals], alpha=0.85)
step = max(1, len(months) // 10)
ax.set_xticks(range(0, len(months), step))
ax.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45,
                   ha="right", fontsize=7)
ax.set_ylabel("Monthly mean ΔBrier vs price (×10⁻⁴)")
ax.set_title("Held-out Brier improvement by snapshot month (GBM full)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_4_monthly.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 5: divergence deciles -> realized directional edge ----
dd = I["divergence_deciles"]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
x = [d["mean_absdiv"] for d in dd]
y = [d["directional_edge_pp"] for d in dd]
e = [1.96 * number(d["se_pp"]) for d in dd]
ax.axhline(0, color=GRAY, lw=1, ls="--")
ax.errorbar(x, y, yerr=e, fmt="o-", color=BLUE, ms=4, lw=1.2, capsize=3)
ax.set_xlabel("Mean |model − price| within decile")
ax.set_ylabel("Realized directional edge (pp)")
ax.set_title("Model–price divergence and realized pricing errors")
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_5_divergence.png", bbox_inches="tight")
plt.close(fig)

# ---- Tables ----
out = []
out.append("## Table 2. Out-of-sample forecast comparison, schedule-anchored\n")
out.append(f"Test rows: {fmt(sched['n'], ',')} ({fmt(sched['n_events'], ',')} events, "
           f"{sched['n_months']} months; {fmt(sched['n_excluded_event_overlap'], ',')} rows "
           f"excluded for event overlap). Price Brier = {fmt(sched['brier_price'], '.5f')}, "
           f"log loss = {fmt(sched['logloss_price'], '.5f')}, base rate = {fmt(sched['base_rate'], '.3f')}.\n")
out.append("| Model | Brier | Log loss | ΔBrier vs price (×10⁻⁴) | t (event) | t (monthly) |")
out.append("|---|---|---|---|---|---|")
for mname, lab in zip(models, ["Isotonic (price)", "Logistic logit(p)", "GBM (p,h)",
                               "GBM +path", "GBM +path+structure", "GBM full"]):
    v = sched["models"][mname]
    out.append(f"| {lab} | {fmt(v['brier'], '.5f')} | {fmt(v['logloss'], '.5f')} | "
               f"{fmt(1e4*number(v['delta_brier_vs_price']), '+.2f')} | {fmt(v['dm_event']['t'], '.2f')} | "
               f"{fmt(v['dm_monthly']['t'], '.2f')} |")

out.append("\n## Table 3. By horizon (GBM full vs price, schedule-anchored)\n")
out.append("| h | n | Brier price | Brier GBM | Δ (×10⁻⁴) | DM t (event) |")
out.append("|---|---|---|---|---|---|")
for h, v in sorted(sched["by_horizon"].items(), key=lambda kv: int(kv[0])):
    out.append(f"| {h} | {fmt(v['n'], ',')} | {fmt(v['brier_price'], '.5f')} | {fmt(v['brier_gbm_full'], '.5f')} | "
               f"{fmt(1e4*(v['brier_price']-v['brier_gbm_full']), '+.2f')} | {fmt(v['dm_t'], '.2f')} |")

res = M["res"]
out.append("\n## Table 4. Closure-proxy-anchored robustness\n")
out.append(f"Test rows: {fmt(res['n'], ',')}; price Brier = {fmt(res['brier_price'], '.5f')}.\n")
out.append("| Model | Brier | ΔBrier (×10⁻⁴) | t (event) | t (monthly) |")
out.append("|---|---|---|---|---|")
for mname in models:
    v = res["models"][mname]
    out.append(f"| {mname} | {fmt(v['brier'], '.5f')} | {fmt(1e4*number(v['delta_brier_vs_price']), '+.2f')} | "
               f"{fmt(v['dm_event']['t'], '.2f')} | {fmt(v['dm_monthly']['t'], '.2f')} |")

out.append("\n## Table 5. Where the model wins (ΔBrier vs price, ×10⁻⁴)\n")
out.append("| Split | Group | n | Δ | t |")
out.append("|---|---|---|---|---|")
for split, groups in I["where"].items():
    for g, v in groups.items():
        out.append(f"| {split} | {g} | {fmt(v['n'], ',')} | {fmt(1e4*number(v['delta_brier']), '+.2f')} | {fmt(v['t'], '.2f')} |")

out.append("\n## Table 6. Divergence-threshold price-based returns (schedule-anchored, held-out)\n")
out.append("One dollar per forecast row, including repeated horizons for a market. Monthly averages use snapshot months; this is not a capital-constrained portfolio or an executable trading simulation.\n")
out.append("| θ | Cost | Trades | Mean ret | t (event) | Monthly mean | t (monthly) | Win |")
out.append("|---|---|---|---|---|---|---|---|")
for theta in ["0.02", "0.05", "0.1"]:
    for ck in ["gross", "net_1c"]:
        k = f"theta{theta}_{ck}"
        if k not in I["backtest"]:
            continue
        v = I["backtest"][k]
        out.append(f"| {theta} | {ck} | {fmt(v['n_trades'], ',')} | {fmt(100*number(v['mean_ret']), '+.2f')}% | "
                   f"{fmt(v['t_clust'], '.2f')} | {fmt(100*number(v['monthly_mean']), '+.2f')}% | "
                   f"{fmt(v['monthly_t'], '.2f')} | {fmt(100*number(v['win_rate']), '.1f')}% |")

S = json.load(open(f"{ROOT}/results/models/supplement.json"))["sched"]
out.append(f"\n## Table 7. Second-half folds (test months ≥ {S['mature_cutoff_month']}) and split-sample blends\n")
out.append("Blend weights use earlier snapshots with closure proxies before the split; evaluation excludes tuning events.\n")
out.append("| Specification | ΔBrier vs price (×10⁻⁴) | t (event) | t (monthly) |")
out.append("|---|---|---|---|")
for m, lab in [("iso", "Isotonic (mature)"), ("logit", "Logistic (mature)"),
               ("gbm_price", "GBM price (mature)"), ("gbm_full", "GBM full (mature)")]:
    v = S["mature"][m]
    out.append(f"| {lab} | {fmt(1e4*number(v['delta_brier']), '+.2f')} | {fmt(v['t_event'], '.2f')} | {fmt(v['t_monthly'], '.2f')} |")
for m, lab in [("logit_honest", "Blend price/logistic (λ* from eligible first-half events)"),
               ("gbm_full_honest", "Blend price/GBM-full (λ* from eligible first-half events)")]:
    v = S["blend"][m]
    out.append(f"| {lab}, λ*={fmt(v['lambda_star'], '.2f')} | {fmt(1e4*number(v['delta_brier']), '+.2f')} | "
               f"{fmt(v['t_event'], '.2f')} | {fmt(v['t_monthly'], '.2f')} |")

out.append("\n## Table 8. Recalibration gains by horizon (schedule-anchored, ΔBrier ×10⁻⁴)\n")
out.append("| h | n | Isotonic (t) | Logistic (t) | GBM full (t) |")
out.append("|---|---|---|---|---|")
for h, v in sorted(S["by_horizon"].items(), key=lambda kv: int(kv[0])):
    out.append(f"| {h} | {fmt(v['n'], ',')} | {fmt(1e4*number(v['iso']['delta_brier']), '+.2f')} ({fmt(v['iso']['t'], '.1f')}) | "
               f"{fmt(1e4*number(v['logit']['delta_brier']), '+.2f')} ({fmt(v['logit']['t'], '.1f')}) | "
               f"{fmt(1e4*number(v['gbm_full']['delta_brier']), '+.2f')} ({fmt(v['gbm_full']['t'], '.1f')}) |")

# Fig 6: recalibration gain by horizon
fig, ax = plt.subplots(figsize=(6.5, 3.6))
hs = sorted(int(h) for h in S["by_horizon"])
ax.axhline(0, color=GRAY, lw=1, ls="--")
for m, col, lab in [("logit", BLUE, "Logistic recalibration"), ("iso", GREEN, "Isotonic"),
                    ("gbm_full", RED, "GBM full")]:
    ax.plot(hs, [1e4 * number(S["by_horizon"][str(h)][m]["delta_brier"]) for h in hs],
            "o-", color=col, ms=4, lw=1.3, label=lab)
ax.set_xscale("log"); ax.set_xticks(hs); ax.set_xticklabels(hs)
ax.set_xlabel("Horizon (days before scheduled end, log scale)")
ax.set_ylabel("ΔBrier vs price (×10⁻⁴)")
ax.set_title("Held-out Brier improvement by forecast horizon")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_6_horizon_gains.png", bbox_inches="tight")
plt.close(fig)

with open(f"{ROOT}/results/models/tables.md", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("figures ->", FIG, "| tables -> results/models/tables.md")
