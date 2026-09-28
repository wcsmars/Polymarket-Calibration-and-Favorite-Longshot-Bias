"""Model-study figures and tables from results/models/{model_metrics,interpretation,
supplement}.json, plus data/processed/ml_dataset.parquet (from 08_features.py) for
the fig2_2 correction curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = f"{ROOT}/figures/models"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
BLUE, RED, GRAY, GREEN = "#2563eb", "#dc2626", "#6b7280", "#059669"

M = json.load(open(f"{ROOT}/results/models/model_metrics.json"))
I = json.load(open(f"{ROOT}/results/models/interpretation.json"))
sched = M["sched"]

# ---- Fig 1: model ladder, delta-Brier vs price with event-clustered CIs ----
models = ["iso", "logit", "gbm_price", "gbm_path", "gbm_notext", "gbm_full"]
labels = ["Isotonic\n(price only)", "Logistic\nlogit(p)", "GBM\n(p, h)",
          "GBM\n+path", "GBM +path\n+structure", "GBM full\n(+text)"]
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(models))
means = [1e4 * sched["models"][m]["dm_event"]["mean"] for m in models]
cis = [1.96e4 * sched["models"][m]["dm_event"]["se"] for m in models]
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

ds = pd.read_parquet(f"{ROOT}/data/processed/ml_dataset.parquet")
ds = ds[(ds["anchor"] == "sched")]
ds["t_res"] = pd.to_datetime(ds["t_res"], utc=True)
tr = ds[ds["t_res"] < pd.Timestamp("2025-01-01", tz="UTC")]
grid = np.linspace(0.01, 0.99, 197)
lgrid = np.log(grid / (1 - grid))
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
axes[0].plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1, label="no correction")
for h, col in [(1, BLUE), (7, GREEN), (30, "#d97706"), (90, RED)]:
    sub = tr[tr["h"] == h]
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(sub[["logit_p"]], sub["y"])
    axes[0].plot(grid, lr.predict_proba(lgrid.reshape(-1, 1))[:, 1], color=col,
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
names = sorted(gpi, key=lambda k: -gpi[k]["mean_dbrier"])
fig, ax = plt.subplots(figsize=(6.5, 3.4))
vals = [1e4 * gpi[k]["mean_dbrier"] for k in names]
errs = [1.96e4 * gpi[k]["sd"] for k in names]
ax.barh(names[::-1], vals[::-1], xerr=errs[::-1], color=BLUE, alpha=0.85,
        error_kw=dict(ecolor="black", lw=0.9, capsize=3))
ax.set_xlabel("Brier degradation when feature group permuted (×10⁻⁴)")
ax.set_title("Group permutation importance (test: 2025 snapshots)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_3_importance.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 4: monthly delta-Brier over time ----
md = I["monthly_delta_brier"]
months = sorted(md)
fig, ax = plt.subplots(figsize=(7, 3.2))
vals = [1e4 * md[m] for m in months]
ax.axhline(0, color=GRAY, lw=1, ls="--")
ax.bar(range(len(months)), vals, color=[GREEN if v > 0 else RED for v in vals], alpha=0.85)
step = max(1, len(months) // 10)
ax.set_xticks(range(0, len(months), step))
ax.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45,
                   ha="right", fontsize=7)
ax.set_ylabel("Monthly mean ΔBrier vs price (×10⁻⁴)")
ax.set_title("Does the edge persist? Out-of-sample improvement by month (GBM full)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_4_monthly.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 5: divergence deciles -> realized directional edge ----
dd = I["divergence_deciles"]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
x = [d["mean_absdiv"] for d in dd]
y = [d["directional_edge_pp"] for d in dd]
e = [1.96 * d["se_pp"] for d in dd]
ax.axhline(0, color=GRAY, lw=1, ls="--")
ax.errorbar(x, y, yerr=e, fmt="o-", color=BLUE, ms=4, lw=1.2, capsize=3)
ax.set_xlabel("Mean |model − price| within decile")
ax.set_ylabel("Realized directional edge (pp)")
ax.set_title("Model–price divergence predicts the direction of pricing errors")
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_5_divergence.png", bbox_inches="tight")
plt.close(fig)

# ---- Tables ----
out = []
out.append("## Table 2. Out-of-sample forecast comparison, schedule-anchored\n")
out.append(f"Test rows: {sched['n']:,} ({sched['n_events']:,} events, "
           f"{sched['n_months']} months; {sched['n_excluded_event_overlap']:,} rows "
           f"excluded for event overlap). Price Brier = {sched['brier_price']:.5f}, "
           f"log loss = {sched['logloss_price']:.5f}, base rate = {sched['base_rate']:.3f}.\n")
out.append("| Model | Brier | Log loss | ΔBrier vs price (×10⁻⁴) | t (event) | t (monthly) |")
out.append("|---|---|---|---|---|---|")
for mname, lab in zip(models, ["Isotonic (price)", "Logistic logit(p)", "GBM (p,h)",
                               "GBM +path", "GBM +path+structure", "GBM full"]):
    v = sched["models"][mname]
    out.append(f"| {lab} | {v['brier']:.5f} | {v['logloss']:.5f} | "
               f"{1e4*v['delta_brier_vs_price']:+.2f} | {v['dm_event']['t']:.2f} | "
               f"{v['dm_monthly']['t']:.2f} |")

out.append("\n## Table 3. By horizon (GBM full vs price, schedule-anchored)\n")
out.append("| h | n | Brier price | Brier GBM | Δ (×10⁻⁴) | DM t (event) |")
out.append("|---|---|---|---|---|---|")
for h, v in sorted(sched["by_horizon"].items(), key=lambda kv: int(kv[0])):
    out.append(f"| {h} | {v['n']:,} | {v['brier_price']:.5f} | {v['brier_gbm_full']:.5f} | "
               f"{1e4*(v['brier_price']-v['brier_gbm_full']):+.2f} | {v['dm_t']:.2f} |")

res = M["res"]
out.append("\n## Table 4. Resolution-anchored robustness\n")
out.append(f"Test rows: {res['n']:,}; price Brier = {res['brier_price']:.5f}.\n")
out.append("| Model | Brier | ΔBrier (×10⁻⁴) | t (event) | t (monthly) |")
out.append("|---|---|---|---|---|")
for mname in models:
    v = res["models"][mname]
    out.append(f"| {mname} | {v['brier']:.5f} | {1e4*v['delta_brier_vs_price']:+.2f} | "
               f"{v['dm_event']['t']:.2f} | {v['dm_monthly']['t']:.2f} |")

out.append("\n## Table 5. Where the model wins (ΔBrier vs price, ×10⁻⁴)\n")
out.append("| Split | Group | n | Δ | t |")
out.append("|---|---|---|---|---|")
for split, groups in I["where"].items():
    for g, v in groups.items():
        out.append(f"| {split} | {g} | {v['n']:,} | {1e4*v['delta_brier']:+.2f} | {v['t']:.2f} |")

out.append("\n## Table 6. Divergence-threshold backtest (schedule-anchored, OOS)\n")
out.append("| θ | Cost | Trades | Mean ret | t (event) | Monthly mean | t (monthly) | Win |")
out.append("|---|---|---|---|---|---|---|---|")
for theta in ["0.02", "0.05", "0.1"]:
    for ck in ["gross", "net_1c"]:
        k = f"theta{theta}_{ck}"
        if k not in I["backtest"]:
            continue
        v = I["backtest"][k]
        out.append(f"| {theta} | {ck} | {v['n_trades']:,} | {100*v['mean_ret']:+.2f}% | "
                   f"{v['t_clust']:.2f} | {100*v['monthly_mean']:+.2f}% | "
                   f"{v['monthly_t']:.2f} | {100*v['win_rate']:.1f}% |")

S = json.load(open(f"{ROOT}/results/models/supplement.json"))["sched"]
out.append("\n## Table 7. Mature folds (test months ≥ 2024-07) and honest blends\n")
out.append("| Specification | ΔBrier vs price (×10⁻⁴) | t (event) | t (monthly) |")
out.append("|---|---|---|---|")
for m, lab in [("iso", "Isotonic (mature)"), ("logit", "Logistic (mature)"),
               ("gbm_price", "GBM price (mature)"), ("gbm_full", "GBM full (mature)")]:
    v = S["mature"][m]
    out.append(f"| {lab} | {1e4*v['delta_brier']:+.2f} | {v['t_event']:.2f} | {v['t_monthly']:.2f} |")
for m, lab in [("logit_honest", "Blend price/logistic (λ* chosen on 1st half)"),
               ("gbm_full_honest", "Blend price/GBM-full (λ* chosen on 1st half)")]:
    v = S["blend"][m]
    out.append(f"| {lab}, λ*={v['lambda_star']:.2f} | {1e4*v['delta_brier']:+.2f} | "
               f"{v['t_event']:.2f} | {v['t_monthly']:.2f} |")

out.append("\n## Table 8. Recalibration gains by horizon (schedule-anchored, ΔBrier ×10⁻⁴)\n")
out.append("| h | n | Isotonic (t) | Logistic (t) | GBM full (t) |")
out.append("|---|---|---|---|---|")
for h, v in sorted(S["by_horizon"].items(), key=lambda kv: int(kv[0])):
    out.append(f"| {h} | {v['n']:,} | {1e4*v['iso']['delta_brier']:+.2f} ({v['iso']['t']:.1f}) | "
               f"{1e4*v['logit']['delta_brier']:+.2f} ({v['logit']['t']:.1f}) | "
               f"{1e4*v['gbm_full']['delta_brier']:+.2f} ({v['gbm_full']['t']:.1f}) |")

# Fig 6: recalibration gain by horizon
fig, ax = plt.subplots(figsize=(6.5, 3.6))
hs = sorted(int(h) for h in S["by_horizon"])
ax.axhline(0, color=GRAY, lw=1, ls="--")
for m, col, lab in [("logit", BLUE, "Logistic recalibration"), ("iso", GREEN, "Isotonic"),
                    ("gbm_full", RED, "GBM full")]:
    ax.plot(hs, [1e4 * S["by_horizon"][str(h)][m]["delta_brier"] for h in hs],
            "o-", color=col, ms=4, lw=1.3, label=lab)
ax.set_xscale("log"); ax.set_xticks(hs); ax.set_xticklabels(hs)
ax.set_xlabel("Horizon (days before scheduled end, log scale)")
ax.set_ylabel("ΔBrier vs price (×10⁻⁴)")
ax.set_title("Out-of-sample gains concentrate at long horizons — and only for simple models")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_6_horizon_gains.png", bbox_inches="tight")
plt.close(fig)

with open(f"{ROOT}/results/models/tables.md", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("figures ->", FIG, "| tables -> results/models/tables.md")
