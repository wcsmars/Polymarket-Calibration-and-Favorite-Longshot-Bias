"""Render calibration-study markdown tables from results/analysis.json."""
import json
from pathlib import Path
import sys

# The preview below prints Greek letters; do not fail on consoles that cannot show them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")
ROOT = Path(__file__).resolve().parents[1]
r = json.load(open(f"{ROOT}/results/analysis.json"))
out = []


def tstars(t):
    a = abs(t)
    return "***" if a > 2.576 else "**" if a > 1.960 else "*" if a > 1.645 else ""


# Table 2: scores by horizon
out.append("## Table 2. Forecast performance by horizon\n")
out.append("| h (days) | N | Events | Base rate | Brier | BSS | REL | RES | MCB (CORP) | DSC (CORP) | Log score |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for h, d in sorted(r["horizon"].items(), key=lambda x: int(x[0])):
    s, c = d["scores"], d["corp"]
    out.append(f"| {h} | {s['n']:,} | {s['n_events']:,} | {s['base_rate']:.3f} | "
               f"{s['brier']:.4f} | {s['bss']:.3f} | {s['reliability']:.4f} | "
               f"{s['resolution']:.4f} | {c['mcb']:.4f} | {c['dsc']:.4f} | {s['log_score']:.4f} |")

# Table 3: calibration regressions by horizon
out.append("\n## Table 3. Calibration regressions by horizon\n")
out.append("| h | Linear α (SE) | Linear β (SE) | Wald p | Log-odds a (SE) | Log-odds b (SE) | b=1 z | Wald p |")
out.append("|---|---|---|---|---|---|---|---|")
for h, d in sorted(r["horizon"].items(), key=lambda x: int(x[0])):
    q = d["regressions"]
    li, lo = q["linear"], q["logodds"]
    out.append(f"| {h} | {li['alpha']:.4f} ({li['se_alpha']:.4f}) | {li['beta']:.4f} ({li['se_beta']:.4f}) | "
               f"{li['wald_p']:.4f} | {lo['a']:.4f} ({lo['se_a']:.4f}) | {lo['b']:.4f} ({lo['se_b']:.4f}) | "
               f"{lo['z_b_vs_1']:.2f} | {lo['wald_p']:.4f} |")

# Table 4: FLB buckets by horizon
out.append("\n## Table 4. Favorite-longshot buckets (y − p, pp) by horizon\n")
out.append("| h | Longshot (p<.10) n | y−p (t) | Mid n | y−p (t) | Favorite (p≥.90) n | y−p (t) |")
out.append("|---|---|---|---|---|---|---|")
for h, d in sorted(r["horizon"].items(), key=lambda x: int(x[0])):
    f = d["flb"]
    def cell(k):
        if k not in f:
            return "— | —"
        v = f[k]
        return f"{v['n']:,} | {100*v['y_minus_p']:+.2f} ({v['t']:.2f}){tstars(v['t'])}"
    out.append(f"| {h} | {cell('longshot_0_10')} | {cell('mid_10_90')} | {cell('favorite_90_100')} |")

# Table 5: moderators
out.append("\n## Table 5. Moderators at h = 7 (log-odds slope b)\n")
out.append("| Group | n | b (SE) | Brier | BSS | REL |")
out.append("|---|---|---|---|---|---|")
for name, groups in r["moderators_h7"].items():
    for g, v in groups.items():
        out.append(f"| {name}: {g} | {v['n']:,} | {v['b']:.3f} ({v['se_b']:.3f}) | "
                   f"{v['brier']:.4f} | {v['bss']:.3f} | {v['reliability']:.4f} |")

# Table 6: backtests
out.append("\n## Table 6. Trading strategies held to resolution\n")
out.append("| Horizon | Strategy | Trades | Gross mean ret (t) | Net-1¢ mean ret (t) | Win rate |")
out.append("|---|---|---|---|---|---|")
for hk in ["h7", "h30"]:
    for strat in ["buy_favorites_yes", "fade_longshots_buy_no"]:
        g = r["backtest"][hk]["gross"].get(strat)
        n = r["backtest"][hk]["net_1c"].get(strat)
        if not g:
            continue
        out.append(f"| {hk[1:]}d | {strat.replace('_',' ')} | {g['n_trades']:,} | "
                   f"{100*g['mean_ret']:+.2f}% ({g['t']:.2f}) | {100*n['mean_ret']:+.2f}% ({n['t']:.2f}) | "
                   f"{100*g['win_rate']:.1f}% |")

# Table 7: robustness
out.append("\n## Table 7. Robustness of the log-odds slope (h = 7 unless noted)\n")
out.append("| Specification | n | b (SE) |")
out.append("|---|---|---|")
base = r["horizon"]["7"]["regressions"]["logodds"]
out.append(f"| Baseline | {r['horizon']['7']['regressions']['n']:,} | {base['b']:.3f} ({base['se_b']:.3f}) |")
for k, v in r["robustness"].items():
    if isinstance(v, dict) and "b" in v:
        out.append(f"| {k.replace('_',' ')} | {v['n']:,} | {v['b']:.3f} ({v['se_b']:.3f}) |")
bp = r.get("balanced_panel_h1_7_30", {})
for hk in ["h1", "h7", "h30"]:
    if hk in bp:
        v = bp[hk]["logodds"]
        n = bp[hk]["scores"]["n"]
        out.append(f"| balanced panel {hk} (n mkts={bp['n_markets']:,}) | {n:,} | {v['b']:.3f} ({v['se_b']:.3f}) |")

# Table 8: schedule-anchored retrospective results
out.append("\n## Table 8. Schedule-anchored calibration (quotes h days before scheduled end)\n")
out.append("| h | n | Log-odds a (SE) | Log-odds b (SE) | Longshot y−p (t) | Mid y−p (t) | Favorite y−p (t) |")
out.append("|---|---|---|---|---|---|---|")
for hk, v in sorted(r.get("schedule_anchored", {}).items(), key=lambda x: int(x[0][1:])):
    lo = v["logodds"]
    f = v["flb"]
    def fc(k):
        if k not in f:
            return "—"
        w = f[k]
        return f"{100*w['y_minus_p']:+.2f} ({w['t']:.2f}){tstars(w['t'])}"
    out.append(f"| {hk[1:]} | {v['n']:,} | {lo['a']:+.4f} ({lo['se_a']:.4f}) | {lo['b']:.4f} ({lo['se_b']:.4f}) | "
               f"{fc('longshot_0_10')} | {fc('mid_10_90')} | {fc('favorite_90_100')} |")
bs = r.get("backtest_sched_h30")
if bs:
    out.append("\n**Schedule-anchored backtest (h = 30):** ")
    out.append("\nRetrospective price-based returns: scheduled anchors can fall after the "
               "closure-time proxy, and the fixed 1¢ cost does not model executable "
               "quotes, depth, or fills. See the README limitations.\n")
    for k in ["buy_favorites_yes", "fade_longshots_buy_no"]:
        g, n = bs["gross"][k], bs["net_1c"][k]
        out.append(f"- {k.replace('_',' ')}: {g['n_trades']:,} trades, gross {100*g['mean_ret']:+.2f}% "
                   f"(t={g['t']:.2f}), net-1¢ {100*n['mean_ret']:+.2f}% (t={n['t']:.2f}), win {100*g['win_rate']:.1f}%")

with open(f"{ROOT}/results/tables.md", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("\n".join(out[:30]))
print(f"\n... written to results/tables.md")
