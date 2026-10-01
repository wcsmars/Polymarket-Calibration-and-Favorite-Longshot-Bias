"""Render calibration-study markdown tables from results/analysis.json."""
import json
import math
import re
from pathlib import Path
import sys

# The preview below prints Greek letters; do not fail on consoles that cannot show them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")
ROOT = Path(__file__).resolve().parents[1]
r = json.load(open(f"{ROOT}/results/analysis.json"))
out = []


def fmt(value, spec=".2f"):
    return "—" if value is None or not math.isfinite(float(value)) else format(value, spec)


def tstars(t):
    if t is None or not math.isfinite(float(t)):
        return ""
    a = abs(t)
    return "***" if a > 2.576 else "**" if a > 1.960 else "*" if a > 1.645 else ""


# Table 2: scores by horizon
out.append("## Table 2. Forecast performance by horizon\n")
out.append("| h (days) | N | Events | Base rate | Brier | BSS | REL | RES | MCB (CORP) | DSC (CORP) | Log score |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for h, d in sorted(r["horizon"].items(), key=lambda x: int(x[0])):
    s, c = d["scores"], d["corp"]
    out.append(f"| {h} | {fmt(s['n'], ',')} | {fmt(s['n_events'], ',')} | {fmt(s['base_rate'], '.3f')} | "
               f"{fmt(s['brier'], '.4f')} | {fmt(s['bss'], '.3f')} | {fmt(s['reliability'], '.4f')} | "
               f"{fmt(s['resolution'], '.4f')} | {fmt(c['mcb'], '.4f')} | {fmt(c['dsc'], '.4f')} | {fmt(s['log_score'], '.4f')} |")

out.append("\nREL and RES use fixed probability bins; their decomposition applies to bin-mean forecasts. CORP uses tie-consistent isotonic recalibration and exactly decomposes the raw Brier score.\n")

# Table 3: calibration regressions by horizon
out.append("\n## Table 3. Calibration regressions by horizon\n")
out.append("| h | Linear α (SE) | Linear β (SE) | Wald p | Log-odds a (SE) | Log-odds b (SE) | b=1 z | Wald p |")
out.append("|---|---|---|---|---|---|---|---|")
for h, d in sorted(r["horizon"].items(), key=lambda x: int(x[0])):
    q = d["regressions"]
    li, lo = q["linear"], q["logodds"]
    out.append(f"| {h} | {fmt(li['alpha'], '.4f')} ({fmt(li['se_alpha'], '.4f')}) | {fmt(li['beta'], '.4f')} ({fmt(li['se_beta'], '.4f')}) | "
               f"{fmt(li['wald_p'], '.4f')} | {fmt(lo['a'], '.4f')} ({fmt(lo['se_a'], '.4f')}) | {fmt(lo['b'], '.4f')} ({fmt(lo['se_b'], '.4f')}) | "
               f"{fmt(lo['z_b_vs_1'], '.2f')} | {fmt(lo['wald_p'], '.4f')} |")

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
        return f"{fmt(v['n'], ',')} | {fmt(100*v['y_minus_p'], '+.2f')} ({fmt(v['t'])}){tstars(v['t'])}"
    out.append(f"| {h} | {cell('longshot_0_10')} | {cell('mid_10_90')} | {cell('favorite_90_100')} |")

# Table 5: moderators
out.append("\n## Table 5. Moderators at h = 7 (log-odds slope b)\n")
out.append("| Group | n | b (SE) | Brier | BSS | REL |")
out.append("|---|---|---|---|---|---|")
for name, groups in r["moderators_h7"].items():
    for g, v in groups.items():
        out.append(f"| {name}: {g} | {fmt(v['n'], ',')} | {fmt(v['b'], '.3f')} ({fmt(v['se_b'], '.3f')}) | "
                   f"{fmt(v['brier'], '.4f')} | {fmt(v['bss'], '.3f')} | {fmt(v['reliability'], '.4f')} |")

# Table 6: backtests
out.append("\n## Table 6. Retrospective price-based returns held to settlement\n")
out.append("One dollar per forecast row; overlapping positions are not a capital-constrained portfolio. Fixed costs do not model executable quotes, liquidity, or fills.\n")
out.append("| Horizon | Strategy | Trades | Gross mean ret (t) | Net-1¢ mean ret (t) | Win rate |")
out.append("|---|---|---|---|---|---|")
for hk in ["h7", "h30"]:
    for strat in ["buy_favorites_yes", "fade_longshots_buy_no"]:
        g = r["backtest"][hk]["gross"].get(strat)
        n = r["backtest"][hk]["net_1c"].get(strat)
        if not g:
            continue
        out.append(f"| {hk[1:]}d | {strat.replace('_',' ')} | {fmt(g['n_trades'], ',')} | "
                   f"{fmt(100*g['mean_ret'], '+.2f')}% ({fmt(g['t'])}) | {fmt(100*n['mean_ret'], '+.2f')}% ({fmt(n['t'])}) | "
                   f"{fmt(100*g['win_rate'], '.1f')}% |")

# Table 7: robustness
out.append("\n## Table 7. Robustness of the log-odds slope (h = 7 unless noted)\n")
out.append("| Specification | n | b (SE) |")
out.append("|---|---|---|")
base_record = r["horizon"].get("7")
base = base_record["regressions"]["logodds"] if base_record else None
if base is not None:
    out.append(f"| Baseline | {fmt(r['horizon']['7']['regressions']['n'], ',')} | {fmt(base['b'], '.3f')} ({fmt(base['se_b'], '.3f')}) |")
for k, v in r["robustness"].items():
    if isinstance(v, dict) and "b" in v:
        out.append(f"| {k.replace('_',' ')} | {fmt(v['n'], ',')} | {fmt(v['b'], '.3f')} ({fmt(v['se_b'], '.3f')}) |")
bp = r.get("balanced_panel_h1_7_30", {})
for hk in ["h1", "h7", "h30"]:
    if hk in bp:
        v = bp[hk]["logodds"]
        n = bp[hk]["scores"]["n"]
        out.append(f"| balanced panel {hk} (n mkts={fmt(bp['n_markets'], ',')}) | {fmt(n, ',')} | {fmt(v['b'], '.3f')} ({fmt(v['se_b'], '.3f')}) |")

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
        return f"{fmt(100*w['y_minus_p'], '+.2f')} ({fmt(w['t'])}){tstars(w['t'])}"
    out.append(f"| {hk[1:]} | {fmt(v['n'], ',')} | {fmt(lo['a'], '+.4f')} ({fmt(lo['se_a'], '.4f')}) | {fmt(lo['b'], '.4f')} ({fmt(lo['se_b'], '.4f')}) | "
               f"{fc('longshot_0_10')} | {fc('mid_10_90')} | {fc('favorite_90_100')} |")
bs = r.get("backtest_sched_h30")
if bs:
    out.append("\n**Schedule-anchored backtest (h = 30):** ")
    out.append("\nRetrospective price-based returns: scheduled anchors can fall after the "
               "historically recorded schedule, and the fixed 1¢ cost does not model executable "
               "quotes, depth, or fills. See the README limitations.\n")
    for k in ["buy_favorites_yes", "fade_longshots_buy_no"]:
        g, n = bs["gross"].get(k), bs["net_1c"].get(k)
        if not g or not n:
            continue
        out.append(f"- {k.replace('_',' ')}: {fmt(g['n_trades'], ',')} trades, gross {fmt(100*g['mean_ret'], '+.2f')}% "
                   f"(t={fmt(g['t'])}), net-1¢ {fmt(100*n['mean_ret'], '+.2f')}% (t={fmt(n['t'])}), win {fmt(100*g['win_rate'], '.1f')}%")

out.append("\nCluster standard errors use the asymptotic CR0 cluster-sum formula. Stars use unadjusted normal-reference cutoffs; repeated comparisons are descriptive. Undefined estimates are shown as —.\n")
text = re.sub(r"(?<![A-Za-z])nan(?![A-Za-z])", "—", "\n".join(out))
with open(f"{ROOT}/results/tables.md", "w", encoding="utf-8") as f:
    f.write(text)
print("\n".join(out[:30]))
print(f"\n... written to results/tables.md")
