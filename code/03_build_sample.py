"""Build the study sample from raw market metadata.

Inclusion criteria, applied in this order:
  - binary Yes/No market, closed with terminal 0/1 prices; reject explicit
    nonfinal UMA statuses (legacy rows with no status remain a qualified proxy)
  - order-book (CLOB) market with token ids
  - parseable closure-time proxy (closedTime, falling back to endDate)
  - parseable scheduled end date (endDate) on or before END_CUTOFF
  - lifetime volume >= MIN_VOLUME
  - at least 1.5 days from createdAt to the closure-time proxy (markets
    without createdAt are kept); like the proxy, this filter is retrospective
Outputs data/processed/sample_markets.csv and prints sample-construction
counts that document sample construction.
"""
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import market_id, parse_flag, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
RAW = f"{ROOT}/data/raw/markets_meta.jsonl"
OUT = f"{ROOT}/data/processed/sample_markets.csv"
MIN_VOLUME = 1000.0
# Study cutoff for scheduled end dates in the archived collection.
# Completed collection windows alone do not prove market-universe completeness.
END_CUTOFF = "2025-07-31 23:59:59+00:00"

CATEGORY_RULES = [
    ("crypto_daily", r"\b(up or down|updown)\b|-up-or-down-"),
    ("crypto", r"bitcoin|btc|ethereum|\beth\b|solana|\bsol\b|crypto|dogecoin|xrp|memecoin|binance|coinbase|microstrategy|satoshi|\bnft\b|opensea|stablecoin|defi|altcoin"),
    ("sports", r"\bnba\b|\bnfl\b|\bmlb\b|\bnhl\b|\bufc\b|\bepl\b|premier league|la liga|serie a|bundesliga|ligue 1|champions league|europa|world cup|super bowl|grand slam|wimbledon|us open|french open|australian open|masters|pga|f1\b|formula 1|grand prix|olympic|fifa|uefa|ncaa|march madness|playoff|finals mvp|world series|stanley cup|heisman|copa|boxing|wrestl|tennis|golf|soccer|football|basketball|baseball|hockey|cricket|rugby|esports|league of legends|csgo|cs2|dota|valorant"),
    ("elections_us", r"president|presidential|electoral|primar(y|ies)|caucus|senate|house seat|congress|governor|mayor|democrat|republican|gop\b|trump|biden|harris|desantis|nominee|nomination|midterm|ballot|swing state|popular vote|veep|vice president|cabinet|impeach"),
    ("politics_world", r"election|parliament|prime minister|chancellor|president of|referendum|coup|ceasefire|war\b|ukraine|russia|israel|gaza|hamas|iran|china|taiwan|north korea|nato|brexit|tariff"),
    ("economics", r"\bfed\b|fomc|rate (hike|cut)|interest rate|inflation|cpi\b|gdp\b|recession|unemployment|nonfarm|payroll|debt ceiling|treasury|powell|ecb\b|stock|s&p|nasdaq|dow\b|ipo\b|tesla|earnings|market cap"),
    ("entertainment", r"oscar|academy award|grammy|emmy|golden globe|box office|movie|album|spotify|billboard|taylor swift|kanye|drake\b|celebrity|bachelor|survivor|big brother|eurovision|game of thrones|stranger things|netflix|tiktok|youtube|mrbeast|twitch"),
    ("science_tech", r"openai|chatgpt|gpt-\d|claude|gemini|deepmind|\bagi\b|spacex|starship|nasa|launch|rocket|apple|iphone|google|microsoft|meta\b|twitter|\bx\b corp|elon|zuckerberg|ai model|artificial intelligence|nobel|covid|vaccine|pandemic|hurricane|earthquake|temperature|climate"),
]


def derive_category(question, slug, raw_cat):
    text = f"{question} {slug}".lower()
    for cat, pattern in CATEGORY_RULES:
        if re.search(pattern, text):
            return cat
    if isinstance(raw_cat, str) and raw_cat.strip():
        c = raw_cat.strip().lower()
        if "sport" in c:
            return "sports"
        if "crypto" in c:
            return "crypto"
        if "politic" in c or "current-affairs" in c:
            return "politics_world"
        if "business" in c or "econom" in c:
            return "economics"
        if "pop" in c or "culture" in c:
            return "entertainment"
        if "science" in c or "tech" in c or "coronavirus" in c:
            return "science_tech"
    return "other"


def final_or_legacy_resolution(status):
    """A proposal/dispute is not final; missing legacy status remains unverified."""
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return True
    return isinstance(status, str) and status.strip().lower() in {"", "resolved"}


def main():
    records = {}
    for row in read_jsonl(RAW):
        identifier = market_id(row.get("id"))
        row["id"] = identifier
        if identifier in records and records[identifier] != row:
            raise ValueError(f"Conflicting metadata records for market {identifier}")
        records[identifier] = row
    if not records:
        raise ValueError("No market metadata records found")
    df = pd.DataFrame(records.values())
    # Optional Gamma fields can be absent from an entire API page or snapshot.
    for column in ["question", "slug", "category", "closedTime", "endDate", "createdAt",
                   "volumeNum", "liquidityNum", "outcomes", "outcomePrices", "clobTokenIds",
                   "enableOrderBook", "eventIds", "eventNegRisk", "negRisk", "closed",
                   "umaResolutionStatus"]:
        if column not in df:
            df[column] = None
    counts = {"all_resolved_markets": len(df)}

    def parse_list(s):
        try:
            value = json.loads(s) if isinstance(s, str) else s
            return value if isinstance(value, list) else None
        except Exception:
            return None

    df["outcomes_l"] = df["outcomes"].map(parse_list)
    df["prices_l"] = df["outcomePrices"].map(parse_list)
    df["tokens_l"] = df["clobTokenIds"].map(parse_list)

    binary = df["outcomes_l"].map(lambda x: isinstance(x, list) and len(x) == 2
                                  and {str(v).strip().lower() for v in x} == {"yes", "no"})
    df = df[binary].copy()
    df["yes_index"] = df["outcomes_l"].map(
        lambda x: [str(v).strip().lower() for v in x].index("yes"))
    counts["binary_yes_no"] = len(df)

    def clean_outcome(pl):
        if not isinstance(pl, list) or len(pl) != 2:
            return None
        try:
            a, b = float(pl[0]), float(pl[1])
        except Exception:
            return None
        if a == 1.0 and b == 0.0:
            return 1
        if a == 0.0 and b == 1.0:
            return 0
        return None  # ambiguous (e.g., 0.5/0.5) or unresolved

    df["y"] = [clean_outcome(prices if index == 0 else
                               list(reversed(prices)) if isinstance(prices, list) else prices)
               for prices, index in zip(df["prices_l"], df["yes_index"])]
    # Neither closure nor a traded price of 0/1 proves a proposed or disputed
    # outcome is final. Older records without UMA status remain a documented
    # limitation rather than being silently treated as verified settlements.
    terminal_closed = df["y"].notna() & df["closed"].map(parse_flag)
    final_status = df["umaResolutionStatus"].map(final_or_legacy_resolution)
    nonfinal_count = int((terminal_closed & ~final_status).sum())
    df = df[terminal_closed & final_status].copy()
    counts["clean_resolution"] = len(df)

    has_tok = df["tokens_l"].map(lambda x: isinstance(x, list) and len(x) == 2
                                and all(isinstance(t, (str, int)) and not isinstance(t, bool)
                                        and str(t).strip() for t in x)
                                and str(x[0]) != str(x[1]))
    enabled = df["enableOrderBook"].map(lambda value: parse_flag(value, None) is not False)
    df = df[has_tok & enabled].copy()
    counts["clob_market"] = len(df)

    df["t_close"] = pd.to_datetime(df["closedTime"], errors="coerce", utc=True, format="mixed")
    df["t_end"] = pd.to_datetime(df["endDate"], errors="coerce", utc=True, format="mixed")
    # t_res is a closure-time proxy, not a verified outcome-availability time.
    df["t_res"] = df["t_close"].fillna(df["t_end"])
    df["t_created"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True, format="mixed")
    df = df[df["t_res"].notna()]
    counts["has_resolution_time"] = len(df)

    df = df[df["t_end"].notna() & (df["t_end"] <= pd.Timestamp(END_CUTOFF))]
    counts["end_date_cutoff"] = len(df)

    df["volumeNum"] = pd.to_numeric(df["volumeNum"], errors="coerce").fillna(0)
    df = df[np.isfinite(df["volumeNum"]) & (df["volumeNum"] >= MIN_VOLUME)].copy()
    counts["volume_filter"] = len(df)

    # Require >= 1.5 days before the closure-time proxy for horizon observations.
    life = (df["t_res"] - df["t_created"]).dt.total_seconds() / 86400
    df = df[life.isna() | (life >= 1.5)]
    counts["lifetime_filter"] = len(df)

    df["yes_token"] = [str(tokens[index]).strip()
                       for tokens, index in zip(df["tokens_l"], df["yes_index"])]
    df["event_id"] = df["eventIds"].map(lambda x: x[0] if isinstance(x, list) and x else None)
    df["event_neg_risk"] = df["eventNegRisk"].map(
        lambda x: parse_flag(x[0]) if isinstance(x, list) and x else False)
    df["negRisk"] = df["negRisk"].map(parse_flag)
    df["cat"] = [derive_category(q, s, c) for q, s, c in
                 zip(df["question"].fillna(""), df["slug"].fillna(""), df["category"])]

    out = df[["id", "question", "slug", "cat", "category", "y", "volumeNum",
              "liquidityNum", "t_res", "t_created", "t_end", "yes_token",
              "event_id", "event_neg_risk", "negRisk"]].copy()
    out["y"] = out["y"].astype(int)
    out = out.sort_values("volumeNum", ascending=False)  # fetch important markets first
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(json.dumps(counts, indent=2))
    print("Closed terminal-price markets rejected for explicit nonfinal resolution status:", nonfinal_count)
    legacy_status_count = int(df["umaResolutionStatus"].map(
        lambda status: not isinstance(status, str) or not status.strip()).sum())
    print("Final sample markets with unverified legacy resolution status:", legacy_status_count)
    print("\nBy derived category:")
    print(out["cat"].value_counts().to_string())
    print("\nClosure-proxy year:")
    print(out["t_res"].dt.year.value_counts().sort_index().to_string())
    print("\nVolume distribution:")
    print(out["volumeNum"].describe(percentiles=[.1, .25, .5, .75, .9, .99]).to_string())
    print("\nBase rate P(Y=1):", round(out["y"].mean(), 4))
    with open(f"{ROOT}/results/sample_construction.json", "w") as f:
        json.dump(counts, f, indent=2)


if __name__ == "__main__":
    main()
