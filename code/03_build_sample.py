"""Build the study sample from raw market metadata.

Inclusion criteria, applied in this order:
  - binary Yes/No market, resolved cleanly to 0 or 1
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = f"{ROOT}/data/raw/markets_meta.jsonl"
OUT = f"{ROOT}/data/processed/sample_markets.csv"
MIN_VOLUME = 1000.0
# Universe cutoff: metadata collection is verified complete for markets with
# scheduled end date on or before this date (window-fetch coverage boundary).
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


def main():
    rows = []
    with open(RAW) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    counts = {"all_resolved_markets": len(df)}

    def parse_list(s):
        try:
            return json.loads(s) if isinstance(s, str) else None
        except Exception:
            return None

    df["outcomes_l"] = df["outcomes"].map(parse_list)
    df["prices_l"] = df["outcomePrices"].map(parse_list)
    df["tokens_l"] = df["clobTokenIds"].map(parse_list)

    binary = df["outcomes_l"].map(lambda x: isinstance(x, list) and len(x) == 2
                                  and str(x[0]).lower() == "yes" and str(x[1]).lower() == "no")
    df = df[binary]
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

    df["y"] = df["prices_l"].map(clean_outcome)
    df = df[df["y"].notna()]
    counts["clean_resolution"] = len(df)

    has_tok = df["tokens_l"].map(lambda x: isinstance(x, list) and len(x) == 2 and all(x))
    df = df[has_tok & (df["enableOrderBook"] != False)]
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
    df = df[df["volumeNum"] >= MIN_VOLUME]
    counts["volume_filter"] = len(df)

    # Require >= 1.5 days before the closure-time proxy for horizon observations.
    life = (df["t_res"] - df["t_created"]).dt.total_seconds() / 86400
    df = df[life.isna() | (life >= 1.5)]
    counts["lifetime_filter"] = len(df)

    df["yes_token"] = df["tokens_l"].map(lambda x: x[0])
    df["event_id"] = df["eventIds"].map(lambda x: x[0] if isinstance(x, list) and x else None)
    df["event_neg_risk"] = df["eventNegRisk"].map(
        lambda x: bool(x[0]) if isinstance(x, list) and x and x[0] is not None else False)
    df["cat"] = [derive_category(q, s, c) for q, s, c in
                 zip(df["question"].fillna(""), df["slug"].fillna(""), df["category"])]

    out = df[["id", "question", "slug", "cat", "category", "y", "volumeNum",
              "liquidityNum", "t_res", "t_created", "t_end", "yes_token",
              "event_id", "event_neg_risk", "negRisk"]].copy()
    out["y"] = out["y"].astype(int)
    out = out.sort_values("volumeNum", ascending=False)  # fetch important markets first
    out.to_csv(OUT, index=False)

    print(json.dumps(counts, indent=2))
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
