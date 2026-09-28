"""Generate the synthetic sample inputs in this folder.

Writes the two raw pipeline inputs in the formats the fetchers produce:

  markets_meta.jsonl     one Gamma-style metadata row per market: the fields in
                         KEEP of code/01_fetch_markets.py plus eventIds,
                         eventSlugs and eventNegRisk
  price_histories.jsonl  one CLOB-style record per sampled market, as written by
                         code/02_fetch_prices.py: {"id", "n", "history"}, where
                         history is a list of [unix_seconds, yes_price]

Every market, event, place, team and price is invented. None of it is
Polymarket data, and statistics computed from it say nothing about real markets.

Price model. A market resolves YES when a latent path W ends above its
threshold c at the closure time T. W is a Brownian motion (unit variance per
day) plus a final news jump at T with variance J, so the calibrated probability
on day t is Phi((W_t - c) / sqrt(T - t + J)). Threshold ladders share one path
across an event, so their outcomes are dependent. Winner events (exactly
one YES among several candidates) update a Dirichlet prior with daily Gaussian
signals. Quoted prices shrink the calibrated log-odds by 1 / FLB_SLOPE, which
plants a favorite-longshot bias, then add small noise and round to 0.001.

Some rows are planted so that each is removed at a chosen step of
code/03_build_sample.py's sequential filters (the first filter it fails), so
every sample-construction count is exercised. Price histories exist only for
markets that pass (02 fetches prices after 03). They include failed (n = -1),
empty and single-point records, and multi-day gaps that trip the 1.5-day
staleness limit.

Usage: python sample/make_sample.py   (rewrites both files in sample/)
"""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 42
FLB_SLOPE = 1.1      # true log-odds = FLB_SLOPE * quoted log-odds
PRICE_NOISE = 0.12   # s.d. of quote noise on the log-odds scale
DAY = 86400
UTC = timezone.utc
END_MIN = datetime(2023, 6, 1, tzinfo=UTC)
END_MAX = datetime(2025, 7, 31, tzinfo=UTC)   # 03 keeps end dates up to 2025-07-31
LATE_MIN = datetime(2025, 8, 3, tzinfo=UTC)
LATE_MAX = datetime(2025, 12, 20, tzinfo=UTC)
LIFE_BANDS = [((2.5, 12.0), 0.10), ((16.0, 35.0), 0.55),
              ((35.0, 70.0), 0.25), ((70.0, 110.0), 0.10)]

TEAMS = ["Harbor City Hawks", "Northfield Rovers", "Eastbrook Comets", "Lakeside Owls",
         "Pine Valley Bears", "Redrock Miners", "Silver Bay Sharks", "Westmoor Wolves",
         "Granite Falls Giants", "Cedar Point Pilots"]
SPORTS = ["basketball", "football", "hockey", "soccer", "cricket", "rugby"]
TOWNS = ["Lakeview", "Stonebridge", "Maple Hollow", "Brightwater", "Oak Harbor", "Fernvale"]
COUNTRIES = ["Norland", "Valdoria", "Estmark", "Corvania", "Tessaly"]
PARTIES = ["Green Alliance", "Civic Union", "Harbor Party", "Reform Bloc", "Unity List"]
TOKENS = ["Orbit", "Lumen", "Quill", "Tidal"]
FILMS = ["The Glass Orchard", "Midnight Ferry", "Copper Sky", "The Last Lighthouse"]
ALBUMS = ["Night Harbor", "Static Bloom", "Blue Meridian"]
MISSIONS = ["Aster-2", "Kestrel", "Halcyon-B"]
RAW_CATEGORY = {"sports": "Sports", "crypto": "Crypto", "crypto_daily": "Crypto",
                "elections_us": "US-current-affairs", "politics_world": "Global Politics",
                "economics": "Business", "entertainment": "Pop-Culture ",
                "science_tech": "Science", "other": None}


def when(t):
    return f"{t:%B} {t.day}, {t.year}"


def singleton_question(rng, cat, t_end):
    """Question text for a stand-alone market in a given derived category."""
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    town, country, team = pick(TOWNS), pick(COUNTRIES), pick(TEAMS)
    templates = {
        "sports": [f"Will the {team} win their {pick(SPORTS)} match on {when(t_end)}?",
                   f"Will the {team} reach the {pick(SPORTS)} playoff semifinal?"],
        "elections_us": [f"Will the incumbent win the {town} mayor race?",
                         f"Will the {town} transit ballot measure pass?",
                         f"Will turnout in the {town} council primary exceed 35%?"],
        "politics_world": [f"Will the {country} parliament pass the budget by {when(t_end)}?",
                           f"Will {country} hold a referendum on regional reform by {when(t_end)}?",
                           f"Will a ceasefire in the {country} border dispute hold until {when(t_end)}?"],
        "crypto": [f"Will the {pick(TOKENS)} stablecoin keep its peg through {when(t_end)}?",
                   f"Will the {pick(TOKENS)} altcoin be listed on a major crypto exchange by {when(t_end)}?"],
        "economics": [f"Will {country} report inflation above 3% for {t_end:%B %Y}?",
                      f"Will {country} unemployment fall below 5% in {t_end:%B %Y}?",
                      f"Will {country} enter a recession by {when(t_end)}?"],
        "entertainment": [f"Will '{pick(FILMS)}' top the weekend box office on {when(t_end)}?",
                          f"Will the album '{pick(ALBUMS)}' debut at number one?"],
        "science_tech": [f"Will the {pick(MISSIONS)} rocket launch before {when(t_end)}?",
                         f"Will a magnitude 6.0+ earthquake strike {country} by {when(t_end)}?",
                         f"Will {town} set a new record high temperature by {when(t_end)}?"],
        "other": [f"Will the {town} marathon draw more than 12,000 runners?",
                  f"Will Lake {town} freeze over before {when(t_end)}?",
                  f"Will the {town} harbor festival sell out by {when(t_end)}?"],
    }
    return pick(templates[cat])


class Builder:
    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)
        self.markets = []   # dicts with metadata fields and an optional price path
        self.n_events = 0

    # ---------- schedules ----------
    def life(self):
        k = self.rng.choice(len(LIFE_BANDS), p=[w for _, w in LIFE_BANDS])
        lo, hi = LIFE_BANDS[k][0]
        return float(self.rng.uniform(lo, hi))

    def schedule(self, life_days, end_lo=END_MIN, end_hi=END_MAX):
        """Scheduled end, closure time (or None) and creation time for one event."""
        rng = self.rng
        span = (end_hi - end_lo).total_seconds()
        t_end = end_lo + timedelta(seconds=span * float(rng.beta(2.0, 1.0)))
        t_end = t_end.replace(hour=int(rng.choice([0, 12, 16])), minute=0,
                              second=0, microsecond=0)
        if t_end > end_hi:
            t_end -= timedelta(days=1)
        kind = rng.choice(3, p=[0.7, 0.2, 0.1])  # on schedule, early, missing
        if kind == 0:
            t_close = t_end + timedelta(days=float(rng.uniform(0.05, 1.5)))
        elif kind == 1:
            t_close = t_end - timedelta(days=float(rng.uniform(2.0, 12.0)))
        else:
            t_close = None
        t_res = t_close or t_end
        created = t_res - timedelta(days=life_days)
        created = created.replace(microsecond=int(rng.integers(1000)) * 1000)
        return t_end, t_close, created

    def obs_times(self, created, t_res):
        """Daily snapshot times (unix seconds) from listing to closure, with gaps."""
        rng = self.rng
        start = created.replace(hour=0, minute=0, second=0, microsecond=0)
        start += timedelta(days=1 + int(rng.integers(0, 2)))
        days = []
        t = start
        while t <= t_res:
            days.append(t.timestamp() + int(rng.integers(0, 5)))
            t += timedelta(days=1)
        if len(days) > 12 and rng.random() < 0.12:  # illiquid stretch, no prints
            gap = int(rng.integers(3, 7))
            at = int(rng.integers(2, len(days) - gap - 1))
            del days[at:at + gap]
        return np.array(days, dtype=float)

    # ---------- price paths ----------
    def quote(self, q):
        q = np.clip(q, 1e-4, 1 - 1e-4)
        z = np.log(q / (1 - q)) / FLB_SLOPE + self.rng.normal(0, PRICE_NOISE, len(q))
        return np.clip(np.round(1 / (1 + np.exp(-z)), 3), 0.001, 0.999)

    def threshold_paths(self, ts, t_res, spreads):
        """One shared latent path; one market per threshold spread."""
        rng = self.rng
        tau = (t_res - ts) / DAY                      # days left at each snapshot
        jump = float(rng.uniform(0.05, 0.6)) * tau[0]  # variance of news at closure
        w = np.cumsum(np.r_[0.0, rng.normal(0, np.sqrt(-np.diff(tau)))])
        w_end = w[-1] + rng.normal(0, math.sqrt(max(tau[-1], 0.0) + jump))
        z0 = rng.normal(0, 1.5)
        out = []
        for s in spreads:
            c = -(z0 + s) * math.sqrt(tau[0] + jump)
            q = ndtr((w - c) / np.sqrt(np.maximum(tau, 0.0) + jump))
            out.append((int(w_end > c), self.quote(q)))
        return out

    def winner_paths(self, ts, t_res, k):
        """Posterior win probabilities for k candidates under daily signals."""
        rng = self.rng
        tau = (t_res - ts) / DAY
        prior = rng.dirichlet(np.full(k, 0.8))
        winner = int(rng.choice(k, p=prior))
        mu = 2.0 / math.sqrt(max(tau[0], 1.0))       # signal drift per day
        steps = np.r_[0.0, -np.diff(tau)]
        drift = np.outer(steps, np.eye(k)[winner]) * mu
        s = np.cumsum(drift + rng.normal(0, 1, (len(ts), k)) * np.sqrt(steps)[:, None], 0)
        logp = np.log(prior) + mu * s
        post = np.exp(logp - logp.max(1, keepdims=True))
        post /= post.sum(1, keepdims=True)
        return [(int(j == winner), self.quote(post[:, j])) for j in range(k)]

    # ---------- markets ----------
    def add(self, question, cat, t_end, t_close, created, event, y=None, prices=None,
            ts=None, group_title=None, neg_risk=False):
        rng = self.rng
        volume = float(np.exp(rng.normal(math.log(20000), 1.4)))
        self.markets.append({
            "question": question, "cat": cat, "t_end": t_end, "t_close": t_close,
            "created": created, "event": event, "y": y, "ts": ts, "prices": prices,
            "volume": max(volume, float(rng.uniform(1000, 1500))),
            "liquidity": 0 if rng.random() < 0.3 else round(float(np.exp(rng.normal(7.5, 1))), 2),
            "tokens": [str(int(rng.integers(10**17, 10**18))) for _ in range(2)],
            "outcomes": ["Yes", "No"], "resolution": None, "order_book": True,
            "group_title": group_title, "neg_risk": neg_risk,
            "neg_risk_field": neg_risk if (neg_risk or rng.random() < 0.6) else None,
            "restricted": bool(rng.random() < 0.3),
        })
        return self.markets[-1]

    def new_event(self, title):
        self.n_events += 1
        return {"n": self.n_events, "slug": slugify(title)}

    def singleton(self, cat, life=None, end_lo=END_MIN, end_hi=END_MAX, priced=True):
        t_end, t_close, created = self.schedule(life or self.life(), end_lo, end_hi)
        question = singleton_question(self.rng, cat, t_end)
        m = self.add(question, cat, t_end, t_close, created, self.new_event(question))
        if priced:
            ts = self.obs_times(created, t_close or t_end)
            if len(ts) >= 2:
                m["y"], m["prices"] = self.threshold_paths(ts, (t_close or t_end).timestamp(), [0.0])[0]
                m["ts"] = ts
            else:
                m["y"] = int(self.rng.random() < 0.5)
        else:
            m["y"] = int(self.rng.random() < 0.5)
        return m

    def ladder(self, cat):
        t_end, t_close, created = self.schedule(self.life())
        t_res = t_close or t_end
        ts = self.obs_times(created, t_res)
        if cat == "crypto":
            token = TOKENS[int(self.rng.integers(len(TOKENS)))]
            base = int(self.rng.integers(2, 9)) * 10
            text = lambda lvl: f"Will the {token} altcoin close above ${lvl} on {when(t_end)}?"
            title = f"{token} altcoin price on {when(t_end)}"
            levels = [base - 10, base, base + 10]
        else:
            country = COUNTRIES[int(self.rng.integers(len(COUNTRIES)))]
            base = int(self.rng.integers(40, 60)) * 100
            text = lambda lvl: f"Will the {country} stock index close above {lvl:,} on {when(t_end)}?"
            title = f"{country} stock index level on {when(t_end)}"
            levels = [base - 200, base, base + 200]
        event = self.new_event(title)
        paths = self.threshold_paths(ts, t_res.timestamp(), [0.9, 0.0, -0.9])
        for lvl, (y, p) in zip(levels, paths):
            self.add(text(lvl), cat, t_end, t_close, created, event, y, p, ts,
                     group_title=f"{lvl:,}")

    def winner_event(self, cat, k):
        t_end, t_close, created = self.schedule(self.life())
        t_res = t_close or t_end
        ts = self.obs_times(created, t_res)
        if cat == "sports":
            sport = SPORTS[int(self.rng.integers(len(SPORTS)))]
            names = list(self.rng.choice(TEAMS, k, replace=False))
            text = lambda nm: f"Will the {nm} win the {t_end.year} {sport} league title?"
            title = f"{t_end.year} {sport} league winner"
        elif cat == "elections_us":
            town = TOWNS[int(self.rng.integers(len(TOWNS)))]
            names = [f"Candidate {chr(65 + j)}" for j in range(k)]
            text = lambda nm: f"Will {nm} win the {town} mayor race?"
            title = f"{town} mayor race winner {t_end.year}"
        else:
            country = COUNTRIES[int(self.rng.integers(len(COUNTRIES)))]
            names = list(self.rng.choice(PARTIES, k, replace=False))
            text = lambda nm: f"Will the {nm} win the most seats in the {country} parliament?"
            title = f"{country} parliament election {t_end.year}"
        event = self.new_event(title)
        for nm, (y, p) in zip(names, self.winner_paths(ts, t_res.timestamp(), k)):
            self.add(text(nm), cat, t_end, t_close, created, event, y, p, ts,
                     group_title=str(nm), neg_risk=True)

    def crypto_daily(self, life):
        t_end, t_close, created = self.schedule(life)
        token = TOKENS[int(self.rng.integers(len(TOKENS)))]
        question = f"{token} Up or Down on {t_end:%B} {t_end.day}?"
        m = self.add(question, "crypto_daily", t_end, t_close, created, self.new_event(question))
        ts = self.obs_times(created, t_close or t_end)
        if len(ts) >= 2:
            m["y"], m["prices"] = self.threshold_paths(ts, (t_close or t_end).timestamp(), [0.0])[0]
            m["ts"] = ts
        else:
            m["y"] = int(self.rng.random() < 0.5)
        return m


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build(seed=SEED):
    """Return (metadata rows, price-history records) for the synthetic sample."""
    b = Builder(seed)
    rng = b.rng
    mix = {"sports": 34, "elections_us": 18, "politics_world": 18, "crypto": 14,
           "economics": 16, "entertainment": 16, "science_tech": 16, "other": 20}
    for cat, n in mix.items():
        for _ in range(n):
            b.singleton(cat)
    for cat in ["crypto"] * 5 + ["economics"] * 5:
        b.ladder(cat)
    for cat, k in [("sports", 4), ("sports", 5), ("sports", 3), ("sports", 4),
                   ("sports", 4), ("sports", 3), ("elections_us", 3), ("elections_us", 4),
                   ("elections_us", 3), ("politics_world", 4), ("politics_world", 3),
                   ("politics_world", 5)]:
        b.winner_event(cat, k)
    for _ in range(4):
        b.crypto_daily(float(rng.uniform(2.6, 3.6)))
    eligible = list(b.markets)

    # Planted exclusions, each removed first by a chosen filter of 03 (filters run in order).
    cats = list(mix)
    pick_cat = lambda: cats[int(rng.integers(len(cats)))]
    planted = {}
    for _ in range(10):   # binary_yes_no: not a Yes/No market
        m = b.singleton("sports", priced=False)
        a, c = rng.choice(TEAMS, 2, replace=False)
        m["question"] = f"{a} vs. {c}: who will win the {SPORTS[0]} match?"
        m["outcomes"] = [str(a), str(c)] if rng.random() < 0.7 else [str(a), str(c), "Draw"]
        planted.setdefault("binary_yes_no", []).append(m)
    for j in range(8):    # clean_resolution: 50-50 or unresolved payout
        m = b.singleton(pick_cat(), priced=False)
        m["resolution"] = ["0.5", "0.5"] if j < 5 else ["0.515", "0.485"]
        planted.setdefault("clean_resolution", []).append(m)
    for j in range(6):    # clob_market: no order book
        m = b.singleton(pick_cat(), priced=False)
        if j < 4:
            m["tokens"], m["order_book"] = None, None
        else:
            m["order_book"] = False
        planted.setdefault("clob_market", []).append(m)
    for _ in range(3):    # has_resolution_time: no closure or end timestamp
        m = b.singleton(pick_cat(), priced=False)
        m["t_close"], m["t_end"] = None, None
        planted.setdefault("has_resolution_time", []).append(m)
    for j in range(12):   # end_date_cutoff: scheduled end after 2025-07-31
        m = b.singleton(pick_cat(), end_lo=LATE_MIN, end_hi=LATE_MAX, priced=False)
        if j < 2:         # closure known, scheduled end missing
            m["t_close"] = m["t_close"] or m["t_end"]
            m["t_end"] = None
        planted.setdefault("end_date_cutoff", []).append(m)
    for _ in range(15):   # volume_filter: lifetime volume below $1,000
        m = b.singleton(pick_cat(), priced=False)
        m["volume"] = float(rng.uniform(20, 990))
        planted.setdefault("volume_filter", []).append(m)
    for j in range(8):    # lifetime_filter: listed for under 1.5 days
        m = (b.crypto_daily(float(rng.uniform(0.4, 1.4))) if j < 5 else
             b.singleton(pick_cat(), life=float(rng.uniform(0.5, 1.4)), priced=False))
        m["ts"] = m["prices"] = None
        planted.setdefault("lifetime_filter", []).append(m)

    # Market and event ids follow listing order, as on the exchange.
    b.markets.sort(key=lambda m: m["created"])
    events = {}
    for i, m in enumerate(b.markets):
        m["id"] = str(510001 + i)
        events.setdefault(m["event"]["n"], str(81001 + len(events)))

    rows, slugs = [], set()
    for m in b.markets:
        row = meta_row(m, events[m["event"]["n"]])
        if row["slug"] in slugs:  # slugs are unique on the exchange
            row["slug"] += "-" + m["id"][-3:]
        slugs.add(row["slug"])
        rows.append(row)
    check_categories(b.markets)

    # Price records only for markets that pass 03, in 03's volume order, with a
    # few fetch failures, empty histories and single prints.
    sampled = sorted(eligible, key=lambda m: -round(m["volume"], 2))
    odd = rng.choice(len(sampled), 7, replace=False)
    histories = []
    for i, m in enumerate(sampled):
        if m["ts"] is None or i in odd[:3]:
            histories.append({"id": m["id"], "n": -1 if i in odd[:3] else 0, "history": []})
        elif i in odd[3:5]:
            histories.append({"id": m["id"], "n": 0, "history": []})
        else:
            pairs = [[int(t), float(p)] for t, p in zip(m["ts"], m["prices"])]
            if i in odd[5:]:
                pairs = pairs[-1:]
            histories.append({"id": m["id"], "n": len(pairs), "history": pairs})
    return rows, histories


def meta_row(m, event_id):
    t_end, t_close, created = m["t_end"], m["t_close"], m["created"]
    y = m["y"] if m["y"] is not None else 0
    prices = m["resolution"] or (["1", "0"] if y == 1 else ["0", "1"])
    if len(m["outcomes"]) != 2 or m["outcomes"][0] != "Yes":
        prices = ["1"] + ["0"] * (len(m["outcomes"]) - 1)
    old = t_end is not None and t_end.year < 2024
    return {
        "id": m["id"], "question": m["question"],
        "slug": slugify(m["question"]),
        "category": RAW_CATEGORY[m["cat"]] if old else None,
        "endDate": t_end.strftime("%Y-%m-%dT%H:%M:%SZ") if t_end else None,
        "endDateIso": t_end.strftime("%Y-%m-%d") if t_end else None,
        "closedTime": t_close.strftime("%Y-%m-%d %H:%M:%S+00") if t_close else None,
        "createdAt": created.strftime("%Y-%m-%dT%H:%M:%S.") + f"{created.microsecond // 1000:03d}Z",
        "startDate": None if old else created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "volumeNum": round(m["volume"], 2), "liquidityNum": m["liquidity"],
        "outcomes": json.dumps(m["outcomes"]), "outcomePrices": json.dumps(prices),
        "clobTokenIds": json.dumps(m["tokens"]) if m["tokens"] else None,
        "marketType": "normal", "enableOrderBook": m["order_book"],
        "closed": True, "archived": False, "restricted": m["restricted"],
        "umaResolutionStatus": None if old else "resolved",
        "negRisk": m["neg_risk_field"], "groupItemTitle": m["group_title"],
        "eventIds": [event_id], "eventSlugs": [m["event"]["slug"]],
        "eventNegRisk": [m["neg_risk_field"]],
    }


def check_categories(markets):
    """Fail loudly if a template no longer maps to its intended category."""
    spec = importlib.util.spec_from_file_location("build_sample", ROOT / "code" / "03_build_sample.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for m in markets:
        raw = RAW_CATEGORY[m["cat"]] if m["t_end"] is not None and m["t_end"].year < 2024 else None
        got = module.derive_category(m["question"], slugify(m["question"]), raw)
        if m["outcomes"] == ["Yes", "No"] and got != m["cat"]:
            raise AssertionError(f"{m['question']!r}: expected {m['cat']}, got {got}")


def write(out_dir=HERE, seed=SEED):
    out_dir = Path(out_dir)
    rows, histories = build(seed)
    with open(out_dir / "markets_meta.jsonl", "w", newline="\n") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(out_dir / "price_histories.jsonl", "w", newline="\n") as f:
        f.writelines(json.dumps(h) + "\n" for h in histories)
    return len(rows), len(histories)


if __name__ == "__main__":
    n_rows, n_hist = write()
    print(f"wrote {n_rows} metadata rows and {n_hist} price histories to {HERE}")
