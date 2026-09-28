"""Fetch metadata for all resolved Polymarket markets via the Gamma API.

The API rejects offset > 10,000, so we paginate by endDate windows
(end_date_min / end_date_max), recursively splitting any window that
approaches the offset cap. Rows are deduped by market id and appended to
data/raw/markets_meta.jsonl. Completed windows are checkpointed so the
script is resumable. Pacing is deliberately polite.
"""
import json
from pathlib import Path
import os
import random
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = f"{ROOT}/data/raw/markets_meta.jsonl"
CHECKPOINT = f"{ROOT}/data/raw/fetch_checkpoint.json"
BASE = "https://gamma-api.polymarket.com/markets"
PAGE = 100
MAX_OFFSET = 9900
PACE = 1.5  # seconds between requests (throttled to avoid IP blocks)

KEEP = [
    "id", "question", "slug", "category", "endDate", "endDateIso",
    "closedTime", "createdAt", "startDate", "volumeNum", "liquidityNum",
    "outcomes", "outcomePrices", "clobTokenIds", "marketType",
    "enableOrderBook", "closed", "archived", "restricted",
    "umaResolutionStatus", "negRisk", "groupItemTitle",
]


def get(params, retries=6):
    import subprocess
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    delay = 5
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sf", "-m", "60", url],
                capture_output=True, timeout=90, check=True).stdout
            data = json.loads(out.decode())
            if not isinstance(data, list):
                raise ValueError(f"non-list response: {str(data)[:120]}")
            time.sleep(PACE)
            return data
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Metadata request failed after {retries} attempts; "
                    "completed windows remain checkpointed. Check the API "
                    "response or use narrower date windows before retrying."
                ) from e
            print(f"request failed ({type(e).__name__}: {e}); backing off {delay}s", flush=True)
            time.sleep(delay + random.random() * 3)
            delay = min(delay * 2, 300)


def month_windows(start_year=2020, end_year=2030):
    months = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            months.append(date(y, m, 1))
    months.append(date(end_year + 1, 1, 1))
    return [(months[i].isoformat(), months[i + 1].isoformat())
            for i in range(len(months) - 1)]


def _utc(bound):
    """Parse a window bound ('2024-01-01' or '2024-01-16T12:00:00Z') as UTC."""
    t = datetime.fromisoformat(bound.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def midpoint(d1, d2):
    a, b = _utc(d1), _utc(d2)
    if b - a < timedelta(seconds=2):
        raise RuntimeError(f"window {d1}..{d2} cannot be split further; "
                           "too many markets share one end time")
    mid = a + (b - a) / 2
    return mid.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_window(d1, d2, seen, out_f):
    """Fetch one [d1, d2) window; recursively split if it hits the offset cap."""
    offset = 0
    n_new = 0
    while True:
        batch = get({
            "closed": "true", "limit": PAGE, "offset": offset,
            "end_date_min": d1, "end_date_max": d2,
        })
        if not batch:
            break
        for m in batch:
            mid = m.get("id")
            if mid in seen:
                continue
            seen.add(mid)
            row = {k: m.get(k) for k in KEEP}
            ev = m.get("events") or []
            row["eventIds"] = [e.get("id") for e in ev]
            row["eventSlugs"] = [e.get("slug") for e in ev]
            row["eventNegRisk"] = [e.get("negRisk") for e in ev]
            out_f.write(json.dumps(row) + "\n")
            n_new += 1
        offset += len(batch)
        if offset > MAX_OFFSET:
            out_f.flush()
            print(f"window {d1}..{d2} hit offset cap; splitting", flush=True)
            m = midpoint(d1, d2)
            n_new += fetch_window(d1, m, seen, out_f)
            n_new += fetch_window(m, d2, seen, out_f)
            return n_new
        if len(batch) < PAGE:
            break
    return n_new


def main():
    done = set()
    if os.path.exists(CHECKPOINT):
        done = set(json.load(open(CHECKPOINT)))
    seen = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["id"])
                except Exception:
                    pass
    print(f"resuming: {len(done)} windows done, {len(seen)} markets already saved", flush=True)

    windows = month_windows()
    total = len(seen)
    with open(OUT, "a") as out_f:
        for d1, d2 in windows:
            key = f"{d1}|{d2}"
            if key in done:
                continue
            n = fetch_window(d1, d2, seen, out_f)
            total += n
            out_f.flush()
            done.add(key)
            json.dump(sorted(done), open(CHECKPOINT, "w"))
            if n:
                print(f"window {d1}..{d2}: +{n} (total {total})", flush=True)
    print(f"DONE: {total} unique resolved markets in {OUT}", flush=True)


if __name__ == "__main__":
    main()
