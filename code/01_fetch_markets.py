"""Fetch metadata for all resolved Polymarket markets via the Gamma API.

API offset limits can change, so we paginate by endDate windows
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
import sys
import urllib.parse
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import market_id, prepare_append, read_jsonl

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


class PaginationLimitError(RuntimeError):
    """The API rejected an offset after accepting this window's first page."""


def get(params, retries=6):
    import subprocess
    if retries < 1:
        raise ValueError("retries must be positive")
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    delay = 5
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sS", "-m", "60", "-w", "\n%{http_code}", url],
                capture_output=True, timeout=90, check=True).stdout
            body, status = out.decode().rsplit("\n", 1)
            if status == "422" and params.get("offset", 0) > 0:
                raise PaginationLimitError(f"HTTP 422 at offset {params['offset']}")
            if not status.isdigit() or not 200 <= int(status) < 300:
                raise ValueError(f"HTTP {status}: {body[:200]}")
            data = json.loads(body)
            if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
                raise ValueError("metadata response must be a list of objects")
            for row in data:
                market_id(row.get("id"))
            time.sleep(PACE)
            return data
        except PaginationLimitError:
            raise
        except (OSError, ValueError, subprocess.SubprocessError) as e:
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
    return t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)


def midpoint(d1, d2):
    a, b = _utc(d1), _utc(d2)
    if b - a < timedelta(seconds=2):
        raise RuntimeError(f"window {d1}..{d2} cannot be split further; "
                           "too many markets share one end time")
    mid = a + (b - a) / 2
    return mid.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_window(d1, d2, seen, out_f):
    """Fetch a date window; overlaps at split boundaries are deduplicated by ID."""
    offset = 0
    n_new = 0
    while True:
        try:
            batch = get({
                "closed": "true", "limit": PAGE, "offset": offset,
                "order": "id", "ascending": "true",
                "end_date_min": _utc(d1).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_date_max": _utc(d2).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
        except PaginationLimitError:
            out_f.flush()
            split = midpoint(d1, d2)
            print(f"window {d1}..{d2} rejected offset {offset}; splitting", flush=True)
            return (n_new + fetch_window(d1, split, seen, out_f)
                    + fetch_window(split, d2, seen, out_f))
        if not batch:
            break
        for m in batch:
            mid = market_id(m.get("id"))
            if mid in seen:
                continue
            row = {k: m.get(k) for k in KEEP}
            row["id"] = mid
            ev = m.get("events") or []
            if not isinstance(ev, list) or any(not isinstance(e, dict) for e in ev):
                raise ValueError(f"Invalid events for market {mid}")
            row["eventIds"] = [e.get("id") for e in ev]
            row["eventSlugs"] = [e.get("slug") for e in ev]
            row["eventNegRisk"] = [e.get("negRisk") for e in ev]
            out_f.write(json.dumps(row, allow_nan=False) + "\n")
            seen.add(mid)
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
    prepare_append(OUT)
    seen = {market_id(row.get("id")) for row in read_jsonl(OUT)} if Path(OUT).exists() else set()
    done = set()
    if os.path.exists(CHECKPOINT) and seen:
        with open(CHECKPOINT, encoding="utf-8") as stream:
            saved = json.load(stream)
        if not isinstance(saved, list) or any(not isinstance(key, str) for key in saved):
            raise ValueError("Invalid metadata checkpoint: expected a list of window keys")
        done = set(saved)
    print(f"resuming: {len(done)} windows done, {len(seen)} markets already saved", flush=True)

    windows = month_windows()
    total = len(seen)
    with open(OUT, "a", encoding="utf-8") as out_f:
        for d1, d2 in windows:
            key = f"{d1}|{d2}"
            if key in done:
                continue
            n = fetch_window(d1, d2, seen, out_f)
            total += n
            out_f.flush()
            os.fsync(out_f.fileno())
            done.add(key)
            checkpoint = Path(CHECKPOINT)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(sorted(done), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, checkpoint)
            if n:
                print(f"window {d1}..{d2}: +{n} (total {total})", flush=True)
    print(f"DONE: {total} unique resolved markets in {OUT}", flush=True)


if __name__ == "__main__":
    main()
