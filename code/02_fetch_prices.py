"""Fetch daily YES-token history for the sampled resolved markets.

Appends validated histories to data/raw/price_histories.jsonl. Successful
records, including genuine empty histories, are skipped on resume; failed
requests are retried. Collection never overwrites the earlier raw records.
"""
import json
from http.client import HTTPException
from pathlib import Path
import sys
import threading
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import load_histories, market_id, normalize_history, prepare_append

BASE = "https://clob.polymarket.com/prices-history"
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = f"{ROOT}/data/processed/sample_markets.csv"
OUT = f"{ROOT}/data/raw/price_histories.jsonl"
WORKERS = 12


def fetch_history(token_id, retries=6):
    if retries < 1:
        raise ValueError("retries must be positive")
    params = urllib.parse.urlencode({
        "market": market_id(token_id), "interval": "max", "fidelity": 1440,
    })
    url = f"{BASE}?{params}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script/1.0"})
            with urllib.request.urlopen(req, timeout=45) as response:
                payload = json.loads(response.read().decode())
            if not isinstance(payload, dict) or "history" not in payload:
                raise ValueError("price response is missing history")
            history = normalize_history(payload["history"])
            return [{"t": int(t), "p": float(p)} for t, p in history]
        except (OSError, HTTPException, ValueError, TypeError):
            if attempt == retries - 1:
                return None
            time.sleep(min(2 ** attempt, 30))
    return None


def main():
    df = pd.read_csv(SAMPLE, dtype={"id": str, "yes_token": str})
    for column in ["id", "yes_token"]:
        df[column] = df[column].map(market_id)
    if df["id"].duplicated().any():
        raise ValueError("sample contains duplicate market IDs")
    prepare_append(OUT)
    seen = set(load_histories(OUT)) if Path(OUT).exists() else set()
    todo = df[~df["id"].isin(seen)]
    total = len(todo)
    print(f"{len(seen)} successfully fetched, {total} to go", flush=True)
    done_count = 0
    failed_count = 0
    lock = threading.Lock()
    with open(OUT, "a", encoding="utf-8") as out_f:
        def work(row):
            nonlocal done_count, failed_count
            history = fetch_history(row.yes_token)
            record = {"id": row.id, "n": len(history) if history is not None else -1,
                      "history": [[h["t"], h["p"]] for h in history] if history else []}
            with lock:
                out_f.write(json.dumps(record, allow_nan=False) + "\n")
                out_f.flush()
                done_count += 1
                failed_count += history is None
                if done_count % 500 == 0:
                    print(f"{done_count}/{total} fetched", flush=True)

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            for _ in executor.map(work, todo.itertuples(index=False)):
                pass
    print(f"DONE: {done_count} attempted, {failed_count} failed; records appended to {OUT}", flush=True)
    if failed_count:
        raise RuntimeError(f"{failed_count} price requests failed; rerun to retry these markets")


if __name__ == "__main__":
    main()
