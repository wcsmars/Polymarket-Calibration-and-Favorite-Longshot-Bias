"""Validation shared by metadata collection and daily-history construction."""
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np


def json_values(value):
    """Convert NumPy values and unavailable numeric estimates to strict JSON."""
    if isinstance(value, np.ndarray):
        return json_values(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.generic):
        return json_values(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key.item() if isinstance(key, np.generic) else key: json_values(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    return value


def write_json(path, value):
    """Atomically replace a UTF-8 result file; undefined estimates are JSON null."""
    # Serialize before opening any output, so unsupported values preserve the
    # last complete result. allow_nan=False also checks dictionary keys.
    text = json.dumps(json_values(value), indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def market_id(value):
    """Keep identifiers as strings; missing identifiers cannot be deduplicated."""
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"Invalid market identifier: {value!r}")
    return str(value).strip()


def parse_flag(value, default=False):
    """Parse boolean metadata without treating the string 'false' as true."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return default


def read_jsonl(path):
    """Read objects without silently skipping corrupted input records."""
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: invalid JSON record") from error
            yield record


def prepare_append(path):
    """Protect resume from a interrupted trailing write, retaining its bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        return
    # Only an unterminated final JSON fragment may be recovered automatically.
    # Complete but malformed records are input errors, caught by read_jsonl.
    with path.open("rb+") as stream:
        end = stream.seek(0, 2)
        stream.seek(end - 1)
        if stream.read(1) == b"\n":
            return
        position = end
        tail = b""
        while position:
            size = min(position, 8192)
            position -= size
            stream.seek(position)
            tail = stream.read(size) + tail
            boundary = tail.rfind(b"\n")
            if boundary >= 0:
                position += boundary + 1
                tail = tail[boundary + 1:]
                break
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            backup = path.with_name(path.name + ".partial")
            # Do not overwrite evidence from an earlier interrupted run.
            suffix = 1
            while backup.exists():
                backup = path.with_name(path.name + f".partial.{suffix}")
                suffix += 1
            backup.write_bytes(tail)
            stream.truncate(position)
            warnings.warn(f"Recovered interrupted final record in {path}; fragment saved to {backup}",
                          RuntimeWarning)
        else:
            stream.seek(0, 2)
            stream.write(b"\n")


def normalize_history(history, *, strict=True):
    """Return sorted unique [timestamp, probability] rows.

    Collection rejects invalid quotes and retries. Existing research snapshots
    retain their raw bytes; construction discards invalid quotes with a warning.
    Conflicting prices at one timestamp are ambiguous and always rejected.
    """
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    points = {}
    invalid = 0
    for point in history:
        try:
            if isinstance(point, dict):
                timestamp, price = point["t"], point["p"]
            elif isinstance(point, (list, tuple)) and len(point) == 2:
                timestamp, price = point
            else:
                raise ValueError("expected timestamp/price pair")
            if isinstance(timestamp, bool) or isinstance(price, bool):
                raise ValueError("boolean quote")
            timestamp, price = float(timestamp), float(price)
            if not (math.isfinite(timestamp) and timestamp >= 0 and timestamp.is_integer()
                    and math.isfinite(price) and 0 <= price <= 1):
                raise ValueError("invalid timestamp or probability")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            if strict:
                raise ValueError(f"Invalid history quote: {point!r}") from error
            invalid += 1
            continue
        if timestamp in points and points[timestamp] != price:
            raise ValueError(f"Conflicting prices at timestamp {timestamp}")
        points[timestamp] = price
    if invalid:
        warnings.warn(f"Discarded {invalid} invalid history quote(s)", RuntimeWarning)
    return np.asarray(sorted(points.items()), dtype=float).reshape(-1, 2)


def load_histories(path):
    """Read saved successes; later failed attempts never erase a valid history."""
    histories = {}
    for record in read_jsonl(path):
        identifier = market_id(record.get("id"))
        history, count = record.get("history"), record.get("n")
        if not isinstance(count, int) or isinstance(count, bool) or count < -1:
            raise ValueError(f"Invalid history count for {identifier}")
        if not isinstance(history, list) or (count >= 0 and count != len(history)):
            raise ValueError(f"History count mismatch for {identifier}")
        if count == -1:
            if history:
                raise ValueError(f"Failed history contains quotes for {identifier}")
            continue
        try:
            histories[identifier] = normalize_history(history, strict=False)
        except ValueError as error:
            raise ValueError(f"Invalid history for market {identifier}: {error}") from error
    return histories
