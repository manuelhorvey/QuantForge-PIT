from __future__ import annotations
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone

class TrialLedger:
    """Append-only JSONL research ledger with a content hash per entry."""
    def __init__(self, path: str | Path): self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def append(self, *, trial_id: str, hypothesis: str, parameters: dict, data_hash: str,
               metrics: dict, status: str = "UNVALIDATED") -> str:
        payload = {"trial_id": trial_id, "hypothesis": hypothesis, "parameters": parameters,
                   "data_hash": data_hash, "metrics": metrics, "status": status,
                   "created_at": datetime.now(timezone.utc).isoformat()}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        record = {**payload, "entry_sha256": digest}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        return digest
