"""Signal storage and memory management utilities."""

from __future__ import annotations

import json
import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import database as db

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SIGNALS_DIR = BASE_DIR / "signals"
MEMORY_PATH = BASE_DIR / "memory.json"
STATUS_PATH = BASE_DIR / "automation_status.json"


class SignalManager:
    """Handles persistence for timeframe snapshots, signals, and Claude memory."""

    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        if not MEMORY_PATH.exists():
            MEMORY_PATH.write_text("[]", encoding="utf-8")
        if not STATUS_PATH.exists():
            STATUS_PATH.write_text("{}", encoding="utf-8")

    @staticmethod
    def _slug_pair(pair: str) -> str:
        return pair.replace("/", "").replace(" ", "_")

    @staticmethod
    def _timestamp_label(ts: datetime | None = None) -> str:
        moment = ts or datetime.utcnow()
        return moment.strftime("%Y%m%dT%H%M%S")

    @staticmethod
    def _generate_signal_id() -> str:
        suffix = "".join(random.choices(string.digits, k=4))
        return f"SIG-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{suffix}"

    def save_timeframe_snapshot(self, pair: str, timeframe: str, snapshot: Dict[str, Any]) -> Path:
        """Persist timeframe data under data/<pair>/<tf>/timestamp.json."""
        pair_dir = DATA_DIR / self._slug_pair(pair) / timeframe
        pair_dir.mkdir(parents=True, exist_ok=True)
        file_path = pair_dir / f"{self._timestamp_label()}.json"
        file_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        return file_path

    def save_signal(self, signal_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Store signal JSON file and record in SQLite database."""
        pair = signal_payload.get("pair", "UNKNOWN")
        signal_id = signal_payload.get("signal_id") or self._generate_signal_id()
        signal_payload["signal_id"] = signal_id
        signal_payload.setdefault("source", "automation")

        pair_dir = SIGNALS_DIR / self._slug_pair(pair)
        pair_dir.mkdir(parents=True, exist_ok=True)
        signal_path = pair_dir / f"{signal_id}.json"
        signal_path.write_text(json.dumps(signal_payload, indent=2, default=str), encoding="utf-8")
        signal_payload.setdefault("file_path", str(signal_path))

        db_payload: Dict[str, Any] = {
            "pair": pair,
            "recommendation": signal_payload.get("signal", "NO TRADE"),
            "confidence": signal_payload.get("confidence"),
            "entry_price": signal_payload.get("entry_price"),
            "stop_loss": signal_payload.get("stop_loss"),
            "take_profit_1": signal_payload.get("take_profit"),
            "take_profit_2": signal_payload.get("secondary_take_profit"),
            "take_profit_3": signal_payload.get("tertiary_take_profit"),
            "lot_size": signal_payload.get("lot_size"),
            "risk_amount": signal_payload.get("risk_amount"),
            "potential_profit_1": signal_payload.get("potential_profit"),
            "potential_profit_2": signal_payload.get("secondary_potential_profit"),
            "potential_profit_3": signal_payload.get("tertiary_potential_profit"),
            "analysis_summary": signal_payload.get("analysis_summary", {}),
            "detailed_explanation": signal_payload.get("commentary", {}),
            "signal_id": signal_payload.get("signal_id"),
        }

        try:
            signal_db_id = db.save_signal(db_payload)
            signal_payload["database_id"] = signal_db_id
        except Exception as exc:  # pragma: no cover - database failures logged only
            signal_payload["database_error"] = str(exc)

        return signal_payload
    def append_memory(self, memory_entry: Dict[str, Any], max_entries: int = 20) -> None:
        """Append new Claude memory to memory.json with a cap."""
        try:
            history = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
        history.append(memory_entry)
        if len(history) > max_entries:
            history = history[-max_entries:]
        MEMORY_PATH.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

    def load_recent_memory(self, pair: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Load most recent memory entries for the specified pair."""
        try:
            history: List[Dict[str, Any]] = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        filtered = [entry for entry in history if entry.get("pair") == pair]
        return filtered[-limit:]

    def load_memory_by_signal_id(self, signal_id: str) -> List[Dict[str, Any]]:
        """Return memory entries linked to a specific signal_id."""
        try:
            history: List[Dict[str, Any]] = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return [entry for entry in history if entry.get("signal_id") == signal_id]

    def update_status(self, pair: str, status: Dict[str, Any]) -> None:
        """Persist latest automation status for a currency pair."""
        try:
            payload: Dict[str, Any] = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        payload[pair] = status
        STATUS_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def load_status(self) -> Dict[str, Any]:
        """Return automation status map."""
        try:
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def list_signals(self, pair: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Load recent stored signals from the filesystem."""
        signals: List[Dict[str, Any]] = []
        pairs = [pair] if pair else [p.name for p in SIGNALS_DIR.iterdir() if p.is_dir()]
        for pair_name in pairs:
            pair_dir = SIGNALS_DIR / pair_name
            if not pair_dir.exists():
                continue
            for file_path in sorted(pair_dir.glob("*.json"), reverse=True):
                try:
                    content = json.loads(file_path.read_text(encoding="utf-8"))
                    content.setdefault("file_path", str(file_path))
                    signals.append(content)
                except json.JSONDecodeError:
                    continue
        signals.sort(key=lambda item: item.get("generated_at", ""), reverse=True)
        return signals[:limit]
