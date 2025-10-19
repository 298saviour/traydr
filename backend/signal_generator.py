"""Signal creation utilities for Claude automation."""

from __future__ import annotations

import random
import string
from datetime import datetime
from typing import Any, Dict, List

from signal_manager import SignalManager


def _generate_signal_id() -> str:
    suffix = "".join(random.choices(string.digits, k=4))
    return f"SIG{suffix}"


class SignalGenerator:
    """Builds and persists signal payloads produced by Claude."""

    def __init__(self, manager: SignalManager | None = None) -> None:
        self.manager = manager or SignalManager()

    def create_signal(self, pair: str, claude_output: Dict[str, Any], metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        signal_id = claude_output.get("signal_id") or _generate_signal_id()
        timestamp = datetime.utcnow().isoformat()
        confidence_pct = self._normalize_confidence(claude_output.get("confidence"))
        # Extract numeric trade parameters
        entry = claude_output.get("entry_price")
        sl = claude_output.get("stop_loss")
        tp1 = claude_output.get("take_profit") or claude_output.get("take_profit_1")
        tp2 = claude_output.get("secondary_take_profit") or claude_output.get("take_profit_2")
        tp3 = claude_output.get("tertiary_take_profit") or claude_output.get("take_profit_3")
        risk_amount = claude_output.get("risk_amount") or 100  # default $100 risk per trade
        lot_size = claude_output.get("lot_size")
        if lot_size is None and entry is not None and sl is not None:
            lot_size = self._estimate_lot_size(pair, float(entry), float(sl), float(risk_amount))
        # Detailed markdown analysis if available
        analysis_md = (
            claude_output.get("ai_analysis_md")
            or claude_output.get("analysis_markdown")
            or claude_output.get("analysis_md")
            or claude_output.get("reasoning")
            or claude_output.get("analysis_summary_markdown")
        )
        # Normalize recommendation field to 'signal'
        recommendation = (
            claude_output.get("direction")
            or claude_output.get("signal")
            or claude_output.get("recommendation")
            or "HOLD"
        )
        payload: Dict[str, Any] = {
            "signal_id": signal_id,
            "pair": pair,
            "generated_at": timestamp,
            "signal": recommendation,
            "confidence": confidence_pct,
            "entry_price": entry,
            "stop_loss": sl,
            "take_profit": tp1,
            "secondary_take_profit": tp2,
            "tertiary_take_profit": tp3,
            "risk_amount": risk_amount,
            "lot_size": lot_size,
            "analysis_summary": {
                "technical": claude_output.get("technical_reasoning"),
                "sentiment": claude_output.get("sentiment_summary"),
            },
            "strategy_notes": claude_output.get("strategy_notes"),
            "commentary": {"ai_analysis_md": analysis_md} if analysis_md else {"ai_analysis_md": "No analysis provided."},
        }
        if metadata:
            payload.update(metadata)

        saved = self.manager.save_signal(payload)
        return saved

    def list_signals(self, pair: str | None = None, limit: int = 20) -> List[Dict[str, Any]]:
        return self.manager.list_signals(pair=pair, limit=limit)

    @staticmethod
    def _normalize_confidence(raw: Any) -> str | None:
        """Convert Claude confidence to percentage string, e.g. '72%'.
        Accepts numeric (0-1 or 0-100), or strings LOW/MEDIUM/HIGH.
        """
        if raw is None:
            return None
        # Numeric
        try:
            val = float(raw)
            if 0.0 <= val <= 1.0:
                return f"{int(round(val * 100))}%"
            if 1.0 < val <= 100.0:
                return f"{int(round(val))}%"
        except Exception:
            pass
        # String categories
        if isinstance(raw, str):
            s = raw.strip().lower()
            mapping = {
                'low': 33,
                'medium': 66,
                'med': 66,
                'high': 85,
                'very high': 92,
                'very_low': 20,
            }
            if s in mapping:
                return f"{mapping[s]}%"
            # Try patterns like '7/10'
            if '/' in s:
                try:
                    num, den = s.split('/')
                    pct = (float(num) / float(den)) * 100.0
                    return f"{int(round(pct))}%"
                except Exception:
                    pass
            # Try trailing % already
            if s.endswith('%'):
                return raw
        return None

    @staticmethod
    def _estimate_lot_size(pair: str, entry: float, stop_loss: float, risk_amount: float) -> float:
        """Estimate lot size to risk approximately $risk_amount based on SL distance.
        Assumptions:
        - FX pairs: ~$10 per pip per 1.0 standard lot
        - JPY pairs treated the same for simplicity
        - XAU/USD and crypto fallback to minimal 0.01 lot
        """
        try:
            distance = abs(entry - stop_loss)
            if distance <= 0:
                return 0.01
            if pair.startswith('XAU/') or pair in ('BTC/USD', 'ETH/USD'):
                return 0.01
            # Pip factor
            factor = 10000.0
            if 'JPY' in pair:
                factor = 100.0
            pips = distance * factor
            if pips <= 0:
                return 0.01
            lot = float(risk_amount) / (pips * 10.0)
            return round(max(lot, 0.01), 2)
        except Exception:
            return 0.01