"""Claude analysis utilities for automated signal generation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
import anthropic

load_dotenv()


@dataclass
class ClaudeConfig:
    model_priority: List[str]
    max_tokens: int = 3500


class ClaudeAnalyzer:
    """Handles prompt construction and Claude API calls for cycle analysis."""

    def __init__(self, config: ClaudeConfig | None = None) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in environment variables")

        self.client = anthropic.Anthropic(api_key=api_key)
        # Allow environment overrides for model priority (comma-separated)
        env_models = os.getenv("ANTHROPIC_MODEL_PRIORITY") or os.getenv("ANTHROPIC_MODEL")
        if env_models:
            model_priority = [m.strip() for m in env_models.split(',') if m.strip()]
        else:
            # Per user request, prioritize Claude 4 models
            model_priority = [
                "claude-opus-4-1-20250805",
                "claude-opus-4-1",
                "claude-sonnet-4-20250514",
                "claude-sonnet-4-0",
            ]
        self.config = config or ClaudeConfig(
            model_priority=model_priority,
            max_tokens=3500,
        )

    REQUIRED_MARKDOWN_HEADERS = [
        "## Overall View",
        "## Multi-timeframe Technical Summary",
        "## Key Levels",
        "## News & Sentiment",
        "## Trade Plan (RR 1:4)",
        "## Strategy Notes",
    ]

    def _truncate_sequence(self, items: Any, max_items: int = 5) -> Any:
        if not isinstance(items, list):
            return items
        if len(items) <= max_items:
            return items
        return items[-max_items:]

    def _condense_indicators(self, indicators: Any) -> Any:
        if not isinstance(indicators, dict):
            return indicators
        trimmed: Dict[str, Any] = {}
        for key, value in indicators.items():
            if isinstance(value, list):
                trimmed[key] = self._truncate_sequence(value)
            elif isinstance(value, dict):
                sub_trim: Dict[str, Any] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        sub_trim[sub_key] = self._truncate_sequence(sub_value)
                    else:
                        sub_trim[sub_key] = sub_value
                trimmed[key] = sub_trim
            else:
                trimmed[key] = value
        return trimmed

    def _condense_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(snapshot, dict):
            return {}
        condensed: Dict[str, Any] = {
            "timeframe": snapshot.get("timeframe"),
            "timestamp": snapshot.get("timestamp"),
            "price": snapshot.get("price"),
        }
        if "indicators" in snapshot:
            condensed["indicators"] = self._condense_indicators(snapshot.get("indicators"))
        if "news" in snapshot and isinstance(snapshot.get("news"), list):
            trimmed_news = snapshot.get("news")[:5]
            sanitized_news: List[Dict[str, Any]] = []
            for item in trimmed_news:
                if isinstance(item, dict):
                    pruned = {k: item.get(k) for k in ("title", "source", "url", "published_at") if k in item}
                    sanitized_news.append(pruned)
            if sanitized_news:
                condensed["news"] = sanitized_news
        sentiment = snapshot.get("sentiment") or {}
        if isinstance(sentiment, dict) and sentiment:
            condensed["sentiment"] = {
                k: sentiment.get(k)
                for k in ("count", "compound", "positive", "negative", "neutral", "modifier")
                if k in sentiment
            }
        for key in ("candles", "rows"):
            values = snapshot.get(key)
            if isinstance(values, list) and values:
                condensed[key] = self._truncate_sequence(values)
        return condensed

    def _is_markdown_rich(self, payload: Dict[str, Any]) -> bool:
        md = (payload.get("ai_analysis_md") or "").strip()
        if not md:
            return False
        words = md.split()
        if len(words) < 220:
            return False
        return all(header in md for header in self.REQUIRED_MARKDOWN_HEADERS)

    def _markdown_retry_instruction(self, previous_markdown: str | None = None) -> str:
        note = (
            "IMPORTANT: Your previous answer did not include the full markdown template with 250-600 words. "
            "Re-run the analysis and strictly follow every heading from the template. Provide detailed bullet points, "
            "explicit indicator references, and ensure confidence justification references data."
        )
        if previous_markdown:
            snippet = previous_markdown.strip()
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            note += f"\nPrevious attempt snippet (for awareness only, do NOT reuse):\n```markdown\n{snippet}\n```"
        return note

    def build_prompt(
        self,
        pair: str,
        phase_a_data: Dict[str, Any],
        phase_b_data: Dict[str, Any],
        sentiment_score: float,
        confirmation_summary: str
    ) -> str:
        """Compose the Claude prompt with phased technical data and sentiment."""
        return f"""As a professional forex analyst AI, generate a structured trading signal for {pair}.

**Input Data:**
- 1H: RSI={phase_a_data.get('1H', {}).get('rsi')}, EMA9={phase_a_data.get('1H', {}).get('ema9')}, EMA21={phase_a_data.get('1H', {}).get('ema21')}, Volume={phase_a_data.get('1H', {}).get('volume')}
- 30M: RSI={phase_a_data.get('30M', {}).get('rsi')}, EMA9={phase_a_data.get('30M', {}).get('ema9')}, EMA21={phase_a_data.get('30M', {}).get('ema21')}, Volume={phase_a_data.get('30M', {}).get('volume')}
- 15M: RSI={phase_b_data.get('15M', {}).get('rsi')}, EMA9={phase_b_data.get('15M', {}).get('ema9')}, EMA21={phase_b_data.get('15M', {}).get('ema21')}, MACD={phase_b_data.get('15M', {}).get('macd')}, Volume={phase_b_data.get('15M', {}).get('volume')}
- Sentiment Score: {sentiment_score:.2f} (-1.0 to 1.0)
- Extended Confirmation: {confirmation_summary}

Output a single JSON object with:
- pair, recommendation (BUY/SELL/NO TRADE), confidence (0-100),
- entry_price, stop_loss, take_profit,
- lot_size_100_risk, lot_size_200_risk,
- analysis_markdown: detailed reasoning.

Keep responses compact and strictly valid JSON."""

    def _safe_json_parse(self, text: str) -> Dict[str, Any] | None:
        if not text:
            return None
        s = str(text).strip()
        if s.startswith("```"):
            parts = s.split("\n", 1)
            s = parts[1] if len(parts) > 1 else s
            s = s.strip()
            if s.endswith("```"):
                s = s[: -3]
            s = s.strip()
            if s.lower().startswith("json"):
                s = s[4:].lstrip("\n").lstrip()
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    snippet = s[start : end + 1]
                    parsed = json.loads(snippet)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
            return None

    def analyze_cycle(
        self,
        pair: str,
        phase_a_data: Dict[str, Any],
        phase_b_data: Dict[str, Any],
        sentiment_score: float,
        confirmation_summary: str
    ) -> Dict[str, Any]:
        """Call Claude to analyze the completed cycle."""
        prompt = self.build_prompt(pair, phase_a_data, phase_b_data, sentiment_score, confirmation_summary)
        last_error: Exception | None = None
        parsed_result: Dict[str, Any] | None = None

        for model in self.config.model_priority:
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=self.config.max_tokens,
                    system="You generate structured trading signals based on supplied data.",
                    messages=[{"role": "user", "content": prompt}],
                )
                if response and response.content:
                    raw_text = response.content[0].text  # type: ignore[index]
                    parsed = self._safe_json_parse(raw_text)
                    if parsed:
                        parsed.setdefault("pair", pair)
                        parsed.setdefault("generated_at", datetime.utcnow().isoformat())
                        if parsed.get("risk_amount") in (None, ""):
                            parsed["risk_amount"] = 100
                        # Map fields so downstream components can render
                        if not parsed.get("signal") and parsed.get("recommendation"):
                            parsed["signal"] = str(parsed.get("recommendation")).upper()
                        # Prefer analysis_markdown when present
                        if parsed.get("analysis_markdown") and not parsed.get("ai_analysis_md"):
                            parsed["ai_analysis_md"] = parsed["analysis_markdown"]
                        if not parsed.get("ai_analysis_md"):
                            parsed["ai_analysis_md"] = parsed.get("analysis_summary", {}).get("text") or ""
                        return parsed
                    parsed_result = {
                        "pair": pair,
                        "signal": "NO TRADE",
                        "confidence": "LOW",
                        "analysis_summary": {
                            "raw_response": raw_text,
                            "error": "Failed to parse Claude response as JSON",
                        },
                        "generated_at": datetime.utcnow().isoformat(),
                    }
                    continue
            except Exception as exc:
                last_error = exc
                continue

        if parsed_result:
            return parsed_result
        raise RuntimeError(f"All Claude model attempts failed: {last_error}")

    def answer_follow_up(self, signal_data: dict, chat_history: list[dict], question: str) -> str:
        """Generate a contextual follow-up answer from Claude. Stays on the same pair/signal."""
        pair = (signal_data or {}).get('pair', '').upper()
        system_prompt = (
            "You are an expert forex trading assistant named Traydr AI. "
            "Answer ONLY about the specific signal and currency pair supplied in the context. "
            "Do not switch instruments or pairs. If the user asks about a different pair, politely steer back to the selected pair. "
            "Be concise, factual, and actionable. Never provide financial advice; instead explain the data-driven reasoning."
        )

        history_formatted = "\n".join([f"{msg.get('role','user')}: {msg.get('content','')}" for msg in (chat_history or [])])

        prompt = f"""You are discussing the signal for PAIR={pair}.

STRICT CONSTRAINTS:
- Talk only about {pair} and this specific signal context.
- If the question is off-topic (different pair or asset), acknowledge and redirect to {pair}.
- When giving levels, repeat the exact numbers from the signal unless the question asks for recalculation.

Original Signal Data (JSON):
```json
{json.dumps(signal_data, indent=2)}
```

Previous Conversation (last 30 days):
{history_formatted}

User Question: {question}

Provide a short answer:
1) One-line direct answer.
2) Up to 3 bullets referencing RSI/EMA/MACD/sentiment from the signal.
3) One risk to watch next."""
        last_error: Exception | None = None
        for model in self.config.model_priority:
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"All Claude follow-up attempts failed: {last_error}")

    def analyze_comprehensive(self, claude_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        NEW: Analyze comprehensive multi-timeframe + fundamental data
        
        Args:
            claude_input: Dictionary containing:
                - pair: Currency pair
                - multi_timeframe_data: Results from MultiTimeframeCollector
                - fundamental_data: Results from FinnhubNewsProvider
                - timestamp: Analysis timestamp
        
        Returns:
            Dictionary with recommendation, confidence, entry/exit levels, and analysis
        """
        pair = claude_input.get("pair", "UNKNOWN")
        mtf_data = claude_input.get("multi_timeframe_data", {})
        fundamental = claude_input.get("fundamental_data")
        integrity_warnings = claude_input.get("integrity_warnings", "")
        integrity_summary = claude_input.get("integrity_summary", {})
        
        # Build data quality notice
        data_quality_notice = ""
        if integrity_warnings:
            data_quality_notice = f"\n\n**DATA QUALITY NOTICE:**\n{integrity_warnings}\n"
        
        # Build comprehensive prompt
        prompt = f"""As a professional forex analyst AI, generate a comprehensive trading signal for {pair}.
{data_quality_notice}

You have access to complete multi-timeframe technical analysis (9 timeframes) and fundamental news data.

**MULTI-TIMEFRAME TECHNICAL DATA:**

Monthly (1M): {self._format_timeframe_data(mtf_data.get("1M"))}
Weekly (1W): {self._format_timeframe_data(mtf_data.get("1W"))}
Daily (1D): {self._format_timeframe_data(mtf_data.get("1D"))}
4-Hour (4H): {self._format_timeframe_data(mtf_data.get("4H"))}
1-Hour (1H): {self._format_timeframe_data(mtf_data.get("1H"))}
30-Minute (30M): {self._format_timeframe_data(mtf_data.get("30M"))}
15-Minute (15M): {self._format_timeframe_data(mtf_data.get("15M"))}
5-Minute (5M): {self._format_timeframe_data(mtf_data.get("5M"))}
1-Minute (1M): {self._format_timeframe_data(mtf_data.get("1M"))}

**FUNDAMENTAL DATA:**

Sentiment: {fundamental.overall_risk if fundamental else "Unknown"} | USD Bias: {fundamental.usd_bias if fundamental else "Unknown"}
Key Theme: {fundamental.key_theme if fundamental else "No major theme"}

Upcoming High-Impact Events: {len([e for e in (fundamental.upcoming_events if fundamental else []) if e.impact == "HIGH"])}
Recent Surprises: {len([e for e in (fundamental.recent_events if fundamental else []) if e.surprise_impact in ["Large", "Moderate"]])}

**INSTRUCTIONS:**

1. Analyze the multi-timeframe alignment (higher timeframes set bias, lower confirm entry)
2. Consider fundamental sentiment and upcoming events
3. Provide a clear BUY/SELL/NO TRADE recommendation
4. Set realistic entry, stop loss, and take profit levels
5. Justify confidence (0-100) based on alignment across timeframes

Output a single JSON object with:
- pair: "{pair}"
- recommendation: "BUY" or "SELL" or "NO TRADE"
- confidence: number 0-100
- entry_price: number
- stop_loss: number  
- take_profit: number
- lot_size_100_risk: string
- lot_size_200_risk: string
- analysis_markdown: detailed multi-timeframe reasoning with markdown formatting

Keep responses valid JSON only."""

        last_error: Exception | None = None
        for model in self.config.model_priority:
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=self.config.max_tokens,
                    system="You generate comprehensive trading signals based on multi-timeframe and fundamental data.",
                    messages=[{"role": "user", "content": prompt}],
                )
                if response and response.content:
                    raw_text = response.content[0].text
                    parsed = self._safe_json_parse(raw_text)
                    if parsed:
                        parsed.setdefault("pair", pair)
                        parsed.setdefault("generated_at", datetime.utcnow().isoformat())
                        if parsed.get("risk_amount") in (None, ""):
                            parsed["risk_amount"] = 100
                        if not parsed.get("signal") and parsed.get("recommendation"):
                            parsed["signal"] = str(parsed.get("recommendation")).upper()
                        if parsed.get("analysis_markdown") and not parsed.get("ai_analysis_md"):
                            parsed["ai_analysis_md"] = parsed["analysis_markdown"]
                        return parsed
            except Exception as exc:
                last_error = exc
                continue
        
        # Fallback
        return {
            "pair": pair,
            "signal": "NO TRADE",
            "recommendation": "NO TRADE",
            "confidence": 0,
            "analysis_summary": {"error": f"All Claude attempts failed: {last_error}"},
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def _format_timeframe_data(self, data) -> str:
        """Format timeframe data for Claude prompt"""
        if not data or not hasattr(data, 'close'):
            return "No data"
        
        parts = [f"Close={data.close:.5f}"]
        
        # Price Action
        if hasattr(data, 'volume') and data.volume:
            parts.append(f"Vol={data.volume:.0f}")
        
        # Momentum
        if hasattr(data, 'rsi_14') and data.rsi_14:
            parts.append(f"RSI={data.rsi_14:.1f}({data.rsi_interpretation})")
        if hasattr(data, 'macd_histogram') and data.macd_histogram:
            parts.append(f"MACD_Hist={data.macd_histogram:.5f}")
        
        # Moving Averages
        if hasattr(data, 'ema_20') and data.ema_20:
            parts.append(f"EMA20={data.ema_20:.5f}")
        if hasattr(data, 'sma_50') and data.sma_50:
            parts.append(f"SMA50={data.sma_50:.5f}")
        
        # Volume Indicators
        if hasattr(data, 'obv') and data.obv:
            parts.append(f"OBV={data.obv:.0f}({data.obv_trend})")
        if hasattr(data, 'volume_ratio') and data.volume_ratio:
            parts.append(f"Vol_Ratio={data.volume_ratio:.2f}x")
        if hasattr(data, 'mfi_14') and data.mfi_14:
            parts.append(f"MFI={data.mfi_14:.1f}({data.mfi_interpretation})")
        if hasattr(data, 'ad_line') and data.ad_line:
            parts.append(f"A/D={data.ad_line:.0f}")
        if hasattr(data, 'cmf_20') and data.cmf_20:
            parts.append(f"CMF={data.cmf_20:.3f}({data.cmf_interpretation})")
        if hasattr(data, 'vwma_20') and data.vwma_20:
            parts.append(f"VWMA={data.vwma_20:.5f}")
        
        # Trend Indicators
        if hasattr(data, 'adx_14') and data.adx_14:
            parts.append(f"ADX={data.adx_14:.1f}({data.adx_interpretation})")
        if hasattr(data, 'plus_di') and data.plus_di and hasattr(data, 'minus_di') and data.minus_di:
            parts.append(f"+DI={data.plus_di:.1f}/-DI={data.minus_di:.1f}")
        if hasattr(data, 'psar') and data.psar:
            parts.append(f"PSAR={data.psar:.5f}({data.psar_trend})")
        if hasattr(data, 'trend_strength') and data.trend_strength:
            parts.append(f"Trend={data.trend_strength}")
        
        # Additional Momentum
        if hasattr(data, 'cci_20') and data.cci_20:
            parts.append(f"CCI={data.cci_20:.1f}({data.cci_interpretation})")
        if hasattr(data, 'williams_r') and data.williams_r:
            parts.append(f"WilliamsR={data.williams_r:.1f}({data.williams_interpretation})")
        
        return ", ".join(parts)

    def answer_general_question(self, chat_history: list[dict], question: str) -> str:
        """Generate a contextual answer for the general chat."""
        system_prompt = (
            "You are an expert forex trading assistant named Traydr AI. "
            "You can discuss forex, financials, money making, and general stock advice. "
            "Never provide financial advice; instead, provide educational information and data-driven analysis."
        )

        history_formatted = "\n".join([f"{msg.get('role','user')}: {msg.get('content','')}" for msg in (chat_history or [])])

        prompt = f"""Previous Conversation (last 30 days):
{history_formatted}

User Question: {question}

Provide a concise and informative answer."""

        last_error: Exception | None = None
        for model in self.config.model_priority:
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"All Claude general chat attempts failed: {last_error}")