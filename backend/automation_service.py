import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import schedule
import random
import os
import math

from claude_analyzer import ClaudeAnalyzer
from forex_analyzer import ForexAnalyzer
from indicators import IndicatorCollector
from technical_analysis import TechnicalAnalysis
from signal_manager import SignalManager
from signal_generator import SignalGenerator
from backend.api_handlers.news_data import NewsDataHandler
from multi_timeframe_collector import MultiTimeframeCollector
from finnhub_news_provider import FinnhubNewsProvider
from final_confidence_layer import FinalConfidenceLayer
import database as db


def _current_time() -> str:
    return datetime.utcnow().isoformat()

SELECTED_MARKETS = [
    "EUR/USD",   # Euro vs US Dollar
    "GBP/USD",   # British Pound vs US Dollar
    "USD/JPY",   # US Dollar vs Japanese Yen
    "XAU/USD",   # Gold vs US Dollar
    "EUR/GBP",   # Euro vs British Pound
    "USD/CAD",   # US Dollar vs Canadian Dollar
    "AUD/USD",   # Australian Dollar vs US Dollar
    "USD/CHF",   # US Dollar vs Swiss Franc
    "NZD/USD",   # New Zealand Dollar vs US Dollar
    "EUR/JPY",   # Euro vs Japanese Yen
    "GBP/JPY",   # British Pound vs Japanese Yen
    "AUD/JPY",   # Australian Dollar vs Japanese Yen
    "ETH/USD",   # ETHEREUM
    "BTC/USD"    # BITCOIN
]
DEFAULT_PAIRS = SELECTED_MARKETS


def _load_selected_pairs() -> List[str]:
    env_value = os.getenv("SCHEDULER_PAIRS")
    if env_value:
        pairs = [p.strip().upper().replace(" ", "") for p in env_value.split(",") if p.strip()]
        return [p if "/" in p else p[:3] + "/" + p[3:] for p in pairs]
    return DEFAULT_PAIRS


def _load_fx_outputsize() -> int:
    try:
        value = int(os.getenv("TWELVEDATA_OUTPUTSIZE", "120"))
        return max(1, min(value, 5000))
    except Exception:
        return 120

class SchedulerService:
    """Manages multi-pair automation cycles with rotation and auto-stop."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_pair: Optional[str] = None
        self._analyzed_pairs: List[str] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._status: Dict[str, Any] = {
            "state": "idle",
            "pair": None,
            "last_cycle_start": None,
            "last_cycle_end": None,
            "last_signal_id": None,
            "last_error": None,
            "progress_log": [],  # list of {ts, msg}
        }

        self.analyzer = ForexAnalyzer()
        self.indicator_collector = IndicatorCollector(self.analyzer.av_provider)
        self.news_fetcher = NewsDataHandler()
        self.signal_manager = SignalManager()
        self.signal_generator = SignalGenerator(self.signal_manager)
        self.claude_analyzer = ClaudeAnalyzer()
        
        # New comprehensive analysis systems
        self.mtf_collector = MultiTimeframeCollector(
            data_provider=self.analyzer,
            technical_analyzer=TechnicalAnalysis
        )
        self.finnhub_provider = FinnhubNewsProvider()
        self.confidence_layer = FinalConfidenceLayer(
            data_provider=self.analyzer,
            technical_analyzer=TechnicalAnalysis,
            news_provider=self.finnhub_provider,
            signal_history=self.signal_manager
        )

        self._scheduler = schedule.Scheduler()
        self._selected_pairs: List[str] = _load_selected_pairs()
        self.fx_outputsize = _load_fx_outputsize()
        if self._selected_pairs:
            self._push_log(f"Configured pairs: {', '.join(self._selected_pairs)}")
        else:
            self._selected_pairs = DEFAULT_PAIRS
        self._apply_fx_outputsize()
        # Daily evaluation job to check outcomes for signals older than 3 days
        self._scheduler.every().day.at("02:00").do(self._evaluate_signals)

    # -------------------- Public API --------------------

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                logging.info("[scheduler] Already running.")
                return
            self.stop() 
            self._stop_event.clear()
            self._analyzed_pairs = []
            self._reset_schedule()
            self._thread = threading.Thread(target=self._run_loop, name="scheduler-loop", daemon=True)
            self._thread.start()
            self._status.update({"state": "running", "pair": None, "last_error": None})
            self._push_log(f"Started scheduler. Will analyze pairs: {', '.join(self._selected_pairs)}")
            logging.info("[scheduler] Started automation service.")
            # Trigger first tick immediately for instant feedback
            try:
                self._push_log("Triggering first analysis tick now…")
                # Run tick in a background thread to avoid blocking request
                threading.Thread(target=self._tick, name="scheduler-initial-tick", daemon=True).start()
            except Exception:
                pass

    def stop(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                logging.info("[scheduler] Stopping current scheduler")
                self._stop_event.set()
                self._thread.join(timeout=5)
            self._scheduler.clear()
            self._thread = None
            self._active_pair = None
            self._status.update({"state": "idle", "pair": None})

    def configure_pairs(self, pairs: List[str]) -> None:
        with self._lock:
            if not pairs:
                self._selected_pairs = DEFAULT_PAIRS
            else:
                dedup = []
                for p in pairs:
                    if p not in dedup:
                        dedup.append(p)
                self._selected_pairs = dedup
            self._push_log(f"Configured pairs: {', '.join(self._selected_pairs)}")

    def configure_fx_outputsize(self, size: int) -> None:
        with self._lock:
            self.fx_outputsize = max(1, min(size, 5000))
            self._apply_fx_outputsize()

    def _apply_fx_outputsize(self) -> None:
        try:
            self.indicator_collector.set_fx_outputsize(self.fx_outputsize)
            logging.info("[scheduler] FX outputsize set to %s", self.fx_outputsize)
        except Exception as exc:
            logging.warning("[scheduler] Failed to apply FX outputsize: %s", exc)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    # -------------------- Internal Logic --------------------

    def _run_loop(self) -> None:
        logging.info("[scheduler] Entering run loop")
        while not self._stop_event.is_set():
            try:
                self._scheduler.run_pending()
            except Exception as exc:
                logging.exception("[scheduler] Error in schedule loop: %s", exc)
                self._status["last_error"] = str(exc)
                time.sleep(5)
            time.sleep(1)
        logging.info("[scheduler] Exit run loop")

    def _reset_schedule(self) -> None:
        self._scheduler.clear()
        self._scheduler.every(1).minutes.do(self._tick)

    def _tick(self) -> None:
        with self._lock:
            if self._status.get("state") in ("cycle_running", "resting"):
                return

            remaining_pairs = [p for p in self._selected_pairs if p not in self._analyzed_pairs]
            if not remaining_pairs:
                self._push_log("All pairs analyzed. Scheduler idle.")
                self.stop()
                return

            next_pair = remaining_pairs[0]
            self._active_pair = next_pair
            logging.info("[scheduler] Tick: selected %s for next cycle.", next_pair)
            self._push_log(f"Tick: preparing next cycle for {next_pair}.")
            thread = threading.Thread(target=self._run_cycle, args=(next_pair,), daemon=True)
            thread.start()

    def _run_phase_a(self, pair: str, from_symbol: str, to_symbol: str) -> Dict[str, int]:
        """Phase A: Collect ALL indicators for 1H and 30M timeframes.
        Returns: dict with success/failure counts for data quality tracking.
        """
        self._push_log(f"[Phase A] Starting for {pair}")
        timeframes = [("1H", "60min"), ("30M", "30min")]
        quality = {"fetched": 0, "saved": 0, "failed": 0}
        
        for label, interval in timeframes:
            if self._stop_event.is_set(): 
                return quality

            self._push_log(f"[Phase A] Fetching ALL indicators for {pair} {label} (RSI, MACD, EMA, BBANDS, ATR, OBV)")
            indicators = self.indicator_collector.collect_all(from_symbol, to_symbol, interval)
            quality["fetched"] += 1
            
            # Extract all indicators
            rsi = indicators.get('RSI', {}).get('value')
            ema = indicators.get('EMA', {})
            ema9 = ema.get('ema9')
            ema21 = ema.get('ema21')
            macd_raw = indicators.get('MACD', {})
            bbands_raw = indicators.get('BBANDS', {})
            atr = indicators.get('ATR', {}).get('value')
            obv = indicators.get('OBV', {}).get('value')
            volume_analysis = indicators.get('Volume', {})
            
            # Log what was successfully retrieved
            retrieved = []
            if rsi: retrieved.append(f"RSI={rsi:.2f}")
            if ema9: retrieved.append(f"EMA9={ema9:.4f}")
            if ema21: retrieved.append(f"EMA21={ema21:.4f}")
            if macd_raw and not macd_raw.get('error'): retrieved.append("MACD")
            if bbands_raw and not bbands_raw.get('error'): retrieved.append("BBANDS")
            if atr: retrieved.append(f"ATR={atr:.4f}")
            if obv: retrieved.append(f"OBV={obv:.2f}")
            self._push_log(f"[Phase A] {label} Retrieved: {', '.join(retrieved) if retrieved else 'None'}")

            # Coerce MACD
            def _coerce_float(value: Any) -> Optional[float]:
                if value is None:
                    return None
                try:
                    coerced = float(value)
                except (TypeError, ValueError):
                    return None
                return coerced if not math.isnan(coerced) else None

            macd = {}
            if isinstance(macd_raw, dict):
                macd_line = _coerce_float(macd_raw.get('macd') or macd_raw.get('MACD'))
                macd_signal = _coerce_float(macd_raw.get('signal') or macd_raw.get('MACD_Signal'))
                macd_hist = _coerce_float(macd_raw.get('hist') or macd_raw.get('MACD_Hist'))
                if all(v is not None for v in (macd_line, macd_signal, macd_hist)):
                    macd = {'macd': macd_line, 'signal': macd_signal, 'hist': macd_hist}

            # Coerce BBANDS
            bbands = {}
            if isinstance(bbands_raw, dict):
                upper = _coerce_float(bbands_raw.get('upper'))
                middle = _coerce_float(bbands_raw.get('middle'))
                lower = _coerce_float(bbands_raw.get('lower'))
                if all(v is not None for v in (upper, middle, lower)):
                    bbands = {'upper': upper, 'middle': middle, 'lower': lower}

            # Check essential indicators
            essential_indicators = [rsi, ema9, ema21]
            if all(essential_indicators):
                saved_indicators = ["RSI", "EMA9", "EMA21"]
                if macd: saved_indicators.append("MACD")
                if bbands: saved_indicators.append("BBANDS")
                if atr: saved_indicators.append("ATR")
                if obv: saved_indicators.append("OBV")
                db.save_phase_a_indicators(pair, label, rsi, ema9, ema21, macd, bbands, atr, obv, volume_analysis)
                self._push_log(f"[Phase A] ✓ Saved {label} indicators: {', '.join(saved_indicators)}")
                quality["saved"] += 1
            else:
                missing = []
                if not rsi: missing.append("RSI")
                if not ema9: missing.append("EMA9")
                if not ema21: missing.append("EMA21")
                self._push_log(f"[Phase A] ✗ Missing essential indicators for {label}: {', '.join(missing)}")
                quality["failed"] += 1
            
            time.sleep(1)  # brief pause to stay within rate limits
        
        return quality

    def _run_phase_b(self, pair: str, from_symbol: str, to_symbol: str) -> Tuple[float, Dict[str, int]]:
        """Phase B: Collect ALL indicators for 15M, 4H, and 1D timeframes plus sentiment.
        Returns: (sentiment_score, quality_dict)
        """
        self._push_log(f"[Phase B] Starting for {pair}")
        quality = {"fetched": 0, "saved": 0, "failed": 0}
        timeframes = [("15M", "15min"), ("4H", "4h"), ("1D", "1day")]
        
        # Fetch news and compute sentiment once
        self._push_log(f"[Phase B] Fetching news for {pair}")
        news = self.news_fetcher.get_news(pair, 5, 3)
        sentiment = self._compute_sentiment(news)
        sentiment_score = sentiment.get('compound', 0.0)

        def _coerce_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                coerced = float(value)
            except (TypeError, ValueError):
                return None
            return coerced if not math.isnan(coerced) else None

        for label, interval in timeframes:
            if self._stop_event.is_set():
                return sentiment_score, quality

            self._push_log(f"[Phase B] Fetching ALL indicators for {pair} {label} (RSI, MACD, EMA, BBANDS, ATR, OBV)")
            indicators = self.indicator_collector.collect_all(from_symbol, to_symbol, interval)
            quality["fetched"] += 1
            
            # Extract all indicators
            rsi = indicators.get('RSI', {}).get('value')
            ema = indicators.get('EMA', {})
            ema9 = ema.get('ema9')
            ema21 = ema.get('ema21')
            macd_raw = indicators.get('MACD', {})
            bbands_raw = indicators.get('BBANDS', {})
            atr = indicators.get('ATR', {}).get('value')
            obv = indicators.get('OBV', {}).get('value')
            volume_analysis = indicators.get('Volume', {})
            
            # Log what was successfully retrieved
            retrieved = []
            if rsi: retrieved.append(f"RSI={rsi:.2f}")
            if ema9: retrieved.append(f"EMA9={ema9:.4f}")
            if ema21: retrieved.append(f"EMA21={ema21:.4f}")
            if macd_raw and not macd_raw.get('error'): retrieved.append("MACD")
            if bbands_raw and not bbands_raw.get('error'): retrieved.append("BBANDS")
            if atr: retrieved.append(f"ATR={atr:.4f}")
            if obv: retrieved.append(f"OBV={obv:.2f}")
            self._push_log(f"[Phase B] {label} Retrieved: {', '.join(retrieved) if retrieved else 'None'}")

            # Coerce MACD
            macd = {}
            if isinstance(macd_raw, dict):
                macd_line = _coerce_float(macd_raw.get('macd') or macd_raw.get('MACD'))
                macd_signal = _coerce_float(macd_raw.get('signal') or macd_raw.get('MACD_Signal'))
                macd_hist = _coerce_float(macd_raw.get('hist') or macd_raw.get('MACD_Hist'))
                if all(v is not None for v in (macd_line, macd_signal, macd_hist)):
                    macd = {'macd': macd_line, 'signal': macd_signal, 'hist': macd_hist}

            # Coerce BBANDS
            bbands = {}
            if isinstance(bbands_raw, dict):
                upper = _coerce_float(bbands_raw.get('upper'))
                middle = _coerce_float(bbands_raw.get('middle'))
                lower = _coerce_float(bbands_raw.get('lower'))
                if all(v is not None for v in (upper, middle, lower)):
                    bbands = {'upper': upper, 'middle': middle, 'lower': lower}

            # Check essential indicators
            essential_indicators = [rsi, ema9, ema21]
            if all(essential_indicators):
                saved_indicators = ["RSI", "EMA9", "EMA21"]
                if macd: saved_indicators.append("MACD")
                if bbands: saved_indicators.append("BBANDS")
                if atr: saved_indicators.append("ATR")
                if obv: saved_indicators.append("OBV")
                db.save_phase_b_indicators(pair, label, rsi, ema9, ema21, macd, bbands, atr, obv, sentiment_score, volume_analysis)
                self._push_log(f"[Phase B] ✓ Saved {label} indicators: {', '.join(saved_indicators)}")
                quality["saved"] += 1
            else:
                missing = []
                if not rsi: missing.append("RSI")
                if not ema9: missing.append("EMA9")
                if not ema21: missing.append("EMA21")
                self._push_log(f"[Phase B] ✗ Missing essential indicators for {label}: {', '.join(missing)}")
                quality["failed"] += 1
            
            time.sleep(1)  # brief pause between timeframes

        return sentiment_score, quality

    def _run_cycle(self, pair: str) -> None:
        """NEW COMPREHENSIVE WORKFLOW: Multi-timeframe + Finnhub + Final Confidence"""
        with self._lock:
            if self._status.get("state") == "cycle_running":
                return
            self._status["state"] = "cycle_running"
            self._status["last_cycle_start"] = _current_time()
            self.signal_manager.update_status(pair, dict(self._status))
        
        stop_after_cycle = False
        last_signal_id = None
        try:
            self._push_log(f"🚀 Starting comprehensive analysis for {pair}")
            
            # STEP 1: Multi-Timeframe Data Collection (Phase 1 + Phase 2)
            self._push_log(f"📊 Collecting multi-timeframe technical data for {pair}")
            mtf_results = self.mtf_collector.collect_full_dataset(
                pair=pair,
                progress_callback=self._push_log
            )
            
            if self._stop_event.is_set(): 
                return
            
            # STEP 2: Fundamental News Collection (Finnhub)
            self._push_log(f"📰 Fetching fundamental news and economic data for {pair}")
            fundamental_data = self.finnhub_provider.fetch_fundamental_data(
                pair=pair,
                progress_callback=self._push_log
            )
            
            if self._stop_event.is_set():
                return
            
            # STEP 3: Initial Claude Analysis
            self._push_log(f"🤖 Running initial AI analysis for {pair}")
            
            # Get data integrity warnings
            integrity_warnings = self.mtf_collector.get_integrity_warnings_for_claude()
            integrity_summary = self.mtf_collector.get_integrity_summary()
            
            # Log integrity summary
            if integrity_summary["finnhub_fallbacks"] > 0:
                self._push_log(f"⚠️ {integrity_summary['finnhub_fallbacks']} timeframe(s) used Finnhub backup")
            if integrity_summary["repaired_feeds"] > 0:
                self._push_log(f"⚠️ {integrity_summary['repaired_feeds']} timeframe(s) had data repaired")
            
            # Prepare comprehensive data for Claude
            claude_input = {
                "pair": pair,
                "multi_timeframe_data": mtf_results,
                "fundamental_data": fundamental_data,
                "integrity_warnings": integrity_warnings,  # Pass warnings to Claude
                "integrity_summary": integrity_summary,
                "timestamp": _current_time()
            }
            
            # Get initial recommendation from Claude
            claude_payload = self.claude_analyzer.analyze_comprehensive(claude_input)
            
            if self._stop_event.is_set():
                return
            
            # STEP 4: Final Confidence Layer (if signal is viable)
            if claude_payload.get("recommendation") not in ["NO TRADE", "HOLD"]:
                self._push_log(f"✨ Computing final confidence for {pair}")
                
                final_output = self.confidence_layer.compute_final_confidence(
                    pair=pair,
                    direction=claude_payload.get("recommendation", "BUY"),
                    entry_price=float(claude_payload.get("entry_price", 0)),
                    stop_loss=float(claude_payload.get("stop_loss", 0)),
                    take_profit=float(claude_payload.get("take_profit", 0)),
                    base_confidence=float(claude_payload.get("confidence", 70)),
                    technical_data=mtf_results,
                    progress_callback=self._push_log
                )
                
                # Update payload with final confidence
                claude_payload["confidence"] = final_output.final_confidence
                claude_payload["invalidation_level"] = final_output.invalidation.price_level
                claude_payload["expiry_time"] = final_output.invalidation.time_expiry
                claude_payload["confirmation_summary"] = final_output.confirmation_summary
                
                self._push_log(f"✓ Final confidence: {final_output.final_confidence:.1f}%")
            
            # STEP 5: Generate and Save Signal
            self._push_log(f"💾 Generating signal for {pair}")
            saved_signal = self.signal_generator.create_signal(pair, claude_payload)
            last_signal_id = saved_signal.get('signal_id')
            
            self._push_log(f"✅ Signal generated (ID: {last_signal_id})")
            self._push_log(f"📈 Recommendation: {claude_payload.get('recommendation', 'N/A')}")
            self._push_log(f"🎯 Confidence: {claude_payload.get('confidence', 0):.1f}%")
            
            self._analyzed_pairs.append(pair)
            self._push_log("🏁 Cycle complete. Scheduler will remain idle until manually restarted.")
            stop_after_cycle = True

        except InterruptedError:
            logging.info("[cycle] Interrupted for pair %s", pair)
        except Exception as exc:
            logging.exception("[cycle] Error during cycle for %s: %s", pair, exc)
            self._status["last_error"] = str(exc)
            self._push_log(f"❌ Error: {str(exc)}")
        finally:
            with self._lock:
                self._status["last_cycle_end"] = _current_time()
                self._status["last_signal_id"] = last_signal_id or self._status.get("last_signal_id")
                self._status["pair"] = None if stop_after_cycle else pair
                self._status["state"] = "idle" if stop_after_cycle or self._stop_event.is_set() else ("running" if self._active_pair else "idle")
                self.signal_manager.update_status(pair, dict(self._status))

        if stop_after_cycle:
            self.stop()
            try:
                self.signal_manager.update_status(pair, dict(self._status))
            except Exception:
                pass

    def _run_extended_confirmation_phase(self, pair: str, from_symbol: str, to_symbol: str) -> Tuple[str, int]:
        """Run the extended confirmation phase for 4H and 1D timeframes."""
        self._push_log("[Confirmation Phase] Starting for 4H and 1D timeframes.")
        confidence_adjustment = 0
        summaries = []

        for timeframe in ["4h", "1d"]:
            try:
                df = self.analyzer.fetch_data(pair, period='1mo', interval=timeframe)
                if df is None or df.empty:
                    self._push_log(f"[Confirmation Phase] No data for {timeframe}, skipping.")
                    continue

                ta = TechnicalAnalysis(df)
                analysis = ta.get_comprehensive_analysis()

                trend_alignment = "aligned" if (analysis['trend']['trend_score'] > 0 and analysis['overall_signal'] == "BUY") or \
                                              (analysis['trend']['trend_score'] < 0 and analysis['overall_signal'] == "SELL") else "conflicting"
                volume_confirmation = "confirms" if (analysis['volume']['obv_trend'] == "Bullish" and analysis['overall_signal'] == "BUY") or \
                                                  (analysis['volume']['obv_trend'] == "Bearish" and analysis['overall_signal'] == "SELL") else "does not confirm"

                summaries.append(f"{timeframe.upper()}: Trend {trend_alignment}, Volume {volume_confirmation}.")

                if trend_alignment == "aligned":
                    confidence_adjustment += 5
                else:
                    confidence_adjustment -= 5
                
                if volume_confirmation == "confirms":
                    confidence_adjustment += 5
                else:
                    confidence_adjustment -= 5

            except Exception as e:
                self._push_log(f"[Confirmation Phase] Error analyzing {timeframe}: {e}")

        if not summaries:
            return "Extended confirmation phase could not be completed.", 0

        return " ".join(summaries), confidence_adjustment

    # -------------------- Logging helper --------------------
    def _push_log(self, message: str) -> None:
        try:
            with self._lock:
                log = self._status.setdefault("progress_log", [])
                log.append({"ts": _current_time(), "msg": message})
                # Keep last 50 messages
                if len(log) > 50:
                    del log[: len(log) - 50]
                # Also mirror to saved status file for persistence per pair
                if self._active_pair:
                    try:
                        # Include latest status snapshot with log for UI
                        snap = dict(self._status)
                        self.signal_manager.update_status(self._active_pair, snap)
                    except Exception:
                        pass
        except Exception:
            logging.exception("[scheduler] Failed to push log message")

    def _enter_rest(self, pair: str, seconds: int) -> None:
        with self._lock:
            self._status.update({"state": "resting"})
            self.signal_manager.update_status(pair, dict(self._status))
        logging.info("[cycle] Resting for %s seconds", seconds)
        rest_notice_given = False
        for i in range(seconds):
            if self._stop_event.is_set():
                raise InterruptedError("Scheduler stopped during rest")
            time.sleep(1)
            if not rest_notice_given and i > 5:
                self._push_log(f"Resting ~{seconds//60} minutes. Scheduler will auto-resume.")
                rest_notice_given = True

    @staticmethod
    def _has_required_indicators(snapshots: List[Dict[str, Any]], required: List[str]) -> bool:
        for s in snapshots:
            inds = s.get("indicators") or {}
            for r in required:
                v = inds.get(r)
                if v is None:
                    return False
                # treat empty dict/list as missing
                if isinstance(v, (list, dict)) and len(v) == 0:
                    return False
        return True

    @staticmethod
    def _normalize_macd_block(phase_b_payload: Dict[str, Any]) -> None:
        if not isinstance(phase_b_payload, dict):
            return
        for timeframe, data in phase_b_payload.items():
            if not isinstance(data, dict):
                continue
            macd_block = data.get('macd')
            if not isinstance(macd_block, dict):
                continue
            for key in ('macd', 'signal', 'hist'):
                value = macd_block.get(key)
                if value is None:
                    continue
                try:
                    coerced = float(value)
                except (TypeError, ValueError):
                    macd_block[key] = None
                    continue
                macd_block[key] = coerced if not math.isnan(coerced) else None

    def _compute_sentiment(self, headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not headlines:
            return {"count": 0, "compound": 0.0, "positive": 0, "negative": 0, "neutral": 0, "modifier": {"bias": "neutral", "adjust": 0}}
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        scores = []
        pos = neg = neu = 0
        for article in headlines:
            title = article.get("title") or ""
            if title:
                s = analyzer.polarity_scores(title)
                scores.append(s.get("compound", 0.0))
                if s.get("compound", 0) >= 0.2:
                    pos += 1
                elif s.get("compound", 0) <= -0.2:
                    neg += 1
                else:
                    neu += 1
        if not scores:
            return {"count": 0, "compound": 0.0, "positive": 0, "negative": 0, "neutral": 0, "modifier": {"bias": "neutral", "adjust": 0}}
        compound = sum(scores) / len(scores)
        total = len(scores)
        pct_pos = pos / total if total else 0
        pct_neg = neg / total if total else 0
        modifier = {"bias": "neutral", "adjust": 0}
        if pct_pos >= 0.6:
            modifier = {"bias": "bullish", "adjust": +10}
        elif pct_neg >= 0.6:
            modifier = {"bias": "bearish", "adjust": +10}
        elif pos and neg and abs(pct_pos - pct_neg) < 0.2:
            modifier = {"bias": "contradictory", "adjust": -15}
        return {
            "count": total,
            "compound": compound,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "modifier": modifier,
        }
    # -------------------- Evaluation logic --------------------
    def _evaluate_signals(self) -> Dict[str, Any]:
        """Mark signals older than 24 hours as success, failed, or neutral."""
        updated = 0
        updated_ids: List[int] = []
        try:
            active = db.get_active_signals()
            cutoff = datetime.utcnow() - timedelta(hours=24)
            for sig in active:
                created_raw = sig.get('created_at')
                try:
                    created_str = str(created_raw)
                    if ' ' in created_str and 'T' not in created_str:
                        created_str = created_str.replace(' ', 'T')
                    created_dt = datetime.fromisoformat(created_str)
                except Exception:
                    continue
                if created_dt > cutoff:
                    continue

                pair = sig.get('pair')
                rec = (sig.get('recommendation') or '').upper()
                entry = sig.get('entry_price')
                tp = sig.get('take_profit_1') or sig.get('take_profit') or None
                sl = sig.get('stop_loss')
                if not pair or entry is None or (tp is None and sl is None):
                    continue

                current = self.analyzer.get_current_price(pair)
                if current is None:
                    continue

                status, pips = self._decide_outcome(pair, rec, entry, tp, sl, current)
                if status:
                    db.update_signal_status(sig['id'], status, pips=pips)
                    db.update_signal_performance(sig['id'], status)
                    logging.info("[evaluate] Marked signal id=%s pair=%s -> %s (pips=%.2f)", sig['id'], pair, status, pips or 0)
                    updated += 1
                    try:
                        updated_ids.append(int(sig['id']))
                    except Exception:
                        pass
        except Exception as exc:
            logging.exception("[evaluate] Error evaluating signals: %s", exc)
        return {"updated": updated, "updated_ids": updated_ids}

    @staticmethod
    def _decide_outcome(pair: str, recommendation: str, entry: float, tp: Optional[float], sl: Optional[float], current: float) -> tuple[Optional[str], Optional[float]]:
        """Return (status, pips) if outcome can be determined, else (None, None). Pips scaled by pair."""
        try:
            if recommendation == 'BUY':
                if tp is not None and current >= entry + (tp - entry) * 0.5:
                    return 'Successful', (current - entry) * 10000
                if sl is not None and current <= sl:
                    return 'Failed', (current - entry) * 10000
            elif recommendation == 'SELL':
                if tp is not None and current <= entry - (entry - tp) * 0.5:
                    return 'Successful', (entry - current) * 10000
                if sl is not None and current >= sl:
                    return 'Failed', (entry - current) * 10000
            return 'Neutral', 0
        except Exception:
            return None, None
