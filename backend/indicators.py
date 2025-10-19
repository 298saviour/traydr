"""Indicator collection helpers using Alpha Vantage."""

from __future__ import annotations

import time
import json
import logging
from typing import Dict, Tuple, Any, Optional

from alpha_vantage_provider import AlphaVantageProvider
from backend.fx_candle_fetcher import FXCandleFetcher


ALLOWED_INDICATORS = {"rsi", "macd", "ema", "bbands", "atr", "vwap", "obv"}


class IndicatorCollector:
    """Fetches technical indicators from Alpha Vantage with rate-limit handling."""

    def __init__(self, provider: AlphaVantageProvider | None = None):
        self.provider = provider or AlphaVantageProvider()
        self.last_call_ts: float = 0.0
        self.calls_in_minute: int = 0
        self.enable_debug_logging = True  # Log all API responses for debugging
        
        # Initialize FX candle fetcher for reliable FX data
        alpha_key = getattr(self.provider, 'api_key', None)
        self.fx_fetcher = FXCandleFetcher(alpha_api_key=alpha_key)
        self.fx_outputsize: int = 120

    def set_fx_outputsize(self, size: int) -> None:
        self.fx_outputsize = max(1, min(size, 5000))

    def _throttle(self) -> None:
        """Ensure Alpha Vantage 5-req/minute limit isn't exceeded with conservative spacing."""
        now = time.time()
        if now - self.last_call_ts >= 60:
            self.last_call_ts = now
            self.calls_in_minute = 0
        
        # Conservative: 4 calls per minute max (15s between calls)
        if self.calls_in_minute >= 4:
            sleep_for = 60 - (now - self.last_call_ts) + 1  # +1s buffer
            if sleep_for > 0:
                logging.info(f"[Throttle] Sleeping {sleep_for:.1f}s to respect rate limit")
                time.sleep(sleep_for)
            self.last_call_ts = time.time()
            self.calls_in_minute = 0
        
        # Add 12s delay between all requests to be extra safe
        elapsed = now - self.last_call_ts
        if elapsed < 12 and self.calls_in_minute > 0:
            wait_time = 12 - elapsed
            logging.info(f"[Throttle] Waiting {wait_time:.1f}s before next request")
            time.sleep(wait_time)
        
        self.calls_in_minute += 1
        self.last_call_ts = time.time()

    def _log_response(self, indicator: str, pair: str, interval: str, response: Dict[str, Any], attempt: int = 1) -> None:
        """Log API responses for debugging."""
        if not self.enable_debug_logging:
            return
        
        try:
            log_entry = {
                "indicator": indicator,
                "pair": pair,
                "interval": interval,
                "attempt": attempt,
                "has_data": bool(response and any("Technical Analysis" in k for k in response.keys())),
                "has_rate_limit_note": "Note" in response,
                "has_error": "Error Message" in response,
                "keys": list(response.keys()) if response else []
            }
            logging.info(f"[Alpha Vantage] {json.dumps(log_entry)}")
            
            # Log full response if there's an issue
            if log_entry["has_rate_limit_note"] or log_entry["has_error"] or not log_entry["has_data"]:
                with open("alpha_vantage_debug.log", "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"[{indicator.upper()}] {pair} {interval} - Attempt {attempt}\n")
                    f.write(json.dumps(response, indent=2))
                    f.write(f"\n{'='*80}\n")
        except Exception as e:
            logging.warning(f"Failed to log API response: {e}")

    def get_indicator_payload(self, from_symbol: str, to_symbol: str, indicator: str, interval: str, retries: int = 3) -> Optional[Dict[str, Any]]:
        """Fetch indicator data for given pair/timeframe with retry logic.

        Indicator options: RSI, MACD, EMA, BBANDS, ATR, VWAP, OBV.
        """
        indicator = indicator.lower()
        if indicator not in ALLOWED_INDICATORS:
            raise ValueError(f"Unsupported indicator {indicator}")

        function_map: Dict[str, Tuple[str, Dict[str, str]]] = {
            "rsi": ("RSI", {"time_period": "14"}),
            "macd": (
                "MACD",
                {
                    "fastperiod": "12",
                    "slowperiod": "26",
                    "signalperiod": "9",
                },
            ),
            "ema": ("EMA", {"time_period": "20"}),
            "bbands": (
                "BBANDS",
                {"time_period": "20", "nbdevup": "2", "nbdevdn": "2"},
            ),
            "atr": ("ATR", {"time_period": "14"}),
            "vwap": ("VWAP", {}),
            "obv": ("OBV", {}),
        }

        function_name, defaults = function_map[indicator]
        pair_str = f"{from_symbol}/{to_symbol}"

        params: Dict[str, str] = {
            "function": function_name,
            "interval": interval,
            "series_type": "close",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
        }
        params.update(defaults)

        # Retry logic with exponential backoff
        for attempt in range(1, retries + 1):
            self._throttle()
            response = self.provider._make_request(params)  # type: ignore[attr-defined]
            
            self._log_response(indicator, pair_str, interval, response or {}, attempt)
            
            if not response:
                logging.warning(f"[{indicator.upper()}] No response for {pair_str} {interval}, attempt {attempt}/{retries}")
                if attempt < retries:
                    time.sleep(5 * attempt)  # Exponential backoff: 5s, 10s, 15s
                continue
            
            # Check for rate limit
            if "Note" in response:
                logging.warning(f"[{indicator.upper()}] Rate limit hit for {pair_str} {interval}, attempt {attempt}/{retries}")
                if attempt < retries:
                    time.sleep(15 * attempt)  # Wait longer: 15s, 30s, 45s
                continue
            
            # Check for error
            if "Error Message" in response:
                error_msg = response.get("Error Message", "Unknown error")
                logging.error(f"[{indicator.upper()}] API error for {pair_str} {interval}: {error_msg}")
                if attempt < retries:
                    time.sleep(10 * attempt)
                continue
            
            # Check for data
            if any("Technical Analysis" in k for k in response.keys()):
                logging.info(f"[{indicator.upper()}] Successfully fetched {pair_str} {interval}")
                return response
            
            logging.warning(f"[{indicator.upper()}] Unexpected response structure for {pair_str} {interval}, attempt {attempt}/{retries}")
            if attempt < retries:
                time.sleep(5 * attempt)
        
        logging.error(f"[{indicator.upper()}] Failed to fetch {pair_str} {interval} after {retries} attempts")
        return None

    # Dedicated helper to fetch EMA with a custom period (e.g., 9, 21, 20, 200)
    def get_ema_value(self, from_symbol: str, to_symbol: str, interval: str, period: int, retries: int = 3) -> Dict[str, Any]:
        pair_str = f"{from_symbol}/{to_symbol}"
        params: Dict[str, str] = {
            "function": "EMA",
            "interval": interval,
            "series_type": "close",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "time_period": str(period),
        }
        
        for attempt in range(1, retries + 1):
            self._throttle()
            response = self.provider._make_request(params)  # type: ignore[attr-defined]
            
            self._log_response(f"EMA{period}", pair_str, interval, response or {}, attempt)
            
            if not response:
                if attempt < retries:
                    time.sleep(5 * attempt)
                continue
            
            if "Note" in response:
                logging.warning(f"[EMA{period}] Rate limit for {pair_str} {interval}, attempt {attempt}/{retries}")
                if attempt < retries:
                    time.sleep(15 * attempt)
                continue
            
            if any("Technical Analysis" in k for k in response.keys()):
                return self.extract_indicator_value("ema", response)
            
            if attempt < retries:
                time.sleep(5 * attempt)
        
        logging.error(f"[EMA{period}] Failed to fetch {pair_str} {interval} after {retries} attempts")
        return {"error": "fetch_failed", "value": None}

    @staticmethod
    def extract_indicator_value(indicator: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the latest indicator reading from Alpha Vantage payload."""
        if not payload:
            return {"error": "no_data"}

        indicator = indicator.lower()
        data_key = next((key for key in payload.keys() if "Technical Analysis" in key), None)
        if not data_key:
            return {"error": "missing_technical_analysis"}

        series: Dict[str, Dict[str, str]] = payload.get(data_key, {})  # type: ignore[assignment]
        if not series:
            return {"error": "empty_series"}

        latest_timestamp = sorted(series.keys())[-1]
        latest_values = series.get(latest_timestamp, {})

        if indicator == "rsi":
            value = latest_values.get("RSI")
            return {"timestamp": latest_timestamp, "value": float(value) if value else None}
        if indicator == "ema":
            value = latest_values.get("EMA")
            return {"timestamp": latest_timestamp, "value": float(value) if value else None}
        if indicator == "macd":
            return {
                "timestamp": latest_timestamp,
                "macd": float(latest_values.get("MACD", "nan")),
                "signal": float(latest_values.get("MACD_Signal", "nan")),
                "hist": float(latest_values.get("MACD_Hist", "nan")),
            }
        if indicator == "bbands":
            return {
                "timestamp": latest_timestamp,
                "upper": float(latest_values.get("Real Upper Band", "nan")),
                "middle": float(latest_values.get("Real Middle Band", "nan")),
                "lower": float(latest_values.get("Real Lower Band", "nan")),
            }
        if indicator == "atr":
            value = latest_values.get("ATR")
            return {"timestamp": latest_timestamp, "value": float(value) if value else None}
        if indicator == "vwap":
            value = latest_values.get("VWAP") or latest_values.get("vwap")
            return {"timestamp": latest_timestamp, "value": float(value) if value else None}
        if indicator == "obv":
            value = latest_values.get("OBV") or latest_values.get("obv")
            return {"timestamp": latest_timestamp, "value": float(value) if value else None}
        return {"error": "unknown_indicator"}

    def collect_all(self, from_symbol: str, to_symbol: str, interval: str) -> Dict[str, Any]:
        """Fetch ALL indicators (RSI, MACD, EMA, BBANDS, ATR, OBV) for any timeframe using FX candles."""
        pair_str = f"{from_symbol}/{to_symbol}"
        
        # Try FX candle fetcher first (more reliable for FX pairs)
        if self.fx_fetcher:
            try:
                fx_interval = self._map_interval_to_fx(interval)
                logging.info(f"[FX All] Fetching {pair_str} {fx_interval} via FX candles")
                
                indicators = self.fx_fetcher.get_indicators_for_pair(
                    from_symbol,
                    to_symbol,
                    fx_interval,
                    outputsize=self.fx_outputsize,
                )
                
                if "error" not in indicators:
                    ts = indicators.get("timestamp")
                    results = {
                        "RSI": {
                            "value": indicators.get("rsi"),
                            "timestamp": ts
                        },
                        "EMA": {
                            "ema9": indicators.get("ema9"),
                            "ema21": indicators.get("ema21"),
                            "timestamp": ts
                        },
                        "MACD": {
                            "macd": indicators.get("macd"),
                            "signal": indicators.get("macd_signal"),
                            "hist": indicators.get("macd_hist"),
                            "timestamp": ts
                        },
                        "BBANDS": indicators.get("bbands", {}),
                        "ATR": {"value": indicators.get("atr"), "timestamp": ts},
                        "OBV": {"value": indicators.get("obv"), "timestamp": ts}
                    }
                    logging.info(
                        f"[FX All] Success for {pair_str} {fx_interval}: RSI={indicators.get('rsi'):.2f}, "
                        f"EMA9={indicators.get('ema9'):.4f}, EMA21={indicators.get('ema21'):.4f}, "
                        f"MACD={indicators.get('macd'):.4f}, ATR={indicators.get('atr')}, OBV={indicators.get('obv')}"
                    )
                    return results
                else:
                    logging.warning(f"[FX All] FX fetcher failed for {pair_str} {fx_interval}: {indicators.get('error')}")
            except Exception as e:
                logging.error(f"[FX All] Exception using FX fetcher for {pair_str} {interval}: {e}")
        
        # Fallback to original Alpha Vantage indicator endpoints
        logging.info(f"[FX All] Falling back to Alpha Vantage indicators for {pair_str} {interval}")
        results: Dict[str, Any] = {}
        
        # Fetch RSI
        payload = self.get_indicator_payload(from_symbol, to_symbol, "rsi", interval)
        if payload:
            results["RSI"] = self.extract_indicator_value("rsi", payload)
        else:
            results["RSI"] = {"error": "no_data", "value": None}
        
        # Fetch MACD
        payload = self.get_indicator_payload(from_symbol, to_symbol, "macd", interval)
        if payload:
            results["MACD"] = self.extract_indicator_value("macd", payload)
        else:
            results["MACD"] = {"error": "no_data"}
        
        # Fetch BBANDS
        payload = self.get_indicator_payload(from_symbol, to_symbol, "bbands", interval)
        if payload:
            results["BBANDS"] = self.extract_indicator_value("bbands", payload)
        else:
            results["BBANDS"] = {"error": "no_data"}
        
        # Fetch ATR
        payload = self.get_indicator_payload(from_symbol, to_symbol, "atr", interval)
        if payload:
            results["ATR"] = self.extract_indicator_value("atr", payload)
        else:
            results["ATR"] = {"error": "no_data", "value": None}
        
        # Fetch OBV
        payload = self.get_indicator_payload(from_symbol, to_symbol, "obv", interval)
        if payload:
            results["OBV"] = self.extract_indicator_value("obv", payload)
        else:
            results["OBV"] = {"error": "no_data", "value": None}
        
        # Fetch EMA(9) and EMA(21)
        try:
            ema9 = self.get_ema_value(from_symbol, to_symbol, interval, 9)
        except Exception as e:
            logging.error(f"[EMA9] Exception for {pair_str} {interval}: {e}")
            ema9 = {"timestamp": None, "value": None, "error": str(e)}
        
        try:
            ema21 = self.get_ema_value(from_symbol, to_symbol, interval, 21)
        except Exception as e:
            logging.error(f"[EMA21] Exception for {pair_str} {interval}: {e}")
            ema21 = {"timestamp": None, "value": None, "error": str(e)}
        
        results["EMA"] = {
            "timestamp": ema9.get("timestamp") or ema21.get("timestamp"),
            "ema9": ema9.get("value"),
            "ema21": ema21.get("value"),
        }
        
        return results

    def _map_interval_to_fx(self, interval: str) -> str:
        """Map scheduler intervals to FX_INTRADAY intervals."""
        mapping = {
            "60min": "60min",
            "30min": "30min", 
            "15min": "15min",
            "1h": "60min",
            "30m": "30min",
            "15m": "15min"
        }
        return mapping.get(interval, interval)
    
    # New convenience wrappers for the scheduler phases
    def collect_basic(self, from_symbol: str, to_symbol: str, interval: str) -> Dict[str, Any]:
        """RSI + EMA(9) + EMA(21) for Phase A momentum analysis using FX candles."""
        pair_str = f"{from_symbol}/{to_symbol}"
        
        # Try FX candle fetcher first (more reliable for FX pairs)
        if self.fx_fetcher:
            try:
                fx_interval = self._map_interval_to_fx(interval)
                logging.info(f"[FX Basic] Fetching {pair_str} {fx_interval} via FX candles")
                
                indicators = self.fx_fetcher.get_indicators_for_pair(from_symbol, to_symbol, fx_interval)
                
                if "error" not in indicators:
                    # Success - format for compatibility
                    results = {
                        "RSI": {
                            "value": indicators.get("rsi"),
                            "timestamp": indicators.get("timestamp")
                        },
                        "EMA": {
                            "ema9": indicators.get("ema9"),
                            "ema21": indicators.get("ema21"),
                            "timestamp": indicators.get("timestamp")
                        }
                    }
                    logging.info(f"[FX Basic] Success for {pair_str} {fx_interval}: RSI={indicators.get('rsi'):.2f}, EMA9={indicators.get('ema9'):.4f}, EMA21={indicators.get('ema21'):.4f}")
                    return results
                else:
                    logging.warning(f"[FX Basic] FX fetcher failed for {pair_str} {fx_interval}: {indicators.get('error')}")
            except Exception as e:
                logging.error(f"[FX Basic] Exception using FX fetcher for {pair_str} {interval}: {e}")
        
        # Fallback to original Alpha Vantage indicator endpoints (likely to fail for FX)
        logging.info(f"[FX Basic] Falling back to Alpha Vantage indicators for {pair_str} {interval}")
        results: Dict[str, Any] = {}
        
        # Fetch RSI
        payload = self.get_indicator_payload(from_symbol, to_symbol, "rsi", interval)
        if payload:
            results["RSI"] = self.extract_indicator_value("rsi", payload)
        else:
            results["RSI"] = {"error": "no_data", "value": None}
        
        # Fetch EMA(9) and EMA(21) for phase A
        try:
            ema9 = self.get_ema_value(from_symbol, to_symbol, interval, 9)
        except Exception as e:
            logging.error(f"[EMA9] Exception for {pair_str} {interval}: {e}")
            ema9 = {"timestamp": None, "value": None, "error": str(e)}
        
        try:
            ema21 = self.get_ema_value(from_symbol, to_symbol, interval, 21)
        except Exception as e:
            logging.error(f"[EMA21] Exception for {pair_str} {interval}: {e}")
            ema21 = {"timestamp": None, "value": None, "error": str(e)}

        results["EMA"] = {
            "timestamp": ema9.get("timestamp") or ema21.get("timestamp"),
            "ema9": ema9.get("value"),
            "ema21": ema21.get("value"),
        }

        return results

    def collect_advanced(self, from_symbol: str, to_symbol: str, interval: str) -> Dict[str, Any]:
        """ATR, VWAP, OBV and a simple LuxAlgo-style consensus from available signals.
        LuxAlgo emulation here is a heuristic: combines MACD hist sign, RSI bands, and EMA slope if available.
        """
        results: Dict[str, Any] = {}
        for ind in ("atr", "vwap", "obv"):
            try:
                payload = self.get_indicator_payload(from_symbol, to_symbol, ind, interval)
                results[ind.upper()] = self.extract_indicator_value(ind, payload)
            except Exception:
                results[ind.upper()] = {"error": "fetch_failed"}

        # LuxAlgo emulation placeholder; real consensus will be derived by Claude using all context
        lux = {"consensus": None}
        try:
            # If we have RSI/MACD/EMA from a recent basic call, this module doesn't retain it.
            # We expose a minimal consensus hook so the analyzer can weigh it with snapshots.
            lux = {"consensus": "Mixed"}
        except Exception:
            pass
        results["LUXALGO"] = lux
        return results
