"""
FX Candle Fetcher with Local Indicator Computation
Replaces Alpha Vantage indicator endpoints with FX_INTRADAY + local computation
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator, MACD
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("ta library not available - will use simple indicator calculations")


class FXCandleFetcher:
    """Fetches FX candles (TwelveData preferred, Alpha Vantage fallback) and computes indicators."""
    
    def __init__(self, alpha_api_key: Optional[str] = None, twelve_api_key: Optional[str] = None):
        self.alpha_api_key = alpha_api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.twelve_api_key = twelve_api_key or os.getenv("TWELVEDATA_API_KEY")
        self.session = requests.Session()
        self.last_call_ts = 0.0
        self.calls_in_minute = 0
        
    def _throttle_alpha(self) -> None:
        """Conservative rate limiting: max 4 calls per minute."""
        now = time.time()
        if now - self.last_call_ts >= 60:
            self.last_call_ts = now
            self.calls_in_minute = 0
        
        if self.calls_in_minute >= 4:
            sleep_for = 60 - (now - self.last_call_ts) + 1
            if sleep_for > 0:
                logging.info(f"[FX Throttle] Sleeping {sleep_for:.1f}s for rate limit")
                time.sleep(sleep_for)
            self.last_call_ts = time.time()
            self.calls_in_minute = 0
        
        # 15s minimum between calls
        elapsed = now - self.last_call_ts
        if elapsed < 15 and self.calls_in_minute > 0:
            wait_time = 15 - elapsed
            logging.info(f"[FX Throttle] Waiting {wait_time:.1f}s before next request")
            time.sleep(wait_time)
        
        self.calls_in_minute += 1
        self.last_call_ts = time.time()

    def _map_interval_to_twelvedata(self, interval: str) -> Optional[str]:
        mapping = {
            "60min": "1h",
            "30min": "30min",
            "15min": "15min",
            "5min": "5min",
            "1min": "1min",
            "1h": "1h",
            "4h": "4h",
            "1day": "1day",
            "1d": "1day",
            "30m": "30min",
            "15m": "15min"
        }
        return mapping.get(interval)
    
    def _log_response(self, pair: str, interval: str, response: Dict[str, Any], attempt: int) -> None:
        """Log API responses for debugging."""
        try:
            log_entry = {
                "pair": pair,
                "interval": interval,
                "attempt": attempt,
                "has_time_series": any("Time Series" in k for k in response.keys()),
                "has_rate_limit": "Note" in response,
                "has_error": "Error Message" in response,
                "keys": list(response.keys())
            }
            logging.info(f"[FX Candles] {json.dumps(log_entry)}")
            
            # Log full response if there's an issue
            if log_entry["has_rate_limit"] or log_entry["has_error"] or not log_entry["has_time_series"]:
                with open("fx_candles_debug.log", "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"[{pair}] {interval} - Attempt {attempt}\n")
                    f.write(json.dumps(response, indent=2))
                    f.write(f"\n{'='*80}\n")
        except Exception as e:
            logging.warning(f"Failed to log FX response: {e}")
    
    def fetch_fx_candles(
        self,
        from_symbol: str,
        to_symbol: str,
        interval: str,
        outputsize: int = 120,
        retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch FX candles using FX_INTRADAY endpoint.
        
        Args:
            from_symbol: Base currency (e.g., 'USD')
            to_symbol: Quote currency (e.g., 'CHF')
            interval: '60min', '30min', '15min', '5min', '1min'
            retries: Number of retry attempts
            
        Returns:
            DataFrame with OHLC data or None if failed
        """
        pair_str = f"{from_symbol}/{to_symbol}"

        # TwelveData is the primary and only data source (includes 4h and 1day)
        td_interval = self._map_interval_to_twelvedata(interval)
        if self.twelve_api_key and td_interval:
            td_df = self._fetch_from_twelvedata(pair_str, td_interval, outputsize=outputsize, retries=retries)
            if td_df is not None:
                return td_df
            logging.error(f"[FX] TwelveData fetch failed for {pair_str} {interval} after {retries} attempts")
            return None

    def _fetch_alpha_intraday(self, from_symbol: str, to_symbol: str, interval: str, outputsize: int, retries: int) -> Optional[pd.DataFrame]:
        pair_str = f"{from_symbol}/{to_symbol}"
        for attempt in range(1, retries + 1):
            try:
                self._throttle_alpha()
                resp = self.session.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "FX_INTRADAY",
                        "from_symbol": from_symbol,
                        "to_symbol": to_symbol,
                        "interval": interval,
                        "outputsize": "full" if outputsize > 100 else "compact",
                        "apikey": self.alpha_api_key
                    },
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                self._log_response(pair_str, interval, data, attempt)
                ts_key = None
                for k in data.keys():
                    if "Time Series" in k:
                        ts_key = k
                        break
                if not ts_key:
                    continue
                ts_data = data[ts_key]
                if not ts_data:
                    continue
                df = pd.DataFrame.from_dict(ts_data, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.rename(columns={
                    '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                    '5. volume': 'volume'
                })
                for col in ['open','high','low','close']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                if 'volume' in df.columns:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df = df.sort_index()
                return df
            except Exception:
                if attempt < retries:
                    time.sleep(10 * attempt)
        return None

    def _fetch_alpha_daily(self, from_symbol: str, to_symbol: str, outputsize: int, retries: int) -> Optional[pd.DataFrame]:
        pair_str = f"{from_symbol}/{to_symbol}"
        for attempt in range(1, retries + 1):
            try:
                self._throttle_alpha()
                resp = self.session.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "FX_DAILY",
                        "from_symbol": from_symbol,
                        "to_symbol": to_symbol,
                        "outputsize": "compact",
                        "apikey": self.alpha_api_key
                    },
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                # Log minimal info
                self._log_response(pair_str, '1day', data, attempt)
                ts_key = "Time Series FX (Daily)"
                if ts_key not in data:
                    logging.warning(f"[FX] No daily time series for {pair_str}, attempt {attempt}/{retries}")
                    if attempt < retries:
                        time.sleep(10 * attempt)
                    continue
                ts_data = data[ts_key]
                if not ts_data:
                    logging.warning(f"[FX] Empty daily series for {pair_str}")
                    continue
                df = pd.DataFrame.from_dict(ts_data, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.rename(columns={
                    '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close'
                }).sort_index()
                for col in ['open','high','low','close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                logging.info(f"[FX] Successfully fetched {len(df)} daily candles for {pair_str}")
                return df.tail(outputsize)
            except Exception as e:
                logging.error(f"[FX] Exception fetching daily for {pair_str}: {e}")
                if attempt < retries:
                    time.sleep(10 * attempt)
        return None

    def _fetch_from_twelvedata(
        self,
        pair: str,
        interval: str,
        outputsize: int,
        retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        base_url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": pair,
            "interval": interval,
            "outputsize": max(1, min(outputsize, 5000)),
            "order": "ASC",
            "apikey": self.twelve_api_key,
            "timezone": "UTC"
        }

        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                self._log_twelvedata_response(pair, interval, data, attempt)

                if data.get("status") == "error":
                    message = data.get("message", "Unknown TwelveData error")
                    logging.error(f"[FX] TwelveData error for {pair} {interval}: {message}")
                    if attempt < retries:
                        time.sleep(10 * attempt)
                    continue

                values = data.get("values")
                if not values:
                    logging.warning(f"[FX] TwelveData empty response for {pair} {interval}, attempt {attempt}/{retries}")
                    if attempt < retries:
                        time.sleep(5 * attempt)
                    continue

                df = pd.DataFrame(values)
                required_cols = {"open", "high", "low", "close", "datetime"}
                if not required_cols.issubset(df.columns):
                    logging.error(f"[FX] TwelveData missing columns for {pair} {interval}: {df.columns.tolist()}")
                    if attempt < retries:
                        time.sleep(5 * attempt)
                    continue

                # Convert types and set index
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col].astype(float)
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df = df.sort_values("datetime").set_index("datetime")
                df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})

                logging.info(f"[FX] Successfully fetched {len(df)} candles for {pair} {interval} (TwelveData)")
                return df

            except Exception as exc:
                logging.error(f"[FX] TwelveData exception for {pair} {interval}, attempt {attempt}/{retries}: {exc}")
                if attempt < retries:
                    time.sleep(10 * attempt)

        logging.error(f"[FX] Failed TwelveData fetch for {pair} {interval} after {retries} attempts")
        return None

    def _convert_alpha_timeseries_to_df(self, ts_data: Dict[str, Dict[str, Any]]) -> Optional[pd.DataFrame]:
        if not ts_data:
            return None

        df = pd.DataFrame.from_dict(ts_data, orient="index")
        if df.empty:
            return None

        df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        column_mapping = {}
        for col in df.columns:
            lower = col.lower()
            if 'open' in lower:
                column_mapping[col] = 'open'
            elif 'high' in lower:
                column_mapping[col] = 'high'
            elif 'low' in lower:
                column_mapping[col] = 'low'
            elif 'close' in lower:
                column_mapping[col] = 'close'

        df = df.rename(columns=column_mapping)
        if not {'open', 'high', 'low', 'close'}.issubset(df.columns):
            logging.error(f"[FX] Alpha Vantage missing OHLC columns: {df.columns.tolist()}")
            return None

        return df

    def _log_twelvedata_response(self, pair: str, interval: str, response: Dict[str, Any], attempt: int) -> None:
        try:
            log_entry = {
                "pair": pair,
                "interval": interval,
                "attempt": attempt,
                "status": response.get("status", "ok"),
                "has_values": bool(response.get("values")),
                "message": response.get("message")
            }
            logging.info(f"[FX TwelveData] {json.dumps(log_entry)}")

            if response.get("status") == "error":
                with open("twelvedata_debug.log", "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"[{pair}] {interval} - Attempt {attempt}\n")
                    f.write(json.dumps(response, indent=2))
                    f.write(f"\n{'='*80}\n")
        except Exception as err:
            logging.warning(f"Failed to log TwelveData response: {err}")
    
    def compute_indicators_simple(self, df: pd.DataFrame, rsi_period: int = 14,
                                ema_short: int = 9, ema_long: int = 21,
                                macd_fast: int = 12, macd_slow: int = 26,
                                macd_signal: int = 9, bbands_period: int = 20,
                                atr_period: int = 14) -> Optional[Dict[str, float]]:
        """Simple indicator computation without ta library - includes ALL indicators."""
        try:
            required_length = max(rsi_period, ema_long, macd_slow, bbands_period, atr_period) + 10
            if len(df) < required_length:
                logging.warning(f"Insufficient data for indicators: {len(df)} candles")
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Simple RSI calculation
            deltas = pd.Series(close).diff()
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)

            avg_gain = gains.rolling(window=rsi_period).mean()
            avg_loss = losses.rolling(window=rsi_period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Simple EMA calculation
            ema9 = pd.Series(close).ewm(span=ema_short, adjust=False).mean()
            ema21 = pd.Series(close).ewm(span=ema_long, adjust=False).mean()

            # MACD
            macd_fast_series = pd.Series(close).ewm(span=macd_fast, adjust=False).mean()
            macd_slow_series = pd.Series(close).ewm(span=macd_slow, adjust=False).mean()
            macd_line = macd_fast_series - macd_slow_series
            macd_signal_series = macd_line.ewm(span=macd_signal, adjust=False).mean()
            macd_hist_series = macd_line - macd_signal_series

            # Bollinger Bands
            sma = pd.Series(close).rolling(window=bbands_period).mean()
            std = pd.Series(close).rolling(window=bbands_period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)

            # ATR (Average True Range)
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            tr1 = high_series - low_series
            tr2 = (high_series - close_series.shift()).abs()
            tr3 = (low_series - close_series.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period).mean()

            # OBV (On-Balance Volume)
            # Use true volume if available; otherwise, use a price-change proxy (unit volume)
            if 'volume' in df.columns:
                volume = pd.to_numeric(df['volume'], errors='coerce').fillna(0).values
                obv_series = pd.Series(index=df.index, dtype=float)
                obv_series.iloc[0] = volume[0]
                for i in range(1, len(close)):
                    if close[i] > close[i-1]:
                        obv_series.iloc[i] = obv_series.iloc[i-1] + volume[i]
                    elif close[i] < close[i-1]:
                        obv_series.iloc[i] = obv_series.iloc[i-1] - volume[i]
                    else:
                        obv_series.iloc[i] = obv_series.iloc[i-1]
                obv_value = float(obv_series.iloc[-1])
            else:
                # Price-change proxy: +1 for up, -1 for down, cumulative sum
                deltas = pd.Series(close).diff().fillna(0)
                unit_volume = deltas.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                obv_series = unit_volume.cumsum()
                obv_value = float(obv_series.iloc[-1])

            # Get latest values
            latest_rsi = rsi.iloc[-1]
            latest_ema9 = ema9.iloc[-1]
            latest_ema21 = ema21.iloc[-1]
            latest_macd = macd_line.iloc[-1]
            latest_macd_signal = macd_signal_series.iloc[-1]
            latest_macd_hist = macd_hist_series.iloc[-1]
            latest_bb_upper = upper_band.iloc[-1]
            latest_bb_middle = sma.iloc[-1]
            latest_bb_lower = lower_band.iloc[-1]
            latest_atr = atr.iloc[-1]

            if any(pd.isna(val) for val in (latest_rsi, latest_ema9, latest_ema21, latest_macd, latest_macd_signal, latest_macd_hist)):
                logging.warning("NaN values in computed indicators")
                return None

            return {
                "rsi": round(float(latest_rsi), 2),
                "ema9": round(float(latest_ema9), 4),
                "ema21": round(float(latest_ema21), 4),
                "macd": round(float(latest_macd), 4),
                "macd_signal": round(float(latest_macd_signal), 4),
                "macd_hist": round(float(latest_macd_hist), 4),
                "bbands": {
                    "upper": round(float(latest_bb_upper), 4) if not pd.isna(latest_bb_upper) else None,
                    "middle": round(float(latest_bb_middle), 4) if not pd.isna(latest_bb_middle) else None,
                    "lower": round(float(latest_bb_lower), 4) if not pd.isna(latest_bb_lower) else None
                },
                "atr": round(float(latest_atr), 4) if not pd.isna(latest_atr) else None,
                "obv": round(obv_value, 2) if obv_value is not None else None
            }

        except Exception as e:
            logging.error(f"Error computing simple indicators: {e}")
            return None
    
    def compute_indicators_ta(self, df: pd.DataFrame, rsi_period: int = 14,
                            ema_short: int = 9, ema_long: int = 21,
                            macd_fast: int = 12, macd_slow: int = 26,
                            macd_signal: int = 9, bbands_period: int = 20,
                            atr_period: int = 14) -> Optional[Dict[str, float]]:
        """Compute ALL indicators using ta library."""
        try:
            from ta.volatility import BollingerBands, AverageTrueRange
            from ta.volume import OnBalanceVolumeIndicator
            
            required_length = max(rsi_period, ema_long, macd_slow, bbands_period, atr_period) + 10
            if len(df) < required_length:
                logging.warning(f"Insufficient data for indicators: {len(df)} candles")
                return None
            
            # Compute indicators
            rsi_indicator = RSIIndicator(df['close'], window=rsi_period)
            ema9_indicator = EMAIndicator(df['close'], window=ema_short)
            ema21_indicator = EMAIndicator(df['close'], window=ema_long)
            macd_indicator = MACD(df['close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
            bbands_indicator = BollingerBands(df['close'], window=bbands_period, window_dev=2)
            atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period)
            
            # OBV
            if 'volume' in df.columns:
                obv_indicator = OnBalanceVolumeIndicator(df['close'], df['volume'].fillna(0))
                obv_value = float(obv_indicator.on_balance_volume().iloc[-1])
            else:
                # Price-change proxy OBV
                deltas = df['close'].diff().fillna(0)
                unit_volume = deltas.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                obv_value = float(unit_volume.cumsum().iloc[-1])

            rsi = rsi_indicator.rsi().iloc[-1]
            ema9 = ema9_indicator.ema_indicator().iloc[-1]
            ema21 = ema21_indicator.ema_indicator().iloc[-1]
            macd_line = macd_indicator.macd().iloc[-1]
            macd_signal_line = macd_indicator.macd_signal().iloc[-1]
            macd_hist_line = macd_indicator.macd_diff().iloc[-1]
            bb_upper = bbands_indicator.bollinger_hband().iloc[-1]
            bb_middle = bbands_indicator.bollinger_mavg().iloc[-1]
            bb_lower = bbands_indicator.bollinger_lband().iloc[-1]
            atr_value = atr_indicator.average_true_range().iloc[-1]

            if any(pd.isna(val) for val in (rsi, ema9, ema21, macd_line, macd_signal_line, macd_hist_line)):
                logging.warning("NaN values in computed indicators")
                return None

            return {
                "rsi": float(rsi),
                "ema9": float(ema9),
                "ema21": float(ema21),
                "macd": float(macd_line),
                "macd_signal": float(macd_signal_line),
                "macd_hist": float(macd_hist_line),
                "bbands": {
                    "upper": float(bb_upper) if not pd.isna(bb_upper) else None,
                    "middle": float(bb_middle) if not pd.isna(bb_middle) else None,
                    "lower": float(bb_lower) if not pd.isna(bb_lower) else None
                },
                "atr": float(atr_value) if not pd.isna(atr_value) else None,
                "obv": obv_value
            }
            
        except Exception as e:
            logging.error(f"Error computing ta indicators: {e}")
            return None
    
    def get_indicators_for_pair(
        self,
        from_symbol: str,
        to_symbol: str,
        interval: str,
        outputsize: int = 120,
    ) -> Dict[str, Any]:
        """
        Fetch candles and compute indicators for a currency pair.
        
        Returns:
            Dict with RSI, EMA9, EMA21 values or error info
        """
        pair_str = f"{from_symbol}/{to_symbol}"
        
        # Fetch candles
        df = self.fetch_fx_candles(from_symbol, to_symbol, interval, outputsize=outputsize)
        if df is None:
            return {"error": "failed_to_fetch_candles", "pair": pair_str, "interval": interval}
        
        # Compute indicators
        if TA_AVAILABLE:
            indicators = self.compute_indicators_ta(df)
        else:
            indicators = self.compute_indicators_simple(df)
        
        if indicators is None:
            return {"error": "failed_to_compute_indicators", "pair": pair_str, "interval": interval}
        
        # Add metadata
        indicators.update({
            "pair": pair_str,
            "interval": interval,
            "timestamp": df.index[-1].isoformat(),
            "candles_count": len(df),
            "method": "ta_library" if TA_AVAILABLE else "simple"
        })
        
        return indicators
