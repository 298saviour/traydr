"""
Data Integrity Layer for TwelveData Feed
Validates, repairs, and ensures data quality before analysis
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class DataIntegrityReport:
    """Report on data quality and actions taken"""
    pair: str
    source: str  # "TwelveData" or "Finnhub"
    feed_integrity: str  # "Clean", "Repaired", or "Corrupted"
    candles_used: int
    avg_gap: float
    timestamp_checked: str
    issues_detected: List[str]
    actions_taken: List[str]
    retry_count: int = 0


class DataIntegrityLayer:
    """
    Validates and ensures data quality from TwelveData with Finnhub fallback
    """
    
    def __init__(self):
        self.twelvedata_key = os.getenv("TWELVE_DATA_API_KEY")
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        
        if not self.twelvedata_key:
            logger.warning("TwelveData API key not found")
        if not self.finnhub_key:
            logger.warning("Finnhub API key not found")
        
        # Validation thresholds
        self.MAX_GAP_PIPS = 50  # 0.005 for EUR/USD (50 pips)
        self.MAX_FLAT_CANDLES = 10
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 2  # seconds
        self.MIN_CANDLES_FOR_INTERPOLATION = 3
    
    def fetch_validated_data(
        self, 
        pair: str, 
        interval: str, 
        outputsize: int = 100,
        progress_callback=None
    ) -> Tuple[Optional[pd.DataFrame], DataIntegrityReport]:
        """
        Fetch data with full validation and automatic fallback
        
        Args:
            pair: Currency pair (e.g., "EUR/USD")
            interval: Timeframe (e.g., "1h", "15min")
            outputsize: Number of candles to fetch
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (DataFrame, IntegrityReport)
        """
        report = DataIntegrityReport(
            pair=pair,
            source="TwelveData",
            feed_integrity="Unknown",
            candles_used=0,
            avg_gap=0.0,
            timestamp_checked=datetime.utcnow().isoformat(),
            issues_detected=[],
            actions_taken=[]
        )
        
        # Try TwelveData first (with retries)
        for attempt in range(self.MAX_RETRIES):
            report.retry_count = attempt + 1
            
            if progress_callback and attempt > 0:
                progress_callback(f"⚠️ Retry {attempt + 1}/{self.MAX_RETRIES} for {pair} {interval}")
            
            df = self._fetch_from_twelvedata(pair, interval, outputsize)
            
            if df is not None and not df.empty:
                # Validate the data
                is_valid, issues = self._validate_data(df, pair)
                report.issues_detected.extend(issues)
                
                if is_valid:
                    report.feed_integrity = "Clean"
                    report.candles_used = len(df)
                    report.avg_gap = self._calculate_avg_gap(df)
                    report.actions_taken.append(f"TwelveData data validated successfully")
                    
                    if progress_callback:
                        progress_callback(f"✓ {pair} {interval} data validated (TwelveData)")
                    
                    return df, report
                else:
                    logger.warning(f"[{pair} {interval}] TwelveData validation failed: {issues}")
                    report.actions_taken.append(f"Attempt {attempt + 1}: Validation failed - {', '.join(issues)}")
                    
                    # Try to repair if possible
                    if self._can_repair(issues):
                        df_repaired = self._repair_data(df, issues)
                        if df_repaired is not None:
                            is_valid_after_repair, _ = self._validate_data(df_repaired, pair)
                            if is_valid_after_repair:
                                report.feed_integrity = "Repaired"
                                report.candles_used = len(df_repaired)
                                report.avg_gap = self._calculate_avg_gap(df_repaired)
                                report.actions_taken.append(f"Data repaired successfully")
                                
                                if progress_callback:
                                    progress_callback(f"⚠️ {pair} {interval} data repaired (TwelveData)")
                                
                                return df_repaired, report
            
            # Wait before retry
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY)
        
        # TwelveData failed, switch to Finnhub
        logger.warning(f"[{pair} {interval}] Switching to Finnhub backup feed")
        report.actions_taken.append(f"TwelveData failed after {self.MAX_RETRIES} attempts, switching to Finnhub")
        
        if progress_callback:
            progress_callback(f"⚠️ Switching to Finnhub backup for {pair} {interval}")
        
        df = self._fetch_from_finnhub(pair, interval, outputsize)
        
        if df is not None and not df.empty:
            is_valid, issues = self._validate_data(df, pair)
            
            if is_valid:
                report.source = "Finnhub"
                report.feed_integrity = "Clean"
                report.candles_used = len(df)
                report.avg_gap = self._calculate_avg_gap(df)
                report.actions_taken.append("Finnhub backup data validated successfully")
                
                if progress_callback:
                    progress_callback(f"✓ {pair} {interval} data validated (Finnhub backup)")
                
                return df, report
            else:
                report.issues_detected.extend(issues)
                report.actions_taken.append(f"Finnhub backup also failed validation: {', '.join(issues)}")
        
        # Both sources failed
        report.feed_integrity = "Corrupted"
        report.actions_taken.append("Both TwelveData and Finnhub failed - no clean data available")
        logger.error(f"[{pair} {interval}] Both data sources failed validation")
        
        if progress_callback:
            progress_callback(f"❌ Failed to get clean data for {pair} {interval}")
        
        return None, report
    
    def _fetch_from_twelvedata(self, pair: str, interval: str, outputsize: int) -> Optional[pd.DataFrame]:
        """Fetch data from TwelveData API"""
        try:
            symbol = pair.replace("/", "")  # EUR/USD -> EURUSD
            
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": self.twelvedata_key,
                "format": "JSON"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "values" not in data or not data["values"]:
                logger.warning(f"No data returned from TwelveData for {pair} {interval}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data["values"])
            df = df.rename(columns={
                "datetime": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })
            
            # Convert types
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df = df.sort_values("Datetime").reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching from TwelveData: {e}")
            return None
    
    def _fetch_from_finnhub(self, pair: str, interval: str, outputsize: int) -> Optional[pd.DataFrame]:
        """Fetch data from Finnhub API as backup"""
        try:
            # Convert pair format: EUR/USD -> OANDA:EUR_USD
            base, quote = pair.split("/")
            symbol = f"OANDA:{base}_{quote}"
            
            # Convert interval to resolution
            resolution_map = {
                "1min": "1",
                "5min": "5",
                "15min": "15",
                "30min": "30",
                "1h": "60",
                "4h": "240",
                "1day": "D",
                "1week": "W",
                "1month": "M"
            }
            resolution = resolution_map.get(interval, "60")
            
            # Calculate time range
            now = int(datetime.utcnow().timestamp())
            
            # Estimate seconds per candle
            seconds_per_candle = {
                "1": 60, "5": 300, "15": 900, "30": 1800,
                "60": 3600, "240": 14400, "D": 86400,
                "W": 604800, "M": 2592000
            }
            seconds = seconds_per_candle.get(resolution, 3600) * outputsize
            from_ts = now - seconds
            
            url = "https://finnhub.io/api/v1/forex/candle"
            params = {
                "symbol": symbol,
                "resolution": resolution,
                "from": from_ts,
                "to": now,
                "token": self.finnhub_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("s") != "ok" or not data.get("c"):
                logger.warning(f"No data returned from Finnhub for {pair} {interval}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame({
                "Datetime": pd.to_datetime(data["t"], unit="s"),
                "Open": data["o"],
                "High": data["h"],
                "Low": data["l"],
                "Close": data["c"],
                "Volume": data.get("v", [0] * len(data["c"]))
            })
            
            df = df.sort_values("Datetime").reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching from Finnhub: {e}")
            return None
    
    def _validate_data(self, df: pd.DataFrame, pair: str) -> Tuple[bool, List[str]]:
        """
        Validate data quality
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check 1: No None or 0 values in OHLC
        for col in ["Open", "High", "Low", "Close"]:
            if df[col].isnull().any():
                issues.append(f"{col} contains None values")
            if (df[col] == 0).any():
                issues.append(f"{col} contains zero values")
        
        # Check 2: Missing or irregular timestamps
        if df["Datetime"].isnull().any():
            issues.append("Missing timestamps detected")
        
        # Check 3: Large gaps (> 50 pips for EUR/USD)
        if len(df) > 1:
            price_diffs = df["Close"].diff().abs()
            max_gap = price_diffs.max()
            
            # Adjust threshold based on pair (rough estimate)
            if "JPY" in pair:
                threshold = 0.5  # 50 pips for JPY pairs
            else:
                threshold = 0.005  # 50 pips for other pairs
            
            if max_gap > threshold:
                issues.append(f"Large gap detected: {max_gap:.5f} ({max_gap * 10000:.0f} pips)")
        
        # Check 4: Flat candles (>10 consecutive)
        if len(df) > self.MAX_FLAT_CANDLES:
            flat_candles = (df["Open"] == df["Close"]).rolling(window=self.MAX_FLAT_CANDLES).sum()
            if (flat_candles == self.MAX_FLAT_CANDLES).any():
                issues.append(f"More than {self.MAX_FLAT_CANDLES} consecutive flat candles")
        
        # Check 5: High == Low (impossible candles)
        impossible_candles = (df["High"] == df["Low"]) & (df["Open"] != df["Close"])
        if impossible_candles.any():
            issues.append("Impossible candles detected (High == Low but Open != Close)")
        
        # Check 6: OHLC logic (High >= Open/Close, Low <= Open/Close)
        invalid_high = (df["High"] < df[["Open", "Close"]].max(axis=1)).any()
        invalid_low = (df["Low"] > df[["Open", "Close"]].min(axis=1)).any()
        
        if invalid_high:
            issues.append("Invalid High values (High < Open or Close)")
        if invalid_low:
            issues.append("Invalid Low values (Low > Open or Close)")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    def _can_repair(self, issues: List[str]) -> bool:
        """Determine if data can be repaired"""
        # Can repair: missing values, small gaps
        # Cannot repair: zero values, impossible candles, large gaps
        
        repairable_keywords = ["None values", "Missing timestamps"]
        unrepairable_keywords = ["zero values", "impossible candles", "Invalid"]
        
        for issue in issues:
            for keyword in unrepairable_keywords:
                if keyword.lower() in issue.lower():
                    return False
        
        return True
    
    def _repair_data(self, df: pd.DataFrame, issues: List[str]) -> Optional[pd.DataFrame]:
        """Attempt to repair data issues"""
        try:
            df_repaired = df.copy()
            
            # Repair 1: Interpolate missing OHLC values
            for col in ["Open", "High", "Low", "Close"]:
                if df_repaired[col].isnull().any():
                    null_count = df_repaired[col].isnull().sum()
                    if null_count <= self.MIN_CANDLES_FOR_INTERPOLATION:
                        df_repaired[col] = df_repaired[col].interpolate(method="linear")
                        logger.info(f"Interpolated {null_count} missing {col} values")
            
            # Repair 2: Forward fill volume
            if df_repaired["Volume"].isnull().any():
                df_repaired["Volume"] = df_repaired["Volume"].fillna(method="ffill")
            
            return df_repaired
            
        except Exception as e:
            logger.error(f"Error repairing data: {e}")
            return None
    
    def _calculate_avg_gap(self, df: pd.DataFrame) -> float:
        """Calculate average price gap between candles"""
        if len(df) < 2:
            return 0.0
        
        gaps = df["Close"].diff().abs()
        return float(gaps.mean())
    
    def format_report_for_claude(self, report: DataIntegrityReport) -> str:
        """Format integrity report for Claude notification"""
        if report.feed_integrity == "Clean":
            return ""  # No notification needed for clean data
        
        if report.feed_integrity == "Repaired":
            return (
                f"⚠️ Data Quality Notice: {report.pair} data had minor anomalies "
                f"({', '.join(report.issues_detected[:2])}). "
                f"Data was automatically repaired and validated. "
                f"Source: {report.source}."
            )
        
        if report.feed_integrity == "Corrupted":
            return (
                f"❌ Data Quality Alert: Unable to obtain clean data for {report.pair}. "
                f"Issues: {', '.join(report.issues_detected[:3])}. "
                f"Analysis may be unreliable."
            )
        
        if report.source == "Finnhub":
            return (
                f"⚠️ TwelveData feed had anomalies for {report.pair}. "
                f"Backup feed (Finnhub) used for this analysis. "
                f"Data validated successfully."
            )
        
        return ""
    
    def get_report_summary(self, report: DataIntegrityReport) -> Dict[str, Any]:
        """Get report as dictionary for logging/storage"""
        return {
            "pair": report.pair,
            "source": report.source,
            "feed_integrity": report.feed_integrity,
            "candles_used": report.candles_used,
            "avg_gap": report.avg_gap,
            "timestamp_checked": report.timestamp_checked,
            "issues_detected": report.issues_detected,
            "actions_taken": report.actions_taken,
            "retry_count": report.retry_count
        }
