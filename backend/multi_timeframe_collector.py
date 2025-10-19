"""
Multi-Timeframe Technical Data Collector
Two-phase collection system with 2-minute delay between phases
Includes Data Integrity Layer for validation and fallback
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from data_integrity_layer import DataIntegrityLayer, DataIntegrityReport

logger = logging.getLogger(__name__)


@dataclass
class TimeframeData:
    """Container for timeframe-specific technical data"""
    timeframe: str
    timestamp: str
    
    # Price Action
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    
    # Moving Averages
    ema_20: Optional[float] = None
    sma_20: Optional[float] = None
    ema_50: Optional[float] = None
    sma_50: Optional[float] = None
    ema_200: Optional[float] = None
    sma_200: Optional[float] = None
    
    # Momentum Indicators
    rsi_14: Optional[float] = None
    rsi_interpretation: str = "Neutral"
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    
    # Volatility Indicators
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    atr_14: Optional[float] = None
    atr_percent: Optional[float] = None
    
    # Key Levels
    support_1: Optional[float] = None
    support_2: Optional[float] = None
    support_3: Optional[float] = None
    resistance_1: Optional[float] = None
    resistance_2: Optional[float] = None
    resistance_3: Optional[float] = None
    pivot_point: Optional[float] = None
    
    # Volume Indicators
    obv: Optional[float] = None
    obv_trend: str = "Neutral"
    volume_ma_20: Optional[float] = None
    volume_ratio: Optional[float] = None  # Current volume vs 20-period MA
    mfi_14: Optional[float] = None  # Money Flow Index
    mfi_interpretation: str = "Neutral"
    ad_line: Optional[float] = None  # Accumulation/Distribution Line
    cmf_20: Optional[float] = None  # Chaikin Money Flow
    cmf_interpretation: str = "Neutral"
    vwma_20: Optional[float] = None  # Volume-Weighted Moving Average
    
    # Trend Indicators
    adx_14: Optional[float] = None  # Average Directional Index
    plus_di: Optional[float] = None  # +DI
    minus_di: Optional[float] = None  # -DI
    adx_interpretation: str = "Weak"  # Strong/Moderate/Weak
    psar: Optional[float] = None  # Parabolic SAR
    psar_trend: str = "Neutral"  # Bullish/Bearish
    
    # Additional Momentum
    cci_20: Optional[float] = None  # Commodity Channel Index
    cci_interpretation: str = "Neutral"
    williams_r: Optional[float] = None  # Williams %R
    williams_interpretation: str = "Neutral"
    
    # Additional Data (for specific timeframes)
    fib_23_6: Optional[float] = None
    fib_38_2: Optional[float] = None
    fib_50_0: Optional[float] = None
    fib_61_8: Optional[float] = None
    fib_78_6: Optional[float] = None
    ma_crossover_status: str = "None"
    trend_strength: str = "Ranging"
    vwap: Optional[float] = None
    order_flow: str = "Neutral"
    bid_ask_spread: Optional[float] = None
    
    # Data Integrity
    data_source: str = "TwelveData"  # TwelveData or Finnhub
    feed_integrity: str = "Unknown"  # Clean, Repaired, or Corrupted
    integrity_issues: List[str] = field(default_factory=list)
    
    # Metadata
    errors: List[str] = field(default_factory=list)
    success: bool = True


class MultiTimeframeCollector:
    """
    Collects technical data across multiple timeframes in two phases
    with a mandatory 2-minute delay between phases
    """
    
    # Phase 1: Macro and Mid-Term (Monthly to 1H)
    PHASE_1_TIMEFRAMES = [
        ("1M", "1month"),
        ("1W", "1week"),
        ("1D", "1day"),
        ("4H", "4h"),
        ("1H", "1h")
    ]
    
    # Phase 2: Intraday and Micro (30M to 1M)
    PHASE_2_TIMEFRAMES = [
        ("30M", "30min"),
        ("15M", "15min"),
        ("5M", "5min"),
        ("1M", "1min")
    ]
    
    # Timeframes requiring additional data
    HIGHER_TIMEFRAMES = ["1M", "1W", "1D"]
    LOWER_TIMEFRAMES = ["1H", "30M", "15M", "5M", "1M"]
    
    def __init__(self, data_provider, technical_analyzer):
        """
        Initialize collector with data provider and technical analyzer
        
        Args:
            data_provider: Object with fetch_data(pair, interval) method
            technical_analyzer: Object with compute_indicators(df) method
        """
        self.data_provider = data_provider
        self.technical_analyzer = technical_analyzer
        self.integrity_layer = DataIntegrityLayer()
        self.results: Dict[str, Dict[str, TimeframeData]] = {}
        self.integrity_reports: List[DataIntegrityReport] = []
    
    def collect_full_dataset(self, pair: str, progress_callback=None) -> Dict[str, TimeframeData]:
        """
        Collect complete multi-timeframe dataset for a pair
        
        Args:
            pair: Currency pair (e.g., "EUR/USD")
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary mapping timeframe labels to TimeframeData objects
        """
        logger.info(f"Starting full multi-timeframe collection for {pair}")
        if progress_callback:
            progress_callback(f"Starting multi-timeframe analysis for {pair}")
        
        results = {}
        
        # PHASE 1: Macro and Mid-Term Data
        logger.info(f"[{pair}] PHASE 1: Collecting macro and mid-term data (1M, 1W, 1D, 4H, 1H)")
        if progress_callback:
            progress_callback(f"[Phase 1] Collecting macro and mid-term data (Monthly to 1H)")
        
        phase_1_results = self._collect_phase(pair, self.PHASE_1_TIMEFRAMES, phase_num=1, progress_callback=progress_callback)
        results.update(phase_1_results)
        
        logger.info(f"[{pair}] ✓ Phase 1 complete. Collected {len(phase_1_results)} timeframes.")
        if progress_callback:
            progress_callback(f"[Phase 1] Complete. Collected {len(phase_1_results)} timeframes.")
        
        # MANDATORY 2-MINUTE DELAY
        logger.info(f"[{pair}] ⏳ Waiting 2 minutes before Phase 2 to avoid rate limits...")
        if progress_callback:
            progress_callback(f"⏳ Waiting 2 minutes before Phase 2 (rate limit protection)...")
        
        time.sleep(120)  # 2 minutes = 120 seconds
        
        logger.info(f"[{pair}] ✓ 2-minute delay complete. Proceeding to Phase 2.")
        if progress_callback:
            progress_callback(f"✓ 2-minute delay complete. Starting Phase 2...")
        
        # PHASE 2: Intraday and Micro Analysis
        logger.info(f"[{pair}] PHASE 2: Collecting intraday and micro data (30M, 15M, 5M, 1M)")
        if progress_callback:
            progress_callback(f"[Phase 2] Collecting intraday and micro data (30M to 1M)")
        
        phase_2_results = self._collect_phase(pair, self.PHASE_2_TIMEFRAMES, phase_num=2, progress_callback=progress_callback)
        results.update(phase_2_results)
        
        logger.info(f"[{pair}] ✓ Phase 2 complete. Collected {len(phase_2_results)} timeframes.")
        if progress_callback:
            progress_callback(f"[Phase 2] Complete. Collected {len(phase_2_results)} timeframes.")
        
        # Final confirmation
        total_timeframes = len(results)
        logger.info(f"[{pair}] ✅ Full multi-timeframe technical dataset ready for AI analysis. Total timeframes: {total_timeframes}")
        if progress_callback:
            progress_callback(f"✅ Full multi-timeframe dataset ready ({total_timeframes} timeframes)")
        
        self.results[pair] = results
        return results
    
    def _collect_phase(self, pair: str, timeframes: List[Tuple[str, str]], phase_num: int, progress_callback=None) -> Dict[str, TimeframeData]:
        """
        Collect data for a specific phase
        
        Args:
            pair: Currency pair
            timeframes: List of (label, interval) tuples
            phase_num: Phase number (1 or 2)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of timeframe data
        """
        results = {}
        
        for label, interval in timeframes:
            logger.info(f"[{pair}] Processing {label} timeframe...")
            if progress_callback:
                progress_callback(f"Processing {label} timeframe...")
            
            try:
                timeframe_data = self._collect_timeframe(pair, label, interval)
                results[label] = timeframe_data
                
                if timeframe_data.success:
                    logger.info(f"[{pair}] ✓ {label} complete")
                else:
                    logger.warning(f"[{pair}] ⚠ {label} completed with errors: {timeframe_data.errors}")
                
                # Small delay between timeframes within same phase
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"[{pair}] ✗ {label} failed: {e}")
                results[label] = TimeframeData(
                    timeframe=label,
                    timestamp=datetime.utcnow().isoformat(),
                    success=False,
                    errors=[str(e)]
                )
        
        return results
    
    def _collect_timeframe(self, pair: str, label: str, interval: str) -> TimeframeData:
        """
        Collect all indicators for a single timeframe
        
        Args:
            pair: Currency pair
            label: Timeframe label (e.g., "1D")
            interval: API interval (e.g., "1day")
            
        Returns:
            TimeframeData object with all indicators
        """
        data = TimeframeData(
            timeframe=label,
            timestamp=datetime.utcnow().isoformat()
        )
        
        try:
            # Fetch price data with validation
            df, integrity_report = self._fetch_with_retry(pair, interval)
            
            # Store integrity information
            if integrity_report:
                data.data_source = integrity_report.source
                data.feed_integrity = integrity_report.feed_integrity
                data.integrity_issues = integrity_report.issues_detected
            
            if df is None or df.empty:
                data.success = False
                data.errors.append("Failed to fetch price data")
                if integrity_report:
                    data.errors.extend(integrity_report.issues_detected)
                return data
            
            # Extract latest price action
            latest = df.iloc[-1]
            data.open = float(latest.get('Open', 0))
            data.high = float(latest.get('High', 0))
            data.low = float(latest.get('Low', 0))
            data.close = float(latest.get('Close', 0))
            data.volume = float(latest.get('Volume', 0))
            
            # Compute all indicators
            self._compute_moving_averages(df, data)
            self._compute_momentum_indicators(df, data)
            self._compute_volatility_indicators(df, data)
            self._compute_volume_indicators(df, data)
            self._compute_trend_indicators(df, data)
            self._compute_additional_momentum(df, data)
            self._compute_key_levels(df, data)
            
            # Additional data based on timeframe
            if label in self.HIGHER_TIMEFRAMES:
                self._compute_fibonacci_levels(df, data)
                self._compute_ma_crossover(df, data)
                self._compute_trend_strength(df, data)
            
            if label in self.LOWER_TIMEFRAMES:
                self._compute_vwap(df, data)
                self._compute_order_flow(df, data)
                self._compute_bid_ask_spread(data)
            
        except Exception as e:
            logger.exception(f"Error collecting {label} for {pair}: {e}")
            data.success = False
            data.errors.append(str(e))
        
        return data
    
    def _fetch_with_retry(self, pair: str, interval: str, max_retries: int = 2) -> Tuple[Optional[pd.DataFrame], Optional[DataIntegrityReport]]:
        """
        Fetch data with validation, retry logic, and automatic fallback
        
        Returns:
            Tuple of (DataFrame, IntegrityReport)
        """
        # Use Data Integrity Layer for validated fetch with automatic fallback
        df, report = self.integrity_layer.fetch_validated_data(
            pair=pair,
            interval=interval,
            outputsize=100,
            progress_callback=None  # Can pass callback if needed
        )
        
        # Store report for later reference
        if report:
            self.integrity_reports.append(report)
        
        return df, report
    
    def get_integrity_warnings_for_claude(self) -> str:
        """
        Get formatted integrity warnings for Claude
        
        Returns:
            String with warnings, or empty string if all data is clean
        """
        warnings = []
        
        for report in self.integrity_reports:
            warning = self.integrity_layer.format_report_for_claude(report)
            if warning:
                warnings.append(warning)
        
        if warnings:
            return "\n".join(warnings)
        return ""
    
    def get_integrity_summary(self) -> Dict[str, Any]:
        """Get summary of all integrity reports"""
        return {
            "total_timeframes": len(self.integrity_reports),
            "clean_feeds": sum(1 for r in self.integrity_reports if r.feed_integrity == "Clean"),
            "repaired_feeds": sum(1 for r in self.integrity_reports if r.feed_integrity == "Repaired"),
            "corrupted_feeds": sum(1 for r in self.integrity_reports if r.feed_integrity == "Corrupted"),
            "finnhub_fallbacks": sum(1 for r in self.integrity_reports if r.source == "Finnhub"),
            "reports": [self.integrity_layer.get_report_summary(r) for r in self.integrity_reports]
        }
    
    def _compute_moving_averages(self, df: pd.DataFrame, data: TimeframeData):
        """Compute all moving averages"""
        try:
            close = df['Close']
            data.ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            data.sma_20 = float(close.rolling(window=20).mean().iloc[-1])
            data.ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            data.sma_50 = float(close.rolling(window=50).mean().iloc[-1])
            data.ema_200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
            data.sma_200 = float(close.rolling(window=200).mean().iloc[-1])
        except Exception as e:
            data.errors.append(f"MA computation error: {e}")
    
    def _compute_momentum_indicators(self, df: pd.DataFrame, data: TimeframeData):
        """Compute RSI, MACD, Stochastic"""
        try:
            close = df['Close']
            
            # RSI (14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            data.rsi_14 = float(rsi.iloc[-1])
            
            # RSI Interpretation
            if data.rsi_14 >= 70:
                data.rsi_interpretation = "Overbought"
            elif data.rsi_14 <= 30:
                data.rsi_interpretation = "Oversold"
            else:
                data.rsi_interpretation = "Neutral"
            
            # MACD (12, 26, 9)
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            
            data.macd_line = float(macd_line.iloc[-1])
            data.macd_signal = float(signal_line.iloc[-1])
            data.macd_histogram = float(histogram.iloc[-1])
            
            # Stochastic (14, 3, 3)
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            k_percent = 100 * ((close - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(window=3).mean()
            
            data.stoch_k = float(k_percent.iloc[-1])
            data.stoch_d = float(d_percent.iloc[-1])
            
        except Exception as e:
            data.errors.append(f"Momentum indicator error: {e}")
    
    def _compute_volatility_indicators(self, df: pd.DataFrame, data: TimeframeData):
        """Compute Bollinger Bands and ATR"""
        try:
            close = df['Close']
            
            # Bollinger Bands (20, 2)
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            data.bb_upper = float(sma_20.iloc[-1] + (2 * std_20.iloc[-1]))
            data.bb_middle = float(sma_20.iloc[-1])
            data.bb_lower = float(sma_20.iloc[-1] - (2 * std_20.iloc[-1]))
            
            # ATR (14)
            high = df['High']
            low = df['Low']
            prev_close = close.shift(1)
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            data.atr_14 = float(atr.iloc[-1])
            data.atr_percent = float((data.atr_14 / data.close) * 100) if data.close > 0 else None
            
        except Exception as e:
            data.errors.append(f"Volatility indicator error: {e}")
    
    def _compute_volume_indicators(self, df: pd.DataFrame, data: TimeframeData):
        """Compute OBV, Volume MA, Volume Ratio, and MFI"""
        try:
            close = df['Close']
            volume = df['Volume']
            
            # On-Balance Volume (OBV)
            obv = (volume * ((close.diff() > 0).astype(int) - (close.diff() < 0).astype(int))).cumsum()
            data.obv = float(obv.iloc[-1])
            
            # OBV Trend (compare current OBV to 20-period MA of OBV)
            obv_ma = obv.rolling(window=20).mean()
            if len(obv_ma) >= 20:
                if data.obv > obv_ma.iloc[-1]:
                    data.obv_trend = "Bullish"
                elif data.obv < obv_ma.iloc[-1]:
                    data.obv_trend = "Bearish"
                else:
                    data.obv_trend = "Neutral"
            
            # Volume Moving Average (20-period)
            data.volume_ma_20 = float(volume.rolling(window=20).mean().iloc[-1])
            
            # Volume Ratio (current volume vs 20-period MA)
            if data.volume_ma_20 and data.volume_ma_20 > 0:
                data.volume_ratio = float(data.volume / data.volume_ma_20)
            
            # Money Flow Index (MFI) - 14 period
            if len(df) >= 14:
                high = df['High']
                low = df['Low']
                
                # Typical Price
                typical_price = (high + low + close) / 3
                
                # Raw Money Flow
                raw_money_flow = typical_price * volume
                
                # Positive and Negative Money Flow
                money_flow_positive = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
                money_flow_negative = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
                
                # 14-period sums
                positive_flow = money_flow_positive.rolling(window=14).sum()
                negative_flow = money_flow_negative.rolling(window=14).sum()
                
                # Money Flow Ratio and MFI
                money_ratio = positive_flow / negative_flow
                mfi = 100 - (100 / (1 + money_ratio))
                
                data.mfi_14 = float(mfi.iloc[-1])
                
                # MFI Interpretation
                if data.mfi_14 >= 80:
                    data.mfi_interpretation = "Overbought"
                elif data.mfi_14 <= 20:
                    data.mfi_interpretation = "Oversold"
                else:
                    data.mfi_interpretation = "Neutral"
            
            # Accumulation/Distribution Line
            if len(df) >= 2:
                mfm = ((close - low) - (high - close)) / (high - low)
                mfm = mfm.fillna(0)  # Handle division by zero
                mfv = mfm * volume
                ad = mfv.cumsum()
                data.ad_line = float(ad.iloc[-1])
            
            # Chaikin Money Flow (20-period)
            if len(df) >= 20:
                mfm = ((close - low) - (high - close)) / (high - low)
                mfm = mfm.fillna(0)
                mfv = mfm * volume
                cmf = mfv.rolling(window=20).sum() / volume.rolling(window=20).sum()
                data.cmf_20 = float(cmf.iloc[-1])
                
                # CMF Interpretation
                if data.cmf_20 > 0.1:
                    data.cmf_interpretation = "Buying Pressure"
                elif data.cmf_20 < -0.1:
                    data.cmf_interpretation = "Selling Pressure"
                else:
                    data.cmf_interpretation = "Neutral"
            
            # Volume-Weighted Moving Average (20-period)
            if len(df) >= 20:
                vwma = (close * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
                data.vwma_20 = float(vwma.iloc[-1])
            
        except Exception as e:
            data.errors.append(f"Volume indicator error: {e}")
    
    def _compute_key_levels(self, df: pd.DataFrame, data: TimeframeData):
        """Compute support, resistance, and pivot points"""
        try:
            high = df['High'].iloc[-1]
            low = df['Low'].iloc[-1]
            close = df['Close'].iloc[-1]
            
            # Pivot Point
            data.pivot_point = float((high + low + close) / 3)
            
            # Support and Resistance levels
            data.resistance_1 = float((2 * data.pivot_point) - low)
            data.support_1 = float((2 * data.pivot_point) - high)
            data.resistance_2 = float(data.pivot_point + (high - low))
            data.support_2 = float(data.pivot_point - (high - low))
            data.resistance_3 = float(high + 2 * (data.pivot_point - low))
            data.support_3 = float(low - 2 * (high - data.pivot_point))
            
        except Exception as e:
            data.errors.append(f"Key levels error: {e}")
    
    def _compute_trend_indicators(self, df: pd.DataFrame, data: TimeframeData):
        """Compute ADX, +DI, -DI, and Parabolic SAR"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            # ADX and Directional Indicators (14-period)
            if len(df) >= 14:
                # True Range
                prev_close = close.shift(1)
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Directional Movement
                up_move = high - high.shift(1)
                down_move = low.shift(1) - low
                
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                
                # Smooth with Wilder's method (14-period)
                atr = tr.rolling(window=14).mean()
                plus_dm_smooth = pd.Series(plus_dm).rolling(window=14).mean()
                minus_dm_smooth = pd.Series(minus_dm).rolling(window=14).mean()
                
                # Directional Indicators
                plus_di = 100 * (plus_dm_smooth / atr)
                minus_di = 100 * (minus_dm_smooth / atr)
                
                # ADX
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = dx.rolling(window=14).mean()
                
                data.adx_14 = float(adx.iloc[-1])
                data.plus_di = float(plus_di.iloc[-1])
                data.minus_di = float(minus_di.iloc[-1])
                
                # ADX Interpretation
                if data.adx_14 > 25:
                    data.adx_interpretation = "Strong Trend"
                elif data.adx_14 > 20:
                    data.adx_interpretation = "Moderate Trend"
                else:
                    data.adx_interpretation = "Weak Trend"
            
            # Parabolic SAR
            if len(df) >= 5:
                af = 0.02  # Acceleration factor
                max_af = 0.2
                
                # Simplified PSAR calculation (last value only)
                is_uptrend = close.iloc[-1] > close.iloc[-5]
                
                if is_uptrend:
                    sar = low.iloc[-5:].min()
                    data.psar_trend = "Bullish"
                else:
                    sar = high.iloc[-5:].max()
                    data.psar_trend = "Bearish"
                
                data.psar = float(sar)
            
        except Exception as e:
            data.errors.append(f"Trend indicator error: {e}")
    
    def _compute_additional_momentum(self, df: pd.DataFrame, data: TimeframeData):
        """Compute CCI and Williams %R"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            # CCI (Commodity Channel Index) - 20 period
            if len(df) >= 20:
                typical_price = (high + low + close) / 3
                sma_tp = typical_price.rolling(window=20).mean()
                mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
                cci = (typical_price - sma_tp) / (0.015 * mad)
                
                data.cci_20 = float(cci.iloc[-1])
                
                # CCI Interpretation
                if data.cci_20 > 100:
                    data.cci_interpretation = "Overbought"
                elif data.cci_20 < -100:
                    data.cci_interpretation = "Oversold"
                else:
                    data.cci_interpretation = "Neutral"
            
            # Williams %R (14-period)
            if len(df) >= 14:
                highest_high = high.rolling(window=14).max()
                lowest_low = low.rolling(window=14).min()
                williams = -100 * ((highest_high - close) / (highest_high - lowest_low))
                
                data.williams_r = float(williams.iloc[-1])
                
                # Williams %R Interpretation
                if data.williams_r > -20:
                    data.williams_interpretation = "Overbought"
                elif data.williams_r < -80:
                    data.williams_interpretation = "Oversold"
                else:
                    data.williams_interpretation = "Neutral"
            
        except Exception as e:
            data.errors.append(f"Additional momentum error: {e}")
    
    def _compute_fibonacci_levels(self, df: pd.DataFrame, data: TimeframeData):
        """Compute Fibonacci retracement levels"""
        try:
            high = df['High'].max()
            low = df['Low'].min()
            diff = high - low
            
            data.fib_23_6 = float(high - (0.236 * diff))
            data.fib_38_2 = float(high - (0.382 * diff))
            data.fib_50_0 = float(high - (0.500 * diff))
            data.fib_61_8 = float(high - (0.618 * diff))
            data.fib_78_6 = float(high - (0.786 * diff))
            
        except Exception as e:
            data.errors.append(f"Fibonacci levels error: {e}")
    
    def _compute_ma_crossover(self, df: pd.DataFrame, data: TimeframeData):
        """Detect 50/200 MA crossover"""
        try:
            if data.sma_50 and data.sma_200:
                close = df['Close']
                sma_50 = close.rolling(window=50).mean()
                sma_200 = close.rolling(window=200).mean()
                
                # Check current and previous crossover state
                if len(sma_50) >= 2 and len(sma_200) >= 2:
                    current_above = sma_50.iloc[-1] > sma_200.iloc[-1]
                    previous_above = sma_50.iloc[-2] > sma_200.iloc[-2]
                    
                    if current_above and not previous_above:
                        data.ma_crossover_status = "Golden Cross"
                    elif not current_above and previous_above:
                        data.ma_crossover_status = "Death Cross"
                    else:
                        data.ma_crossover_status = "None"
        except Exception as e:
            data.errors.append(f"MA crossover error: {e}")
    
    def _compute_trend_strength(self, df: pd.DataFrame, data: TimeframeData):
        """Determine trend strength"""
        try:
            close = df['Close']
            sma_20 = close.rolling(window=20).mean()
            sma_50 = close.rolling(window=50).mean()
            
            # Calculate trend strength based on price position relative to MAs
            if data.close > sma_20.iloc[-1] > sma_50.iloc[-1]:
                # Strong uptrend
                strength = (data.close - sma_50.iloc[-1]) / sma_50.iloc[-1] * 100
                if strength > 5:
                    data.trend_strength = "Strong Uptrend"
                elif strength > 2:
                    data.trend_strength = "Moderate Uptrend"
                else:
                    data.trend_strength = "Weak Uptrend"
            elif data.close < sma_20.iloc[-1] < sma_50.iloc[-1]:
                # Strong downtrend
                strength = (sma_50.iloc[-1] - data.close) / sma_50.iloc[-1] * 100
                if strength > 5:
                    data.trend_strength = "Strong Downtrend"
                elif strength > 2:
                    data.trend_strength = "Moderate Downtrend"
                else:
                    data.trend_strength = "Weak Downtrend"
            else:
                data.trend_strength = "Ranging"
        except Exception as e:
            data.errors.append(f"Trend strength error: {e}")
    
    def _compute_vwap(self, df: pd.DataFrame, data: TimeframeData):
        """Compute Volume-Weighted Average Price"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
            data.vwap = float(vwap.iloc[-1])
        except Exception as e:
            data.errors.append(f"VWAP error: {e}")
    
    def _compute_order_flow(self, df: pd.DataFrame, data: TimeframeData):
        """Determine order flow balance"""
        try:
            # Simple order flow based on price and volume
            close_change = df['Close'].diff()
            volume = df['Volume']
            
            buying_volume = volume.where(close_change > 0, 0).sum()
            selling_volume = volume.where(close_change < 0, 0).sum()
            
            if buying_volume > selling_volume * 1.2:
                data.order_flow = "Buyers"
            elif selling_volume > buying_volume * 1.2:
                data.order_flow = "Sellers"
            else:
                data.order_flow = "Neutral"
        except Exception as e:
            data.errors.append(f"Order flow error: {e}")
    
    def _compute_bid_ask_spread(self, data: TimeframeData):
        """Estimate bid/ask spread (simplified)"""
        try:
            # Simplified spread estimation (0.1% of price)
            data.bid_ask_spread = float(data.close * 0.001)
        except Exception as e:
            data.errors.append(f"Bid/ask spread error: {e}")
    
    def format_output(self, pair: str) -> str:
        """
        Format collected data for display/logging
        
        Args:
            pair: Currency pair
            
        Returns:
            Formatted string output
        """
        if pair not in self.results:
            return f"No data collected for {pair}"
        
        output = []
        output.append("=" * 80)
        output.append(f"MULTI-TIMEFRAME TECHNICAL ANALYSIS: {pair}")
        output.append("=" * 80)
        output.append("")
        
        results = self.results[pair]
        
        # Order: 1M → 1W → 1D → 4H → 1H → 30M → 15M → 5M → 1M
        timeframe_order = ["1M", "1W", "1D", "4H", "1H", "30M", "15M", "5M", "1M"]
        
        for tf in timeframe_order:
            if tf not in results:
                continue
            
            data = results[tf]
            output.append(f"{pair} – {tf}")
            output.append("=" * 40)
            
            # Price Action
            output.append(f"Open: {data.open:.5f} | High: {data.high:.5f} | Low: {data.low:.5f} | Close: {data.close:.5f}")
            output.append(f"Volume: {data.volume:,.0f}")
            output.append("")
            
            # Moving Averages
            output.append("Moving Averages:")
            output.append(f"  EMA(20): {data.ema_20:.5f} | SMA(20): {data.sma_20:.5f}")
            output.append(f"  EMA(50): {data.ema_50:.5f} | SMA(50): {data.sma_50:.5f}")
            output.append(f"  EMA(200): {data.ema_200:.5f} | SMA(200): {data.sma_200:.5f}")
            output.append("")
            
            # Momentum
            output.append("Momentum Indicators:")
            output.append(f"  RSI(14): {data.rsi_14:.2f} - {data.rsi_interpretation}")
            output.append(f"  MACD: Line={data.macd_line:.5f}, Signal={data.macd_signal:.5f}, Hist={data.macd_histogram:.5f}")
            output.append(f"  Stochastic: %K={data.stoch_k:.2f}, %D={data.stoch_d:.2f}")
            output.append("")
            
            # Volatility
            output.append("Volatility Indicators:")
            output.append(f"  Bollinger Bands: Upper={data.bb_upper:.5f}, Middle={data.bb_middle:.5f}, Lower={data.bb_lower:.5f}")
            output.append(f"  ATR(14): {data.atr_14:.5f} ({data.atr_percent:.2f}% of price)")
            output.append("")
            
            # Key Levels
            output.append("Key Levels:")
            output.append(f"  Pivot: {data.pivot_point:.5f}")
            output.append(f"  Resistance: R1={data.resistance_1:.5f}, R2={data.resistance_2:.5f}, R3={data.resistance_3:.5f}")
            output.append(f"  Support: S1={data.support_1:.5f}, S2={data.support_2:.5f}, S3={data.support_3:.5f}")
            output.append("")
            
            # Additional data for higher timeframes
            if tf in self.HIGHER_TIMEFRAMES and data.fib_50_0:
                output.append("Fibonacci Levels:")
                output.append(f"  23.6%: {data.fib_23_6:.5f} | 38.2%: {data.fib_38_2:.5f} | 50.0%: {data.fib_50_0:.5f}")
                output.append(f"  61.8%: {data.fib_61_8:.5f} | 78.6%: {data.fib_78_6:.5f}")
                output.append(f"  MA Crossover: {data.ma_crossover_status}")
                output.append(f"  Trend Strength: {data.trend_strength}")
                output.append("")
            
            # Additional data for lower timeframes
            if tf in self.LOWER_TIMEFRAMES and data.vwap:
                output.append("Microstructure:")
                output.append(f"  VWAP: {data.vwap:.5f}")
                output.append(f"  Order Flow: {data.order_flow}")
                output.append(f"  Bid/Ask Spread: {data.bid_ask_spread:.5f}")
                output.append("")
            
            # Errors if any
            if data.errors:
                output.append(f"⚠ Errors: {', '.join(data.errors)}")
                output.append("")
            
            output.append("")
        
        output.append("=" * 80)
        output.append("✅ Full multi-timeframe technical dataset ready for AI analysis.")
        output.append("=" * 80)
        
        return "\n".join(output)
