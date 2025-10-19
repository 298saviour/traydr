"""
Final Confidence Layer Integration
Combines execution context, patterns, divergence, and performance analytics
for maximum confidence signal output
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import talib

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Live trading conditions"""
    current_spread: float
    average_spread_24h: float
    current_volume: float
    average_volume_24h: float
    liquidity_condition: str  # Normal, Thin, Heavy
    slippage_average: float
    broker_type: str = "Hybrid (ECN/STP)"
    confidence_modifier: float = 0.0
    
    def compute_modifier(self):
        """Compute confidence modifier based on execution context"""
        modifier = 0.0
        
        # Liquidity condition
        if self.liquidity_condition == "Normal" and "ECN" in self.broker_type:
            modifier += 2.0
        elif self.liquidity_condition == "Thin":
            modifier -= 3.0
        
        # Slippage impact
        if self.slippage_average > 2.0:  # pips
            modifier -= 2.0
        
        self.confidence_modifier = modifier
        return modifier


@dataclass
class PatternDetection:
    """Chart pattern analysis"""
    detected_patterns: List[str] = field(default_factory=list)
    completion_percentage: Dict[str, float] = field(default_factory=dict)
    measured_targets: Dict[str, float] = field(default_factory=dict)
    failed_patterns: List[str] = field(default_factory=list)
    confidence_modifier: float = 0.0
    
    def compute_modifier(self, trade_direction: str):
        """Compute confidence modifier based on patterns"""
        modifier = 0.0
        
        # Valid continuation patterns
        bullish_patterns = ["FLAG", "PENNANT", "ASCENDING_TRIANGLE", "CUP_HANDLE"]
        bearish_patterns = ["BEAR_FLAG", "DESCENDING_TRIANGLE", "HEAD_SHOULDERS"]
        
        if trade_direction == "BUY":
            for pattern in self.detected_patterns:
                if any(bp in pattern.upper() for bp in bullish_patterns):
                    modifier += 5.0
                    break
        elif trade_direction == "SELL":
            for pattern in self.detected_patterns:
                if any(bp in pattern.upper() for bp in bearish_patterns):
                    modifier += 5.0
                    break
        
        # Failed patterns penalty
        if len(self.failed_patterns) > 0:
            modifier -= 5.0
        
        self.confidence_modifier = modifier
        return modifier


@dataclass
class TimeBasedFactors:
    """Time-based market factors"""
    next_high_impact_event: Optional[str] = None
    hours_until_event: Optional[float] = None
    seasonal_bias: str = "Neutral"  # Bullish, Bearish, Neutral
    trend_duration_candles: int = 0
    confidence_modifier: float = 0.0
    
    def compute_modifier(self, trade_direction: str):
        """Compute confidence modifier based on time factors"""
        modifier = 0.0
        
        # Upcoming event risk
        if self.hours_until_event and self.hours_until_event < 4:
            modifier -= 5.0
        
        # Seasonal alignment
        if self.seasonal_bias == trade_direction.capitalize() + "ish":
            modifier += 3.0
        
        self.confidence_modifier = modifier
        return modifier


@dataclass
class DivergenceTracking:
    """Divergence analysis across timeframes"""
    rsi_divergence_1h: str = "None"  # Bullish, Bearish, None
    rsi_divergence_4h: str = "None"
    volume_divergence: str = "None"
    cross_timeframe_conflict: bool = False
    currency_strength_divergence: float = 0.0
    confidence_modifier: float = 0.0
    
    def compute_modifier(self, trade_direction: str):
        """Compute confidence modifier based on divergences"""
        modifier = 0.0
        
        # RSI divergence alignment
        if trade_direction == "BUY":
            if self.rsi_divergence_4h == "Bullish":
                modifier += 4.0
        elif trade_direction == "SELL":
            if self.rsi_divergence_4h == "Bearish":
                modifier += 4.0
        
        # Cross-timeframe conflict
        if self.cross_timeframe_conflict:
            modifier -= 4.0
        
        self.confidence_modifier = modifier
        return modifier


@dataclass
class InvalidationLevels:
    """Trade invalidation thresholds"""
    price_level: float
    time_expiry: str  # ISO format
    structural_reference: str  # e.g., "Last swing low at 1.0820"
    fibonacci_level: Optional[float] = None
    
    def format_output(self) -> str:
        """Format invalidation info"""
        return f"Price: {self.price_level:.5f} | Time: {self.time_expiry} | Ref: {self.structural_reference}"


@dataclass
class PerformanceAnalytics:
    """Historical performance tracking"""
    last_5_outcomes: List[str] = field(default_factory=list)  # "WIN", "LOSS"
    win_loss_ratio: float = 0.0
    average_confidence: float = 0.0
    best_timeframe: str = "4H"
    common_failures: List[str] = field(default_factory=list)
    confidence_modifier: float = 0.0
    
    def compute_modifier(self):
        """Compute confidence modifier based on performance"""
        modifier = 0.0
        
        # Recent success rate
        if len(self.last_5_outcomes) >= 3:
            recent_wins = sum(1 for o in self.last_5_outcomes[:3] if o == "WIN")
            if recent_wins == 3:
                modifier += 3.0
            elif recent_wins < 2:  # Less than 50%
                modifier -= 3.0
        
        self.confidence_modifier = modifier
        return modifier


@dataclass
class FinalConfidenceOutput:
    """Final signal output with maximum confidence"""
    pair: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Confidence components
    base_confidence: float
    execution_modifier: float
    pattern_modifier: float
    time_modifier: float
    divergence_modifier: float
    performance_modifier: float
    final_confidence: float
    
    # Context
    execution_context: ExecutionContext
    pattern_detection: PatternDetection
    time_factors: TimeBasedFactors
    divergence_tracking: DivergenceTracking
    invalidation: InvalidationLevels
    performance: PerformanceAnalytics
    
    # Summary
    confirmation_summary: str
    timestamp: str


class FinalConfidenceLayer:
    """
    Integrates all confidence factors after 4H confirmation
    """
    
    def __init__(self, data_provider, technical_analyzer, news_provider, signal_history):
        """
        Initialize confidence layer
        
        Args:
            data_provider: Market data provider
            technical_analyzer: Technical analysis engine
            news_provider: Finnhub news provider
            signal_history: Historical signal database
        """
        self.data_provider = data_provider
        self.technical_analyzer = technical_analyzer
        self.news_provider = news_provider
        self.signal_history = signal_history
    
    def compute_final_confidence(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        base_confidence: float,
        technical_data: Dict,
        progress_callback=None
    ) -> FinalConfidenceOutput:
        """
        Compute final confidence after 4H confirmation
        
        Args:
            pair: Currency pair
            direction: Trade direction (BUY/SELL)
            entry_price: Entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            base_confidence: Initial confidence from technical analysis
            technical_data: Technical indicator data
            progress_callback: Optional progress callback
            
        Returns:
            FinalConfidenceOutput with all confidence factors
        """
        logger.info(f"Computing final confidence for {pair} {direction}")
        if progress_callback:
            progress_callback(f"Computing final confidence for {pair}")
        
        # Stabilization delay
        logger.info("Waiting 1 minute for stabilization...")
        if progress_callback:
            progress_callback("Waiting 1 minute for market stabilization...")
        time.sleep(60)
        
        # Module 1: Execution Context
        if progress_callback:
            progress_callback("Analyzing execution context...")
        execution_ctx = self._analyze_execution_context(pair)
        
        # Module 2: Pattern Detection
        if progress_callback:
            progress_callback("Detecting chart patterns...")
        pattern_det = self._detect_patterns(pair, direction, technical_data)
        
        # Module 3: Time-Based Factors
        if progress_callback:
            progress_callback("Analyzing time-based factors...")
        time_factors = self._analyze_time_factors(pair, direction)
        
        # Module 4: Divergence Tracking
        if progress_callback:
            progress_callback("Tracking divergences...")
        divergence = self._track_divergences(pair, direction, technical_data)
        
        # Module 5: Invalidation Levels
        if progress_callback:
            progress_callback("Computing invalidation levels...")
        invalidation = self._compute_invalidation(pair, direction, entry_price, technical_data)
        
        # Module 6: Performance Analytics
        if progress_callback:
            progress_callback("Analyzing historical performance...")
        performance = self._analyze_performance(pair)
        
        # Compute modifiers
        exec_mod = execution_ctx.compute_modifier()
        pattern_mod = pattern_det.compute_modifier(direction)
        time_mod = time_factors.compute_modifier(direction)
        div_mod = divergence.compute_modifier(direction)
        perf_mod = performance.compute_modifier()
        
        # Calculate final confidence
        total_modifier = exec_mod + pattern_mod + time_mod + div_mod + perf_mod
        final_confidence = min(base_confidence + total_modifier, 95.0)  # Cap at 95%
        
        # Generate confirmation summary
        summary = self._generate_summary(
            direction, execution_ctx, pattern_det, time_factors,
            divergence, performance, final_confidence
        )
        
        output = FinalConfidenceOutput(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            base_confidence=base_confidence,
            execution_modifier=exec_mod,
            pattern_modifier=pattern_mod,
            time_modifier=time_mod,
            divergence_modifier=div_mod,
            performance_modifier=perf_mod,
            final_confidence=final_confidence,
            execution_context=execution_ctx,
            pattern_detection=pattern_det,
            time_factors=time_factors,
            divergence_tracking=divergence,
            invalidation=invalidation,
            performance=performance,
            confirmation_summary=summary,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Final confidence computed: {final_confidence:.1f}%")
        if progress_callback:
            progress_callback(f"✓ Final confidence: {final_confidence:.1f}%")
        
        return output
    
    def _analyze_execution_context(self, pair: str) -> ExecutionContext:
        """Analyze live trading conditions"""
        try:
            # Fetch current market data
            current_data = self.data_provider.fetch_current_quote(pair)
            historical_data = self.data_provider.fetch_data(pair, interval="1h", limit=24)
            
            # Calculate spreads
            current_spread = current_data.get("spread", 0.0002)  # Default 2 pips
            avg_spread_24h = historical_data["Spread"].mean() if "Spread" in historical_data else current_spread
            
            # Calculate volumes
            current_volume = current_data.get("volume", 0)
            avg_volume_24h = historical_data["Volume"].mean()
            
            # Determine liquidity condition
            if current_spread > 2 * avg_spread_24h:
                liquidity = "Thin"
            elif current_volume > 1.5 * avg_volume_24h:
                liquidity = "Heavy"
            else:
                liquidity = "Normal"
            
            # Estimate slippage (simplified)
            slippage_avg = current_spread * 10000 * 0.5  # Half spread in pips
            
            return ExecutionContext(
                current_spread=current_spread,
                average_spread_24h=avg_spread_24h,
                current_volume=current_volume,
                average_volume_24h=avg_volume_24h,
                liquidity_condition=liquidity,
                slippage_average=slippage_avg
            )
        
        except Exception as e:
            logger.exception(f"Error analyzing execution context: {e}")
            return ExecutionContext(
                current_spread=0.0002,
                average_spread_24h=0.0002,
                current_volume=0,
                average_volume_24h=0,
                liquidity_condition="Normal",
                slippage_average=1.0
            )
    
    def _detect_patterns(self, pair: str, direction: str, technical_data: Dict) -> PatternDetection:
        """Detect chart patterns using TA-Lib"""
        try:
            df = self.data_provider.fetch_data(pair, interval="4h", limit=100)
            
            open_prices = df["Open"].values
            high_prices = df["High"].values
            low_prices = df["Low"].values
            close_prices = df["Close"].values
            
            detected = []
            completion = {}
            targets = {}
            
            # Detect patterns using TA-Lib
            patterns_to_check = [
                ("CDLDOJI", "Doji"),
                ("CDLENGULFING", "Engulfing"),
                ("CDLHAMMER", "Hammer"),
                ("CDLSHOOTINGSTAR", "Shooting Star"),
                ("CDLMORNINGSTAR", "Morning Star"),
                ("CDLEVENINGSTAR", "Evening Star"),
                ("CDL3WHITESOLDIERS", "Three White Soldiers"),
                ("CDL3BLACKCROWS", "Three Black Crows")
            ]
            
            for pattern_func, pattern_name in patterns_to_check:
                if hasattr(talib, pattern_func):
                    result = getattr(talib, pattern_func)(open_prices, high_prices, low_prices, close_prices)
                    if result[-1] != 0:
                        detected.append(pattern_name)
                        completion[pattern_name] = 100.0  # Pattern completed
            
            # Check for failed patterns (simplified)
            failed = []
            if len(detected) == 0 and close_prices[-1] < close_prices[-5]:
                failed.append("Failed breakout")
            
            return PatternDetection(
                detected_patterns=detected,
                completion_percentage=completion,
                measured_targets=targets,
                failed_patterns=failed
            )
        
        except Exception as e:
            logger.exception(f"Error detecting patterns: {e}")
            return PatternDetection()
    
    def _analyze_time_factors(self, pair: str, direction: str) -> TimeBasedFactors:
        """Analyze time-based factors"""
        try:
            # Fetch economic calendar
            fundamental_data = self.news_provider.fetch_fundamental_data(pair)
            
            # Find next high-impact event
            next_event = None
            hours_until = None
            
            for event in fundamental_data.upcoming_events:
                if event.impact == "HIGH":
                    next_event = event.event_name
                    event_time = datetime.fromisoformat(event.datetime_utc.replace(" UTC", ""))
                    hours_until = (event_time - datetime.utcnow()).total_seconds() / 3600
                    break
            
            # Seasonal bias (simplified - would use historical data)
            current_month = datetime.utcnow().month
            seasonal_bias = "Neutral"
            if current_month in [11, 12, 1]:  # Winter months
                seasonal_bias = "Bullish" if "USD" in pair else "Neutral"
            
            # Trend duration
            df = self.data_provider.fetch_data(pair, interval="4h", limit=50)
            sma_20 = df["Close"].rolling(window=20).mean()
            sma_50 = df["Close"].rolling(window=50).mean()
            
            # Count candles since crossover
            trend_duration = 0
            for i in range(len(df) - 1, 0, -1):
                if (sma_20.iloc[i] > sma_50.iloc[i]) != (sma_20.iloc[i-1] > sma_50.iloc[i-1]):
                    trend_duration = len(df) - i
                    break
            
            return TimeBasedFactors(
                next_high_impact_event=next_event,
                hours_until_event=hours_until,
                seasonal_bias=seasonal_bias,
                trend_duration_candles=trend_duration
            )
        
        except Exception as e:
            logger.exception(f"Error analyzing time factors: {e}")
            return TimeBasedFactors()
    
    def _track_divergences(self, pair: str, direction: str, technical_data: Dict) -> DivergenceTracking:
        """Track divergences across timeframes"""
        try:
            # Fetch data for multiple timeframes
            df_1h = self.data_provider.fetch_data(pair, interval="1h", limit=50)
            df_4h = self.data_provider.fetch_data(pair, interval="4h", limit=50)
            
            # RSI divergence detection
            rsi_1h_div = self._detect_rsi_divergence(df_1h)
            rsi_4h_div = self._detect_rsi_divergence(df_4h)
            
            # Volume divergence
            volume_div = self._detect_volume_divergence(df_4h)
            
            # Cross-timeframe conflict
            conflict = (rsi_1h_div == "Bullish" and rsi_4h_div == "Bearish") or \
                      (rsi_1h_div == "Bearish" and rsi_4h_div == "Bullish")
            
            return DivergenceTracking(
                rsi_divergence_1h=rsi_1h_div,
                rsi_divergence_4h=rsi_4h_div,
                volume_divergence=volume_div,
                cross_timeframe_conflict=conflict
            )
        
        except Exception as e:
            logger.exception(f"Error tracking divergences: {e}")
            return DivergenceTracking()
    
    def _detect_rsi_divergence(self, df: pd.DataFrame) -> str:
        """Detect RSI divergence"""
        try:
            close = df["Close"].values
            rsi = talib.RSI(close, timeperiod=14)
            
            # Find recent highs/lows
            price_highs = []
            rsi_highs = []
            
            for i in range(2, len(close) - 2):
                if close[i] > close[i-1] and close[i] > close[i+1]:
                    price_highs.append(close[i])
                    rsi_highs.append(rsi[i])
            
            # Check for divergence
            if len(price_highs) >= 2:
                if price_highs[-1] > price_highs[-2] and rsi_highs[-1] < rsi_highs[-2]:
                    return "Bearish"
                elif price_highs[-1] < price_highs[-2] and rsi_highs[-1] > rsi_highs[-2]:
                    return "Bullish"
            
            return "None"
        
        except:
            return "None"
    
    def _detect_volume_divergence(self, df: pd.DataFrame) -> str:
        """Detect volume divergence"""
        try:
            close = df["Close"].values
            volume = df["Volume"].values
            
            # Compare price and volume trends
            price_trend = close[-1] - close[-10]
            volume_trend = volume[-10:].mean() - volume[-20:-10].mean()
            
            if price_trend > 0 and volume_trend < 0:
                return "Bearish"
            elif price_trend < 0 and volume_trend > 0:
                return "Bullish"
            
            return "None"
        
        except:
            return "None"
    
    def _compute_invalidation(self, pair: str, direction: str, entry_price: float, technical_data: Dict) -> InvalidationLevels:
        """Compute invalidation levels"""
        try:
            df = self.data_provider.fetch_data(pair, interval="4h", limit=50)
            
            # Find structural levels
            if direction == "BUY":
                # Last swing low
                recent_lows = df["Low"].tail(20)
                swing_low = recent_lows.min()
                invalidation_price = swing_low * 0.999  # Just below swing low
                reference = f"Last swing low at {swing_low:.5f}"
            else:
                # Last swing high
                recent_highs = df["High"].tail(20)
                swing_high = recent_highs.max()
                invalidation_price = swing_high * 1.001  # Just above swing high
                reference = f"Last swing high at {swing_high:.5f}"
            
            # Time expiry (4 hours from now)
            expiry_time = (datetime.utcnow() + timedelta(hours=4)).isoformat()
            
            return InvalidationLevels(
                price_level=invalidation_price,
                time_expiry=expiry_time,
                structural_reference=reference
            )
        
        except Exception as e:
            logger.exception(f"Error computing invalidation: {e}")
            return InvalidationLevels(
                price_level=entry_price * 0.99 if direction == "BUY" else entry_price * 1.01,
                time_expiry=(datetime.utcnow() + timedelta(hours=4)).isoformat(),
                structural_reference="Default invalidation"
            )
    
    def _analyze_performance(self, pair: str) -> PerformanceAnalytics:
        """Analyze historical performance"""
        try:
            # Fetch last 5 signals for this pair
            recent_signals = self.signal_history.get_recent_signals(pair, limit=5)
            
            outcomes = []
            confidences = []
            
            for signal in recent_signals:
                if signal.get("outcome"):
                    outcomes.append(signal["outcome"])
                if signal.get("confidence"):
                    confidences.append(signal["confidence"])
            
            # Calculate metrics
            win_loss = sum(1 for o in outcomes if o == "WIN") / len(outcomes) if outcomes else 0.5
            avg_conf = sum(confidences) / len(confidences) if confidences else 70.0
            
            return PerformanceAnalytics(
                last_5_outcomes=outcomes,
                win_loss_ratio=win_loss,
                average_confidence=avg_conf,
                best_timeframe="4H"
            )
        
        except Exception as e:
            logger.exception(f"Error analyzing performance: {e}")
            return PerformanceAnalytics()
    
    def _generate_summary(
        self,
        direction: str,
        execution: ExecutionContext,
        patterns: PatternDetection,
        time_factors: TimeBasedFactors,
        divergence: DivergenceTracking,
        performance: PerformanceAnalytics,
        final_confidence: float
    ) -> str:
        """Generate confirmation summary"""
        lines = []
        lines.append(f"FINAL CONFIDENCE: {final_confidence:.1f}%")
        lines.append(f"Direction: {direction}")
        lines.append("")
        
        lines.append("EXECUTION CONTEXT:")
        lines.append(f"  Liquidity: {execution.liquidity_condition}")
        lines.append(f"  Spread: {execution.current_spread * 10000:.1f} pips (avg: {execution.average_spread_24h * 10000:.1f})")
        lines.append(f"  Modifier: {execution.confidence_modifier:+.1f}%")
        lines.append("")
        
        lines.append("PATTERN ANALYSIS:")
        if patterns.detected_patterns:
            lines.append(f"  Detected: {', '.join(patterns.detected_patterns)}")
        else:
            lines.append("  No significant patterns")
        lines.append(f"  Modifier: {patterns.confidence_modifier:+.1f}%")
        lines.append("")
        
        lines.append("TIME FACTORS:")
        if time_factors.next_high_impact_event:
            lines.append(f"  Next event: {time_factors.next_high_impact_event} in {time_factors.hours_until_event:.1f}h")
        lines.append(f"  Seasonal bias: {time_factors.seasonal_bias}")
        lines.append(f"  Modifier: {time_factors.confidence_modifier:+.1f}%")
        lines.append("")
        
        lines.append("DIVERGENCE:")
        lines.append(f"  4H RSI: {divergence.rsi_divergence_4h}")
        lines.append(f"  Volume: {divergence.volume_divergence}")
        lines.append(f"  Modifier: {divergence.confidence_modifier:+.1f}%")
        lines.append("")
        
        lines.append("PERFORMANCE:")
        lines.append(f"  Recent W/L: {performance.win_loss_ratio:.1%}")
        lines.append(f"  Modifier: {performance.confidence_modifier:+.1f}%")
        
        return "\n".join(lines)
    
    def format_output(self, output: FinalConfidenceOutput) -> str:
        """Format final confidence output for display"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"FINAL CONFIDENCE ANALYSIS: {output.pair} {output.direction}")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append("SIGNAL DETAILS:")
        lines.append(f"  Entry: {output.entry_price:.5f}")
        lines.append(f"  Stop Loss: {output.stop_loss:.5f}")
        lines.append(f"  Take Profit: {output.take_profit:.5f}")
        lines.append("")
        
        lines.append("CONFIDENCE BREAKDOWN:")
        lines.append(f"  Base Confidence: {output.base_confidence:.1f}%")
        lines.append(f"  Execution Context: {output.execution_modifier:+.1f}%")
        lines.append(f"  Pattern Detection: {output.pattern_modifier:+.1f}%")
        lines.append(f"  Time Factors: {output.time_modifier:+.1f}%")
        lines.append(f"  Divergence: {output.divergence_modifier:+.1f}%")
        lines.append(f"  Performance: {output.performance_modifier:+.1f}%")
        lines.append(f"  ─────────────────────")
        lines.append(f"  FINAL CONFIDENCE: {output.final_confidence:.1f}%")
        lines.append("")
        
        lines.append("INVALIDATION:")
        lines.append(f"  {output.invalidation.format_output()}")
        lines.append("")
        
        lines.append("CONFIRMATION SUMMARY:")
        lines.append(output.confirmation_summary)
        lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
