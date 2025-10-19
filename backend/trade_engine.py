import numpy as np
from datetime import datetime

class TradeEngine:
    """Generates trade recommendations with risk management"""
    
    def __init__(self, account_balance=10000, risk_percent=2):
        """
        Initialize trade engine
        
        Args:
            account_balance: Trading account balance in USD
            risk_percent: Maximum risk per trade as percentage (default 2%)
        """
        self.account_balance = account_balance
        self.risk_percent = risk_percent
        
    def calculate_position_size(self, entry_price, stop_loss, pair_name):
        """
        Calculate position size based on risk management
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            pair_name: Currency pair name
        """
        risk_amount = self.account_balance * (self.risk_percent / 100)
        pip_distance = abs(entry_price - stop_loss)
        
        if 'JPY' in pair_name:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        num_pips = pip_distance / pip_value
        
        if num_pips > 0:
            position_size_units = risk_amount / (num_pips * pip_value * 100000)
            lot_size = round(position_size_units, 2)
        else:
            lot_size = 0.01
        
        lot_size = max(0.01, min(lot_size, 10))
        return lot_size
    
    def generate_detailed_explanation(self, pair, analysis, recommendation_type):
        """Generate detailed explanation for the trade recommendation"""
        
        trend = analysis['trend']
        momentum = analysis['momentum']
        volatility = analysis['volatility']
        
        explanation = {
            'summary': f"Based on technical analysis of {pair}, the market shows {trend['trend'].lower()} conditions.",
            'trend_explanation': self._explain_trend(trend),
            'momentum_explanation': self._explain_momentum(momentum),
            'volatility_explanation': self._explain_volatility(volatility),
            'why_this_trade': self._explain_trade_logic(recommendation_type, analysis),
            'risk_factors': self._identify_risks(analysis),
            'confidence_reason': self._explain_confidence(analysis)
        }
        
        return explanation

    def _explain_trend(self, trend):
        """Explain trend analysis"""
        price = trend['price']
        sma_20 = trend['sma_20']
        sma_50 = trend['sma_50']
        
        if trend['trend'] == "STRONG UPTREND":
            return f"Price ({price:.5f}) is trading above both the 20-day SMA ({sma_20:.5f}) and 50-day SMA ({sma_50:.5f}), indicating strong bullish momentum. The 20-day SMA is also above the 50-day SMA, confirming the uptrend."
        elif trend['trend'] == "UPTREND":
            return f"Price ({price:.5f}) is above the 20-day SMA ({sma_20:.5f}), showing bullish pressure, though the 50-day SMA ({sma_50:.5f}) suggests caution."
        elif trend['trend'] == "STRONG DOWNTREND":
            return f"Price ({price:.5f}) is below both moving averages (20-SMA: {sma_20:.5f}, 50-SMA: {sma_50:.5f}), indicating strong bearish momentum."
        elif trend['trend'] == "DOWNTREND":
            return f"Price ({price:.5f}) is below the 20-day SMA ({sma_20:.5f}), showing bearish pressure."
        else:
            return f"Price ({price:.5f}) is consolidating around the moving averages, indicating indecision in the market."

    def _explain_momentum(self, momentum):
        """Explain momentum indicators"""
        rsi = momentum['rsi']
        macd_signal = "bullish" if momentum['macd'] > momentum['macd_signal'] else "bearish"
        
        explanation = f"RSI is at {rsi:.2f}. "
        
        if rsi < 30:
            explanation += "This is in oversold territory, suggesting the price may have fallen too far and could bounce back. "
        elif rsi > 70:
            explanation += "This is in overbought territory, suggesting the price may have risen too far and could pull back. "
        else:
            explanation += "This is in neutral territory, indicating balanced buying and selling pressure. "
        
        explanation += f"The MACD is currently {macd_signal}, "
        if macd_signal == "bullish":
            explanation += "meaning short-term momentum is stronger than long-term, favoring buyers."
        else:
            explanation += "meaning short-term momentum is weaker than long-term, favoring sellers."
        
        return explanation

    def _explain_volatility(self, volatility):
        """Explain volatility and price levels"""
        atr = volatility['atr']
        support = volatility['support']
        resistance = volatility['resistance']
        bb_signal = volatility['bb_signal']
        
        return (f"The Average True Range (ATR) is {atr:.5f}, indicating the typical price movement range. "
                f"Current support level is at {support:.5f} and resistance is at {resistance:.5f}. "
                f"Bollinger Bands show the price is {bb_signal.lower()}, which helps identify potential reversal points.")

    def _explain_trade_logic(self, recommendation_type, analysis):
        """Explain why this specific trade is recommended"""
        if recommendation_type in ["BUY", "STRONG BUY"]:
            return ("This BUY recommendation is based on multiple bullish signals aligning: "
                    "the trend is upward, momentum indicators are positive, and the price is positioned "
                    "favorably for further gains. The combination of these factors suggests a high probability "
                    "of price appreciation.")
        elif recommendation_type in ["SELL", "STRONG SELL"]:
            return ("This SELL recommendation is based on multiple bearish signals aligning: "
                    "the trend is downward, momentum indicators are negative, and the price is positioned "
                    "for potential decline. The combination of these factors suggests a high probability "
                    "of price depreciation.")
        else:
            return ("No trade is recommended because the technical indicators are sending mixed signals. "
                    "In such conditions, it's safer to wait for clearer market direction before entering a position.")

    def _identify_risks(self, analysis):
        """Identify potential risks for this trade"""
        risks = []
        
        trend_score = analysis['trend']['trend_score']
        momentum_score = analysis['momentum']['momentum_score']
        rsi = analysis['momentum']['rsi']
        
        if abs(trend_score) < 2:
            risks.append("Weak trend strength - price could reverse easily")
        
        if abs(momentum_score) < 2:
            risks.append("Weak momentum - lack of strong buying/selling pressure")
        
        if rsi > 70:
            risks.append("Overbought conditions - potential for pullback")
        elif rsi < 30:
            risks.append("Oversold conditions - price may continue falling before bouncing")
        
        if trend_score * momentum_score < 0:
            risks.append("Trend and momentum are diverging - conflicting signals")
        
        if not risks:
            risks.append("Market conditions are favorable with aligned signals")
        
        return risks

    def _explain_confidence(self, analysis):
        """Explain why confidence is high, medium, or low"""
        total_score = abs(analysis['total_score'])
        
        if total_score >= 4:
            return "HIGH confidence: Multiple strong indicators are aligned in the same direction, reducing uncertainty."
        elif total_score >= 2:
            return "MEDIUM confidence: Several indicators support this direction, but some conflicting signals exist."
        else:
            return "LOW confidence: Mixed signals from technical indicators suggest waiting for clearer market direction."
    
    def generate_trade_recommendation(self, pair, analysis, current_price):
        """
        Generate complete trade recommendation
        
        Args:
            pair: Currency pair name
            analysis: Technical analysis results
            current_price: Current market price
        """
        signal = analysis['overall_signal']
        atr = analysis['volatility']['atr']
        
        # Generate detailed explanation
        detailed_explanation = self.generate_detailed_explanation(pair, analysis, signal)
        
        if signal == "NEUTRAL":
            return {
                'pair': pair,
                'recommendation': 'NO TRADE',
                'reason': 'Market conditions are neutral. Wait for clearer signals.',
                'confidence': 'LOW',
                'analysis_summary': self._format_analysis_summary(analysis),
                'detailed_explanation': detailed_explanation
            }
        
        is_buy = signal in ['BUY', 'STRONG BUY']
        confidence = 'HIGH' if 'STRONG' in signal else 'MEDIUM'
        
        if is_buy:
            entry_price = current_price
            stop_loss = current_price - (2 * atr)
            take_profit_1 = current_price + (2 * atr)
            take_profit_2 = current_price + (4 * atr)
            take_profit_3 = current_price + (6 * atr)
            direction = "BUY"
        else:
            entry_price = current_price
            stop_loss = current_price + (2 * atr)
            take_profit_1 = current_price - (2 * atr)
            take_profit_2 = current_price - (4 * atr)
            take_profit_3 = current_price - (6 * atr)
            direction = "SELL"
        
        lot_size = self.calculate_position_size(entry_price, stop_loss, pair)
        
        if 'JPY' in pair:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        pip_distance = abs(entry_price - stop_loss)
        num_pips = pip_distance / pip_value
        risk_amount = self.account_balance * (self.risk_percent / 100)
        potential_profit_1 = risk_amount
        potential_profit_2 = risk_amount * 2
        potential_profit_3 = risk_amount * 3
        
        return {
            'pair': pair,
            'recommendation': direction,
            'confidence': confidence,
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit_1': round(take_profit_1, 5),
            'take_profit_2': round(take_profit_2, 5),
            'take_profit_3': round(take_profit_3, 5),
            'lot_size': lot_size,
            'risk_amount': round(risk_amount, 2),
            'potential_profit_1': round(potential_profit_1, 2),
            'potential_profit_2': round(potential_profit_2, 2),
            'potential_profit_3': round(potential_profit_3, 2),
            'risk_reward_ratio': '1:1, 1:2, 1:3',
            'pip_distance': round(num_pips, 1),
            'analysis_summary': self._format_analysis_summary(analysis),
            'detailed_explanation': detailed_explanation,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _format_analysis_summary(self, analysis):
        trend = analysis['trend']
        momentum = analysis['momentum']
        volatility = analysis['volatility']
        
        summary = {
            'trend': trend['trend'],
            'current_price': round(trend['price'], 5),
            'rsi': round(momentum['rsi'], 2),
            'macd_signal': 'Bullish' if momentum['macd'] > momentum['macd_signal'] else 'Bearish',
            'support': round(volatility['support'], 5),
            'resistance': round(volatility['resistance'], 5),
            'volatility': round(volatility['atr'], 5),
            'key_signals': momentum['signals']
        }
        
        return summary
    
    def analyze_multiple_pairs(self, analyzer, pairs=None):
        from technical_analysis import TechnicalAnalysis
        
        if pairs is None:
            # Use new Alpha Vantage mapping; fallback to PAIRS if present
            default_mapping = getattr(analyzer, 'AV_CURRENCIES', getattr(analyzer, 'PAIRS', {}))
            pairs = list(default_mapping.keys())
        
        recommendations = []
        
        for pair in pairs:
            print(f"\nAnalyzing {pair}...")
            data = analyzer.fetch_data(pair, period='1mo', interval='1h')
            if data is None or data.empty:
                print(f"Skipping {pair} - no data available")
                continue
            
            ta = TechnicalAnalysis(data)
            analysis = ta.get_comprehensive_analysis()
            current_price = analyzer.get_current_price(pair)
            recommendation = self.generate_trade_recommendation(pair, analysis, current_price)
            recommendations.append(recommendation)
        
        return recommendations