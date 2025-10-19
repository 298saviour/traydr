import pandas as pd
import numpy as np

class TechnicalAnalysis:
    """Performs technical analysis using manual calculations"""
    
    def __init__(self, data):
        self.data = data.copy()
        if 'Volume' not in self.data.columns:
            self.data['Volume'] = 0
        self.signals = {}
    
    def calculate_sma(self, period):
        """Simple Moving Average"""
        return self.data['Close'].rolling(window=period).mean()
    
    def calculate_ema(self, period):
        """Exponential Moving Average"""
        return self.data['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, period=14):
        """Relative Strength Index"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self):
        """MACD Indicator"""
        ema_12 = self.calculate_ema(12)
        ema_26 = self.calculate_ema(26)
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, macd - signal
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = self.calculate_sma(period)
        std = self.data['Close'].rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    def calculate_obv(self):
        """On-Balance Volume"""
        return (np.sign(self.data['Close'].diff()) * self.data['Volume']).fillna(0).cumsum()

    def calculate_average_volume(self, period=20):
        """Average Volume"""
        return self.data['Volume'].rolling(window=period).mean()

    def calculate_pivot_points(self):
        """Calculate Pivot Points, Support, and Resistance"""
        high = self.data['High'].iloc[-1]
        low = self.data['Low'].iloc[-1]
        close = self.data['Close'].iloc[-1]

        pivot = (high + low + close) / 3
        s1 = (2 * pivot) - high
        r1 = (2 * pivot) - low
        s2 = pivot - (high - low)
        r2 = pivot + (high - low)
        s3 = low - 2 * (high - pivot)
        r3 = high + 2 * (pivot - low)

        return {'pivot': pivot, 's1': s1, 's2': s2, 's3': s3, 'r1': r1, 'r2': r2, 'r3': r3}
    
    def calculate_atr(self, period=14):
        """Average True Range"""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def calculate_stochastic(self, period=14):
        """Stochastic Oscillator"""
        low_min = self.data['Low'].rolling(window=period).min()
        high_max = self.data['High'].rolling(window=period).max()
        
        k = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=3).mean()
        return k, d
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        self.data['SMA_20'] = self.calculate_sma(20)
        self.data['SMA_50'] = self.calculate_sma(50)
        self.data['EMA_12'] = self.calculate_ema(12)
        self.data['EMA_26'] = self.calculate_ema(26)
        
        macd, signal, diff = self.calculate_macd()
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = signal
        self.data['MACD_Diff'] = diff
        
        self.data['RSI'] = self.calculate_rsi(14)
        
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands()
        self.data['BB_High'] = bb_upper
        self.data['BB_Mid'] = bb_mid
        self.data['BB_Low'] = bb_lower
        
        self.data['ATR'] = self.calculate_atr(14)
        
        stoch_k, stoch_d = self.calculate_stochastic()
        self.data['Stoch_K'] = stoch_k
        self.data['Stoch_D'] = stoch_d

        self.data['OBV'] = self.calculate_obv()
        self.data['Avg_Volume'] = self.calculate_average_volume()
        
        return self.data
    
    def analyze_trend(self):
        """Analyze overall trend"""
        latest = self.data.iloc[-1]
        
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            trend = "STRONG UPTREND"
            trend_score = 2
        elif latest['Close'] > latest['SMA_20']:
            trend = "UPTREND"
            trend_score = 1
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            trend = "STRONG DOWNTREND"
            trend_score = -2
        elif latest['Close'] < latest['SMA_20']:
            trend = "DOWNTREND"
            trend_score = -1
        else:
            trend = "SIDEWAYS"
            trend_score = 0
        
        return {
            'trend': trend,
            'trend_score': trend_score,
            'price': latest['Close'],
            'sma_20': latest['SMA_20'],
            'sma_50': latest['SMA_50']
        }
    
    def analyze_momentum(self):
        """Analyze momentum indicators"""
        latest = self.data.iloc[-1]
        
        signals = []
        score = 0
        
        rsi = latest['RSI']
        if rsi < 30:
            signals.append("RSI Oversold (Bullish)")
            score += 2
        elif rsi < 40:
            signals.append("RSI Approaching Oversold")
            score += 1
        elif rsi > 70:
            signals.append("RSI Overbought (Bearish)")
            score -= 2
        elif rsi > 60:
            signals.append("RSI Approaching Overbought")
            score -= 1
        else:
            signals.append("RSI Neutral")
        
        if latest['MACD'] > latest['MACD_Signal']:
            signals.append("MACD Bullish")
            score += 1
        else:
            signals.append("MACD Bearish")
            score -= 1
        
        if latest['Stoch_K'] < 20:
            signals.append("Stochastic Oversold")
            score += 1
        elif latest['Stoch_K'] > 80:
            signals.append("Stochastic Overbought")
            score -= 1
        
        return {
            'signals': signals,
            'momentum_score': score,
            'rsi': rsi,
            'macd': latest['MACD'],
            'macd_signal': latest['MACD_Signal'],
            'stoch_k': latest['Stoch_K']
        }

    def analyze_volume(self):
        """Analyze volume indicators"""
        latest = self.data.iloc[-1]
        previous = self.data.iloc[-2]
        
        obv_trend = "Neutral"
        if latest['OBV'] > previous['OBV']:
            obv_trend = "Bullish"
        elif latest['OBV'] < previous['OBV']:
            obv_trend = "Bearish"

        volume_strength = "Average"
        if latest['Volume'] > latest['Avg_Volume'] * 1.5:
            volume_strength = "High"
        elif latest['Volume'] < latest['Avg_Volume'] * 0.5:
            volume_strength = "Low"

        return {
            'obv_trend': obv_trend,
            'volume_strength': volume_strength,
            'volume': latest['Volume'],
            'avg_volume': latest['Avg_Volume']
        }
    
    def analyze_volatility_and_support_resistance(self):
        """Analyze volatility and support/resistance"""
        latest = self.data.iloc[-1]
        
        bb_position = (latest['Close'] - latest['BB_Low']) / (latest['BB_High'] - latest['BB_Low'])
        
        if bb_position > 0.8:
            bb_signal = "Near Upper Band (Overbought)"
        elif bb_position < 0.2:
            bb_signal = "Near Lower Band (Oversold)"
        else:
            bb_signal = "Middle Range"
        
        pivots = self.calculate_pivot_points()
        
        return {
            'atr': latest['ATR'],
            'bb_upper': latest['BB_High'],
            'bb_middle': latest['BB_Mid'],
            'bb_lower': latest['BB_Low'],
            'bb_signal': bb_signal,
            'pivot_points': pivots
        }
    
    def get_comprehensive_analysis(self):
        """Get complete technical analysis"""
        self.calculate_all_indicators()
        
        trend_analysis = self.analyze_trend()
        momentum_analysis = self.analyze_momentum()
        volatility_analysis = self.analyze_volatility_and_support_resistance()
        volume_analysis = self.analyze_volume()
        
        total_score = trend_analysis['trend_score'] + momentum_analysis['momentum_score']
        
        if total_score >= 3:
            overall_signal = "STRONG BUY"
        elif total_score >= 1:
            overall_signal = "BUY"
        elif total_score <= -3:
            overall_signal = "STRONG SELL"
        elif total_score <= -1:
            overall_signal = "SELL"
        else:
            overall_signal = "NEUTRAL"
        
        return {
            'overall_signal': overall_signal,
            'total_score': total_score,
            'trend': trend_analysis,
            'momentum': momentum_analysis,
            'volatility': volatility_analysis,
            'volume': volume_analysis,
            'timestamp': self.data.index[-1]
        }