import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import random
from typing import Dict, Tuple
from alpha_vantage_provider import AlphaVantageProvider
from backend.api_handlers.market_data import TwelveDataProvider
import database as db

class ForexAnalyzer:
    """Fetches and manages forex market data"""

    # Alpha Vantage currency codes for real data
    AV_CURRENCIES = {
        'EUR/USD': ('EUR', 'USD'),
        'GBP/USD': ('GBP', 'USD'),
        'USD/JPY': ('USD', 'JPY'),
        'XAU/USD': ('XAU', 'USD'),
        'NAS100': ('NAS100', 'NAS100'),
        'SPX500': ('SPX500', 'SPX500'),
        'GER40': ('GER40', 'GER40'),
        'ETH/USD': ('ETH', 'USD'),
        'BTC/USD': ('BTC', 'USD'),
        # Forex pairs supported by automation defaults
        'AUD/USD': ('AUD', 'USD'),
        'NZD/USD': ('NZD', 'USD'),
        'USD/CAD': ('USD', 'CAD'),
        'USD/CHF': ('USD', 'CHF'),
        'EUR/GBP': ('EUR', 'GBP'),
        'EUR/JPY': ('EUR', 'JPY'),
        'GBP/JPY': ('GBP', 'JPY'),
        'AUD/JPY': ('AUD', 'JPY'),
    }

    def __init__(self):
        self.data = {}
        self.td_provider = TwelveDataProvider()
        self.av_provider = AlphaVantageProvider()
        self._price_cache: Dict[str, Tuple[float, datetime]] = {}
        print(f"TwelveData API available: {self.td_provider.is_available()}")
        print(f"Alpha Vantage API available: {self.av_provider.is_available()}")

    def _record_price(self, pair: str, price: float) -> None:
        value = float(price)
        self._price_cache[pair] = (value, datetime.utcnow())
        existing = self.data.get(pair)
        if existing is not None and not getattr(existing, 'empty', True) and 'Close' in existing.columns:
            try:
                existing.iloc[-1, existing.columns.get_loc('Close')] = value
            except Exception:
                pass

    def _fetch_yahoo_quote(self, symbol: str) -> float | None:
        try:
            response = requests.get(
                "https://query1.finance.yahoo.com/v7/finance/quote",
                params={"symbols": symbol},
                timeout=6,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            payload = response.json() or {}
            result = (payload.get("quoteResponse") or {}).get("result") or []
            if not result:
                return None
            quote = result[0]
            for key in ("regularMarketPrice", "regularMarketPreviousClose", "bid", "ask"):
                value = quote.get(key)
                if value not in (None, 0):
                    return float(value)
        except Exception:
            return None
        return None

    def fetch_data(self, pair, period='1mo', interval='1h'):
        """
        Fetch historical data for a forex pair

        Args:
            pair: Currency pair symbol (e.g., 'EUR/USD')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
        """
        # Try TwelveData first if available
        if self.td_provider.is_available():
            data_td = self._fetch_twelvedata_data(pair, period, interval)
            if data_td is not None:
                print(f"Using TwelveData for {pair}")
                self.data[pair] = data_td
                return data_td
            else:
                print(f"TwelveData unavailable for {pair}; attempting Alpha Vantage fallback")

        # Fallback to Alpha Vantage
        if self.av_provider.is_available():
            currencies = self.AV_CURRENCIES.get(pair)
            if currencies:
                real_data = self._fetch_alpha_vantage_data(currencies[0], currencies[1], period, interval)
                if real_data is not None:
                    print(f"Using real Alpha Vantage data for {pair}")
                    self.data[pair] = real_data
                    return real_data

        # Fall back to simulated data
        print(f"Using simulated data for {pair} (Primary and fallback APIs unavailable)")
        simulated = self._generate_simulated_data(pair, period, interval)
        # Cache it so other methods (e.g., get_current_price) can use it
        self.data[pair] = simulated
        return simulated

    def _fetch_twelvedata_data(self, pair, period, interval):
        """Fetch real data from TwelveData API"""
        try:
            # Map period to outputsize
            period_map = {'1d': 24, '5d': 120, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
            outputsize = period_map.get(period, 30)

            # Map interval to TwelveData timeframe
            timeframe_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1day'}
            td_interval = timeframe_map.get(interval, '1day')

            historical_data = self.td_provider.get_historical_data(pair, td_interval, outputsize)
            if historical_data:
                df = pd.DataFrame(historical_data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df = df.apply(pd.to_numeric)
                return df.sort_index()
            return None
        except Exception as e:
            print(f"Error fetching TwelveData for {pair}: {e}")
            return None

    def _fetch_alpha_vantage_data(self, from_currency, to_currency, period, interval):
        """Fetch real data from Alpha Vantage API"""
        try:
            # Map period to days
            period_days = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
            days = period_days.get(period, 30)

            # Map interval to Alpha Vantage timeframe
            timeframe_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1hour', '1d': '1day'}
            timeframe = timeframe_map.get(interval, '1day')

            # Fetch historical data from Alpha Vantage
            historical_data = self.av_provider.get_historical_data(from_currency, to_currency, timeframe)

            if historical_data:
                # Convert to DataFrame format
                candles = self.av_provider.format_candles_for_dataframe(historical_data, timeframe)
                if candles:
                    # Limit to requested period
                    df = pd.DataFrame(candles[-days:])  # Get last N days
                    df['Date'] = pd.to_datetime(df['time'])
                    df = df.set_index('Date')
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    return df

            return None

        except Exception as e:
            print(f"Error fetching Alpha Vantage data for {from_currency}/{to_currency}: {e}")
            return None

    # -------------------- 4H Aggregation & Local Indicators --------------------
    def _fetch_intraday_60min_df(self, from_currency: str, to_currency: str, lookback_days: int = 90) -> pd.DataFrame | None:
        """Fetch 60min candles from Alpha Vantage and return as DataFrame with DateTime index."""
        try:
            intraday = self.av_provider.get_historical_data(from_currency, to_currency, '60min')
            if not intraday:
                return None
            candles = self.av_provider.format_candles_for_dataframe(intraday, '60min')
            if not candles:
                return None
            df = pd.DataFrame(candles)
            df['Date'] = pd.to_datetime(df['time'])
            df = df.set_index('Date')
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            # Keep recent window
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            df = df[df.index >= cutoff]
            return df.sort_index()
        except Exception as e:
            print(f"Error fetching 60min data: {e}")
            return None

    @staticmethod
    def _aggregate_to_4h(df_60: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 60min OHLCV to 4H bars."""
        rule = '4H'
        agg = df_60.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        }).dropna()
        return agg

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _bbands(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
        ma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = ma + std_mult * std
        lower = ma - std_mult * std
        return pd.DataFrame({'middle': ma, 'upper': upper, 'lower': lower})

    def compute_4h_basic_indicators(self, pair: str) -> dict:
        """Compute RSI(14), MACD(12/26/9), EMA(20/200), BBANDS(20,2) on aggregated 4H candles.
        Returns a dict shaped like IndicatorCollector.collect_all() output.
        Also caches aggregated 4H candles and indicators to DB.
        """
        try:
            currencies = self.AV_CURRENCIES.get(pair)
            if not currencies:
                return {}
            from_symbol, to_symbol = currencies
            df60 = self._fetch_intraday_60min_df(from_symbol, to_symbol, lookback_days=120)
            if df60 is None or df60.empty:
                return {}
            df4 = self._aggregate_to_4h(df60)
            if df4.empty:
                return {}
            # Cache aggregated candles
            try:
                payload = {
                    'meta': {'pair': pair, 'source_interval': '60min', 'aggregated': '4H'},
                    'rows': df4.tail(300).reset_index().assign(time=lambda d: d['Date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ'))[
                        ['time','Open','High','Low','Close','Volume']
                    ].to_dict(orient='records')
                }
                db.cache_candles(pair, '4H', payload)
            except Exception:
                pass

            close = df4['Close']
            ts_str = datetime.utcnow().isoformat()

            # RSI(14)
            rsi_series = self._rsi(close, 14)
            rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else None
            # EMA(20/200)
            ema20 = self._ema(close, 20).iloc[-1]
            ema200 = self._ema(close, 200).iloc[-1]
            # MACD(12/26/9)
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd_line = ema12 - ema26
            signal = macd_line.ewm(span=9, adjust=False).mean()
            hist = macd_line - signal
            macd = float(macd_line.iloc[-1]) if not np.isnan(macd_line.iloc[-1]) else None
            macd_sig = float(signal.iloc[-1]) if not np.isnan(signal.iloc[-1]) else None
            macd_hist = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else None
            # BBANDS(20,2)
            bb = self._bbands(close, 20, 2.0)
            bb_last = bb.iloc[-1]

            result = {
                'RSI': { 'timestamp': ts_str, 'value': rsi_val },
                'EMA': { 'timestamp': ts_str, 'ema20': float(ema20) if not np.isnan(ema20) else None,
                         'ema200': float(ema200) if not np.isnan(ema200) else None },
                'MACD': { 'timestamp': ts_str, 'macd': macd, 'signal': macd_sig, 'hist': macd_hist },
                'BBANDS': { 'timestamp': ts_str, 'upper': float(bb_last['upper']) if not np.isnan(bb_last['upper']) else None,
                            'middle': float(bb_last['middle']) if not np.isnan(bb_last['middle']) else None,
                            'lower': float(bb_last['lower']) if not np.isnan(bb_last['lower']) else None },
            }

            # Cache indicators
            try:
                db.cache_indicator(pair, '4H', 'RSI', result['RSI'])
                db.cache_indicator(pair, '4H', 'EMA', result['EMA'])
                db.cache_indicator(pair, '4H', 'MACD', result['MACD'])
                db.cache_indicator(pair, '4H', 'BBANDS', result['BBANDS'])
            except Exception:
                pass

            return result
        except Exception as e:
            print(f"Error computing 4H indicators for {pair}: {e}")
            return {}

    def _generate_simulated_data(self, pair, period='1mo', interval='1d'):
        """Generate simulated forex data for demonstration"""
        # Determine number of data points based on period and interval
        period_map = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
        interval_map = {'1m': 1440, '5m': 288, '15m': 96, '1h': 24, '1d': 1}

        days = period_map.get(period, 30)
        points_per_day = interval_map.get(interval, 24)
        total_points = days * points_per_day

        # Base prices for different pairs (updated to current market levels)
        base_prices = {
            'EUR/USD': 1.0850,  # Euro around 1.08-1.09
            'GBP/USD': 1.3150,  # Pound around 1.31-1.32
            'USD/JPY': 149.50,  # Yen around 149-150
            'AUD/USD': 0.6750,  # Aussie around 0.67-0.68
            'USD/CAD': 1.3550,  # CAD around 1.35-1.36
            'EUR/GBP': 0.8250,  # EUR/GBP around 0.82-0.83
            'GBP/JPY': 196.50,  # GBP/JPY around 196-197
            'USD/CHF': 0.8550,  # CHF around 0.85-0.86
            'XAU/USD': 3886.455,  # Gold at user's specified price
            'NZD/USD': 0.6150,  # Kiwi around 0.61-0.62
            'BTC/USD': 65000.0,  # Approximate Bitcoin price placeholder
            'ETH/USD': 3200.0    # Approximate Ethereum price placeholder
        }

        base_price = base_prices.get(pair, 1.0)

        # Generate dates
        end_date = datetime.now()
        if interval == '1d':
            dates = pd.date_range(end=end_date, periods=total_points, freq='D')
        else:
            dates = pd.date_range(end=end_date, periods=total_points, freq='H')

        # Generate realistic price movement
        prices = [base_price]
        for i in range(1, total_points):
            change = random.gauss(0, base_price * 0.002)  # 0.2% std deviation
            new_price = prices[-1] + change
            prices.append(max(new_price, base_price * 0.95))  # Floor at 95% of base

        # Create OHLCV data
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + random.uniform(0, 0.005)) for p in prices],
            'Low': [p * (1 - random.uniform(0, 0.005)) for p in prices],
            'Close': [p * (1 + random.gauss(0, 0.002)) for p in prices],
            'Volume': [random.randint(100000, 1000000) for _ in prices]
        })

        print(f"Generated {len(data)} simulated data points for {pair}")
        return data

    def get_current_price(self, pair):
        """Get the most recent price for a pair"""
        cached = self._price_cache.get(pair)
        if cached:
            price, timestamp = cached
            if datetime.utcnow() - timestamp < timedelta(seconds=45):
                return price

        # Try TwelveData first if available
        if self.td_provider.is_available():
            price_td = self.td_provider.get_real_time_quote(pair)
            if price_td is not None:
                print(f"Using TwelveData real-time price for {pair}: {price_td}")
                self._record_price(pair, price_td)
                return price_td
            else:
                print(f"TwelveData real-time price unavailable for {pair}; trying Alpha Vantage")

        # Fallback to Alpha Vantage
        if self.av_provider.is_available():
            currencies = self.AV_CURRENCIES.get(pair)
            if currencies:
                print(f"Fetching real-time price for {pair} from Alpha Vantage...")
                current_price = self.av_provider.get_real_time_quote(currencies[0], currencies[1])
                if current_price:
                    print(f"Successfully got real-time price for {pair}: {current_price}")
                    value = float(current_price)
                    self._record_price(pair, value)
                    return value
                else:
                    print(f"Alpha Vantage real-time quote failed for {pair}, trying historical fallback...")

                # Fallback: try latest historical close from Alpha Vantage
                print(f"Trying historical data fallback for {pair}...")
                hist = self.av_provider.get_historical_data(currencies[0], currencies[1], '1day')
                if hist and len(hist) > 0:
                    try:
                        fallback_price = float(hist[-1]['close'])
                        print(f"Got historical fallback price for {pair}: {fallback_price}")
                        self._record_price(pair, fallback_price)
                        return fallback_price
                    except Exception as e:
                        print(f"Error parsing historical fallback price for {pair}: {e}")
                else:
                    print(f"No historical data available for fallback for {pair}")

        # YFinance fallback for crypto/metals pairs to avoid stale simulated values
        yf_symbol_map = {
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'JPY=X',
            'XAU/USD': 'GC=F',
            'WTI/USD': 'CL=F',
            'NAS100': '^IXIC',
            'SPX500': '^GSPC',
            'GER40': '^GDAXI',
            'AAPL': 'AAPL',
            'MSFT': 'MSFT',
            'ETH/USD': 'ETH-USD',
            # Keep old ones for compatibility
            'AUD/USD': 'AUDUSD=X',
            'USD/CAD': 'CAD=X',
            'EUR/GBP': 'EURGBP=X',
            'GBP/JPY': 'GBPJPY=X',
            'USD/CHF': 'CHF=X',
            'NZD/USD': 'NZDUSD=X',
            'BTC/USD': 'BTC-USD',
        }
        yf_symbol = yf_symbol_map.get(pair)
        if yf_symbol:
            try:
                yahoo_quote = self._fetch_yahoo_quote(yf_symbol)
                if yahoo_quote is not None:
                    print(f"Yahoo Finance quote API price for {pair}: {yahoo_quote}")
                    self._record_price(pair, yahoo_quote)
                    return yahoo_quote
                ticker = yf.Ticker(yf_symbol)
                price = None
                # fast_info is quick but may not exist on old yfinance
                fast_info = getattr(ticker, 'fast_info', None)
                if fast_info:
                    price = fast_info.get('last_price') or fast_info.get('last_trade_price')
                if price is None:
                    hist = ticker.history(period='1d', interval='1m')
                    if not hist.empty:
                        price = float(hist['Close'].iloc[-1])
                if price is not None:
                    print(f"YFinance fallback price for {pair}: {price}")
                    value = float(price)
                    self._record_price(pair, value)
                    return value
            except Exception as exc:
                print(f"YFinance fallback failed for {pair}: {exc}")

        print(f"Falling back to cached/simulated data for {pair}")
        # Fall back to cached data or fetch new data
        if pair not in self.data or getattr(self.data[pair], 'empty', True):
            print(f"No cached data for {pair}, fetching new data...")
            self.fetch_data(pair, period='1d', interval='1m')

        if pair in self.data and not getattr(self.data[pair], 'empty', True):
            price = self.data[pair]['Close'].iloc[-1]
            print(f"Using cached data price for {pair}: {price}")
            value = float(price)
            self._record_price(pair, value)
            return value

        print(f"No price data available for {pair}")
        return None

    def get_all_pairs_data(self, period='1mo', interval='1h'):
        """Fetch data for all supported pairs"""
        results = {}
        for pair in self.AV_CURRENCIES.keys():
            print(f"Fetching {pair}...")
            data = self.fetch_data(pair, period, interval)
            if data is not None:
                results[pair] = data
        return results
