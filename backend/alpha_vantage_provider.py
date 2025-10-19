import os
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

class AlphaVantageProvider:
    """Real-time forex data provider using Alpha Vantage API.

    Free tier: 25 requests/day, 5 requests/minute
    Supports: Real-time quotes, historical data, multiple timeframes
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.session = requests.Session()

    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make request to Alpha Vantage API and return JSON if available.

        Falls back to CSV parsing only when JSON is not available.
        """
        if not self.api_key:
            return None

        params['apikey'] = self.api_key

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()

            # If JSON is returned, use it
            try:
                data = response.json()
                # Handle API limit and errors
                if isinstance(data, dict) and (
                    'Note' in data or 'Information' in data or 'Error Message' in data
                ):
                    return None
                return data
            except ValueError:
                pass

            data = response.text
            if 'Thank you for using Alpha Vantage!' in data:
                return None  # API limit reached

            return self._parse_response(data)
        except requests.exceptions.RequestException as e:
            print(f"Alpha Vantage API request failed: {e}")
            return None

    def _parse_response(self, response_text: str) -> Dict:
        """Parse Alpha Vantage CSV-like response"""
        lines = response_text.strip().split('\n')
        if len(lines) < 2:
            return {}

        headers = lines[0].split(',')
        data = []

        for line in lines[1:]:
            if line.strip():
                values = line.split(',')
                data.append(dict(zip(headers, values)))

        return {
            'headers': headers,
            'data': data
        }

    def get_real_time_quote(self, from_symbol: str, to_symbol: str = 'USD') -> Optional[float]:
        """Get real-time forex quote (JSON endpoint)."""
        symbol_map = {
            'EUR': 'EUR', 'GBP': 'GBP', 'JPY': 'JPY', 'AUD': 'AUD',
            'CAD': 'CAD', 'CHF': 'CHF', 'NZD': 'NZD', 'XAU': 'XAU',
            'BTC': 'BTC', 'ETH': 'ETH'
        }

        from_curr = symbol_map.get(from_symbol)
        if not from_curr:
            print(f"Alpha Vantage: Unsupported currency symbol: {from_symbol}")
            return None

        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': from_curr,
            'to_currency': to_symbol
        }

        data = self._make_request(params)
        if data is None:
            print(f"Alpha Vantage: No data returned for {from_curr}/{to_symbol}")
            return None

        # JSON shape: {'Realtime Currency Exchange Rate': {'5. Exchange Rate': '1.2345', ...}}
        if isinstance(data, dict):
            info = data.get('Realtime Currency Exchange Rate') or data.get('Realtime Currency Exchange Rate '.strip())
            if isinstance(info, dict):
                rate = info.get('5. Exchange Rate') or info.get('5. Exchange Rate '.strip())
                try:
                    if rate:
                        rate_float = float(rate)
                        print(f"Alpha Vantage: Got real-time quote for {from_curr}/{to_symbol}: {rate_float}")
                        return rate_float
                    else:
                        print(f"Alpha Vantage: No exchange rate found in response for {from_curr}/{to_symbol}")
                        print(f"Response info keys: {list(info.keys()) if info else 'No info'}")
                except (ValueError, TypeError) as e:
                    print(f"Alpha Vantage: Error parsing exchange rate for {from_curr}/{to_symbol}: {rate} - {e}")
            else:
                print(f"Alpha Vantage: Invalid response format for {from_curr}/{to_symbol}")
                print(f"Expected dict but got: {type(info)}")
        else:
            print(f"Alpha Vantage: Expected dict response but got: {type(data)}")

        return None

    def get_historical_data(self, from_symbol: str, to_symbol: str = 'USD', timeframe: str = '1day') -> Optional[List[Dict]]:
        """Get historical forex data as a list of OHLC dicts (JSON parsing)."""
        tf = (timeframe or '1day').lower()
        intraday = tf in ('1min', '5min', '15min', '30min', '1hour', '4hour')

        # Alpha Vantage intraday supports: 1,5,15,30,60 minutes
        interval_map = {
            '1min': '1min', '5min': '5min', '15min': '15min', '30min': '30min',
            '1hour': '60min', '4hour': '60min'  # approximate 4h with 60min series
        }

        symbol_map = {
            'EUR': 'EUR', 'GBP': 'GBP', 'JPY': 'JPY', 'AUD': 'AUD',
            'CAD': 'CAD', 'CHF': 'CHF', 'NZD': 'NZD', 'XAU': 'XAU',
            'BTC': 'BTC', 'ETH': 'ETH'
        }
        from_curr = symbol_map.get(from_symbol)
        if not from_curr:
            return None

        if from_curr in {'BTC', 'ETH'}:
            ts = self._get_crypto_series(from_curr, to_symbol, tf, intraday, interval_map)
        elif intraday:
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': from_curr,
                'to_symbol': to_symbol,
                'interval': interval_map.get(tf, '60min'),
                'outputsize': 'compact'
            }
            data = self._make_request(params)
            if not isinstance(data, dict):
                return None
            series_key = next((k for k in data.keys() if k.startswith('Time Series FX')), None)
            ts = data.get(series_key, {}) if series_key else {}
        else:
            fn_map = {'1day': 'FX_DAILY', '1week': 'FX_WEEKLY', '1month': 'FX_MONTHLY'}
            function = fn_map.get(tf, 'FX_DAILY')
            params = {
                'function': function,
                'from_symbol': from_curr,
                'to_symbol': to_symbol,
                'outputsize': 'compact'
            }
            data = self._make_request(params)
            if not isinstance(data, dict):
                return None
            series_key = next((k for k in data.keys() if k.startswith('Time Series FX')), None)
            ts = data.get(series_key, {}) if series_key else {}

        if not isinstance(ts, dict) or not ts:
            return None

        candles: List[Dict] = []
        # ts is dict: { '2025-10-05 15:00:00': {'1. open':'...', '2. high':...} }
        for timestamp, ohlc in ts.items():
            try:
                candles.append({
                    'timestamp': timestamp,
                    'open': ohlc.get('1. open'),
                    'high': ohlc.get('2. high'),
                    'low':  ohlc.get('3. low'),
                    'close': ohlc.get('4. close'),
                    'volume': '0'
                })
            except Exception:
                continue

        # Sort by time ascending
        candles.sort(key=lambda x: x['timestamp'])
        return candles

    def _get_crypto_series(self, symbol: str, market: str, tf: str, intraday: bool, interval_map: Dict[str, str]) -> Dict:
        """Retrieve time series data for crypto pairs."""
        if intraday:
            crypto_interval_map = {
                '1min': '1min', '5min': '5min', '15min': '15min', '30min': '30min',
                '1hour': '60min', '4hour': '60min'
            }
            params = {
                'function': 'DIGITAL_CURRENCY_INTRADAY',
                'symbol': symbol,
                'market': market,
                'interval': crypto_interval_map.get(tf, '60min')
            }
        else:
            params = {
                'function': 'DIGITAL_CURRENCY_DAILY',
                'symbol': symbol,
                'market': market
            }

        data = self._make_request(params)
        if not isinstance(data, dict):
            return {}

        series_key = next((k for k in data.keys() if k.startswith('Time Series')), None)
        return data.get(series_key, {}) if series_key else {}

    def format_candles_for_dataframe(self, historical_data: List[Dict], timeframe: str = '1day') -> List[Dict]:
        """Convert Alpha Vantage data to OHLCV format"""
        formatted_candles = []

        for entry in historical_data:
            try:
                formatted_candles.append({
                    'time': entry.get('timestamp', ''),
                    'Open': float(entry.get('open', '0')),
                    'High': float(entry.get('high', '0')),
                    'Low': float(entry.get('low', '0')),
                    'Close': float(entry.get('close', '0')),
                    'Volume': int(float(entry.get('volume', '0')))
                })
            except (ValueError, KeyError):
                continue

        return formatted_candles

    def is_available(self) -> bool:
        """Check if Alpha Vantage API is available and configured"""
        return bool(self.api_key)

    def get_available_pairs(self) -> List[str]:
        """Get list of supported forex pairs"""
        return [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD',
            'EUR/GBP', 'GBP/JPY', 'USD/CHF', 'XAU/USD', 'NZD/USD'
        ]

    def get_account_info(self) -> Dict:
        """Get API usage info (Alpha Vantage doesn't provide account info like OANDA)"""
        return {
            'provider': 'Alpha Vantage',
            'free_tier': '25 requests/day, 5 requests/minute',
            'status': 'active' if self.api_key else 'inactive'
        }
