import os
import requests
from typing import Dict, List, Optional

class TwelveDataProvider:
    BASE_URL = "https://api.twelvedata.com"

    # Map internal symbols to TwelveData symbols
    TD_SYMBOLS: Dict[str, str] = {
        "EUR/USD": "EUR/USD",
        "GBP/USD": "GBP/USD",
        "USD/JPY": "USD/JPY",
        "XAU/USD": "XAU/USD",
        "NAS100": "NDX",
        "SPX500": "SPX",
        "GER40": "DAX",
        "ETH/USD": "ETH/USD",
        "BTC/USD": "BTC/USD",
        "AUD/USD": "AUD/USD",
        "NZD/USD": "NZD/USD",
        "USD/CAD": "USD/CAD",
        "USD/CHF": "USD/CHF",
        "EUR/GBP": "EUR/GBP",
        "EUR/JPY": "EUR/JPY",
        "GBP/JPY": "GBP/JPY",
        "AUD/JPY": "AUD/JPY",
    }

    def __init__(self):
        self.api_key = os.getenv("TWELVEDATA_API_KEY")
        self.session = requests.Session()

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _map_symbol(self, symbol: str) -> Optional[str]:
        mapped = self.TD_SYMBOLS.get(symbol)
        if not mapped:
            print(f"[TwelveData] Unsupported symbol mapping for '{symbol}'")
        return mapped or symbol

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        if not self.is_available():
            print("[TwelveData] API key not configured; skipping request")
            return None
        
        params['apikey'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get('status') == 'error':
                print(f"[TwelveData] API error ({endpoint}): {data.get('message')}")
                return None
            return data
        except requests.exceptions.RequestException as e:
            print(f"[TwelveData] Request failed for {endpoint}: {e}")
            return None

    def get_real_time_quote(self, symbol: str) -> Optional[float]:
        mapped_symbol = self._map_symbol(symbol)
        if not mapped_symbol:
            return None

        params = {'symbol': mapped_symbol}
        data = self._make_request('price', params)
        if data and 'price' in data:
            try:
                return float(data['price'])
            except (ValueError, TypeError):
                print(f"[TwelveData] Invalid price format for {mapped_symbol}: {data.get('price')}")
                return None
        print(f"[TwelveData] No price returned for {mapped_symbol}")
        return None

    def get_historical_data(self, symbol: str, interval: str, outputsize: int = 100) -> Optional[List[Dict]]:
        mapped_symbol = self._map_symbol(symbol)
        if not mapped_symbol:
            return None

        params = {
            'symbol': mapped_symbol,
            'interval': interval,
            'outputsize': outputsize
        }
        data = self._make_request('time_series', params)
        if data and isinstance(data.get('values'), list):
            if not data['values']:
                print(f"[TwelveData] Empty time series returned for {mapped_symbol} interval={interval}")
                return None
            return data['values']
        print(f"[TwelveData] No historical data returned for {mapped_symbol}")
        return None
