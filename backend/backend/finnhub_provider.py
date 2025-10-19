import logging
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None


class FinnhubNewsProvider:
    """Primary news provider using Finnhub.io API."""

    BASE_URL = "https://finnhub.io/api/v1"

    SYMBOL_CONFIG: Dict[str, Dict] = {
        # Major FX pairs (map to OANDA symbols)
        "EUR/USD": {"endpoint": "forex/news", "params": {"symbol": "OANDA:EUR_USD"}},
        "GBP/USD": {"endpoint": "forex/news", "params": {"symbol": "OANDA:GBP_USD"}},
        "USD/JPY": {"endpoint": "forex/news", "params": {"symbol": "OANDA:USD_JPY"}},
        "AUD/USD": {"endpoint": "forex/news", "params": {"symbol": "OANDA:AUD_USD"}},
        "NZD/USD": {"endpoint": "forex/news", "params": {"symbol": "OANDA:NZD_USD"}},
        "USD/CAD": {"endpoint": "forex/news", "params": {"symbol": "OANDA:USD_CAD"}},
        "USD/CHF": {"endpoint": "forex/news", "params": {"symbol": "OANDA:USD_CHF"}},
        "EUR/GBP": {"endpoint": "forex/news", "params": {"symbol": "OANDA:EUR_GBP"}},
        "EUR/JPY": {"endpoint": "forex/news", "params": {"symbol": "OANDA:EUR_JPY"}},
        "GBP/JPY": {"endpoint": "forex/news", "params": {"symbol": "OANDA:GBP_JPY"}},
        "AUD/JPY": {"endpoint": "forex/news", "params": {"symbol": "OANDA:AUD_JPY"}},
        "XAU/USD": {"endpoint": "forex/news", "params": {"symbol": "OANDA:XAU_USD"}},
        "NAS100": {
            "endpoint": "news",
            "params": {"category": "general"},
            "keywords": ["Nasdaq", "NDX", "Nasdaq 100"],
        },
        "SPX500": {
            "endpoint": "news",
            "params": {"category": "general"},
            "keywords": ["S&P 500", "SPX", "SP500", "Standard & Poor"],
        },
        "GER40": {
            "endpoint": "news",
            "params": {"category": "general"},
            "keywords": ["DAX", "Germany 40", "GER40"],
        },
        # Crypto
        "BTC/USD": {"endpoint": "crypto/news", "params": {"symbol": "BINANCE:BTCUSDT"}},
        "ETH/USD": {"endpoint": "crypto/news", "params": {"symbol": "BINANCE:ETHUSDT"}},
    }

    CURRENCY_KEYWORDS: Dict[str, List[str]] = {
        "EUR": ["EUR", "Euro", "Eurozone", "ECB", "European Central Bank", "Europe economy", "Euro area"],
        "USD": ["USD", "US Dollar", "Federal Reserve", "Fed", "United States economy", "US inflation", "US jobs report", "US GDP"],
        "GBP": ["GBP", "British Pound", "UK economy", "Bank of England", "BoE", "UK inflation", "UK interest rates"],
        "JPY": ["JPY", "Japanese Yen", "Bank of Japan", "BoJ", "Japan economy", "Japan inflation"],
        "AUD": ["AUD", "Australian Dollar", "RBA", "Reserve Bank of Australia", "Australia economy"],
        "NZD": ["NZD", "New Zealand Dollar", "RBNZ", "Reserve Bank of New Zealand", "New Zealand economy"],
        "CAD": ["CAD", "Canadian Dollar", "Bank of Canada", "BoC", "Canada economy", "Canada inflation"],
        "CHF": ["CHF", "Swiss Franc", "Swiss National Bank", "SNB", "Switzerland economy"],
        "CNY": ["CNY", "Chinese Yuan", "PBoC", "People's Bank of China", "China economy", "China inflation"],
        "SEK": ["SEK", "Swedish Krona", "Riksbank", "Sweden economy"],
        "NOK": ["NOK", "Norwegian Krone", "Norges Bank", "Norway economy"],
        "XAU": ["gold", "gold price", "precious metals", "safe haven"],
        "WTI": ["oil", "crude", "WTI", "energy markets"],
        "BTC": ["Bitcoin", "BTC", "cryptocurrency"],
        "ETH": ["Ethereum", "ETH", "smart contracts"],
    }

    SENTIMENT_ANALYZER = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None

    def __init__(self) -> None:
        self.api_key = os.getenv("FINNHUB_API_KEY")
        self.session = requests.Session()

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _build_keywords(self, pair: str) -> Set[str]:
        keywords: Set[str] = set()
        normalized = pair.upper().replace("-", "/")
        if "/" in normalized:
            base, quote = normalized.split("/", 1)
            for token in (base, quote):
                keywords.update(self.CURRENCY_KEYWORDS.get(token, []))
        else:
            keywords.update(self.CURRENCY_KEYWORDS.get(normalized, []))
        # Include raw pair references for broader matching
        keywords.update({pair.upper(), pair.replace("/", " ")})
        return keywords

    def _build_request(self, pair: str, lookback_hours: int) -> Optional[Dict]:
        config = self.SYMBOL_CONFIG.get(pair, {"endpoint": "news", "params": {"category": "general"}})
        endpoint = config.get("endpoint", "news")
        params = dict(config.get("params", {}))

        if not self.api_key:
            return None

        params["token"] = self.api_key

        # Finnhub company news needs date window (YYYY-MM-DD)
        if endpoint == "company-news":
            to_dt = datetime.utcnow()
            from_dt = to_dt - timedelta(hours=lookback_hours)
            params["from"] = from_dt.date().isoformat()
            params["to"] = to_dt.date().isoformat()

        keywords = set(config.get("keywords", []))
        keywords.update(self._build_keywords(pair))

        return {
            "endpoint": endpoint,
            "params": params,
            "keywords": sorted(keywords),
            "pair": pair,
        }

    def _request(self, endpoint: str, params: Dict) -> Optional[List[Dict]]:
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
            # company-news returns list; others may wrap in dict
            if isinstance(payload, dict) and "data" in payload:
                return payload.get("data") or []
            if isinstance(payload, list):
                return payload
            return []
        except requests.RequestException as exc:
            print(f"[Finnhub] Request failed for {endpoint}: {exc}")
            return None
        except ValueError as exc:
            print(f"[Finnhub] Invalid JSON for {endpoint}: {exc}")
            return None

    @staticmethod
    def _matches_keywords(item: Dict, keywords: Optional[List[str]]) -> bool:
        if not keywords:
            return True
        haystack = " ".join(
            str(item.get(field, "")) for field in ("headline", "summary", "category")
        ).lower()
        return any(keyword.lower() in haystack for keyword in keywords)

    @staticmethod
    def _adapt_item(item: Dict) -> Optional[Dict]:
        headline = item.get("headline") or item.get("title")
        if not headline:
            return None
        summary = item.get("summary") or item.get("text")
        source = item.get("source") or item.get("category")
        url = item.get("url") or item.get("link")
        timestamp = item.get("datetime") or item.get("time")
        published_at = None
        if isinstance(timestamp, (int, float)):
            published_at = datetime.utcfromtimestamp(timestamp).isoformat()
        elif isinstance(timestamp, str) and timestamp:
            published_at = timestamp

        return {
            "title": headline,
            "description": summary,
            "source": source,
            "url": url,
            "published_at": published_at,
        }

    def _filter_items(
        self,
        raw_items: List[Dict],
        keywords: List[str],
        lookback_hours: int,
        limit: int,
    ) -> List[Dict]:
        if not raw_items:
            return []

        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        filtered: List[Dict] = []
        seen_titles: Set[str] = set()

        for item in raw_items:
            if not self._matches_keywords(item, keywords):
                continue
            adapted = self._adapt_item(item)
            if not adapted:
                continue

            published_at = adapted.get("published_at")
            if published_at:
                try:
                    published_dt = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
                except ValueError:
                    published_dt = None
                if published_dt and published_dt < cutoff:
                    continue

            title_key = adapted["title"].strip().lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            if self.SENTIMENT_ANALYZER:
                text_blob = " ".join(filter(None, [adapted.get("title"), adapted.get("description")]))
                if text_blob:
                    sentiment = self.SENTIMENT_ANALYZER.polarity_scores(text_blob)
                    adapted["sentiment"] = sentiment.get("compound")

            filtered.append(adapted)
            if len(filtered) >= limit:
                break

        return filtered

    def get_news_for_pair(
        self, pair: str, limit: int = 8, lookback_hours: int = 48
    ) -> List[Dict]:
        if not self.is_available():
            return []

        request_config = self._build_request(pair, lookback_hours)
        if not request_config:
            return []

        endpoint = request_config["endpoint"]
        params = request_config["params"]
        keywords = request_config.get("keywords")

        raw_items = self._request(endpoint, params)
        if raw_items is None:
            logging.warning(f"[Finnhub] No response for endpoint={endpoint} pair={pair}")
            return []

        filtered = self._filter_items(raw_items, keywords or [], lookback_hours, limit)

        # Fallback: broaden to general news when forex endpoint returns nothing
        if not filtered and endpoint != "news":
            general_params = {"category": "general", "token": self.api_key}
            logging.info(f"[Finnhub] No direct news for {pair}, falling back to general category filter")
            general_items = self._request("news", general_params) or []
            filtered = self._filter_items(general_items, keywords or [], lookback_hours, limit)

        if not filtered:
            logging.info(f"[Finnhub] No relevant news found for {pair} within {lookback_hours}h window")

        return filtered
