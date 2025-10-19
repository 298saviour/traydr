"""
Finnhub News & Economic Data Provider
Fetches fundamental news and economic calendar data for forex analysis
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class EconomicEvent:
    """Container for economic calendar event"""
    datetime_utc: str
    currency: str
    event_name: str
    impact: str  # HIGH, MEDIUM, LOW
    previous: Optional[str] = None
    forecast: Optional[str] = None
    actual: Optional[str] = None
    surprise_impact: str = "None"  # Large, Moderate, Small, None
    
    def format_output(self) -> str:
        """Format event for display"""
        parts = [
            f"{self.datetime_utc}",
            f"{self.currency}",
            f"{self.event_name}",
            f"Impact: {self.impact}"
        ]
        
        if self.previous:
            parts.append(f"Previous: {self.previous}")
        if self.forecast:
            parts.append(f"Forecast: {self.forecast}")
        if self.actual:
            parts.append(f"Actual: {self.actual}")
        
        if self.surprise_impact != "None":
            parts.append(f"[SURPRISE: {self.surprise_impact}]")
        
        return " | ".join(parts)


@dataclass
class NewsHeadline:
    """Container for market news headline"""
    datetime_utc: str
    category: str
    headline: str
    summary: str
    source: str
    related_currencies: List[str] = field(default_factory=list)
    sentiment: str = "Neutral"  # Bullish, Bearish, Neutral
    
    def format_output(self) -> str:
        """Format headline for display"""
        currencies = ", ".join(self.related_currencies) if self.related_currencies else "General"
        return f"[{self.datetime_utc}] {self.category} | {currencies} | {self.headline}"


@dataclass
class FundamentalDataset:
    """Complete fundamental data for a currency pair"""
    pair: str
    timestamp: str
    
    # Economic Events
    upcoming_events: List[EconomicEvent] = field(default_factory=list)
    recent_events: List[EconomicEvent] = field(default_factory=list)
    
    # News Headlines
    recent_news: List[NewsHeadline] = field(default_factory=list)
    
    # Sentiment Summary
    overall_risk: str = "Neutral"  # Risk-on, Risk-off, Neutral
    usd_bias: str = "Neutral"  # Bullish, Bearish, Neutral
    key_theme: str = "No major theme"
    
    # Metadata
    data_source: str = "Finnhub"
    fetch_success: bool = True
    errors: List[str] = field(default_factory=list)


class FinnhubNewsProvider:
    """
    Fetches and organizes fundamental news and economic data from Finnhub
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    # Currency mapping for Finnhub
    CURRENCY_CODES = {
        "EUR": "EUR",
        "USD": "USD",
        "GBP": "GBP",
        "JPY": "JPY",
        "CHF": "CHF",
        "CAD": "CAD",
        "AUD": "AUD",
        "NZD": "NZD"
    }
    
    # Impact level keywords
    HIGH_IMPACT_KEYWORDS = [
        "interest rate", "rate decision", "nfp", "non-farm payroll",
        "cpi", "inflation", "gdp", "fomc", "ecb", "boe", "boj",
        "employment", "unemployment", "central bank"
    ]
    
    MEDIUM_IMPACT_KEYWORDS = [
        "pmi", "ism", "retail sales", "trade balance", "jobless claims",
        "ppi", "housing", "consumer confidence", "manufacturing"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub provider
        
        Args:
            api_key: Finnhub API key (defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            logger.warning("Finnhub API key not found. News fetching will be limited.")
        
        self.session = requests.Session()
        self.cache: Dict[str, FundamentalDataset] = {}
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
    
    def fetch_fundamental_data(self, pair: str, progress_callback=None) -> FundamentalDataset:
        """
        Fetch complete fundamental dataset for a currency pair
        
        Args:
            pair: Currency pair (e.g., "EUR/USD")
            progress_callback: Optional callback for progress updates
            
        Returns:
            FundamentalDataset with all news and economic data
        """
        logger.info(f"Fetching fundamental data for {pair} from Finnhub")
        if progress_callback:
            progress_callback(f"Fetching fundamental news and economic data for {pair}")
        
        dataset = FundamentalDataset(
            pair=pair,
            timestamp=datetime.utcnow().isoformat()
        )
        
        try:
            # Extract currencies from pair
            currencies = self._extract_currencies(pair)
            
            # Fetch economic calendar events
            if progress_callback:
                progress_callback(f"Fetching economic calendar for {', '.join(currencies)}")
            
            upcoming, recent = self._fetch_economic_calendar(currencies)
            dataset.upcoming_events = upcoming
            dataset.recent_events = recent
            
            # Wait to respect rate limits
            time.sleep(2)
            
            # Fetch market news
            if progress_callback:
                progress_callback(f"Fetching market news for {pair}")
            
            news = self._fetch_market_news(currencies)
            dataset.recent_news = news
            
            # Compute sentiment summary
            if progress_callback:
                progress_callback(f"Analyzing sentiment for {pair}")
            
            self._compute_sentiment_summary(dataset, currencies)
            
            # Cache the result
            self.cache[pair] = dataset
            
            logger.info(f"Successfully fetched fundamental data for {pair}")
            if progress_callback:
                progress_callback(f"✓ Fundamental data collection complete for {pair}")
            
        except Exception as e:
            logger.exception(f"Error fetching fundamental data for {pair}: {e}")
            dataset.fetch_success = False
            dataset.errors.append(str(e))
        
        return dataset
    
    def _extract_currencies(self, pair: str) -> List[str]:
        """Extract currency codes from pair"""
        # Remove slash and split
        clean_pair = pair.replace("/", "").replace("-", "")
        
        # Handle special cases
        if "XAU" in clean_pair:
            return ["XAU", "USD"]  # Gold
        if "BTC" in clean_pair or "ETH" in clean_pair:
            return ["USD"]  # Crypto pairs
        
        # Standard forex pairs (6 characters)
        if len(clean_pair) == 6:
            base = clean_pair[:3]
            quote = clean_pair[3:]
            return [base, quote]
        
        return ["USD"]  # Fallback
    
    def _fetch_economic_calendar(self, currencies: List[str]) -> Tuple[List[EconomicEvent], List[EconomicEvent]]:
        """
        Fetch economic calendar events
        
        Returns:
            Tuple of (upcoming_events, recent_events)
        """
        upcoming = []
        recent = []
        
        try:
            # Calculate date ranges
            now = datetime.utcnow()
            start_recent = (now - timedelta(hours=24)).strftime("%Y-%m-%d")
            end_upcoming = (now + timedelta(hours=48)).strftime("%Y-%m-%d")
            
            # Fetch calendar data
            url = f"{self.BASE_URL}/calendar/economic"
            params = {
                "token": self.api_key,
                "from": start_recent,
                "to": end_upcoming
            }
            
            self._rate_limit_wait()
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get("economicCalendar", [])
                
                for event_data in events:
                    event = self._parse_economic_event(event_data, currencies)
                    if event:
                        event_time = datetime.fromisoformat(event.datetime_utc.replace("Z", "+00:00"))
                        
                        if event_time > now:
                            upcoming.append(event)
                        else:
                            recent.append(event)
                
                # Sort by time
                upcoming.sort(key=lambda e: e.datetime_utc)
                recent.sort(key=lambda e: e.datetime_utc, reverse=True)
                
                logger.info(f"Fetched {len(upcoming)} upcoming and {len(recent)} recent events")
            else:
                logger.warning(f"Economic calendar request failed: {response.status_code}")
        
        except Exception as e:
            logger.exception(f"Error fetching economic calendar: {e}")
        
        return upcoming, recent
    
    def _parse_economic_event(self, data: Dict, target_currencies: List[str]) -> Optional[EconomicEvent]:
        """Parse economic event from Finnhub data"""
        try:
            # Extract currency
            country = data.get("country", "")
            currency = self._country_to_currency(country)
            
            # Filter by target currencies
            if currency not in target_currencies:
                return None
            
            # Extract event details
            event_name = data.get("event", "Unknown Event")
            
            # Determine impact level
            impact = self._determine_impact(event_name)
            
            # Parse values
            previous = data.get("previous")
            forecast = data.get("estimate")
            actual = data.get("actual")
            
            # Format datetime
            datetime_str = data.get("time", "")
            if datetime_str:
                dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
                datetime_utc = dt.strftime("%Y-%m-%d %H:%M UTC")
            else:
                datetime_utc = "Unknown"
            
            event = EconomicEvent(
                datetime_utc=datetime_utc,
                currency=currency,
                event_name=event_name,
                impact=impact,
                previous=str(previous) if previous is not None else None,
                forecast=str(forecast) if forecast is not None else None,
                actual=str(actual) if actual is not None else None
            )
            
            # Calculate surprise impact
            if actual is not None and forecast is not None:
                try:
                    actual_val = float(actual)
                    forecast_val = float(forecast)
                    diff_pct = abs((actual_val - forecast_val) / forecast_val * 100) if forecast_val != 0 else 0
                    
                    if diff_pct > 50:
                        event.surprise_impact = "Large"
                    elif diff_pct > 20:
                        event.surprise_impact = "Moderate"
                    elif diff_pct > 5:
                        event.surprise_impact = "Small"
                except:
                    pass
            
            return event
        
        except Exception as e:
            logger.debug(f"Error parsing economic event: {e}")
            return None
    
    def _fetch_market_news(self, currencies: List[str]) -> List[NewsHeadline]:
        """Fetch recent market news headlines"""
        headlines = []
        
        try:
            # Fetch general forex news
            url = f"{self.BASE_URL}/news"
            params = {
                "token": self.api_key,
                "category": "forex",
                "minId": 0
            }
            
            self._rate_limit_wait()
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Filter news from last 24 hours
                cutoff = datetime.utcnow() - timedelta(hours=24)
                
                for article in data[:50]:  # Limit to 50 most recent
                    headline = self._parse_news_headline(article, currencies)
                    if headline:
                        article_time = datetime.fromtimestamp(article.get("datetime", 0))
                        if article_time > cutoff:
                            headlines.append(headline)
                
                logger.info(f"Fetched {len(headlines)} relevant news headlines")
            else:
                logger.warning(f"News request failed: {response.status_code}")
        
        except Exception as e:
            logger.exception(f"Error fetching market news: {e}")
        
        return headlines
    
    def _parse_news_headline(self, data: Dict, target_currencies: List[str]) -> Optional[NewsHeadline]:
        """Parse news headline from Finnhub data"""
        try:
            headline_text = data.get("headline", "")
            summary = data.get("summary", "")
            
            # Check if relevant to target currencies
            related = []
            for currency in target_currencies:
                if currency in headline_text.upper() or currency in summary.upper():
                    related.append(currency)
            
            # Also check for central bank mentions
            if not related:
                for keyword in ["ECB", "FED", "BOE", "BOJ", "FOMC"]:
                    if keyword in headline_text.upper():
                        related.append(self._central_bank_to_currency(keyword))
            
            if not related and target_currencies:
                # Skip if not relevant
                return None
            
            # Format datetime
            timestamp = data.get("datetime", 0)
            dt = datetime.fromtimestamp(timestamp)
            datetime_utc = dt.strftime("%Y-%m-%d %H:%M UTC")
            
            # Determine category
            category = data.get("category", "General")
            
            # Determine sentiment (basic keyword analysis)
            sentiment = self._analyze_sentiment(headline_text + " " + summary)
            
            return NewsHeadline(
                datetime_utc=datetime_utc,
                category=category,
                headline=headline_text,
                summary=summary,
                source=data.get("source", "Unknown"),
                related_currencies=related,
                sentiment=sentiment
            )
        
        except Exception as e:
            logger.debug(f"Error parsing news headline: {e}")
            return None
    
    def _compute_sentiment_summary(self, dataset: FundamentalDataset, currencies: List[str]):
        """Compute overall sentiment summary"""
        try:
            # Analyze recent events for surprises
            high_impact_surprises = [e for e in dataset.recent_events 
                                    if e.impact == "HIGH" and e.surprise_impact in ["Large", "Moderate"]]
            
            # Analyze news sentiment
            bullish_news = [n for n in dataset.recent_news if n.sentiment == "Bullish"]
            bearish_news = [n for n in dataset.recent_news if n.sentiment == "Bearish"]
            
            # Determine overall risk
            if len(bullish_news) > len(bearish_news) * 1.5:
                dataset.overall_risk = "Risk-on"
            elif len(bearish_news) > len(bullish_news) * 1.5:
                dataset.overall_risk = "Risk-off"
            else:
                dataset.overall_risk = "Neutral"
            
            # Determine USD bias
            if "USD" in currencies:
                usd_events = [e for e in dataset.recent_events if e.currency == "USD"]
                positive_usd = sum(1 for e in usd_events if e.surprise_impact in ["Large", "Moderate"] 
                                  and e.actual and e.forecast and float(e.actual) > float(e.forecast))
                
                if positive_usd > len(usd_events) / 2:
                    dataset.usd_bias = "Bullish"
                elif positive_usd < len(usd_events) / 2:
                    dataset.usd_bias = "Bearish"
            
            # Determine key theme
            themes = []
            for event in dataset.recent_events + dataset.upcoming_events:
                if "inflation" in event.event_name.lower() or "cpi" in event.event_name.lower():
                    themes.append("Inflation")
                elif "rate" in event.event_name.lower() or "fomc" in event.event_name.lower():
                    themes.append("Rate Policy")
                elif "employment" in event.event_name.lower() or "nfp" in event.event_name.lower():
                    themes.append("Employment")
                elif "gdp" in event.event_name.lower():
                    themes.append("Growth")
            
            if themes:
                # Most common theme
                dataset.key_theme = max(set(themes), key=themes.count) + " focus"
        
        except Exception as e:
            logger.exception(f"Error computing sentiment summary: {e}")
    
    def _determine_impact(self, event_name: str) -> str:
        """Determine impact level of economic event"""
        event_lower = event_name.lower()
        
        for keyword in self.HIGH_IMPACT_KEYWORDS:
            if keyword in event_lower:
                return "HIGH"
        
        for keyword in self.MEDIUM_IMPACT_KEYWORDS:
            if keyword in event_lower:
                return "MEDIUM"
        
        return "LOW"
    
    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis of text"""
        text_lower = text.lower()
        
        bullish_keywords = ["rise", "gain", "growth", "strong", "positive", "optimism", "rally", "surge"]
        bearish_keywords = ["fall", "decline", "weak", "negative", "concern", "drop", "plunge", "crisis"]
        
        bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in text_lower)
        
        if bullish_count > bearish_count:
            return "Bullish"
        elif bearish_count > bullish_count:
            return "Bearish"
        return "Neutral"
    
    def _country_to_currency(self, country: str) -> str:
        """Map country code to currency"""
        mapping = {
            "US": "USD",
            "EU": "EUR",
            "GB": "GBP",
            "JP": "JPY",
            "CH": "CHF",
            "CA": "CAD",
            "AU": "AUD",
            "NZ": "NZD"
        }
        return mapping.get(country, country)
    
    def _central_bank_to_currency(self, bank: str) -> str:
        """Map central bank to currency"""
        mapping = {
            "FED": "USD",
            "FOMC": "USD",
            "ECB": "EUR",
            "BOE": "GBP",
            "BOJ": "JPY",
            "SNB": "CHF",
            "BOC": "CAD",
            "RBA": "AUD",
            "RBNZ": "NZD"
        }
        return mapping.get(bank.upper(), "USD")
    
    def _rate_limit_wait(self):
        """Ensure minimum time between API requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def format_output(self, dataset: FundamentalDataset) -> str:
        """
        Format fundamental dataset for display/logging
        
        Args:
            dataset: FundamentalDataset to format
            
        Returns:
            Formatted string output
        """
        output = []
        output.append("=" * 80)
        output.append(f"FUNDAMENTAL ANALYSIS: {dataset.pair}")
        output.append(f"Data Source: {dataset.data_source} | Timestamp: {dataset.timestamp}")
        output.append("=" * 80)
        output.append("")
        
        # Upcoming Events
        output.append("UPCOMING EVENTS (Next 24-48 hours):")
        output.append("-" * 80)
        if dataset.upcoming_events:
            high_impact = [e for e in dataset.upcoming_events if e.impact == "HIGH"]
            medium_impact = [e for e in dataset.upcoming_events if e.impact == "MEDIUM"]
            
            if high_impact:
                output.append("\n🔴 HIGH IMPACT:")
                for event in high_impact:
                    output.append(f"  {event.format_output()}")
            
            if medium_impact:
                output.append("\n🟡 MEDIUM IMPACT:")
                for event in medium_impact[:5]:  # Limit to 5
                    output.append(f"  {event.format_output()}")
        else:
            output.append("  No major events scheduled")
        
        output.append("")
        
        # Recent Events
        output.append("RECENT EVENTS (Last 24 hours):")
        output.append("-" * 80)
        if dataset.recent_events:
            for event in dataset.recent_events[:10]:  # Limit to 10
                output.append(f"  {event.format_output()}")
                
                # Add reaction note for surprise events
                if event.surprise_impact in ["Large", "Moderate"]:
                    if event.actual and event.forecast:
                        try:
                            actual_val = float(event.actual)
                            forecast_val = float(event.forecast)
                            direction = "above" if actual_val > forecast_val else "below"
                            output.append(f"    → Market reaction: Data came in {direction} expectations")
                        except:
                            pass
        else:
            output.append("  No major news affecting this pair in the last 24 hours.")
        
        output.append("")
        
        # Recent News Headlines
        if dataset.recent_news:
            output.append("RECENT NEWS HEADLINES:")
            output.append("-" * 80)
            for headline in dataset.recent_news[:10]:  # Limit to 10
                output.append(f"  {headline.format_output()}")
                if headline.summary:
                    # Truncate summary to 100 chars
                    summary = headline.summary[:100] + "..." if len(headline.summary) > 100 else headline.summary
                    output.append(f"    {summary}")
            output.append("")
        
        # Sentiment Summary
        output.append("SENTIMENT SUMMARY:")
        output.append("-" * 80)
        output.append(f"  Overall Risk: {dataset.overall_risk}")
        output.append(f"  USD Bias: {dataset.usd_bias}")
        output.append(f"  Key Theme: {dataset.key_theme}")
        output.append("")
        
        # Errors if any
        if dataset.errors:
            output.append("⚠ ERRORS:")
            for error in dataset.errors:
                output.append(f"  {error}")
            output.append("")
        
        output.append("=" * 80)
        
        return "\n".join(output)
