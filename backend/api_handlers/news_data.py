try:
    from backend.finnhub_provider import FinnhubNewsProvider  # type: ignore
except ImportError:
    from ..finnhub_provider import FinnhubNewsProvider  # type: ignore

try:
    from news_api_provider import NewsApiProvider  # type: ignore
except ImportError:
    NewsApiProvider = None  # type: ignore

class NewsDataHandler:
    def __init__(self):
        self.finnhub_provider = FinnhubNewsProvider()
        self.news_api_provider = NewsApiProvider() if NewsApiProvider else None

    def get_news(self, pair, limit, lookback_hours):
        items = self.finnhub_provider.get_news_for_pair(pair, limit, lookback_hours)
        if items:
            return items

        if self.news_api_provider and self.news_api_provider.is_available():
            fallback_query = pair.replace('/', ' ')
            articles = self.news_api_provider.get_news(fallback_query, limit=limit, lookback_hours=lookback_hours)
            adapted = []
            for article in articles:
                title = article.get('title') or article.get('headline')
                if not title:
                    continue
                adapted.append({
                    'title': title,
                    'description': article.get('description') or article.get('summary'),
                    'source': (article.get('source') or {}).get('name') if isinstance(article.get('source'), dict) else article.get('source'),
                    'url': article.get('url'),
                    'published_at': article.get('publishedAt') or article.get('published_at')
                })
                if len(adapted) >= limit:
                    break
            return adapted

        return []
