import requests
from typing import List, Dict, Optional

class GDELTProvider:
    """Provides geopolitical and news context from the GDELT Project."""
    # Using GDELT DOC 2.0 API
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def is_available(self) -> bool:
        # GDELT is a free, open API, so it's always 'available'
        return True

    def get_news(self, keywords: List[str], limit: int = 20) -> List[Dict]:
        """Search for news articles matching keywords."""
        if not keywords:
            return []

        query = ' OR '.join([f'\"{k}\"' for k in keywords])
        params = {
            'query': query,
            'mode': 'artlist',
            'maxrecords': limit,
            'sort': 'datedesc',
            'format': 'json'
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])
            return [
                {
                    'title': a.get('title'),
                    'description': a.get('seendate'), # GDELT lacks descriptions, use date as placeholder
                    'source': a.get('sourcecountry'),
                    'url': a.get('url'),
                    'published_at': a.get('seendate'),
                }
                for a in articles
            ]
        except requests.exceptions.RequestException as e:
            print(f"[GDELTProvider] Request failed: {e}")
            return []
        except ValueError:
            print(f"[GDELTProvider] Failed to parse JSON response.")
            return []
