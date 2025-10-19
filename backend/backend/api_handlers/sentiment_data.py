from ..gdelt_provider import GDELTProvider

class SentimentDataHandler:
    def __init__(self):
        self.gdelt_provider = GDELTProvider()

    def get_sentiment(self, keywords):
        return self.gdelt_provider.get_news(keywords)
