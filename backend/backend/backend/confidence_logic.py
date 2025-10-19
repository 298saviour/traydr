def calculate_confidence(tech_score, indicator_score, sentiment_score, volatility_score):
    """Calculate a weighted confidence score."""
    confidence = (
        (tech_score * 0.5) +
        (indicator_score * 0.2) +
        (sentiment_score * 0.2) +
        (volatility_score * 0.1)
    )
    return confidence
