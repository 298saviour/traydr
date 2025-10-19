import json
from .api_handlers.market_data import TwelveDataProvider
from .api_handlers.sentiment_data import SentimentDataHandler
from .api_handlers.news_data import NewsDataHandler
from .confidence_logic import calculate_confidence

class ClaudeManager:
    def __init__(self, client):
        self.client = client
        self.market_data_handler = TwelveDataProvider()
        self.sentiment_data_handler = SentimentDataHandler()
        self.news_data_handler = NewsDataHandler()

    def analyze_trade(self, analysis_request):
        """Orchestrates the iterative analysis loop with Claude."""
        system_prompt = self._get_system_prompt()
        iterations = []
        confidence = 0
        MAX_ITERATIONS = 3

        # Initial data fetch
        market_data = self._fetch_market_data(analysis_request['pair'], analysis_request['timeframes'])
        news_data = self.news_data_handler.get_news(analysis_request['pair'], 10, 48)
        sentiment_data = self.sentiment_data_handler.get_sentiment(analysis_request['news_keywords'])

        current_data = {
            **analysis_request,
            "market_data": market_data,
            "news_data": news_data,
            "sentiment_data": sentiment_data
        }

        for i in range(MAX_ITERATIONS):
            prompt = self._construct_prompt(current_data, iterations)
            raw_response = self._call_claude(system_prompt, prompt)

            try:
                response_json = json.loads(raw_response)
            except json.JSONDecodeError:
                # If Claude returns text instead of JSON, wrap it
                response_json = {"reasoning": raw_response, "final_confidence": 0}

            if 'claude_data_request' in response_json:
                iterations.append({
                    "iteration": i + 1,
                    "action": response_json['claude_data_request'].get('reason'),
                    "new_confidence": response_json.get('confidence_score')
                })
                
                # Fetch additional data
                data_request = response_json['claude_data_request']
                if 'add_timeframes' in data_request:
                    current_data['timeframes'].extend(data_request['add_timeframes'])
                if 'add_news_keywords' in data_request:
                    current_data['news_keywords'].extend(data_request['add_news_keywords'])
                
                # Re-fetch data with new parameters
                market_data = self._fetch_market_data(current_data['pair'], current_data['timeframes'])
                news_data = self.news_data_handler.get_news(current_data['pair'], 10, 48)
                sentiment_data = self.sentiment_data_handler.get_sentiment(current_data['news_keywords'])

                current_data['market_data'] = market_data
                current_data['news_data'] = news_data
                current_data['sentiment_data'] = sentiment_data

            elif 'final_confidence' in response_json:
                response_json['iterations'] = iterations
                return response_json # Loop complete

        # If loop finishes without a final signal, return the last response
        return {"reasoning": "Max iterations reached without reaching confidence threshold.", "iterations": iterations}

    def _fetch_market_data(self, pair, timeframes):
        market_data = {}
        for tf in timeframes:
            # A more robust solution would map UI timeframes to API intervals
            interval = tf.replace('m', 'min').replace('h', 'h') # Basic mapping
            market_data[tf] = self.market_data_handler.get_historical_data(pair, interval, 200)
        return market_data

    def _construct_prompt(self, data, iterations):
        return json.dumps({
            "pair": data['pair'],
            "timeframes": data['timeframes'],
            "indicators": data['indicators'],
            "news_keywords": data['news_keywords'],
            "market_data": data['market_data'],
            "news_data": data['news_data'],
            "sentiment_data": data['sentiment_data'],
            "previous_iterations": iterations
        }, default=str)

    def _get_system_prompt(self):
        return """You are an expert financial analyst and trading strategist.
Your task is to analyze the selected market pair using technical, sentiment, and volatility data.
You must reason step by step, compute confidence, and if your confidence is below 75%, request additional data intelligently.
You can request:
More timeframes (up to 3 total)
More indicators (up to 3 total)
More sentiment/news data with specific keywords
Once your confidence >= 75%, return your final structured trade decision in JSON.
Each iteration, provide:
Current confidence score
Reasoning summary
If needed, specify a “claude_data_request” JSON describing new data needed.
If after 3 iterations confidence still < 75%, return your best possible conclusion with reasoning.
Your final output must include a JSON object with the following structure:
{
  "pair": "EUR/USD",
  "chosen_timeframes": ["1H", "4H", "1D"],
  "chosen_indicators": ["RSI", "EMA"],
  "technical_summary": "RSI(14)=45 rising, EMA(50)>EMA(100) bullish crossover.",
  "sentiment_summary": "Positive tone on ECB tightening and USD weakness.",
  "volatility": "Medium, ATR(14)=45 pips.",
  "initial_confidence": 63,
  "iterations": [
     {"iteration": 1, "action": "Add timeframe 15M", "new_confidence": 68},
     {"iteration": 2, "action": "Add indicator MACD", "new_confidence": 74}
  ],
  "final_confidence": 78,
  "signal": "BUY",
  "entry_price": 1.0850,
  "stop_loss": 1.0810,
  "take_profit": 1.0925,
  "risk_reward_ratio": "1:3",
  "reasoning": "Confluence across 3 timeframes, RSI recovery, EMA crossover, and bullish ECB sentiment all align upward."
}
"""

    def _call_claude(self, system_prompt, user_prompt):
        models = [
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307",
        ]
        last_error = None
        for model in models:
            try:
                msg = self.client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                )
                return msg.content[0].text
            except Exception as e:
                print(f"Anthropic model fallback: {model} failed: {e}")
                last_error = e
                continue
        raise last_error if last_error else RuntimeError("All Claude models failed")
