from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from forex_analyzer import ForexAnalyzer
from technical_analysis import TechnicalAnalysis
from trade_engine import TradeEngine
import database as db
import os
import anthropic
import json
from dotenv import load_dotenv
import numpy as np
import math
import pandas as pd
from datetime import datetime, date
from automation_service import SchedulerService
from signal_manager import SignalManager
import logging
from logging.handlers import TimedRotatingFileHandler
from claude_analyzer import ClaudeAnalyzer
from backend.claude_manager import ClaudeManager
from backend.api_handlers.market_data import TwelveDataProvider
from backend.api_handlers.news_data import NewsDataHandler

app = Flask(__name__)
CORS(app)

# Ensure console can print Unicode on Windows to avoid UnicodeEncodeError (cp1252)
import sys
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Load environment variables from .env if present
load_dotenv()

analyzer = ForexAnalyzer()
trade_engine = TradeEngine(account_balance=10000, risk_percent=2)
signal_manager = SignalManager()
scheduler_service = SchedulerService()
claude_analyzer = ClaudeAnalyzer()
news_data_handler = NewsDataHandler()

# Store recommendations for chat context
current_recommendations = {}

def json_safe(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        try:
            # Replace NaN/Inf with None
            if not math.isfinite(float(obj)):
                return None
            return float(obj)
        except Exception:
            return None
    elif isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    elif obj is None:
        return None
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return json_safe(obj.to_list())
    elif isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))
    elif isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe(item) for item in obj]
    elif isinstance(obj, (set, tuple)):
        return [json_safe(x) for x in obj]
    return obj

def call_claude_with_fallback(system_prompt: str, user_text: str, max_tokens: int = 1024) -> str:
    """Call Anthropic with model fallback for compatibility across accounts."""
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-5-opus-20241010",
        "claude-3-haiku-20240307"
    ]
    last_error = None
    for model in models:
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}],
                    }
                ],
            )
            return msg.content[0].text
        except Exception as e:
            print(f"Anthropic model fallback: {model} failed: {e}")
            last_error = e
            continue
    # If all models fail, raise last error
    raise last_error if last_error else RuntimeError("All Claude models failed")

# Claude API setup
try:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in environment.")
    
    # Trim whitespace from the key
    api_key = api_key.strip()
    
    # Initialize with explicit parameters only
    client = anthropic.Anthropic(
        api_key=api_key,
        max_retries=2,
        timeout=60.0
    )
    
    print("Anthropic client initialized successfully.")
    print(f"Using anthropic version: {anthropic.__version__}")
except Exception as e:
    client = None
    print(f"Error initializing Anthropic client: {e}")
    print(f"Please set ANTHROPIC_API_KEY in your environment or in a .env file at the project root.")
    import traceback
    traceback.print_exc()

# ========= Logging (daily rotating) =========
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)
log_path = os.path.join(logs_dir, 'app.log')
handler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=7, encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
root_logger = logging.getLogger()
if not any(isinstance(h, TimedRotatingFileHandler) for h in root_logger.handlers):
    root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

# ============ PAGE ROUTES ============

@app.route('/')
def live_signals():
    """Live Signals page"""
    return render_template('live_signals.html')

@app.route('/trading-assistant')
def trading_assistant():
    """Trading Assistant page"""
    return render_template('trading_assistant.html')

@app.route('/signal-history')
def signal_history():
    """Signal History page"""
    return render_template('signal_history.html')

@app.route('/performance')
def performance():
    """Performance Analytics page"""
    return render_template('performance.html')

@app.route('/ai-analyzer')
def ai_analyzer():
    """AI Trading Analyzer page"""
    return render_template('analyzer_dashboard.html')

# ============ API ROUTES ============

# ----- Automation control -----
@app.route('/status', methods=['GET'])
def api_status():
    try:
        return jsonify({'success': True, 'status': scheduler_service.status()})
    except Exception as e:
        logging.exception("/status error: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def api_analyze():
    try:
        payload = request.get_json(silent=True) or {}
        pair = payload.get('pair')
        outputsize = payload.get('outputsize')

        pairs_to_use = []
        if pair:
            raw = str(pair).strip().upper().replace(' ', '')
            # If the provided symbol exactly matches a supported key, use it as-is
            if raw in analyzer.AV_CURRENCIES.keys():
                normalized = raw
            else:
                # Only auto-insert a slash for 6-letter alphabetic forex codes (e.g., EURUSD)
                if ('/' not in raw) and (len(raw) == 6) and raw.isalpha():
                    normalized = f"{raw[:3]}/{raw[3:]}"
                else:
                    normalized = raw
            pairs_to_use = [normalized]

        if pairs_to_use:
            scheduler_service.configure_pairs(pairs_to_use)
        if outputsize:
            try:
                scheduler_service.configure_fx_outputsize(int(outputsize))
            except Exception:
                pass

        scheduler_service.start()
        target_pairs = pairs_to_use if pairs_to_use else scheduler_service._selected_pairs
        return jsonify({'success': True, 'message': f"Started scheduler for {', '.join(target_pairs)}"})
    except Exception as e:
        logging.exception("/analyze error: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stop', methods=['POST'])
def api_stop():
    try:
        scheduler_service.stop()
        return jsonify({'success': True, 'message': 'Scheduler stopped'})
    except Exception as e:
        logging.exception("/stop error: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ask-claude', methods=['POST'])
def ask_claude():
    data = request.json
    signal_id = data.get('signal_id')
    question = data.get('question')

    if not signal_id or not question:
        return jsonify({'error': 'signal_id and question are required'}), 400

    try:
        signal_data = db.get_signal(signal_id)
        if not signal_data:
            return jsonify({'error': 'Signal not found'}), 404

        chat_history = db.get_chat_history(signal_id)
        db.save_chat_message(signal_id, 'user', question)

        analyzer = ClaudeAnalyzer()
        answer = analyzer.answer_follow_up(signal_data, chat_history, question)

        db.save_chat_message(signal_id, 'assistant', answer)

        # Return the latest answer
        return jsonify({'answer': answer})

    except Exception as e:
        logging.exception(f"Error asking Claude: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/general-chat', methods=['GET', 'POST'])
def general_chat():
    if request.method == 'GET':
        history = db.get_general_chat_history()
        return jsonify(history)

    if request.method == 'POST':
        data = request.json
        question = data.get('question')
        if not question:
            return jsonify({'error': 'Question is required'}), 400

        try:
            db.save_general_chat_message('user', question)
            chat_history = db.get_general_chat_history()

            analyzer = ClaudeAnalyzer()
            # This method will need to be created in ClaudeAnalyzer
            answer = analyzer.answer_general_question(chat_history, question)

            db.save_general_chat_message('assistant', answer)
            return jsonify({'answer': answer})
        except Exception as e:
            logging.exception(f"Error in general chat: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/chat-history', methods=['GET'])
def api_chat_history():
    try:
        signal_id = request.args.get('signal_id', type=int)
        if not signal_id:
            return jsonify({'success': False, 'error': 'signal_id is required'}), 400
        if not db.get_signal(signal_id):
            return jsonify({'success': False, 'error': f'Signal {signal_id} not found.'}), 404
        history = db.get_chat_history(signal_id)
        return jsonify({'success': True, 'chat_history': history})
    except Exception as e:
        logging.exception("/api/chat-history error: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/signals/<path:pair>', methods=['GET'])
def api_signals_pair(pair):
    pair = pair.replace('%2F', '/')
    try:
        items = signal_manager.list_signals(pair=pair)
        return jsonify({'success': True, 'signals': items})
    except Exception as e:
        logging.exception("/signals/<pair> error: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/evaluate', methods=['POST'])
def admin_evaluate():
    """Manually trigger 3-day evaluation job and return summary."""
    try:
        summary = scheduler_service._evaluate_signals()  # returns {updated, updated_ids}
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        logging.exception("/admin/evaluate error: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-signals', methods=['POST'])
def generate_signals():
    """Generate new trading signals"""
    try:
        data = request.get_json() or {}
        pairs = data.get('pairs', list(analyzer.AV_CURRENCIES.keys()))
        account_balance = data.get('account_balance', 10000)
        risk_percent = data.get('risk_percent', 2)

        trade_engine.account_balance = account_balance
        trade_engine.risk_percent = risk_percent

        print(f"[generate-signals] Incoming pairs: {pairs}")
        recommendations = trade_engine.analyze_multiple_pairs(analyzer, pairs)
        print(f"[generate-signals] Recommendations generated: {len(recommendations)}")
        
        # Save signals to database and make JSON-safe
        saved_signals = []
        for rec in recommendations:
            # Make JSON-safe before saving
            rec = json_safe(rec)

            # If AI analysis metadata was provided from the frontend, attach it before saving
            ai_md = (data.get('ai_analysis') or '').strip()
            timeframe = data.get('timeframe')
            analysis_ts = data.get('analysis_timestamp')

            # Ensure dict containers exist
            detailed = rec.get('detailed_explanation') or {}
            summary = rec.get('analysis_summary') or {}

            if ai_md:
                detailed['ai_analysis_md'] = ai_md
            if timeframe:
                summary['timeframe'] = timeframe
            if analysis_ts:
                summary['analysis_timestamp'] = analysis_ts

            rec['detailed_explanation'] = detailed
            rec['analysis_summary'] = summary
            
            signal_id = db.save_signal(rec)
            print(f"[generate-signals] Saved signal id={signal_id} pair={rec.get('pair')} rec={rec.get('recommendation')}")
            rec['id'] = signal_id
            saved_signals.append(rec)
            
            # Also store in memory for chat context
            current_recommendations[str(signal_id)] = rec

        print(f"[generate-signals] Total saved: {len(saved_signals)}")
        try:
            all_after = db.get_all_signals()
            active_after = db.get_active_signals()
        except Exception as _e:
            all_after, active_after = [], []
        return jsonify({
            'success': True,
            'signals': saved_signals,
            'count': len(saved_signals),
            'saved_ids': [s.get('id') for s in saved_signals],
            'history_count': len(all_after),
            'active_count': len(active_after)
        })
    except Exception as e:
        print(f"Error generating signals: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Get all active signals"""
    try:
        signals = db.get_active_signals()
        print(f"[get-signals] Active signals returned: {len(signals)}")
        return jsonify({'success': True, 'signals': signals})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals/history', methods=['GET'])
def get_signal_history():
    """Get signal history with optional status filter"""
    try:
        status = request.args.get('status', 'all')
        pair_filter = request.args.get('pair')
        
        if status == 'all':
            signals = db.get_all_signals()
        else:
            signals = db.get_signals_by_status(status)
        # Optional pair filter
        if pair_filter:
            signals = [s for s in signals if str(s.get('pair')) == pair_filter]
        # Normalize timestamps for frontend parsing
        for s in signals:
            try:
                if s.get('created_at'):
                    s['created_at'] = str(s['created_at']).replace(' ', 'T')
                if s.get('updated_at'):
                    s['updated_at'] = str(s['updated_at']).replace(' ', 'T')
                # Backward compatibility: normalize confidence to percentage if missing
                conf = s.get('confidence')
                if conf is not None and isinstance(conf, (int, float)):
                    s['confidence'] = f"{int(round(float(conf)))}%"
            except Exception:
                pass
        print(f"[get-signal-history] Status={status} Count={len(signals)}")
        
        return jsonify({'success': True, 'signals': signals})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals/<int:signal_id>/update-status', methods=['POST'])
def update_signal_status(signal_id):
    """Update signal status"""
    try:
        data = request.get_json()
        status = data.get('status')
        pips = data.get('pips')
        
        db.update_signal_status(signal_id, status, pips)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/performance/stats', methods=['GET'])
def get_performance_stats():
    """Get performance statistics"""
    try:
        stats = db.get_performance_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals/<int:signal_id>/discuss', methods=['POST'])
def discuss_signal(signal_id):
    """Discuss a signal with Claude."""
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'success': False, 'error': 'Missing question'}), 400

    try:
        answer = get_claude_answer(question, db.get_signal(signal_id))
        return jsonify({'success': True, 'answer': answer})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/performance/pairs', methods=['GET'])
def get_pair_performance():
    """Get performance by currency pair"""
    try:
        pairs = db.get_pair_performance()
        return jsonify({'success': True, 'pairs': pairs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pairs', methods=['GET'])
def get_pairs():
    """Get available currency pairs"""
    return jsonify({'pairs': list(analyzer.AV_CURRENCIES.keys())})

@app.route('/api/ai-analyze-pair', methods=['POST'])
def ai_analyze_pair():
    """Use Claude AI to analyze a specific currency pair with fresh data and news"""
    if not client:
        return jsonify({
            'success': False,
            'error': 'AI assistant is not configured. Please check the server logs.'
        })

    try:
        data = request.get_json()
        pair = data.get('pair')
        timeframe = data.get('timeframe', '1d')  # Default to daily timeframe

        if not pair:
            return jsonify({'success': False, 'error': 'Missing currency pair'})

        print(f"Analyzing {pair} with {timeframe} timeframe, fresh data and news...")

        # Map timeframe to appropriate period and interval (define early for reuse)
        timeframe_config = {
            '1h': {'period': '5d', 'interval': '1h'},
            '4h': {'period': '1mo', 'interval': '4h'},
            '1d': {'period': '3mo', 'interval': '1d'},
            '1w': {'period': '6mo', 'interval': '1w'},
            '1M': {'period': '1y', 'interval': '1M'}
        }

        config = timeframe_config.get(timeframe, timeframe_config['1d'])

        # Step 1: Fetch the MOST CURRENT price data from Alpha Vantage (with fallback)
        current_price = analyzer.get_current_price(pair)
        df = None
        if current_price is None:
            # Fall back to last close from historical data for selected timeframe
            df = analyzer.fetch_data(pair, period=config['period'], interval=config['interval'])
            if df is not None and not df.empty:
                try:
                    current_price = float(df['Close'].iloc[-1])
                except Exception:
                    current_price = None

        if current_price is None:
            # Final fallback: try 1d/1h window
            df_fallback = analyzer.fetch_data(pair, period='1d', interval='1h')
            if df_fallback is not None and not df_fallback.empty:
                try:
                    current_price = float(df_fallback['Close'].iloc[-1])
                    if df is None:
                        df = df_fallback
                except Exception:
                    pass

        if current_price is None:
            return jsonify({'success': False, 'error': f'Unable to fetch current price for {pair}'})

        print(f"Current {pair} price: {current_price}")

        # Step 2: Fetch comprehensive historical data for technical analysis (reuse df if available)
        if df is None:
            df = analyzer.fetch_data(pair, period=config['period'], interval=config['interval'])

        if df is None or df.empty:
            return jsonify({'success': False, 'error': f'No data available for {pair} with {timeframe} timeframe'})

        # Step 3: Perform technical analysis on the fresh data
        ta = TechnicalAnalysis(df)
        analysis = ta.get_comprehensive_analysis()
        analysis = json_safe(analysis)  # Ensure JSON serializable

        # Step 4: Fetch the LATEST news for this pair (last 48 hours)
        news = news_data_handler.get_news(pair, 5, 48)
        print(f"Fetched {len(news)} recent news items for {pair}")

        # Step 5: Prepare recent candles data (ensure fresh)
        df_recent = df.tail(10).copy().reset_index()
        date_col = 'Date'
        if date_col not in df_recent.columns:
            if 'index' in df_recent.columns:
                date_col = 'index'
            else:
                date_col = df_recent.columns[0]

        recent_candles = []
        for _, row in df_recent.iterrows():
            recent_candles.append({
                'date': json_safe(row[date_col]),
                'close': float(row['Close']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'volume': int(row['Volume'])
            })

        # Step 6: Create comprehensive context for Claude
        context = {
            'pair': pair,
            'current_price': float(current_price),
            'analysis': analysis,
            'recent_candles': recent_candles,
            'news': news,
            'analysis_timestamp': datetime.now().isoformat(),
            'timeframe': timeframe,
            'timeframe_config': config
        }

        print(f"Sending to Claude: {pair} analysis with {timeframe} timeframe, {len(news)} news items and current price {current_price}")

        # Step 7: Get AI analysis from Claude with all fresh data
        ai_analysis = get_ai_pair_analysis(pair, context)

        return jsonify({
            'success': True,
            'pair': pair,
            'analysis': ai_analysis,
            'current_price': float(current_price),
            'news_count': len(news),
            'timeframe': timeframe,
            'analysis_timestamp': context['analysis_timestamp'],
            'signal_id': None
        })

    except Exception as e:
        print(f"Error in AI pair analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/assistant/chat', methods=['POST'])
def assistant_chat():
    """General trading assistant chat (not tied to specific signal)"""
    if not client:
        return jsonify({
            'success': False,
            'answer': 'The AI assistant is not configured. Please check the server logs.'
        })

    try:
        data = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({'success': False, 'answer': 'Missing question.'})

        # Get recent signals for context and make JSON-safe
        recent_signals = db.get_active_signals()[:5]  # Get last 5 active signals
        
        # Convert timestamp fields to strings
        safe_signals = []
        for signal in recent_signals:
            safe_signal = {
                'pair': signal.get('pair'),
                'recommendation': signal.get('recommendation'),
                'confidence': signal.get('confidence'),
                'status': signal.get('status'),
                'created_at': str(signal.get('created_at', ''))
            }
            safe_signals.append(safe_signal)
        
        answer = get_assistant_answer(question, safe_signals)
        return jsonify({'success': True, 'answer': answer})
    except Exception as e:
        print(f"Error in assistant chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'answer': 'Sorry, I encountered an error while processing your question.'
        })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat questions about specific signal recommendations using Claude API"""
    if not client:
        return jsonify({
            'success': False,
            'answer': 'The AI assistant is not configured. Please check the server logs.'
        })

    data = request.get_json()
    question = data.get('question', '')
    signal_id = data.get('signal_id')

    if not question:
        return jsonify({'success': False, 'answer': 'Missing question.'})

    # Try to get signal from database if ID provided
    if signal_id:
        signals = db.get_all_signals()
        rec = next((s for s in signals if s['id'] == int(signal_id)), None)
        if not rec:
            return jsonify({'success': False, 'answer': 'Signal not found.'})
        
        # Make signal JSON-safe by converting timestamps
        rec = json_safe(rec)
        if 'created_at' in rec:
            rec['created_at'] = str(rec['created_at'])
        if 'updated_at' in rec:
            rec['updated_at'] = str(rec['updated_at'])
    else:
        return jsonify({'success': False, 'answer': 'Missing signal ID.'})
    
    try:
        answer = get_claude_answer(question, rec)
        return jsonify({'success': True, 'answer': answer})
    except Exception as e:
        print(f"Error getting answer from Claude: {e}")
        return jsonify({
            'success': False,
            'answer': 'Sorry, I encountered an error while processing your question.'
        })

def get_claude_answer(question, rec):
    """Generate an intelligent answer using the Claude API based on the question and recommendation context."""
    # Sanitize recommendation data for the prompt
    contextual_rec = json_safe({
        'pair': rec.get('pair'),
        'recommendation': rec.get('recommendation'),
        'confidence': rec.get('confidence'),
        'status': rec.get('status'),
        'analysis_summary': rec.get('analysis_summary'),
        'trade_parameters': {
            'entry_price': rec.get('entry_price'),
            'stop_loss': rec.get('stop_loss'),
            'take_profit_1': rec.get('take_profit_1'),
            'take_profit_2': rec.get('take_profit_2'),
            'take_profit_3': rec.get('take_profit_3'),
            'lot_size': rec.get('lot_size'),
            'risk_amount': rec.get('risk_amount'),
            'potential_profit_1': rec.get('potential_profit_1'),
            'potential_profit_2': rec.get('potential_profit_2'),
            'potential_profit_3': rec.get('potential_profit_3'),
        },
        'explanation': rec.get('detailed_explanation')
    })

    # Remove None values for a cleaner prompt
    contextual_rec = {k: v for k, v in contextual_rec.items() if v is not None}

    system_prompt = (
        "You are an expert forex trading assistant. Your name is 298sav.ai AI. "
        "You are providing insights to a user about a specific trade recommendation generated by a technical analysis engine. "
        "Be helpful, concise, and clear in your explanations. Use the provided data to answer the user's question. "
        "Do not give financial advice, but explain the reasoning behind the trade based on the data. "
        "Format your answers cleanly using markdown, such as bullet points for lists."
    )

    user_text = (
        "Here is the trade recommendation data:\n\n```json\n"
        + json.dumps(contextual_rec, indent=2, default=str)
        + "\n```\n\nMy question is: "
        + question
    )

    return call_claude_with_fallback(system_prompt, user_text, max_tokens=1024)

def get_assistant_answer(question, recent_signals=None):
    """Generate general forex trading assistant answers using Claude API"""
    
    context_info = ""
    if recent_signals and len(recent_signals) > 0:
        context_info = f"\n\nRecent active signals:\n"
        for sig in recent_signals:
            context_info += f"- {sig['pair']}: {sig['recommendation']} (Confidence: {sig.get('confidence', 'N/A')})\n"
    
    system_prompt = (
        "You are an expert forex trading assistant named 298sav.ai AI. "
        "You help traders understand market conditions, technical analysis, and trading strategies. "
        "Be helpful, concise, and educational. Use markdown formatting for clarity. "
        "Do not provide specific financial advice, but educate on trading concepts and analysis."
    )

    return call_claude_with_fallback(system_prompt, question + context_info, max_tokens=1536)

def get_ai_pair_analysis(pair, context):
    """Get detailed AI analysis for a specific currency pair with confidence score"""
    # Pull timeframe info from context (used in prompt below)
    timeframe = context.get('timeframe', '1d')
    timeframe_config = context.get('timeframe_config', {})

    system_prompt = (
        "You are 298sav.ai AI, an expert forex market analyst. "
        "Analyze the provided currency pair data and technical indicators to give a comprehensive trading assessment. "
        "Your analysis should include:\n"
        "1. **Overall Market Sentiment** - Bullish, Bearish, or Neutral\n"
        "2. **Confidence Score** - Rate from 1-10 how confident you are in this analysis\n"
        "3. **Key Technical Insights** - What the indicators are telling us\n"
        "4. **Trend Analysis** - Short-term and medium-term trend direction\n"
        "5. **Support & Resistance Levels** - Key price levels to watch\n"
        "6. **Trading Recommendation** - BUY, SELL, or NO TRADE with reasoning\n"
        "7. **Risk Factors** - What could invalidate this analysis\n"
        "8. **News Impact** - How recent news may affect this pair\n\n"
        "IMPORTANT: Consider both technical indicators AND recent news headlines when making your assessment. "
        "News can significantly impact short-term price movements and should influence your confidence level and risk assessment."
        "Format your response in clear markdown with headings and bullet points. "
        "Be specific, data-driven, and educational. Do not give financial advice, but provide analysis based on the data."
    )

    # Prepare the context message
    news_items = context.get('news', []) or []
    news_bullets = "\n".join([
        f"- {n.get('title','').strip()} ({n.get('source','')}, {n.get('published_at','')[:10]})" for n in news_items if n.get('title')
    ])

    prompt = f"""Please analyze the following forex pair data with FRESH market data and recent news:

**Currency Pair:** {pair}
**Current Price:** {context.get('current_price', 'N/A')}
**Analysis Timeframe:** {context.get('timeframe', '1d')} ({context.get('timeframe_config', {}).get('period', 'N/A')} period, {context.get('timeframe_config', {}).get('interval', 'N/A')} interval)
**Analysis Timestamp:** {context.get('analysis_timestamp', 'N/A')}

**Technical Analysis Summary:**
```json
{json.dumps(context.get('analysis', {}), indent=2, default=str)}
```

**Recent Price Action (Last 10 Candles):**
```json
{json.dumps(context.get('recent_candles', []), indent=2, default=str)}
```

**Recent News Headlines (last 48h):**
{news_bullets or '- No recent news found.'}

**ANALYSIS INSTRUCTIONS:**
- Consider how recent news may impact technical levels and trends for this specific timeframe
- Factor in current market conditions when assessing support/resistance levels
- Use news sentiment to adjust your confidence score for this timeframe
- Explain how news events might create trading opportunities or risks within this timeframe
- Provide insights specific to {timeframe} trading (intraday vs daily vs weekly patterns)

Provide a comprehensive analysis with a confidence score (1-10) and actionable insights."""

    return call_claude_with_fallback(system_prompt, prompt, max_tokens=2048)


@app.route('/api/ai-trade-analyzer', methods=['POST'])
def ai_trade_analyzer():
    """Endpoint to trigger the AI Trading Analyzer"""
    if not client:
        return jsonify({'error': 'Anthropic client not configured'}), 500

    try:
        request_data = request.json
        claude_manager = ClaudeManager(client)
        # This is a simplified call. The full implementation would handle the iterative analysis.
        result = claude_manager.analyze_trade(request_data)
        # The result from ClaudeManager is already a dictionary
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 5000)), debug=False)