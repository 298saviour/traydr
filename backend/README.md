# Traydr Backend

Flask API for AI-powered forex trading platform.

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run server
python app.py
```

### Deploy to Railway

1. Push to GitHub
2. Connect to Railway
3. Add environment variables in Railway dashboard
4. Deploy automatically

## 📋 Environment Variables

Required:
- `TWELVE_DATA_API_KEY` - TwelveData API key
- `ALPHA_VANTAGE_API_KEY` - Alpha Vantage API key
- `FINNHUB_API_KEY` - Finnhub API key
- `ANTHROPIC_API_KEY` - Anthropic Claude API key

Optional:
- `PORT` - Server port (default: 5000)
- `FLASK_ENV` - Environment (production/development)
- `FRONTEND_URL` - Frontend URL for CORS

## 🔌 API Endpoints

### Health Check
- `GET /health` - Server health status

### Forex Analysis
- `GET /api/pairs` - Get available pairs
- `POST /api/analyze` - Start analysis for a pair
- `GET /api/status` - Get automation status
- `GET /api/progress` - Get progress logs

### Signals
- `GET /api/signals` - Get signal history
- `GET /api/signals/:id` - Get specific signal

### AI Chat
- `POST /api/ask-claude` - Ask about a signal
- `POST /api/general-chat` - General forex chat

### Control
- `POST /api/stop` - Stop automation

## 📦 Dependencies

- Flask - Web framework
- SQLAlchemy - Database ORM
- Anthropic - Claude AI
- Pandas/NumPy - Data processing
- TA-Lib - Technical analysis
- Gunicorn - Production server

## 🗄️ Database

SQLite for development, PostgreSQL for production (Railway provides this).

## 🔒 Security

- API keys in environment variables
- CORS configured for frontend
- Input validation on all endpoints
- Error handling and logging

## 📊 Features

- Multi-timeframe technical analysis (9 timeframes)
- 50+ technical indicators
- Data integrity validation
- Automatic fallback to Finnhub
- AI-powered signal generation
- Real-time progress tracking

## 🛠️ File Structure

```
backend/
├── app.py                          # Main Flask app
├── automation_service.py           # Automation orchestration
├── claude_analyzer.py              # Claude AI integration
├── data_integrity_layer.py         # Data validation
├── database.py                     # Database operations
├── final_confidence_layer.py       # Confidence calculation
├── finnhub_news_provider.py        # News integration
├── forex_analyzer.py               # Forex data provider
├── indicators.py                   # Technical indicators
├── multi_timeframe_collector.py    # Multi-TF analysis
├── signal_generator.py             # Signal creation
├── signal_manager.py               # Signal management
├── technical_analysis.py           # TA utilities
├── trade_engine.py                 # Trade calculations
├── backend/
│   ├── api_handlers/
│   │   ├── market_data.py         # TwelveData client
│   │   └── news_data.py           # News handler
│   └── claude_manager.py          # Claude manager
├── requirements.txt               # Python dependencies
├── Procfile                       # Railway/Heroku config
└── runtime.txt                    # Python version
```

## 🚨 Troubleshooting

### Database Issues
```bash
# Reset database
rm traydr.db
python -c "import database; database.init_db()"
```

### API Key Issues
- Verify keys in .env file
- Check Railway environment variables
- Ensure keys are active and have credits

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📈 Monitoring

- Check `/health` endpoint
- View logs in Railway dashboard
- Monitor API usage in provider dashboards

## 🔄 Updates

To update production:
```bash
git add .
git commit -m "Update message"
git push origin main
```

Railway will auto-deploy.

---

**Version**: 6.0.0  
**Python**: 3.11+  
**License**: Proprietary
