# Traydr - AI-Powered Forex Trading Platform

Production-ready deployment package for Railway.app

## 🚀 Quick Deploy to Railway

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO
   git push -u origin main
   ```

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your Traydr repository
   - Railway will auto-detect and deploy both services

## 📦 Project Structure

```
Traydr/
├── backend/          # Flask API (Python)
├── frontend/         # Next.js App (React/TypeScript)
├── railway.json      # Railway configuration
└── README.md         # This file
```

## 🔑 Environment Variables

### Backend (.env)
```
TWELVE_DATA_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
FINNHUB_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
DATABASE_URL=sqlite:///traydr.db
FLASK_ENV=production
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
```

## 🌐 Mobile Optimization

- Responsive design for all screen sizes
- Touch-optimized controls
- Progressive Web App (PWA) ready
- Fast loading with optimized assets

## 📱 Features

- Real-time forex signal generation
- AI-powered analysis (Claude)
- Multi-timeframe technical analysis
- Economic calendar integration
- Signal history and chat
- Mobile-first design

## 🛠️ Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📊 Tech Stack

- **Backend**: Flask, SQLAlchemy, Anthropic Claude
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Database**: SQLite (production: PostgreSQL on Railway)
- **APIs**: TwelveData, Alpha Vantage, Finnhub

## 🔒 Security

- API keys stored in environment variables
- CORS configured for production
- Rate limiting implemented
- Data validation on all inputs

## 📈 Monitoring

- Health check endpoint: `/health`
- Logs available in Railway dashboard
- Error tracking and reporting

## 🆘 Support

For issues or questions, check the documentation in each service folder.

---

**Version**: 6.0.0  
**Last Updated**: October 18, 2025
