# Traydr Deployment Guide - Railway.app

Complete guide to deploy Traydr on Railway.app

## 📋 Prerequisites

- GitHub account
- Railway account ([railway.app](https://railway.app))
- API keys:
  - TwelveData
  - Alpha Vantage
  - Finnhub
  - Anthropic Claude

## 🚀 Step-by-Step Deployment

### 1. Prepare Repository

```bash
cd C:\Users\sayve\Desktop\Traydr

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Traydr v6.0.0"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/traydr.git
git branch -M main
git push -u origin main
```

### 2. Deploy Backend on Railway

1. **Go to Railway Dashboard**
   - Visit [railway.app/dashboard](https://railway.app/dashboard)
   - Click "New Project"

2. **Deploy from GitHub**
   - Select "Deploy from GitHub repo"
   - Choose your `traydr` repository
   - Select `backend` folder as root directory

3. **Add Environment Variables**
   Click "Variables" tab and add:
   ```
   TWELVE_DATA_API_KEY=your_key_here
   ALPHA_VANTAGE_API_KEY=your_key_here
   FINNHUB_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   FLASK_ENV=production
   PORT=5000
   ```

4. **Deploy**
   - Railway will auto-detect Flask app
   - Click "Deploy"
   - Wait for build to complete
   - Copy the generated URL (e.g., `https://traydr-backend.railway.app`)

### 3. Deploy Frontend on Railway

1. **Create New Service**
   - In same project, click "New Service"
   - Select "Deploy from GitHub repo"
   - Choose `traydr` repository again
   - Select `frontend` folder as root directory

2. **Add Environment Variables**
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
   ```
   (Use the backend URL from step 2.4)

3. **Deploy**
   - Railway will auto-detect Next.js app
   - Click "Deploy"
   - Wait for build to complete
   - Copy the generated URL (e.g., `https://traydr.railway.app`)

### 4. Update Backend CORS

1. **Go to Backend Service**
   - Click on backend service
   - Go to "Variables"

2. **Add Frontend URL**
   ```
   FRONTEND_URL=https://your-frontend-url.railway.app
   ```

3. **Redeploy**
   - Click "Deploy" to restart with new variable

### 5. Test Deployment

1. **Visit Frontend URL**
   - Open `https://your-frontend-url.railway.app`
   - Should see Traydr homepage

2. **Test Health Check**
   - Visit `https://your-backend-url.railway.app/health`
   - Should return JSON with status

3. **Test Signal Generation**
   - Select a pair (e.g., EUR/USD)
   - Click "Analyze with AI"
   - Watch progress logs
   - Verify signal generation

## 🔧 Configuration

### Backend Configuration

**Procfile** (already included):
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

**Runtime** (already included):
```
python-3.11.7
```

### Frontend Configuration

**Build Command** (auto-detected):
```bash
npm install && npm run build
```

**Start Command** (auto-detected):
```bash
npm start
```

## 📊 Database Setup

Railway provides PostgreSQL for production:

1. **Add PostgreSQL**
   - In project, click "New"
   - Select "Database" → "PostgreSQL"
   - Railway will provision database

2. **Update Backend**
   - Go to backend variables
   - Railway auto-adds `DATABASE_URL`
   - Backend will use PostgreSQL instead of SQLite

3. **Initialize Database**
   - Database auto-initializes on first run
   - Check logs to verify

## 🔒 Security Checklist

- [ ] All API keys in environment variables
- [ ] CORS configured with frontend URL
- [ ] HTTPS enabled (automatic on Railway)
- [ ] Database credentials secure
- [ ] No sensitive data in code
- [ ] `.env` files in `.gitignore`

## 📱 Mobile Testing

### iOS (Safari)
1. Visit site on iPhone
2. Tap Share button
3. Select "Add to Home Screen"
4. App installs as PWA

### Android (Chrome)
1. Visit site on Android
2. Tap menu (3 dots)
3. Select "Install app"
4. App installs as PWA

## 🔄 Updates & Redeployment

### Automatic Deployment
Railway auto-deploys on git push:

```bash
# Make changes
git add .
git commit -m "Update description"
git push origin main
```

Railway will:
1. Detect changes
2. Build new version
3. Deploy automatically
4. Zero-downtime deployment

### Manual Deployment
In Railway dashboard:
1. Go to service
2. Click "Deploy"
3. Select commit to deploy

## 📊 Monitoring

### Railway Dashboard
- **Logs**: View real-time logs
- **Metrics**: CPU, memory, network usage
- **Deployments**: History of all deployments

### Health Checks
- Backend: `https://backend-url/health`
- Frontend: Visit homepage

### Error Tracking
Check Railway logs for:
- API errors
- Database errors
- Application crashes

## 🚨 Troubleshooting

### Backend Won't Start
1. Check logs in Railway dashboard
2. Verify all environment variables set
3. Check Python version (3.11+)
4. Verify requirements.txt

### Frontend Won't Build
1. Check Node version (18+)
2. Verify `NEXT_PUBLIC_API_URL` set
3. Check build logs
4. Try local build: `npm run build`

### CORS Errors
1. Verify `FRONTEND_URL` in backend
2. Check backend logs
3. Ensure URLs match exactly
4. Redeploy backend after changes

### Database Errors
1. Check `DATABASE_URL` variable
2. Verify PostgreSQL service running
3. Check database logs
4. Try reinitializing database

### API Rate Limits
1. Monitor API usage in provider dashboards
2. Implement caching if needed
3. Upgrade API plans if necessary

## 💰 Cost Estimation

### Railway Pricing
- **Hobby Plan**: $5/month
  - 500 hours execution time
  - $0.000231/GB-hour for RAM
  - $0.000463/vCPU-hour

- **Pro Plan**: $20/month
  - Unlimited execution time
  - Same resource pricing
  - Priority support

### Typical Monthly Cost
- **Backend**: ~$3-5
- **Frontend**: ~$2-3
- **Database**: ~$2-3
- **Total**: ~$7-11/month

### API Costs
- TwelveData: Free tier (8 calls/min)
- Alpha Vantage: Free tier (5 calls/min)
- Finnhub: Free tier (60 calls/min)
- Anthropic: Pay-as-you-go (~$0.01/signal)

**Estimated**: $10-30/month for moderate usage

## 📈 Scaling

### Horizontal Scaling
Railway supports multiple instances:
1. Go to service settings
2. Increase replica count
3. Load balancing automatic

### Vertical Scaling
Increase resources:
1. Upgrade Railway plan
2. More CPU/RAM allocated
3. Better performance

### Optimization
- Enable caching
- Optimize database queries
- Compress responses
- Use CDN for static assets

## 🔐 Custom Domain (Optional)

1. **Purchase Domain**
   - From any registrar (Namecheap, GoDaddy, etc.)

2. **Add to Railway**
   - Go to frontend service
   - Click "Settings" → "Domains"
   - Click "Custom Domain"
   - Enter your domain

3. **Update DNS**
   - Add CNAME record:
     ```
     CNAME  @  your-app.railway.app
     ```
   - Wait for propagation (5-60 minutes)

4. **SSL Certificate**
   - Railway auto-provisions SSL
   - HTTPS enabled automatically

## ✅ Post-Deployment Checklist

- [ ] Backend health check passes
- [ ] Frontend loads correctly
- [ ] Can generate signals
- [ ] AI chat works
- [ ] Signal history displays
- [ ] Mobile responsive
- [ ] PWA installable
- [ ] All API keys working
- [ ] Database connected
- [ ] Logs are clean
- [ ] Performance acceptable

## 📞 Support

### Railway Support
- Documentation: [docs.railway.app](https://docs.railway.app)
- Discord: [discord.gg/railway](https://discord.gg/railway)
- Email: team@railway.app

### Traydr Issues
- Check logs first
- Review this guide
- Test locally
- Check API status

## 🎉 Success!

Your Traydr platform is now live!

**Frontend**: `https://your-app.railway.app`  
**Backend**: `https://your-backend.railway.app`

Share with users and start generating signals! 🚀

---

**Version**: 6.0.0  
**Last Updated**: October 18, 2025  
**Platform**: Railway.app
