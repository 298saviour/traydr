# Traydr Frontend

Next.js 14 frontend for AI-powered forex trading platform.

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your backend URL

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Deploy to Railway

1. Push to GitHub
2. Connect to Railway
3. Add environment variable: `NEXT_PUBLIC_API_URL`
4. Deploy automatically

## 📱 Mobile Optimization

- **Responsive Design**: Mobile-first approach
- **Touch Optimized**: Large touch targets (min 44x44px)
- **Fast Loading**: Optimized assets and code splitting
- **PWA Ready**: Can be installed as app
- **Safe Areas**: Respects device notches and home indicators

## 🎨 Features

- Real-time forex signal generation
- AI chat with Claude
- Signal history with search
- Progress tracking with live updates
- Dark mode optimized
- Smooth animations
- Offline support (PWA)

## 📋 Environment Variables

Required:
- `NEXT_PUBLIC_API_URL` - Backend API URL

## 🗂️ Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Home page (Live Signals)
│   ├── history/
│   │   └── page.tsx        # Signal History
│   ├── chat/
│   │   └── page.tsx        # Trading Assistant
│   └── globals.css         # Global styles
├── components/
│   ├── Navigation.tsx      # Mobile-optimized nav
│   ├── SignalCard.tsx      # Signal display
│   ├── ChatInterface.tsx   # AI chat
│   └── ProgressLog.tsx     # Live progress
├── lib/
│   ├── api.ts              # API client
│   └── utils.ts            # Utilities
├── public/
│   ├── manifest.json       # PWA manifest
│   └── icons/              # App icons
├── next.config.mjs         # Next.js config
├── tailwind.config.ts      # Tailwind config
└── package.json            # Dependencies
```

## 🎯 Pages

### Live Forex Signals (`/`)
- Select currency pair
- Start AI analysis
- View real-time progress
- See generated signals

### Signal History (`/history`)
- Browse past signals
- Filter by pair/date
- View detailed analysis
- Chat about signals

### Trading Assistant (`/chat`)
- General forex questions
- Market analysis
- Trading strategies
- Educational content

## 🔧 API Integration

All API calls go through `lib/api.ts`:

```typescript
// Example usage
import { analyzeSignal, getSignals } from '@/lib/api'

// Analyze pair
const result = await analyzeSignal('EUR/USD')

// Get signals
const signals = await getSignals()
```

## 📱 Mobile Features

### Touch Optimization
- Minimum 44x44px touch targets
- Swipe gestures for navigation
- Pull-to-refresh on lists
- Haptic feedback (where supported)

### Performance
- Code splitting per route
- Image optimization
- Lazy loading
- Prefetching

### Responsive Breakpoints
- `xs`: 475px (small phones)
- `sm`: 640px (phones)
- `md`: 768px (tablets)
- `lg`: 1024px (laptops)
- `xl`: 1280px (desktops)
- `2xl`: 1536px (large screens)

## 🎨 Styling

Using Tailwind CSS with custom configuration:

```tsx
// Mobile-first approach
<div className="
  p-4           // Mobile: 16px padding
  md:p-6        // Tablet: 24px padding
  lg:p-8        // Desktop: 32px padding
">
```

## 🚨 Error Handling

- Network errors: Retry with exponential backoff
- API errors: User-friendly messages
- Loading states: Skeleton screens
- Empty states: Helpful guidance

## 🔄 State Management

Using React hooks and context:
- Signal state
- Chat history
- Progress logs
- User preferences

## 🌐 PWA Features

### Installable
Users can install as native app on:
- iOS (Safari)
- Android (Chrome)
- Desktop (Chrome, Edge)

### Offline Support
- Cached static assets
- Offline page
- Service worker

### App-like Experience
- Full-screen mode
- Custom splash screen
- App icons

## 🛠️ Development

### Run Tests
```bash
npm run lint
```

### Build for Production
```bash
npm run build
npm start
```

### Environment Variables
- Development: `.env.local`
- Production: Railway dashboard

## 📊 Performance Targets

- **First Contentful Paint**: < 1.5s
- **Time to Interactive**: < 3.5s
- **Lighthouse Score**: > 90
- **Bundle Size**: < 200KB (initial)

## 🔒 Security

- Environment variables for sensitive data
- HTTPS only in production
- XSS protection
- CSRF protection
- Content Security Policy

## 🚀 Deployment

### Railway
1. Connect GitHub repo
2. Select `frontend` folder as root
3. Add `NEXT_PUBLIC_API_URL` env var
4. Deploy

### Vercel (Alternative)
```bash
npm install -g vercel
vercel --prod
```

## 📈 Monitoring

- Check Railway logs
- Monitor API response times
- Track error rates
- User analytics (optional)

## 🔄 Updates

To update production:
```bash
git add .
git commit -m "Update message"
git push origin main
```

Railway auto-deploys on push.

## 🎯 Best Practices

### Mobile
- Touch targets ≥ 44x44px
- Readable font sizes (≥ 16px)
- Sufficient contrast ratios
- Avoid horizontal scrolling

### Performance
- Optimize images
- Minimize JavaScript
- Use code splitting
- Enable compression

### Accessibility
- Semantic HTML
- ARIA labels
- Keyboard navigation
- Screen reader support

---

**Version**: 6.0.0  
**Framework**: Next.js 14  
**License**: Proprietary
