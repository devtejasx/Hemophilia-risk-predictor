# Hemophilia AI Frontend

Modern React-based clinical decision support system for hemophilia risk prediction and management.

## Quick Start

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Setup environment
cp .env.example .env.local
# Edit .env.local and set VITE_API_URL to your FastAPI backend URL
```

### Development

```bash
# Start dev server on http://localhost:3000
npm run dev
```

### Build

```bash
# Build for production
npm run build

# Preview built app
npm run preview
```

## Project Structure

```
src/
├── components/        # Reusable UI components
│   ├── Navbar.tsx
│   ├── Sidebar.tsx
│   ├── MetricCard.tsx
│   ├── PatientCard.tsx
│   ├── ChatBox.tsx
│   ├── FormField.tsx
│   └── Charts.tsx
├── pages/            # Page components
│   ├── Dashboard.tsx
│   ├── AddPatient.tsx
│   ├── Predictions.tsx
│   ├── SHAPAnalysis.tsx
│   ├── Chatbot.tsx
│   └── Analytics.tsx
├── services/         # API integration
│   ├── api.ts        # Axios client
│   └── api-client.ts # API methods
├── store/            # State management (Zustand)
│   └── appStore.ts
├── styles/           # Global styles
│   └── index.css
├── App.tsx           # Main app with routing
└── main.tsx          # Entry point
```

## Features

### 📊 Dashboard
- Real-time KPI metrics
- Risk distribution visualization
- Trend analysis
- Recent patients overview

### ➕ Add Patient
- Multi-step form wizard
- Demographics, clinical, treatment, and history sections
- Form validation
- Patient database integration

### 🧠 Risk Predictions
- Interactive patient data input
- ML-based inhibitor risk prediction
- Feature importance visualization
- Clinical recommendations

### 📈 SHAP Analysis
- Model explainability with SHAP values
- Multiple view modes (Basic, Advanced, Detailed)
- Feature contribution breakdown
- Clinical guidance

### 💬 Chatbot
- Clinical question answering
- Multiple conversation modes
- Treatment planning assistance
- Educational content

### 📉 Analytics
- Comprehensive patient cohort analysis
- Advanced filtering and sorting
- Risk distribution charts
- Data export (CSV)

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling with dark mode
- **React Router v6** - Client routing
- **Axios** - HTTP client
- **Recharts** - Data visualization
- **Zustand** - State management
- **TypeScript** - Type safety
- **Lucide React** - Icons

## Environment Variables

Create `.env.local`:

```env
VITE_API_URL=http://localhost:8000/api
```

## API Requirements

The backend should provide:

- `POST /api/predict` - Risk prediction endpoint
- `POST /api/chat` - Chatbot endpoint
- `GET /api/patients` - Patient list
- `POST /api/patients` - Create patient
- `GET /api/analytics/dashboard` - Analytics data
- `GET /api/analytics/risk-distribution` - Risk metrics
- `GET /api/analytics/severity-distribution` - Severity metrics

## Development

### Code Style
- ESLint for linting
- TypeScript for type checking
- Tailwind CSS for styling

### Component Patterns

```typescript
// Functional component with hooks
const MyComponent: React.FC<Props> = ({ prop1 }) => {
  const [state, setState] = useState<Type>(initialValue)
  return <div>{state}</div>
}
```

### State Management

Use Zustand store for app-wide state:

```typescript
import { useAppStore } from '@/store/appStore'

const { theme, setTheme } = useAppStore()
```

## Deployment

### Vercel (Recommended)

```bash
# Push to GitHub, then:
# Connect repo on Vercel dashboard
# Set VITE_API_URL environment variable
# Deploy
```

### Self-Hosted

```bash
npm run build
# Serve dist/ folder with any static server
```

### Docker

```bash
docker build -t hemophilia-frontend .
docker run -p 3000:3000 hemophilia-frontend
```

## Troubleshooting

### API Connection Issues
- Verify VITE_API_URL is correct
- Check CORS settings on backend
- Ensure backend is running on correct port

### Build Errors
- Delete `node_modules` and `dist`
- Run `npm install` again
- Check Node.js version (16+)

### Dark Mode Not Working
- Clear browser cache
- Check localStorage for theme value
- Verify Tailwind dark mode config

## Performance

- Code splitting with React Router
- Image optimization
- CSS minification via Tailwind
- Build optimization with Vite

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

Medical research - Proprietary

## Support

For issues and questions, contact the development team.
