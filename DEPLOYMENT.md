# 🏥 Hemophilia Clinical AI - Full Stack Setup & Deployment Guide

## Overview

Complete production-ready full-stack system for Hemophilia Clinical Decision Support:
- **Frontend**: Streamlit single-page dashboard
- **Backend**: FastAPI REST API with JWT authentication
- **Database**: SQLite with proper schemas
- **Deployment**: Docker + docker-compose

---

## 📋 Quick Start (5 minutes)

### Option 1: Docker (Recommended - Easiest)

```bash
# Clone/enter project directory
cd /path/to/Capstone

# Build and start all services
docker-compose up -d

# Wait 30 seconds for services to start

# Access services:
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Local Development

```bash
# Terminal 1: Start Backend API
cd /path/to/Capstone
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend_api:app --reload

# Terminal 2: Start Frontend
cd /path/to/Capstone
source .venv/Scripts/activate
streamlit run app_frontend.py

# Frontend opens at: http://localhost:8501
# Backend API at: http://localhost:8000
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│           STREAMLIT FRONTEND (Port 8501)                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Dashboard | KPIs | Forms | Chat | Analytics     │  │
│  │  - Login/Registration                             │  │
│  │  - Patient Management                            │  │
│  │  - Risk Predictions                              │  │
│  │  - Clinical Chat                                 │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │  HTTP/REST API (JSON)           │
        │  JWT Authentication             │
        │  CORS Enabled                   │
        └─────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│           FASTAPI BACKEND (Port 8000)                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │  /api/auth/register                              │  │
│  │  /api/auth/login                                 │  │
│  │  /api/patients (CRUD)                            │  │
│  │  /api/predictions (ML)                           │  │
│  │  /api/chat (AI Assistant)                        │  │
│  │  /api/analytics (Reports)                        │  │
│  │  /docs (Interactive API docs)                    │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │  SQLite Database                │
        │  hemophilia_clinic.db           │
        │                                 │
        │  Tables:                        │
        │  - users                        │
        │  - patients                     │
        │  - predictions                  │
        │  - chat_history                 │
        └─────────────────────────────────┘
```

---

## 🔐 Authentication Flow

```
1. User Registration
   POST /api/auth/register
   → Hash password (SHA256)
   → Create user record
   → Return JWT token

2. User Login
   POST /api/auth/login
   → Verify credentials
   → Generate JWT token
   → Return token + user data

3. Protected Requests
   GET /api/patients
   Headers: Authorization: Bearer {token}
   → Validate JWT
   → Return user-specific data

4. Token Expiry
   Default: 30 minutes
   Can be extended by re-login
```

---

## 📊 API Endpoints

### Authentication
```
POST   /api/auth/register        Create new user account
POST   /api/auth/login            Login with email/password
```

### Patients
```
POST   /api/patients              Create new patient
GET    /api/patients              Get all patients for user
GET    /api/patients/{id}         Get specific patient
```

### Predictions
```
POST   /api/predictions           Get risk prediction for patient
```

### Chat
```
POST   /api/chat                  Send chat message to AI
GET    /api/chat-history          Get user's chat history
```

### Analytics
```
GET    /api/analytics             Get dashboard analytics
```

### System
```
GET    /health                    Health check endpoint
GET    /docs                      Interactive API documentation
```

---

## 🔌 API Usage Examples

### Register User
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "doctor_smith",
    "email": "smith@hospital.com",
    "password": "SecurePass123!",
    "full_name": "Dr. John Smith"
  }'

# Response:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "username": "doctor_smith",
    "email": "smith@hospital.com",
    "full_name": "Dr. John Smith"
  }
}
```

### Login
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "smith@hospital.com",
    "password": "SecurePass123!"
  }'
```

### Create Patient
```bash
curl -X POST http://localhost:8000/api/patients \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {token}" \
  -d '{
    "name": "John Doe",
    "age": 45,
    "gender": "Male",
    "clotting_factor": 65.0,
    "previous_bleeds": 5,
    "activity_level": 6,
    "medication_compliance": 0.85,
    "treatment_type": "Factor VIII",
    "notes": "Patient presents with..."
  }'
```

### Get Patient Predictions
```bash
curl -X POST http://localhost:8000/api/predictions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {token}" \
  -d '{
    "patient_id": 1
  }'

# Response:
{
  "risk_score": 0.678,
  "risk_label": "MEDIUM RISK",
  "risk_probability": 0.678,
  "factors": {
    "age": 45,
    "clotting_factor": 65,
    "previous_bleeds": 5,
    "activity_level": 6,
    "compliance": 0.85
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### Chat with AI
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {token}" \
  -d '{
    "role": "user",
    "message": "What are the risk factors for a patient with low clotting factor?"
  }'
```

---

## 🐳 Docker Commands

### Build
```bash
# Build backend
docker build -f Dockerfile.backend -t hemophilia-api .

# Build frontend
docker build -f Dockerfile.frontend -t hemophilia-frontend .

# Build both with compose
docker-compose build
```

### Run
```bash
# Start all services
docker-compose up -d

# Start with logs
docker-compose up

# Start specific service
docker-compose up -d backend
docker-compose up -d frontend
```

### Monitor
```bash
# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Check status
docker-compose ps

# Check health
docker-compose ps --format "table {{.Names}}\t{{.Status}}"
```

### Stop/Clean
```bash
# Stop services (keeps data)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove everything (including volumes)
docker-compose down -v

# Restart
docker-compose restart
```

---

## 🔧 Environment Configuration

Create `.env` file in project root:

```
# Backend Configuration
BACKEND_URL=http://localhost:8000
SECRET_KEY=your-super-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=hemophilia_clinic.db

# Logging
LOG_LEVEL=INFO

# Features
ENABLE_AUTH=true
ENABLE_CHAT=true
ENABLE_ANALYTICS=true

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:8501","http://localhost:8502"]
```

---

## 📦 Project Structure

```
Capstone/
├── backend_api.py              # FastAPI main app
├── app_frontend.py             # Streamlit frontend
├── app.py                      # Original Streamlit (reference)
├── database.py                 # Database functions
│
├── Dockerfile.backend          # Backend container definition
├── Dockerfile.frontend         # Frontend container definition
├── docker-compose.yml          # Orchestration
│
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (created manually)
├── .env.example                # Example configuration
│
├── hemophilia_clinic.db        # SQLite database (auto-created)
├── models/                     # ML models (rf.pkl, xgb.pkl, etc)
│
├── DEPLOYMENT.md               # This file
└── README.md                   # Project documentation
```

---

## 🚀 Production Deployment

### AWS EC2
```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. Clone repository
git clone <repo-url>
cd Capstone

# 4. Set environment variables
cp .env.example .env
# Edit .env with production values

# 5. Start services
docker-compose up -d

# 6. Configure reverse proxy (Nginx)
# Point to http://localhost:8501 for frontend
# Point to http://localhost:8000 for API
```

### Heroku
```bash
# Install Heroku CLI
# Heroku doesn't support docker-compose, deploy separately

# Backend
heroku create hemophilia-api
git subtree push --prefix . heroku main

# Frontend  
heroku create hemophilia-frontend
# Configure streamlit as procfile app
```

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml hemophilia

# Scale services
docker service scale hemophilia_backend=3

# Monitor
docker stack ps hemophilia
```

### Kubernetes
```bash
# Build images
docker build -f Dockerfile.backend -t hemophilia-api:latest .
docker build -f Dockerfile.frontend -t hemophilia-frontend:latest .

# Push to registry
docker push your-registry/hemophilia-api:latest
docker push your-registry/hemophilia-frontend:latest

# Deploy with kubectl
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## 🔒 Security Best Practices

1. **Secrets Management**
   - Never commit `.env` to git
   - Use `export SECRET_KEY=$(openssl rand -hex 32)` for production
   - Rotate keys regularly

2. **HTTPS/TLS**
   - Use Nginx reverse proxy with Let's Encrypt
   - Enable SSL in production

3. **Database**
   - Use PostgreSQL instead of SQLite in production
   - Enable row-level security
   - Regular backups

4. **API Security**
   - Rate limiting (implement in FastAPI)
   - CORS configuration (only allow known origins)
   - Input validation (already in Pydantic models)

5. **Authentication**
   - Implement refresh tokens
   - Add 2FA support
   - Password complexity requirements

---

## 📊 Monitoring & Logging

### Docker Logs
```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific container
docker-compose logs -f backend
```

### Application Logging
- All requests logged to console
- Errors logged with traceback
- Configure log file in `.env`

### Health Checks
- Backend: http://localhost:8000/health
- Frontend: http://localhost:8501

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check logs
docker-compose logs backend

# Common issues:
# 1. Port 8000 already in use
#    → Change port in docker-compose.yml

# 2. Database locked
#    → Delete hemophilia_clinic.db and restart

# 3. Missing dependencies
#    → Rebuild: docker-compose build backend
```

### Frontend can't connect to backend
```bash
# Check backend status
curl http://localhost:8000/health

# Check frontend logs
docker-compose logs frontend

# Verify API_BASE_URL in app_frontend.py
# Should be http://backend:8000 (internal) or http://localhost:8000 (local)
```

### Database issues
```bash
# Delete and recreate
docker-compose down -v
docker-compose up -d

# Check database
sqlite3 hemophilia_clinic.db ".tables"
```

---

## 📈 Performance Optimization

1. **Frontend Caching**
   - Implement @st.cache_data decorators
   - Cache API responses

2. **Database**
   - Add indexes on frequently queried columns
   - Use connection pooling
   - Archive old predictions

3. **API**
   - Implement pagination for large datasets
   - Use gzip compression
   - Add Redis caching

4. **Docker**
   - Multi-stage builds for smaller images
   - Health checks enabled
   - Resource limits defined

---

## 📚 Additional Resources

- **FastAPI Docs**: http://localhost:8000/docs
- **Streamlit Docs**: https://docs.streamlit.io
- **Docker Compose**: https://docs.docker.com/compose

---

## 📞 Support & Contributors

For issues or contributions, please create an issue or submit a pull request.

---

**Version**: 2.0  
**Last Updated**: January 2024  
**Status**: Production Ready ✅
