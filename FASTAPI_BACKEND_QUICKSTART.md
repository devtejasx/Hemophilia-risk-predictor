# FastAPI Backend Quick Start

## 🚀 In 5 Minutes

### 1. Install Backend (30 seconds)
```bash
cd backend
pip install -r requirements.txt
```

### 2. Start Backend Server (10 seconds)
```bash
python -m uvicorn main:app --reload
```

You should see:
```
╔════════════════════════════════════════════════════════════════╗
║  🏥 Hemophilia AI Platform - FastAPI Backend                  ║
║  Starting on: http://localhost:8000                           ║
║  Documentation: http://localhost:8000/docs                    ║
╚════════════════════════════════════════════════════════════════╝
```

### 3. Open API Docs (5 seconds)
```
http://localhost:8000/docs
```

You'll see interactive Swagger UI with all endpoints!

### 4. Start Streamlit (Optional - new terminal)
```bash
streamlit run app.py
```

Streamlit will automatically use the FastAPI backend!

---

## 📋 Quick Endpoints Overview

### 🔮 Predict Risk
```
POST /predict
Input: Patient clinical data
Output: Risk score, category, recommendations
```

### 💬 Chat with AI
```
POST /chat
Input: Question, mode (diagnosis/treatment/risk/monitoring)
Output: AI-generated response + recommendations
```

### 👥 Manage Patients
```
GET/POST/PUT/DELETE /patients
Get, create, update, delete patient records
```

### 📊 Get Analytics
```
GET /analytics/dashboard
Dashboard statistics, risk distribution, trends
```

---

## 🧪 Quick Test

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

### Test 2: Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 12,
    "dose": 2000,
    "exposure": 90,
    "severity": "Moderate",
    "mutation": "Intron 22 Inversion"
  }'
```

### Test 3: Chat with AI
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What should I monitor?",
    "mode": "monitoring_analysis"
  }'
```

---

## 📁 Backend Structure

```
backend/
├── main.py                 ← FastAPI app entry point
├── models.py               ← Request/response models
├── ml_utils.py             ← Prediction logic
├── gpt_utils.py            ← Chat logic
└── routers/
    ├── predict.py          ← /predict endpoints
    ├── chat.py             ← /chat endpoints
    ├── patients.py         ← /patients endpoints
    └── analytics.py        ← /analytics endpoints
```

---

## 🎯 Key Features

✅ **4 Prediction REST Endpoints** - ML risk predictions
✅ **4 Chat Modes** - Clinical AI with context
✅ **Full CRUD Patient Management** - Create, read, update, delete
✅ **Dashboard Analytics** - Statistics and trends
✅ **Auto-generated Docs** - Swing UI + ReDoc
✅ **Error Handling** - Clean error responses
✅ **CORS Enabled** - Works with Streamlit + React

---

## 🚨 Common Issues

| Issue | Fix |
|-------|-----|
| Port 8000 in use | Change port: `--port 8001` |
| "Connection refused" | Check backend started: `curl http://localhost:8000/health` |
| CORS error | Already fixed in `main.py` |
| API key error | Set `OPENAI_API_KEY` environment variable |

---

## 📖 Full Documentation

See `FASTAPI_BACKEND_GUIDE.md` for complete API reference!

---

**Ready to go? Start with:**
```bash
cd backend && python -m uvicorn main:app --reload
```

Then open: http://localhost:8000/docs

Enjoy! 🚀

