# Integrated Chatbot - Quick Start (5 Minutes)

Get the Express + MongoDB chatbot running in your Streamlit app.

## Prerequisites

✅ **Already installed?**
- Python 3.11+ 
- Streamlit
- MongoDB or MongoDB Atlas account

❌ **Need to install?**
- Node.js 16+: https://nodejs.org/

## Quick Setup

### Windows Users

1. **Run setup script:**
   ```batch
   setup_integrated_chatbot.bat
   ```
   
2. **In Terminal 1 - Start MongoDB:**
   ```batch
   mongod
   ```

3. **In Terminal 2 - Start Backend:**
   ```batch
   cd chatbot-service
   npm start
   ```

4. **In Terminal 3 - Start Frontend:**
   ```batch
   .venv\Scripts\activate.bat
   streamlit run app.py
   ```

### macOS/Linux Users

1. **Run commands:**
   ```bash
   # Install dependencies
   cd chatbot-service
   npm install
   cd ..
   
   # Create backend config
   cd chatbot-service
   cp .env.example .env
   cd ..
   ```

2. **Start services:**
   ```bash
   # Terminal 1: MongoDB
   mongod
   
   # Terminal 2: Backend
   cd chatbot-service
   npm start
   
   # Terminal 3: Frontend
   source .venv/bin/activate
   streamlit run app.py
   ```

## Verify It Works

✅ **Check Backend:**
```bash
curl http://localhost:5001/health
# Should return: {"status":"ok","service":"hemophilia-chatbot"}
```

✅ **Open Streamlit:**
- Go to http://localhost:8501
- Click "💬 Quick Chat" tab
- Should see "✅ Connected to chatbot service"

✅ **Try a Question:**
- Ask: "What is hemophilia?"
- Should get instant answer from knowledge base

## MongoDB Options

### Option 1: Local MongoDB (Default)
- File created: `chatbot-service/.env`
- Already configured for local MongoDB
- Just run `mongod` before starting backend

### Option 2: MongoDB Atlas (Cloud)
1. Go to https://www.mongodb.com/cloud/atlas
2. Create free cluster
3. Get connection string: `mongodb+srv://user:pass@cluster.mongodb.net/hemophilia-chatbot`
4. Edit `chatbot-service/.env`:
   ```
   MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/hemophilia-chatbot
   ```
5. Restart backend: `npm start`

## Troubleshooting

### "Cannot connect to service"
- ✅ Backend running? Check Terminal 2 for `mongod` message
- ✅ Port 5001 free? `netstat -ano | findstr :5001` (Windows)
- ✅ MongoDB running? Terminal 1 should show `mongod` waiting

### "MongoDB connection error"
- ✅ Local MongoDB? Run `mongod` in separate terminal
- ✅ MongoDB Atlas? Update `.env` with correct connection string
- ✅ Connection string copied correctly? Check for special characters

### "No knowledge base responses"
- ✅ Backend logs show errors? Check npm start output
- ✅ Try simple question: "Tell me about hemophilia"
- ✅ Restart backend: Stop and `npm start` again

## File Structure

```
├── chatbot-service/          ← Express backend
│   └── .env                  ← MongoDB configuration
├── chatbot_service_client.py ← Python bridge
├── app.py                    ← Streamlit app
└── INTEGRATED_CHATBOT_SETUP.md ← Full guide
```

## What's Included

✅ **Knowledge Base** - 9 medical topics
✅ **MongoDB Persistence** - All chats saved
✅ **REST API** - 7 endpoints
✅ **Session Management** - Unique conversation IDs
✅ **Search & Analytics** - Query conversations
✅ **Zero API Keys** - Works offline

## Next Steps

1. Run the setup script (or manual commands above)
2. Start all 3 services (MongoDB, Backend, Frontend)
3. Go to http://localhost:8501
4. Click "💬 Quick Chat"
5. Ask your first question!

## Production Deployment

See: `INTEGRATED_CHATBOT_SETUP.md` → "Deployment Options"

---

**Questions?** See the full guide: `INTEGRATED_CHATBOT_SETUP.md`
