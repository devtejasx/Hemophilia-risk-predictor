# Integrated Hemophilia Chatbot Setup Guide

Complete guide to run the integrated Express backend + Streamlit frontend with MongoDB persistence.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit App (Python)                                 │
│  - User Interface (QT Chat Page)                        │
│  - Patient Prediction & Analysis                        │
│  - Dashboard & Monitoring                               │
└──────────────────────┬──────────────────────────────────┘
                       │ (HTTP Requests)
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Express Backend (Node.js) - Port 5001                  │
│  - Knowledge Base Service                               │
│  - API Endpoints                                        │
│  - Session Management                                   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  MongoDB                                                 │
│  - Conversation History                                 │
│  - Chat Sessions                                        │
│  - Metadata & Analytics                                 │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### Required Software

- ✅ **Python 3.11+** - Already installed for Streamlit app
- ✅ **Node.js 16+** - For Express backend
  - Download: https://nodejs.org/
  - Verify: `node --version` and `npm --version`

- ✅ **MongoDB** - Local or Cloud
  - **Option A: Local Installation**
    - Windows: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-windows/
    - macOS: `brew install mongodb-community`
    - Verify: `mongod --version`
  
  - **Option B: MongoDB Atlas (Cloud)**
    - Sign up: https://www.mongodb.com/cloud/atlas
    - Create cluster and get connection string

### Check Prerequisites

```bash
# Check Node.js
node --version  # Should be v16+
npm --version   # Should be 7+

# Check MongoDB (if local)
mongod --version  # Should show version

# Check Python
python --version  # Should be 3.11+
```

## Installation Steps

### Step 1: Start MongoDB

#### If using Local MongoDB:

**Windows (PowerShell):**
```powershell
# Start MongoDB service
net start MongoDB

# Or run mongod directly
mongod
```

**macOS/Linux:**
```bash
brew services start mongodb-community
# Or
mongod
```

#### If using MongoDB Atlas (Cloud):
- Create cluster at https://www.mongodb.com/cloud/atlas
- Get connection string and note it for Step 3

**Verify MongoDB is running:**
```bash
mongo
# Should connect successfully
exit
```

---

### Step 2: Set Up Express Backend

```bash
# Navigate to backend directory
cd chatbot-service

# Install dependencies
npm install

# Create .env file with configuration
cp .env.example .env
```

**Edit `.env` file:**

If using **Local MongoDB:**
```
PORT=5001
NODE_ENV=development
MONGODB_URI=mongodb://localhost:27017/hemophilia-chatbot
CORS_ORIGIN=http://localhost:8501
```

If using **MongoDB Atlas (Cloud):**
```
PORT=5001
NODE_ENV=development
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/hemophilia-chatbot
CORS_ORIGIN=http://localhost:8501
```

**Start the backend:**

```bash
# Terminal 1: Start Express server
npm start

# You should see:
# ✅ MongoDB connected successfully
# 🚀 Chatbot Service started on port 5001
```

**Verify backend is running:**
```bash
# In another terminal, test the health endpoint
curl http://localhost:5001/health

# Should return:
# {"status":"ok","service":"hemophilia-chatbot",...}
```

---

### Step 3: Set Up Streamlit Frontend

```bash
# Return to main project directory
cd ..

# Ensure Python environment is active
# Windows:
.venv\Scripts\Activate.ps1

# macOS/Linux:
source .venv/bin/activate

# Install/update requirements
pip install -r requirements.txt

# Ensure requests library is installed
pip install requests
```

**Check if chatbot_service_client is properly placed:**
```bash
# File should exist:
ls chatbot_service_client.py
# or
dir chatbot_service_client.py  # Windows
```

---

### Step 4: Run the Integrated System

**Terminal 1: Express Backend (keep running)**
```bash
cd chatbot-service
npm start
```

**Terminal 2: Streamlit Frontend**
```bash
# Ensure .venv is activated
streamlit run app.py

# Should open browser to http://localhost:8501
```

---

## Verification Checklist

✅ **Before you start, verify:**

- [ ] Node.js installed: `node --version` returns v16+
- [ ] MongoDB running: `mongod` shows connection ready OR MongoDB Atlas cluster is accessible
- [ ] Backend dependencies installed: `npm install` completed in `chatbot-service/`
- [ ] `.env` file created with correct MongoDB URI
- [ ] Streamlit dependencies installed: `pip install -r requirements.txt`
- [ ] `chatbot_service_client.py` exists in main directory

✅ **After starting services:**

- [ ] Backend running: `curl http://localhost:5001/health` returns 200
- [ ] Streamlit running: Browser opened at `http://localhost:8501`
- [ ] Quick Chat page shows "Connected to chatbot service ✅"
- [ ] Can start a new conversation
- [ ] Can send a message and receive a response
- [ ] Response metadata shows category and match type

---

## Usage Guide

### In Streamlit App

1. **Navigate to "💬 Quick Chat" page**
   - Green button indicates service is connected
   - Red warning indicates service is not running

2. **Start Conversation**
   - Click "Retry Connection" if needed
   - Greeting message appears

3. **Send Messages**
   - Type your question in the chat input
   - Service searches knowledge base for matching answer
   - Response appears with category and match information

4. **Available Topics**
   - Click topic buttons to ask pre-formatted questions
   - Responses are automatically fetched and displayed
   - All conversations are saved to MongoDB

5. **Clear Chat**
   - Click "🔄 Clear" button to reset conversation
   - All previous messages remain in database
   - Starts fresh greeting

### Database Features

- **Persistent Storage** - All conversations saved to MongoDB
- **Session Management** - Each user gets unique sessionId
- **Search & Analytics** - View all conversations in admin panel
- **Message Metadata** - Each message tagged with category and timestamp

---

## Troubleshooting

### Issue: "Cannot connect to chatbot service"

**Solution:**
```bash
# Verify backend is running on correct port
curl http://localhost:5001/health

# If failed, check:
# 1. Is backend still running? (check Terminal 1)
# 2. Is port 5001 already in use?

# Find process using port 5001:
# Windows:
netstat -ano | findstr :5001

# macOS/Linux:
lsof -i :5001

# Kill process if needed:
# Windows:
taskkill /PID <process_id> /F

# macOS/Linux:
kill -9 <process_id>
```

### Issue: "MongoDB connection error"

**Solution:**
```bash
# 1. Verify MongoDB is running
mongod
# Should show "waiting for connections"

# 2. Check connection string in .env
# Should be: mongodb://localhost:27017/hemophilia-chatbot

# 3. If using MongoDB Atlas, verify:
# - Cluster is running
# - IP whitelist includes your computer
# - Username/password are correct
# - Connection string has correct database name
```

### Issue: "No responses from chatbot"

**Solution:**
1. Check backend logs for errors: `npm start` output
2. Verify knowledge base is loaded: Check `knowledge/knowledgeBase.js`
3. Test API directly:
```bash
curl -X POST http://localhost:5001/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "test_123",
    "message": "What is hemophilia?"
  }'
```

### Issue: "Port already in use"

**Solution:**
```bash
# For port 5001 (backend):

# Windows PowerShell:
Get-Process -Id (Get-NetTCPConnection -LocalPort 5001).OwningProcess

# Kill it:
Stop-Process -Id <PID> -Force

# macOS/Linux:
lsof -i :5001
kill -9 <PID>

# Then restart: cd chatbot-service && npm start
```

---

## Advanced Configuration

### Custom Knowledge Base

Edit `chatbot-service/knowledge/knowledgeBase.js` to add new topics:

```javascript
const KNOWLEDGE_BASE = {
  newtopic: {
    keywords: ['keyword1', 'keyword2', 'keyword3'],
    response: `**Your Topic**
    
Your detailed response here...
`
  }
}
```

Restart backend: `npm start`

### MongoDB Atlas Setup

1. Create account: https://www.mongodb.com/cloud/atlas
2. Create cluster (free tier available)
3. Add user credentials
4. Whitelist IP address
5. Get connection string
6. Add to `.env`:
```
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/hemophilia-chatbot
```

### Rate Limiting & Production

For production deployment:
```bash
# Install production dependencies
npm install express-rate-limit helmet

# Update server.js to add:
app.use(rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100
}));
```

---

## Deployment Options

### Deploy Backend (Express)

**Render.com (Recommended)**
```bash
# 1. Push to GitHub
git add .
git commit -m "Add chatbot service"
git push

# 2. Create new service on Render
# - Connect GitHub repo
# - Set build command: cd chatbot-service && npm install
# - Set start command: npm start
# - Add environment variables from .env
# - Deploy

# 3. Get public URL (e.g., https://chatbot-service-xyz.onrender.com)
```

**Heroku**
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create hemophilia-chatbot-service

# Set MongoDB URI
heroku config:set MONGODB_URI=mongodb+srv://...

# Deploy
git push heroku main
```

### Update Streamlit for Deployed Backend

When backend is deployed, update connection URL:

**In app.py or environment:**
```python
# Instead of:
client = init_chatbot_service()

# Use:
backend_url = os.getenv("CHATBOT_SERVICE_URL", "http://localhost:5001")
client = init_chatbot_service(backend_url)
```

Or set environment variable:
```bash
export CHATBOT_SERVICE_URL=https://chatbot-service-xyz.onrender.com
```

---

## File Structure

```
Capstone/
├── chatbot-service/                 # Express Backend
│   ├── server.js                   # Main server
│   ├── package.json                # Dependencies
│   ├── .env                        # Configuration (git ignored)
│   ├── .env.example                # Template
│   ├── config/
│   │   └── db.js                   # MongoDB connection
│   ├── models/
│   │   └── Conversation.js         # Chat schema
│   ├── controllers/
│   │   └── chatController.js       # API handlers
│   ├── routes/
│   │   └── chatRoutes.js           # API endpoints
│   ├── knowledge/
│   │   └── knowledgeBase.js        # Medical knowledge
│   └── README.md                   # Backend docs
│
├── app.py                          # Streamlit main app
├── chatbot_service_client.py       # Python client library
├── simple_chatbot.py               # Legacy chatbot (still available)
├── requirements.txt                # Python dependencies
└── ...other files...
```

---

## Support & Documentation

- **Backend Docs**: `chatbot-service/README.md`
- **API Reference**: See POST `/api/chat/message` examples above
- **Knowledge Base Topics**: 9 categories in `knowledge/knowledgeBase.js`
- **Database Schema**: See `models/Conversation.js`

## Quick Commands Reference

```bash
# Start MongoDB
mongod

# Start Express Backend
cd chatbot-service
npm start

# Start Streamlit Frontend (in new terminal)
streamlit run app.py

# Test backend health
curl http://localhost:5001/health

# Test chat endpoint
curl -X POST http://localhost:5001/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"test","message":"What is hemophilia?"}'

# View MongoDB collections
mongo
# > show dbs
# > use hemophilia-chatbot
# > db.conversations.find()
# > exit
```

---

## Next Steps

1. ✅ All systems running?
   - Backend shows "✅ MongoDB connected"
   - Streamlit shows "✅ Connected to chatbot service"

2. Try asking questions in Quick Chat:
   - "What is hemophilia?"
   - "What are inhibitors?"
   - "Can I exercise?"

3. Check MongoDB to see conversations:
   ```bash
   mongo hemophilia-chatbot
   db.conversations.findOne()
   ```

4. (Optional) Deploy to production:
   - Backend to Render or Heroku
   - Update Streamlit with production URL

---

**Happy chatting! 🤖💬**
