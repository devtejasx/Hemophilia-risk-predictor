# MongoDB Quick Reference

TL;DR guide to add MongoDB persistence to your chatbot.

## Current State
- ✅ Chatbot works with in-memory storage
- ❌ Conversations disappear when server restarts
- ⏱️ Takes ~5-10 minutes to set up MongoDB

---

## Option 1: Local MongoDB (Fastest)

### Windows

**1. Download & Install (2 min)**
```
Go to: https://www.mongodb.com/try/download/community
Download MSI → Run → Next → Next → Install
```

**2. Start MongoDB**
```powershell
mongod
# or as service:
net start MongoDB
```

**3. Update config** (File: `chatbot-service/.env`)
```
MONGODB_URI=mongodb://localhost:27017/hemophilia-chatbot
```

**4. Restart backend**
```bash
cd chatbot-service
npm start
# Should show: ✅ MongoDB connected successfully
```

**5. Done!** Test in Streamlit app

---

### macOS

```bash
# Install
brew install mongodb-community

# Start
mongod

# Or start as service:
brew services start mongodb-community

# Rest same as Windows (steps 3-5)
```

---

### Linux (Ubuntu/Debian)

```bash
# Install
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update && sudo apt-get install -y mongodb-org

# Start
sudo systemctl start mongod

# Rest same as Windows (steps 3-5)
```

---

## Option 2: MongoDB Atlas (Cloud, No Installation)

### Setup (3 minutes)

**1. Create Account**
```
https://www.mongodb.com/cloud/atlas → Sign up
```

**2. Create Cluster**
```
Create Cluster → AWS → M0 (FREE) → Create
Wait 2-3 minutes...
```

**3. Add User**
```
Database Access → Add New Database User
Username: hemophilia_user
Password: YourStrongPassword123
Role: Atlas Admin → Add User
```

**4. Whitelist IP**
```
Network Access → Add Current IP → Confirm
```

**5. Get Connection String**
```
Clusters → Connect → Connect your application
Copy: mongodb+srv://hemophilia_user:PASSWORD@cluster0.xxxxx.mongodb.net/hemophilia-chatbot
```

**6. Update `.env`**
```
MONGODB_URI=mongodb+srv://hemophilia_user:PASSWORD@cluster0.xxxxx.mongodb.net/hemophilia-chatbot
```

**7. Restart backend**
```bash
cd chatbot-service
npm start
```

**Done!** Automatic cloud backup included.

---

## Verify It Works

### Check Connection

```bash
# Option 1: Check backend logs
cd chatbot-service
npm start
# Should show: ✅ MongoDB connected successfully

# Option 2: Query MongoDB directly
mongosh hemophilia-chatbot
db.conversations.find()
exit
```

### Test Full Flow

1. Open http://localhost:8502
2. Click "💬 Quick Chat"
3. Ask: "What is hemophilia?"
4. Check MongoDB:

```bash
mongosh hemophilia-chatbot
db.conversations.count()  # Should show 1+ conversations
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "MongoDB not found" | Not installed. Download from mongodb.com |
| "Connection refused" | MongoDB not running. Run `mongod` or `net start MongoDB` |
| "Authentication failed" | Wrong password. Check connection string |
| "Connection timeout" | For Atlas: Check IP whitelisted in Network Access |

---

## Quick Commands

```bash
# Start MongoDB
mongod                    # Terminal 1 - keeps running

# Start Backend
cd chatbot-service
npm start                 # Terminal 2 - keeps running

# Start Streamlit
streamlit run app.py      # Terminal 3 - keeps running

# Query your data
mongosh hemophilia-chatbot
db.conversations.findOne()
db.conversations.countDocuments()
exit

# Backup your data (Local MongoDB)
mongodump --db hemophilia-chatbot --out ./backup/
```

---

## Comparison

| Feature | Local MongoDB | MongoDB Atlas | In-Memory |
|---------|---------------|---------------|-----------|
| **Setup time** | 5 min | 3 min | 0 min |
| **Installation** | Required | None | None |
| **Persistence** | ✅ Yes | ✅ Yes | ❌ No |
| **Backup** | Manual | Automatic | N/A |
| **Best for** | Development | Production | Testing |
| **Cost** | Free | Free tier | Free |

---

## Next Steps

1. Choose: **Local MongoDB** or **MongoDB Atlas**
2. Follow quick setup above
3. Run: `cd chatbot-service && npm start`
4. Test in Streamlit app
5. Watch conversations persist! 🎉

---

**Full Setup Guide:** See `MONGODB_SETUP.md` for detailed instructions
