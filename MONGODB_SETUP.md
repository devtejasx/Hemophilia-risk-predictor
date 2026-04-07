# MongoDB Setup Guide for Hemophilia Chatbot

Complete instructions to add persistent MongoDB storage to your chatbot service.

## Current vs. With MongoDB

### Current Setup (In-Memory)
✅ **Advantages:**
- Works immediately, no installation needed
- Fast responses
- Perfect for testing

❌ **Limitations:**
- Conversations lost when server restarts
- Only current session stored
- Can't search historical conversations

### With MongoDB
✅ **Advantages:**
- Persistent storage - conversations saved permanently
- Search & analytics on all past chats
- Can scale to many users
- Professional backup/recovery

---

## Option A: MongoDB Community Edition (Local)

### Step 1: Download & Install

**Windows:**

1. Go to: https://www.mongodb.com/try/download/community
2. Select:
   - **Version:** Latest (e.g., 7.0.0)
   - **OS:** Windows
   - **Package:** MSI
3. Click **Download**
4. Run the installer (.msi file)
5. Choose **Complete** installation
6. Check box: **Install MongoDB Compass** (GUI tool, optional but helpful)
7. Click **Install**
8. Installation completes in ~5 minutes

**macOS:**
```bash
# Using Homebrew
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB as service
brew services start mongodb-community
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
```

### Step 2: Verify Installation

**Windows (PowerShell):**
```powershell
mongod --version
# Should output: db version v7.0.0 (or similar)

# Check if MongoDB is running, start if needed:
net start MongoDB
# or
mongod
```

**macOS/Linux:**
```bash
mongod --version
# Should output version info

# Check if running
brew services list  # macOS
```

### Step 3: Update Chatbot Config

Edit `chatbot-service/.env`:

```env
PORT=5001
NODE_ENV=development
MONGODB_URI=mongodb://localhost:27017/hemophilia-chatbot
CORS_ORIGIN=http://localhost:8501
```

**Verify connection string is correct:**
- `localhost:27017` = default MongoDB port
- `hemophilia-chatbot` = database name (will auto-create)

### Step 4: Restart Backend

**Stop current backend:**
- Find the terminal running `npm start`
- Press `Ctrl+C`

**Start with MongoDB:**
```bash
cd chatbot-service
npm start
```

**Should see:**
```
✅ MongoDB connected successfully
   Database: hemophilia-chatbot
   Host: localhost
🚀 Chatbot Service started on port 5001
```

### Step 5: Verify Data Persistence

Test in another terminal:

```bash
# Start MongoDB shell
mongosh  # or 'mongo' in older versions

# In mongo shell:
use hemophilia-chatbot
db.conversations.find()  # See all conversations

# Should show your chat history!
exit
```

---

## Option B: MongoDB Atlas (Cloud)

Better for production, free tier available.

### Step 1: Create Free Account

1. Go to: https://www.mongodb.com/cloud/atlas
2. Click **Create Free Account**
3. Sign up with email
4. Verify email
5. Create free organization and project

### Step 2: Create Cluster

1. Click **Create Cluster**
2. Choose:
   - **Provider:** AWS/Google Cloud (any)
   - **Region:** Closest to you
   - **Cluster Tier:** M0 (FREE - perfect for testing)
3. Click **Create Cluster**
   - Takes 2-3 minutes to initialize

### Step 3: Set Up Access

**Create Database User:**
1. Go to **Database Access** (left menu)
2. Click **Add New Database User**
3. Choose **Password** authentication
4. Username: `hemophilia_user` (or any name)
5. Password: Create strong password (save it!)
6. Choose **Built-in Role:** `Atlas Admin`
7. Click **Add User**

**Whitelist Your IP:**
1. Go to **Network Access** (left menu)
2. Click **Add IP Address**
3. Click **Add Current IP Address** (auto-detected)
4. Description: "My Computer"
5. Click **Confirm**

### Step 4: Get Connection String

1. Go back to **Clusters** page
2. Click **Connect** button on your cluster
3. Choose **Connect your application**
4. Copy the connection string
   - Looks like: `mongodb+srv://user:password@cluster.mongodb.net/dbname`

**Important:** Replace `<password>` with your actual password (from Step 3)

### Step 5: Update Config

Edit `chatbot-service/.env`:

```env
PORT=5001
NODE_ENV=development
MONGODB_URI=mongodb+srv://hemophilia_user:YOUR_PASSWORD@cluster0.mongodb.net/hemophilia-chatbot
CORS_ORIGIN=http://localhost:8501
```

**Example (with real values):**
```env
MONGODB_URI=mongodb+srv://hemophilia_user:MySecurePass123@cluster0.abcdef.mongodb.net/hemophilia-chatbot
```

### Step 6: Restart Backend

```bash
cd chatbot-service
npm start
```

**Should see:**
```
✅ MongoDB connected successfully
   Database: hemophilia-chatbot
   Host: cluster0.abcdef.mongodb.net
🚀 Chatbot Service started on port 5001
```

### Step 7: Verify in Atlas Dashboard

1. Go back to MongoDB Atlas
2. Click **Collections** on your cluster
3. Should see `hemophilia-chatbot` database
4. Click it → `conversations` collection
5. Should see your chat messages!

---

## Troubleshooting

### Issue: "MongoDB connection refused"

**Solution:**
```bash
# 1. Verify MongoDB is running:
mongod --version

# 2. Start MongoDB (Windows):
net start MongoDB
# or
mongod

# 3. macOS:
brew services start mongodb-community

# 4. Linux:
sudo systemctl start mongod

# 5. Check it's listening:
netstat -ano | findstr :27017  # Windows
lsof -i :27017  # macOS/Linux
```

### Issue: "Authentication failed" (Atlas only)

**Solution:**
- Check `.env` has correct password
- Special characters in password need URL encoding:
  - `@` → `%40`
  - `#` → `%23`
  - `:` → `%3A`
  - Example: `pass@word#123` → `pass%40word%23123`

### Issue: "Connection timeout"

**For Atlas:**
1. Check IP is whitelisted in **Network Access**
2. Add `0.0.0.0/0` to allow all IPs (for testing only)
3. Check internet connection

### Issue: "Database not found"

**Solution:**
The database auto-creates on first write. This is normal. Just send a chat message and it will be created.

---

## Switching Back to In-Memory

If you want to revert to in-memory storage (no MongoDB needed):

**Edit `.env`:**
```env
MONGODB_URI=invalid://url  # Invalid connection string
```

Or just comment it out. Backend will detect MongoDB is unavailable and use in-memory storage automatically.

---

## Monitoring Your Data

### With Local MongoDB

**View conversations:**
```bash
mongosh hemophilia-chatbot
db.conversations.find().pretty()
```

**Count conversations:**
```bash
db.conversations.countDocuments()
```

**Search by keyword:**
```bash
db.conversations.find(
  { "messages.content": { $regex: "hemophilia", $options: "i" } }
).pretty()
```

**Delete old conversations (older than 30 days):**
```bash
db.conversations.deleteMany({
  updatedAt: { $lt: new Date(Date.now() - 30*24*60*60*1000) }
})
```

### With MongoDB Atlas

1. Go to your cluster
2. Click **Collections**
3. Browse your data in web interface
4. Or use MongoDB Compass app (installed with Windows MongoDB)

---

## Performance Tips

### Create Index for Faster Searches

```bash
# With MongoDB running:
mongosh hemophilia-chatbot

# Create index on sessionId (for quick lookups):
db.conversations.createIndex({ sessionId: 1 })

# Create index on createdAt (for time-based queries):
db.conversations.createIndex({ createdAt: -1 })

# View all indexes:
db.conversations.getIndexes()
```

### Backup Your Data

**Local MongoDB:**
```bash
# Backup to folder:
mongodump --db hemophilia-chatbot --out ./backups/

# Restore from backup:
mongorestore --db hemophilia-chatbot ./backups/hemophilia-chatbot
```

**MongoDB Atlas:**
1. Go to **Backup** section in cluster settings
2. Click **Create On-Demand Backup**
3. Backups are automatic (daily)
4. Can restore with one click

---

## Quick Decision Guide

### Use Local MongoDB if:
- ✅ Testing/development
- ✅ Want full control
- ✅ Offline access needed
- ✅ Have server infrastructure

### Use MongoDB Atlas if:
- ✅ Production deployment
- ✅ Want managed service
- ✅ Don't want to manage server
- ✅ Need automatic backups
- ✅ Planning to scale

### Keep In-Memory if:
- ✅ Just testing chatbot
- ✅ Don't need persistence
- ✅ Single-user/session testing
- ✅ Temporary setup

---

## After Installing MongoDB

### Restart Backend

```bash
cd chatbot-service
npm start

# Should show:
# ✅ MongoDB connected successfully
```

### Test with Chat

1. Go to http://localhost:8502
2. Click **Quick Chat**
3. Ask a question
4. Check MongoDB for saved conversation:

```bash
mongosh hemophilia-chatbot
db.conversations.findOne()
```

---

## Integration Summary

Your chatbot backend automatically:
1. ✅ Detects MongoDB availability at startup
2. ✅ Uses MongoDB if available
3. ✅ Falls back to in-memory if not available
4. ✅ Reports storage type in API responses (`"storage": "MongoDB"` or `"storage": "In-Memory"`)

**No code changes needed!** Just install MongoDB and restart the backend.

---

## Next Steps

1. Choose: Local MongoDB or MongoDB Atlas
2. Follow installation steps for your choice
3. Update `.env` with connection string
4. Restart backend: `cd chatbot-service && npm start`
5. Test in Streamlit app
6. Verify data in MongoDB

**Questions?** Check troubleshooting section or run:
```bash
cd chatbot-service
npm start
# Backend logs will show any connection issues
```
