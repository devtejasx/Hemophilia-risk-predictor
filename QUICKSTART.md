# ⚡ Quick Start Guide

## 1️⃣ Installation (2 minutes)

```bash
# Install all dependencies
pip install -r requirements.txt
```

## 2️⃣ Configure OpenAI API (1 minute)

Create `.env` file in project root:
```env
OPENAI_API_KEY=sk-your_actual_key_here
```

Get key: https://platform.openai.com/api-keys

## 3️⃣ Launch App (30 seconds)

```bash
streamlit run app.py
```

Open browser to: `http://localhost:8501`

---

## 🎯 First Steps

### Step 1: Create a Patient
1. Click **📋 Patient Form** tab
2. Fill in patient details (name, age, severity, mutation, etc.)
3. Click **🚀 Run Advanced Risk Analysis**
4. View results instantly

### Step 2: Chat with AI Doctor
1. Click **🤖 Chatbot** tab
2. Ask anything: "What's the inhibitor risk?" "What treatment should we use?"
3. Get GPT-4 powered clinical recommendations
4. Save important notes with doctor

### Step 3: Review Analytics  
1. Click **🏥 Dashboard** tab
2. Explore patient directory
3. Add clinical notes
4. View population analytics

---

## 📊 Key Features

| Feature | Location | What It Does |
|---------|----------|-------------|
| 🤖 AI Chatbot | Chatbot Tab | Real-time clinical Q&A powered by GPT-4 |
| 📊 Risk Prediction | Patient Form | ML-based inhibitor risk calculation |
| 🏥 Doctor Dashboard | Dashboard Tab | Manage patients, notes, analytics |
| 💾 Database | Background | Auto-saves all data (SQLite) |
| 📈 Analytics | Dashboard → Analytics | Population trends and statistics |
| 🔍 Search | Dashboard → Search | Find patients by name/risk/mutation |

---

## 💬 Chatbot Examples

**"Generate clinical recommendations"**
→ Complete treatment plan based on patient profile

**"What's the inhibitor risk?"**
→ Analysis of mutation, severity, exposure factors

**"Should treatment be adjusted?"**
→ Recommendations based on adherence and outcomes

**"What monitoring frequency?"**
→ Specific monitoring schedule for this patient

---

## 🔑 Important

⚠️ **Never share your OpenAI API key!**
- Keep `.env` file private
- Don't commit to GitHub
- Add to `.gitignore`

💰 **Monitor costs**
- Check usage: https://platform.openai.com/account/usage
- GPT-4 costs more than GPT-3.5

---

## ❓ Need Help?

1. **Check README.md** - Comprehensive documentation
2. **Check .env.example** - Environment template
3. **Review database.py** - Database functions
4. **Review gpt_chatbot.py** - AI integration

---

## ✅ You're Ready!

Start by creating a patient and asking the AI doctor a question. Enjoy!
