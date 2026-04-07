# 🔑 Authentication Quick Start Guide

**Get your Hemophilia AI platform authentication running in 5 minutes**

---

## ⚡ 60-Second Setup

### 1. Install New Packages
```bash
cd backend
pip install -r requirements.txt
```

❌ Missing packages: `bcrypt`, `PyJWT`, `passlib` (now included)

### 2. Start Backend
```bash
python -m uvicorn main:app --reload
```

✅ Should see:
```
✅ User authentication database initialized
✅ Admin user verified
  Username: admin
  Password: Admin@2026
```

### 3. Test in Browser
```
http://localhost:8000/docs
```

✅ Click "Authorize" button (top right)  
✅ Enter credentials:
- Username: `admin`
- Password: `Admin@2026`  
✅ Click "Try it out" on any endpoint

---

## 🎯 Quick Tests

### Test 1: Login
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "Admin@2026"
  }'
```

✅ Expected: Returns access_token, refresh_token, user info

### Test 2: Get Profile
```bash
# First, copy access_token from login response, then:

curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

✅ Expected: Shows your admin profile

### Test 3: Make Prediction (Requires Auth)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 12,
    "dose": 2000,
    "exposure": 90,
    "severity": "Moderate",
    "mutation": "Intron 22"
  }'
```

✅ Expected: Risk prediction with score

### Test 4: Register New User
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "patient_john",
    "email": "john@example.com",
    "password": "SecurePass123!",
    "full_name": "John Doe",
    "role": "patient"
  }'
```

✅ Expected: New account created, automatic login

---

## 👥 User Roles Explained

### ADMIN (Full Access)
- Default username: `admin`
- Default password: `Admin@2026`
- Can: Delete patients, export data, manage users
- Use for: System setup, user management

### DOCTOR
- Can: View/edit patients, analytics, predictions
- Use for: Hospital staff, clinicians

### PATIENT
- Can: View own data, make predictions, chat with AI
- Use for: Patient apps

---

## 📝 Common Tasks

### Create Doctor Account
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "dr_smith",
    "email": "smith@hospital.com",
    "password": "DoctorPass123!",
    "full_name": "Dr. Jane Smith",
    "role": "doctor"
  }'
```

### Change Your Password
```bash
curl -X POST http://localhost:8000/auth/change-password \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "current_password": "Admin@2026",
    "new_password": "NewAdminPass123!"
  }'
```

### Get New Token (When Expired)
```bash
curl -X POST http://localhost:8000/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "YOUR_REFRESH_TOKEN"
  }'
```

---

## 🔒 Security Checklist

- [ ] Changed admin password from default
- [ ] Set unique JWT_SECRET_KEY in .env
- [ ] HTTPS enabled in production
- [ ] CORS_ORIGINS updated for your domain
- [ ] Regular database backups

---

## 📊 What's Protected Now?

| Endpoint | Before | After |
|----------|--------|-------|
| POST /predict | Public | 🔒 Token Required |
| POST /chat | Public | 🔒 Token Required |
| GET /patients | Public | 🔒 Doctor/Admin Only |
| POST /patients | Public | 🔒 Doctor/Admin Only |
| GET /analytics | Public | 🔒 Doctor/Admin Only |

---

## 🆘 If Something Breaks

### "Port 8000 in use"
```bash
# Kill process
lsof -i :8000
kill -9 <PID>
```

### "Database error"
```bash
# Reset database
rm hemophilia_users.db
python -m uvicorn main:app --reload  # Will recreate
```

### "Invalid token"
- Login again to get fresh token
- Token expires after 30 minutes
- Use refresh endpoint to get new token without logging in

---

## 📈 Next Steps

1. **Change admin password** ⚠️ IMPORTANT
2. **Create doctor accounts** for your staff
3. **Set CORS properly** for your frontend
4. **Update JWT_SECRET_KEY** in production
5. **Integrate with Streamlit** (backend_client.py ready!)

---

## 🎓 Learn More

See: [AUTHENTICATION_IMPLEMENTATION.md](AUTHENTICATION_IMPLEMENTATION.md)

For complete details on:
- JWT tokens
- Role-based access
- Protected endpoints
- User management
- Security best practices

---

**Status: ✅ Authentication System Active**

Your API is now secure and ready for production! 🎉

