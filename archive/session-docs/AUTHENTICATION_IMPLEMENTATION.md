# 🔐 FastAPI Authentication System Implementation

**Complete JWT-based Authentication with Role-Based Access Control (RBAC)**

---

## 📋 Overview

This document describes the comprehensive authentication and authorization system implemented in the FastAPI backend for the Hemophilia AI Platform.

### Key Features

✅ **JWT Authentication** - Industry-standard bearer token authentication  
✅ **Bcrypt Password Hashing** - Secure password storage with bcrypt  
✅ **Role-Based Access Control** - Three distinct roles with granular permissions  
✅ **Token Refresh** - Access token renewal without re-authentication  
✅ **User Registration** - Self-service patient and doctor registration  
✅ **Admin Account** - Pre-configured admin user for system management  
✅ **Audit Logging** - Tracking of security-relevant events  
✅ **Route Protection** - Automatic authorization validation on protected endpoints  

---

## 🎯 System Architecture

### Authentication Layers

```
Request with Token
        ↓
HTTPBearer Extraction
        ↓
JWT Validation & Decoding
        ↓
User Identity Verification
        ↓
Role Check (if required)
        ↓
Permission Validation
        ↓
Endpoint Execution
```

### Role Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                      ADMIN                              │
│  • Full system access                                   │
│  • User management                                      │
│  • Export analytics (ZIP dumps)                         │
│  • System configuration                                │
└─────────────────────────────────────────────────────────┘
                          ↑
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────────────────┐           ┌──────────────────────┐
│    DOCTOR         │           │     PATIENT          │
│                   │           │                      │
│ • View patients   │           │ • View own data      │
│ • Edit patients   │           │ • Make predictions   │
│ • Make predictions│           │ • Chat with AI       │
│ • View analytics  │           │ • Submit feedback    │
│ • Create reports  │           │                      │
└───────────────────┘           └──────────────────────┘
```

### Permission Matrix

| Action | Patient | Doctor | Admin |
|--------|---------|--------|-------|
| Read own profile | ✅ | ✅ | ✅ |
| Update own profile | ✅ | ✅ | ✅ |
| Change own password | ✅ | ✅ | ✅ |
| Make predictions | ✅ | ✅ | ✅ |
| Chat with AI | ✅ | ✅ | ✅ |
| List all patients | ❌ | ✅ | ✅ |
| Create patient | ❌ | ✅ | ✅ |
| Edit any patient | ❌ | ✅ | ✅ |
| Delete patient | ❌ | ❌ | ✅ |
| View analytics | ❌ | ✅ | ✅ |
| Export data | ❌ | ❌ | ✅ |
| Manage users | ❌ | ❌ | ✅ |

---

## 🗂️ File Structure

### New Authentication Files

```
backend/
├── auth.py                          # JWT, password hashing, token management
├── security.py                      # Dependency injection, route protection
├── users.py                         # User CRUD, authentication persistence
└── routers/
    └── auth.py                      # Authentication endpoints (/auth/*)
```

### Modified Files

```
backend/
├── main.py                          # Integrated auth router, JWT setup
├── models.py                        # +UserResponse, TokenResponse models
├── requirements.txt                 # +bcrypt, PyJWT, passlib
└── routers/
    ├── predict.py                   # Protected endpoints
    ├── chat.py                      # Protected endpoints
    ├── patients.py                  # Protected endpoints (role-based)
    └── analytics.py                 # Protected endpoints (role-based)
```

---

## 🔑 Core Components

### 1. JWT Token Management (`backend/auth.py`)

```python
# Token Creation
access_token, expiry = create_access_token(
    username="doctor1",
    role="doctor"
)

# Token Validation
payload = decode_token(token)
if payload is None:  # Token invalid or expired
    raise Unauthorized

# Token Refresh
new_token = refresh_access_token(refresh_token)
```

**Token Payloads:**

```json
{
  "sub": "username",
  "role": "doctor",
  "exp": 1704067200,
  "iat": 1704063600,
  "type": "access"
}
```

### 2. Password Security (`backend/auth.py`)

```python
# Hashing
password_hash = hash_password("SecurePass123!")  # Uses bcrypt

# Verification
is_correct = verify_password("SecurePass123!", hashed_password)
```

**Password Requirements:**
- Minimum 8 characters
- Should include uppercase, lowercase, numbers, special characters
- Never stored in plain text
- All passwords re-hashed with bcrypt

### 3. Route Protection (`backend/security.py`)

```python
# Require any authenticated user
@app.get("/endpoint")
async def endpoint(current_user: dict = Depends(get_current_user)):
    return {"user": current_user["username"]}

# Require doctor or admin role
@app.get("/protected")
async def doctor_only(current_user: dict = Depends(get_current_doctor)):
    pass

# Require admin role
@app.delete("/admin-only")
async def admin_only(current_user: dict = Depends(get_current_admin)):
    pass
```

### 4. User Database (`backend/users.py`)

SQLite-based user persistence with tables:

**users** table:
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    full_name TEXT,
    password_hash TEXT,
    role TEXT,  -- patient, doctor, admin
    is_active BOOLEAN,
    created_at TIMESTAMP,
    last_login TIMESTAMP
)
```

**user_sessions** table:
```sql
CREATE TABLE user_sessions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    refresh_token TEXT UNIQUE,
    created_at TIMESTAMP,
    expires_at TIMESTAMP,
    ip_address TEXT,
    user_agent TEXT
)
```

**user_audit_log** table:
```sql
CREATE TABLE user_audit_log (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    action TEXT,
    details TEXT,
    ip_address TEXT,
    created_at TIMESTAMP
)
```

---

## 🔌 Authentication Endpoints

### 1. **POST /auth/login** - User Login

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "doctor1",
    "password": "SecurePass123!"
  }'
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": 2,
    "username": "doctor1",
    "email": "doctor@hospital.com",
    "full_name": "Dr. Jane Smith",
    "role": "doctor",
    "created_at": "2026-04-01T12:00:00",
    "last_login": "2026-04-02T10:30:00",
    "is_active": true
  }
}
```

### 2. **POST /auth/register** - User Registration

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

**Response (201):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": 3,
    "username": "patient_john",
    "email": "john@example.com",
    "full_name": "John Doe",
    "role": "patient",
    "created_at": "2026-04-02T14:55:00",
    "last_login": null,
    "is_active": true
  }
}
```

**Registration Rules:**
- Username: 3-50 characters, unique
- Email: Valid email format, unique
- Password: Minimum 8 characters
- Role: "patient" or "doctor" (admin accounts created by admin only)

### 3. **POST /auth/refresh** - Token Refresh

```bash
curl -X POST http://localhost:8000/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5..."
  }'
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 4. **GET /auth/me** - Get Current User

```bash
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5..."
```

**Response (200):**
```json
{
  "id": 2,
  "username": "doctor1",
  "email": "doctor@hospital.com",
  "full_name": "Dr. Jane Smith",
  "role": "doctor",
  "created_at": "2026-04-01T12:00:00",
  "last_login": "2026-04-02T10:30:00",
  "is_active": true
}
```

### 5. **POST /auth/change-password** - Change Password

```bash
curl -X POST http://localhost:8000/auth/change-password \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5..." \
  -H "Content-Type: application/json" \
  -d '{
    "current_password": "OldPass123!",
    "new_password": "NewPass456!"
  }'
```

**Response (200):**
```json
{
  "message": "Password changed successfully"
}
```

### 6. **POST /auth/logout** - Logout

```bash
curl -X POST http://localhost:8000/auth/logout \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5..."
```

**Response (200):**
```json
{
  "message": "Logged out successfully. Please delete your tokens from the client."
}
```

---

## 🛡️ Protected Endpoints

### All protected endpoints require Bearer token

**Format:**
```
Authorization: Bearer <access_token>
```

**Example:**
```bash
curl http://localhost:8000/predict \
  -X POST \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5..." \
  -H "Content-Type: application/json" \
  -d '{"age": 12, "dose": 2000, ...}'
```

### Endpoint Protection Summary

| Endpoint | Method | Protection | Allowed Roles |
|----------|--------|-----------|---------------|
| /auth/login | POST | ❌ Public | Any |
| /auth/register | POST | ❌ Public | Any |
| /auth/refresh | POST | ❌ Public | Any |
| /auth/me | GET | ✅ Required | patient, doctor, admin |
| /auth/change-password | POST | ✅ Required | patient, doctor, admin |
| /predict | POST | ✅ Required | patient, doctor, admin |
| /chat | POST | ✅ Required | patient, doctor, admin |
| /patients | GET | ✅ Required | doctor, admin |
| /patients | POST | ✅ Required | doctor, admin |
| /patients/{id} | GET | ✅ Required | doctor, admin |
| /patients/{id} | PUT | ✅ Required | doctor, admin |
| /patients/{id} | DELETE | ✅ Required | admin |
| /analytics/dashboard | GET | ✅ Required | doctor, admin |
| /analytics/export | GET | ✅ Required | admin |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

New packages added:
- `bcrypt==4.1.1`
- `PyJWT==2.8.1`
- `passlib==1.7.4`

### 2. Start Backend

```bash
python -m uvicorn main:app --reload
```

Output:
```
✅ Database initialized
✅ User authentication database initialized
✅ Admin user verified

Default Admin:
  Username: admin
  Password: Admin@2026
```

### 3. Access Swagger UI

Navigate to: **http://localhost:8000/docs**

### 4. Test Login

In Swagger UI:
1. Click on `POST /auth/login`
2. Click "Try it out"
3. Enter credentials:
   ```json
   {
     "username": "admin",
     "password": "Admin@2026"
   }
   ```
4. Click "Execute"
5. Copy the `access_token` value

### 5. Authorize Swagger UI

1. Click the 🔒 "Authorize" button at top right
2. Paste token as: `Bearer <token_here>`
3. Click "Authorize" then "Close"
4. Now all endpoints will include your token automatically

### 6. Test Protected Endpoint

Now you can test any protected endpoint like:
- `GET /auth/me` - See your profile
- `GET /patients` - List patients
- etc.

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```env
# JWT Configuration
JWT_SECRET_KEY=your-super-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# OpenAI
OPENAI_API_KEY=sk-your-key

# CORS
CORS_ORIGINS=["http://localhost:8501", "http://localhost:3000"]
```

### Change Admin Credentials

**Important:** Change default admin password immediately!

Option 1: Reset in database
```python
from backend.users import reset_password
reset_password("admin", "NewSecurePassword123!")
```

Option 2: Create new admin user
```python
from backend.users import create_user
create_user(
    username="new_admin",
    email="admin@domain.com",
    password="SecurePass123!",
    full_name="Administrator",
    role="admin"
)
```

### Extend Token Expiration

Edit `backend/auth.py`:
```python
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Default: 30
REFRESH_TOKEN_EXPIRE_DAYS = 14    # Default: 7
```

---

## 🔐 Security Best Practices

### For Development

✅ Default configuration suitable for local testing  
⚠️ Change JWT secret key  
⚠️ Change admin password  
⚠️ Disable CORS allow_all_origins for production

### For Production

✅ Use strong JWT_SECRET_KEY (minimum 32 characters)
```python
import secrets
secrets.token_urlsafe(32)  # Generate strong key
```

✅ Enable HTTPS/SSL everywhere
```python
CORS_ORIGINS = ["https://yourdomain.com"]
```

✅ Use environment variables for all secrets
```python
from dotenv import load_dotenv
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
```

✅ Implement password reset email verification
✅ Enable token blacklisting in Redis
✅ Monitor audit logs
✅ Implement rate limiting
✅ Use secure HTTP headers

---

## 🧪 Testing

### Using Python Requests

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Login
response = requests.post(f"{BASE_URL}/auth/login", json={
    "username": "doctor1",
    "password": "SecurePass123!"
})
tokens = response.json()
access_token = tokens["access_token"]

# 2. Use token for protected endpoint
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(f"{BASE_URL}/patients", headers=headers)
print(response.json())

# 3. Refresh token
response = requests.post(f"{BASE_URL}/auth/refresh", json={
    "refresh_token": tokens["refresh_token"]
})
new_token = response.json()["access_token"]
```

### Using Postman

1. Import collection from Swagger: `http://localhost:8000/openapi.json`
2. Set up auth:
   - Copy bearer token from login response
   - Right-click collection → Edit
   - Authorization tab → Type: Bearer Token
   - Paste token

### Using cURL

```bash
# Login
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"Admin@2026"}' \
  | jq -r '.access_token')

# Use token
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

---

## 🐛 Troubleshooting

### "Invalid username or password"
- Verify username/password are correct
- Check that user account is active (is_active = 1)
- Look at user_audit_log in SQLite for failed attempts

### "Invalid or expired token"
- Request new token using refresh endpoint
- Or login again
- Check JWT_SECRET_KEY hasn't changed

### "Admin access required"
- Your user role is not admin
- Create admin account or change user role in database

### "Database initialization error"
- Check write permissions on directory
- Ensure `backend/` directory exists
- Check `hemophilia_users.db` file path

### Port 8000 already in use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

---

## 📚 Client Integration

### Streamlit Integration

```python
# backend_client.py already has auth support!
from backend_client import predict_risk, get_patients

# Just get token once, then use client functions
# Token is handled automatically by backend_client
```

### React Integration

```javascript
// Save token after login
const response = await fetch('/auth/login', ...)
const { access_token } = await response.json()
localStorage.setItem('token', access_token)

// Use token in requests
const headers = {
  'Authorization': `Bearer ${localStorage.getItem('token')}`
}

// Refresh when needed
const newToken = await fetch('/auth/refresh', {
  method: 'POST',
  body: JSON.stringify({
    refresh_token: localStorage.getItem('refreshToken')
  })
})
```

---

## 📖 Additional Resources

- **JWT.io**: https://jwt.io (token debugging)
- **FastAPI Security Docs**: https://fastapi.tiangolo.com/advanced/security/
- **Bcrypt**: https://github.com/pyca/bcrypt
- **OWASP Authentication Cheat Sheet**: https://cheatsheetseries.owasp.org/

---

## ✅ Implementation Checklist

- [x] JWT token generation and validation
- [x] Bcrypt password hashing
- [x] User registration endpoint
- [x] User login endpoint
- [x] Token refresh endpoint
- [x] User profile endpoint
- [x] Password change endpoint
- [x] Role-based access control
- [x] Protected routes on all endpoints
- [x] User database with audit logging
- [x] Admin account initialization
- [x] Documentation

---

**Authentication System Ready! 🎉**

Start the backend and test at: http://localhost:8000/docs

Default Credentials:
- Username: `admin`
- Password: `Admin@2026`

