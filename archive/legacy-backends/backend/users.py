"""
User database management for authentication system.
Handles user CRUD operations, password hashing, and user queries.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from backend.auth import hash_password, verify_password

DATABASE_FILE = "hemophilia_users.db"


def init_user_database():
    """Initialize user database tables"""
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'patient',
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            created_by INTEGER,
            FOREIGN KEY (created_by) REFERENCES users(id)
        )
    ''')
    
    # User sessions table (for refresh tokens)
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            refresh_token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    # User audit log table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        )
    ''')
    
    conn.commit()
    conn.close()


def create_user(
    username: str,
    email: str,
    password: str,
    full_name: str,
    role: str = "patient",
    created_by: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Create new user"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        password_hash = hash_password(password)
        
        c.execute('''
            INSERT INTO users (username, email, full_name, password_hash, role, created_by)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, email, full_name, password_hash, role, created_by))
        
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Log action
        log_audit("create_user", f"User created: {username} (role: {role})")
        
        return get_user_by_id(user_id)
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            raise ValueError(f"Username '{username}' already exists")
        elif "email" in str(e):
            raise ValueError(f"Email '{email}' already exists")
        raise
    except Exception as e:
        print(f"Error creating user: {e}")
        return None


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get user by username"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('''
            SELECT id, username, email, full_name, password_hash, role, is_active, 
                   created_at, last_login
            FROM users
            WHERE username = ?
        ''', (username,))
        
        row = c.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Error getting user by username: {e}")
        return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('''
            SELECT id, username, email, full_name, role, is_active, created_at, last_login
            FROM users
            WHERE id = ?
        ''', (user_id,))
        
        row = c.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Error getting user by ID: {e}")
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('''
            SELECT id, username, email, full_name, password_hash, role, is_active,
                   created_at, last_login
            FROM users
            WHERE email = ?
        ''', (email,))
        
        row = c.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Error getting user by email: {e}")
        return None


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with username and password"""
    user = get_user_by_username(username)
    
    if not user:
        return None
    
    if not user.get("is_active"):
        return None
    
    if not verify_password(password, user.get("password_hash", "")):
        return None
    
    return user


def update_last_login(user_id: int):
    """Update user's last login timestamp"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating last login: {e}")


def change_password(user_id: int, old_password: str, new_password: str) -> bool:
    """Change user password"""
    try:
        user = get_user_by_id(user_id)
        if not user:
            return False
        
        # Verify old password
        if not verify_password(old_password, user.get("password_hash", "")):
            return False
        
        # Hash new password
        password_hash = hash_password(new_password)
        
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute('''
            UPDATE users SET password_hash = ? WHERE id = ?
        ''', (password_hash, user_id))
        
        conn.commit()
        conn.close()
        
        log_audit(f"Password changed for user ID {user_id}")
        return True
    except Exception as e:
        print(f"Error changing password: {e}")
        return False


def reset_password(email: str, new_password: str) -> bool:
    """Reset user password by email"""
    try:
        user = get_user_by_email(email)
        if not user:
            return False
        
        password_hash = hash_password(new_password)
        
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute('''
            UPDATE users SET password_hash = ? WHERE id = ?
        ''', (password_hash, user["id"]))
        
        conn.commit()
        conn.close()
        
        log_audit(f"Password reset for user {user['username']}")
        return True
    except Exception as e:
        print(f"Error resetting password: {e}")
        return False


def get_all_users(role: Optional[str] = None, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """Get all users with optional filtering by role"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        if role:
            c.execute('''
                SELECT id, username, email, full_name, role, is_active, created_at, last_login
                FROM users
                WHERE role = ?
                LIMIT ? OFFSET ?
            ''', (role, limit, skip))
        else:
            c.execute('''
                SELECT id, username, email, full_name, role, is_active, created_at, last_login
                FROM users
                LIMIT ? OFFSET ?
            ''', (limit, skip))
        
        rows = c.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"Error getting all users: {e}")
        return []


def update_user(user_id: int, **kwargs) -> Optional[Dict[str, Any]]:
    """Update user fields"""
    try:
        allowed_fields = {"email", "full_name", "role", "is_active"}
        update_fields = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not update_fields:
            return get_user_by_id(user_id)
        
        set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
        values = list(update_fields.values()) + [user_id]
        
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute(f'''
            UPDATE users SET {set_clause} WHERE id = ?
        ''', values)
        
        conn.commit()
        conn.close()
        
        return get_user_by_id(user_id)
    except Exception as e:
        print(f"Error updating user: {e}")
        return None


def deactivate_user(user_id: int) -> bool:
    """Deactivate user account"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute('''
            UPDATE users SET is_active = 0 WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        log_audit(f"User deactivated: {user_id}")
        return True
    except Exception as e:
        print(f"Error deactivating user: {e}")
        return False


def delete_user(user_id: int) -> bool:
    """Delete user account"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute('DELETE FROM users WHERE id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        
        log_audit(f"User deleted: {user_id}")
        return True
    except Exception as e:
        print(f"Error deleting user: {e}")
        return False


def log_audit(action: str, details: str = "", user_id: Optional[int] = None):
    """Log audit action"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO user_audit_log (user_id, action, details)
            VALUES (?, ?, ?)
        ''', (user_id, action, details))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging audit: {e}")


def get_user_count() -> int:
    """Get total number of users"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM users')
        count = c.fetchone()[0]
        conn.close()
        
        return count
    except Exception as e:
        print(f"Error getting user count: {e}")
        return 0


def user_exists(username: str) -> bool:
    """Check if user exists"""
    return get_user_by_username(username) is not None


def create_admin_if_not_exists(
    username: str = "admin",
    email: str = "admin@hemophilia.local",
    password: str = "Admin@2026",
    full_name: str = "Administrator"
) -> bool:
    """Create default admin user if none exists"""
    try:
        if get_user_count() > 0:
            return False  # Users already exist
        
        create_user(username, email, password, full_name, role="admin")
        log_audit("Admin user created", "Default admin account initialized")
        return True
    except Exception as e:
        print(f"Error creating admin user: {e}")
        return False
