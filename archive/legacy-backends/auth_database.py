"""
User Database Manager
CRUD operations for user management
"""

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import Optional, List, Tuple
from datetime import datetime
from auth_models import User, UserRole, RefreshToken
from auth_schemas import UserCreate, UserUpdate, AdminUserUpdate
from auth_security import hash_password, generate_user_id, generate_token_id


class UserManager:
    """User database management"""
    
    @staticmethod
    def create_user(db: Session, user_create: UserCreate) -> Tuple[Optional[User], Optional[str]]:
        """Create new user
        
        Args:
            db: Database session
            user_create: User creation schema
            
        Returns:
            Tuple of (user, error_message)
        """
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.email == user_create.email) | 
                (User.username == user_create.username)
            ).first()
            
            if existing_user:
                if existing_user.email == user_create.email:
                    return None, "Email already registered"
                else:
                    return None, "Username already taken"
            
            # Create new user
            new_user = User(
                user_id=generate_user_id(),
                email=user_create.email,
                username=user_create.username,
                full_name=user_create.full_name,
                hashed_password=hash_password(user_create.password),
                role=UserRole(user_create.role),
                is_active=True,
                is_verified=False,
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            return new_user, None
            
        except IntegrityError as e:
            db.rollback()
            return None, f"Database error: {str(e)}"
        except Exception as e:
            db.rollback()
            return None, f"Error creating user: {str(e)}"
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email
        
        Args:
            db: Database session
            email: User email
            
        Returns:
            User if found, None otherwise
        """
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username
        
        Args:
            db: Database session
            username: User username
            
        Returns:
            User if found, None otherwise
        """
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """Get user by ID
        
        Args:
            db: Database session
            user_id: User unique identifier
            
        Returns:
            User if found, None otherwise
        """
        return db.query(User).filter(User.user_id == user_id).first()
    
    @staticmethod
    def update_user(
        db: Session,
        user: User,
        user_update: UserUpdate
    ) -> Tuple[Optional[User], Optional[str]]:
        """Update user information
        
        Args:
            db: Database session
            user: User to update
            user_update: Update data
            
        Returns:
            Tuple of (updated_user, error_message)
        """
        try:
            if user_update.full_name is not None:
                user.full_name = user_update.full_name
            
            if user_update.email is not None:
                # Check if email already exists
                existing = db.query(User).filter(
                    (User.email == user_update.email) &
                    (User.user_id != user.user_id)
                ).first()
                
                if existing:
                    return None, "Email already in use"
                
                user.email = user_update.email
            
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            
            return user, None
            
        except Exception as e:
            db.rollback()
            return None, f"Error updating user: {str(e)}"
    
    @staticmethod
    def admin_update_user(
        db: Session,
        user: User,
        admin_update: AdminUserUpdate
    ) -> Tuple[Optional[User], Optional[str]]:
        """Admin update user (with role/status changes)
        
        Args:
            db: Database session
            user: User to update
            admin_update: Admin update data
            
        Returns:
            Tuple of (updated_user, error_message)
        """
        try:
            if admin_update.full_name is not None:
                user.full_name = admin_update.full_name
            
            if admin_update.email is not None:
                existing = db.query(User).filter(
                    (User.email == admin_update.email) &
                    (User.user_id != user.user_id)
                ).first()
                if existing:
                    return None, "Email already in use"
                user.email = admin_update.email
            
            if admin_update.role is not None:
                user.role = UserRole(admin_update.role)
            
            if admin_update.is_active is not None:
                user.is_active = admin_update.is_active
            
            if admin_update.is_verified is not None:
                user.is_verified = admin_update.is_verified
            
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            
            return user, None
            
        except Exception as e:
            db.rollback()
            return None, f"Error updating user: {str(e)}"
    
    @staticmethod
    def update_last_login(db: Session, user: User) -> None:
        """Update user last login timestamp
        
        Args:
            db: Database session
            user: User to update
        """
        try:
            user.last_login = datetime.utcnow()
            db.commit()
        except Exception as e:
            print(f"Error updating last login: {e}")
            db.rollback()
    
    @staticmethod
    def list_users(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        role: Optional[str] = None
    ) -> Tuple[List[User], int]:
        """List users with optional filtering
        
        Args:
            db: Database session
            skip: Skip first N records
            limit: Maximum records to return
            role: Filter by role
            
        Returns:
            Tuple of (users, total_count)
        """
        query = db.query(User)
        
        if role:
            query = query.filter(User.role == UserRole(role))
        
        total = query.count()
        users = query.offset(skip).limit(limit).all()
        
        return users, total
    
    @staticmethod
    def delete_user(db: Session, user: User) -> Tuple[bool, Optional[str]]:
        """Delete user (soft delete - set inactive)
        
        Args:
            db: Database session
            user: User to delete
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            user.is_active = False
            user.updated_at = datetime.utcnow()
            db.commit()
            return True, None
        except Exception as e:
            db.rollback()
            return False, f"Error deleting user: {str(e)}"
    
    @staticmethod
    def count_users(db: Session) -> int:
        """Count total active users
        
        Args:
            db: Database session
            
        Returns:
            Total user count
        """
        return db.query(User).filter(User.is_active == True).count()


class RefreshTokenManager:
    """Refresh token management"""
    
    @staticmethod
    def create_refresh_token(
        db: Session,
        user_id: str,
        token: str,
        expires_at: datetime
    ) -> Tuple[Optional[RefreshToken], Optional[str]]:
        """Create refresh token record
        
        Args:
            db: Database session
            user_id: User ID
            token: Token string
            expires_at: Expiration datetime
            
        Returns:
            Tuple of (token_record, error_message)
        """
        try:
            refresh_token = RefreshToken(
                token_id=generate_token_id(),
                user_id=user_id,
                token=token,
                expires_at=expires_at,
            )
            
            db.add(refresh_token)
            db.commit()
            db.refresh(refresh_token)
            
            return refresh_token, None
            
        except Exception as e:
            db.rollback()
            return None, f"Error creating refresh token: {str(e)}"
    
    @staticmethod
    def get_refresh_token(db: Session, token: str) -> Optional[RefreshToken]:
        """Get refresh token by token string
        
        Args:
            db: Database session
            token: Token string
            
        Returns:
            RefreshToken if found, None otherwise
        """
        return db.query(RefreshToken).filter(RefreshToken.token == token).first()
    
    @staticmethod
    def revoke_refresh_token(db: Session, refresh_token: RefreshToken) -> bool:
        """Revoke refresh token
        
        Args:
            db: Database session
            refresh_token: Token to revoke
            
        Returns:
            True if successful, False otherwise
        """
        try:
            refresh_token.is_revoked = True
            db.commit()
            return True
        except Exception as e:
            print(f"Error revoking token: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def revoke_user_tokens(db: Session, user_id: str) -> bool:
        """Revoke all tokens for a user
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            True if successful
        """
        try:
            db.query(RefreshToken).filter(
                (RefreshToken.user_id == user_id) &
                (RefreshToken.is_revoked == False)
            ).update({RefreshToken.is_revoked: True})
            db.commit()
            return True
        except Exception as e:
            print(f"Error revoking user tokens: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def cleanup_expired_tokens(db: Session) -> int:
        """Clean up expired tokens
        
        Args:
            db: Database session
            
        Returns:
            Number of tokens deleted
        """
        try:
            result = db.query(RefreshToken).filter(
                RefreshToken.expires_at < datetime.utcnow()
            ).delete()
            db.commit()
            return result
        except Exception as e:
            print(f"Error cleaning up tokens: {e}")
            db.rollback()
            return 0
