"""
Authentication Integration Tests
Test all authentication endpoints and flows
Run with: pytest auth_test.py -v
Or run directly: python auth_test.py
"""

import requests
import json
from typing import Dict, Optional
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8000"
API_AUTH_URL = f"{API_BASE_URL}/api/auth"

# Test user credentials
TEST_USERS = {
    "admin": {
        "email": "admin@medical-ai.com",
        "password": "AdminPassword123!"
    },
    "doctor": {
        "email": "doctor1@medical-ai.com",
        "password": "DoctorPassword123!"
    },
    "patient": {
        "email": "patient1@medical-ai.com",
        "password": "PatientPassword123!"
    }
}

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


# ============================================================================
# TEST HELPERS
# ============================================================================

def print_section(title: str):
    """Print test section header"""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"{title.center(70)}")
    print(f"{'='*70}{Colors.END}\n")


def print_test(test_name: str, passed: bool, message: str = ""):
    """Print test result"""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} | {test_name}")
    if message:
        print(f"       {message}")


def print_json(data: Dict, indent: int = 2):
    """Pretty print JSON"""
    print(Colors.YELLOW + json.dumps(data, indent=indent, default=str) + Colors.END)


def get_headers(token: Optional[str] = None) -> Dict:
    """Get request headers with optional auth token"""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# ============================================================================
# TEST SUITE: LOGIN & SIGNUP
# ============================================================================

def test_signup():
    """Test user signup endpoint"""
    print_section("TEST SUITE: SIGNUP")
    
    # Test 1: Successful signup
    new_user = {
        "email": f"testuser_{int(time.time())}@test.com",
        "username": f"testuser_{int(time.time())}",
        "password": "TestPassword123!",
        "role": "patient"
    }
    
    response = requests.post(
        f"{API_AUTH_URL}/signup",
        json=new_user,
        headers=get_headers()
    )
    
    passed = response.status_code == 201
    print_test(
        "Signup - Create new user",
        passed,
        f"Status: {response.status_code}"
    )
    
    if passed:
        data = response.json()
        print(f"       Email: {new_user['email']}")
        print(f"       Access Token: {data['access_token'][:20]}...")
        return data['access_token'], data['refresh_token']
    
    # Test 2: Duplicate email
    response = requests.post(
        f"{API_AUTH_URL}/signup",
        json=new_user,
        headers=get_headers()
    )
    
    print_test(
        "Signup - Reject duplicate email",
        response.status_code == 409,
        f"Status: {response.status_code}"
    )
    
    # Test 3: Weak password
    weak_password_user = {
        "email": f"weak_{int(time.time())}@test.com",
        "username": f"weak_{int(time.time())}",
        "password": "weak",
        "role": "patient"
    }
    
    response = requests.post(
        f"{API_AUTH_URL}/signup",
        json=weak_password_user,
        headers=get_headers()
    )
    
    print_test(
        "Signup - Reject weak password",
        response.status_code == 400,
        f"Status: {response.status_code}"
    )


def test_login():
    """Test user login endpoint"""
    print_section("TEST SUITE: LOGIN")
    
    tokens = {}
    
    for role, creds in TEST_USERS.items():
        # Test: Successful login
        response = requests.post(
            f"{API_AUTH_URL}/login",
            json=creds,
            headers=get_headers()
        )
        
        passed = response.status_code == 200
        print_test(
            f"Login - {role.upper()} user",
            passed,
            f"Status: {response.status_code}"
        )
        
        if passed:
            data = response.json()
            tokens[role] = {
                "access_token": data['access_token'],
                "refresh_token": data['refresh_token']
            }
            print(f"       User: {data['user']['email']}")
            print(f"       Role: {data['user']['role']}")
    
    # Test: Invalid credentials
    response = requests.post(
        f"{API_AUTH_URL}/login",
        json={"email": "admin@medical-ai.com", "password": "wrong"},
        headers=get_headers()
    )
    
    print_test(
        "Login - Reject invalid password",
        response.status_code == 401,
        f"Status: {response.status_code}"
    )
    
    # Test: Non-existent user
    response = requests.post(
        f"{API_AUTH_URL}/login",
        json={"email": "nonexistent@test.com", "password": "password"},
        headers=get_headers()
    )
    
    print_test(
        "Login - Reject non-existent user",
        response.status_code == 401,
        f"Status: {response.status_code}"
    )
    
    return tokens


# ============================================================================
# TEST SUITE: TOKEN MANAGEMENT
# ============================================================================

def test_token_refresh(tokens: Dict):
    """Test token refresh endpoint"""
    print_section("TEST SUITE: TOKEN REFRESH")
    
    for role, token_pair in tokens.items():
        response = requests.post(
            f"{API_AUTH_URL}/refresh",
            json={"refresh_token": token_pair['refresh_token']},
            headers=get_headers()
        )
        
        passed = response.status_code == 200
        print_test(
            f"Token Refresh - {role.upper()} user",
            passed,
            f"Status: {response.status_code}"
        )
        
        if passed:
            data = response.json()
            print(f"       New Access Token: {data['access_token'][:20]}...")
    
    # Test: Invalid refresh token
    response = requests.post(
        f"{API_AUTH_URL}/refresh",
        json={"refresh_token": "invalid.token.here"},
        headers=get_headers()
    )
    
    print_test(
        "Token Refresh - Reject invalid token",
        response.status_code == 401,
        f"Status: {response.status_code}"
    )


# ============================================================================
# TEST SUITE: USER PROFILE
# ============================================================================

def test_user_profile(tokens: Dict):
    """Test user profile endpoints"""
    print_section("TEST SUITE: USER PROFILE")
    
    for role, token_pair in tokens.items():
        # Get profile
        response = requests.get(
            f"/api/protected/user-stats",
            headers=get_headers(token_pair['access_token'])
        )
        
        # Note: This endpoint requires the full URL
        response = requests.get(
            f"{API_BASE_URL}/api/protected/user-stats",
            headers=get_headers(token_pair['access_token'])
        )
        
        passed = response.status_code == 200
        print_test(
            f"Get Profile - {role.upper()} user",
            passed,
            f"Status: {response.status_code}"
        )
        
        if passed and response.status_code == 200:
            data = response.json()
            print(f"       Email: {data['email']}")
            print(f"       Role: {data['role']}")


def test_profile_without_auth():
    """Test that profile requires authentication"""
    print_section("TEST SUITE: AUTHENTICATION REQUIRED")
    
    # Test: Without token
    response = requests.get(
        f"{API_BASE_URL}/api/protected/user-stats",
        headers=get_headers()
    )
    
    print_test(
        "Protected endpoint - Reject unauthorized",
        response.status_code == 403,
        f"Status: {response.status_code}"
    )
    
    # Test: With invalid token
    response = requests.get(
        f"{API_BASE_URL}/api/protected/user-stats",
        headers=get_headers("invalid.token.here")
    )
    
    print_test(
        "Protected endpoint - Reject invalid token",
        response.status_code == 403,
        f"Status: {response.status_code}"
    )


# ============================================================================
# TEST SUITE: ROLE-BASED ACCESS
# ============================================================================

def test_role_based_access(tokens: Dict):
    """Test role-based access control"""
    print_section("TEST SUITE: ROLE-BASED ACCESS CONTROL")
    
    # Test: Admin can access admin endpoints
    response = requests.get(
        f"{API_BASE_URL}/api/admin/stats",
        headers=get_headers(tokens['admin']['access_token'])
    )
    
    print_test(
        "Admin Access - Admin user can access",
        response.status_code == 200,
        f"Status: {response.status_code}"
    )
    
    # Test: Non-admin cannot access admin endpoints
    response = requests.get(
        f"{API_BASE_URL}/api/admin/stats",
        headers=get_headers(tokens['patient']['access_token'])
    )
    
    print_test(
        "Admin Access - Patient user rejected",
        response.status_code == 403,
        f"Status: {response.status_code}"
    )
    
    # Test: Doctor can access medical endpoints
    response = requests.post(
        f"{API_BASE_URL}/api/medical/predict",
        json={"patient_id": "P001"},
        headers=get_headers(tokens['doctor']['access_token'])
    )
    
    print_test(
        "Doctor Access - Doctor user can access",
        response.status_code in [200, 422],  # 422 if validation fails but auth passed
        f"Status: {response.status_code}"
    )


# ============================================================================
# TEST SUITE: LOGOUT & REVOCATION
# ============================================================================

def test_logout(tokens: Dict):
    """Test logout endpoint"""
    print_section("TEST SUITE: LOGOUT")
    
    # Test: Logout revokes tokens
    response = requests.post(
        f"{API_AUTH_URL}/logout",
        headers=get_headers(tokens['patient']['access_token'])
    )
    
    print_test(
        "Logout - Logout successful",
        response.status_code == 200,
        f"Status: {response.status_code}"
    )
    
    # Test: Refresh token no longer works after logout
    time.sleep(0.5)  # Small delay
    response = requests.post(
        f"{API_AUTH_URL}/refresh",
        json={"refresh_token": tokens['patient']['refresh_token']},
        headers=get_headers()
    )
    
    print_test(
        "Logout - Refresh token revoked",
        response.status_code == 401,
        f"Status: {response.status_code}"
    )


# ============================================================================
# TEST SUITE: ADMIN OPERATIONS
# ============================================================================

def test_admin_operations(tokens: Dict):
    """Test admin user management endpoints"""
    print_section("TEST SUITE: ADMIN OPERATIONS")
    
    admin_token = tokens['admin']['access_token']
    
    # Test: List users
    response = requests.get(
        f"{API_BASE_URL}/api/admin/users",
        headers=get_headers(admin_token)
    )
    
    passed = response.status_code == 200
    print_test(
        "Admin - List users",
        passed,
        f"Status: {response.status_code}"
    )
    
    if passed:
        data = response.json()
        print(f"       Total users: {data['total']}")
    
    # Test: Create user
    new_admin_user = {
        "email": f"newadmin_{int(time.time())}@test.com",
        "username": f"newadmin_{int(time.time())}",
        "password": "NewAdminPass123!",
        "role": "admin"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/admin/users",
        json=new_admin_user,
        headers=get_headers(admin_token)
    )
    
    print_test(
        "Admin - Create user",
        response.status_code == 201,
        f"Status: {response.status_code}"
    )
    
    if response.status_code == 201:
        user_id = response.json()['user_id']
        
        # Test: Get user
        response = requests.get(
            f"{API_BASE_URL}/api/admin/users/{user_id}",
            headers=get_headers(admin_token)
        )
        
        print_test(
            "Admin - Get user",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
        
        # Test: Update user
        response = requests.put(
            f"{API_BASE_URL}/api/admin/users/{user_id}",
            json={"full_name": "Updated Name"},
            headers=get_headers(admin_token)
        )
        
        print_test(
            "Admin - Update user",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run complete test suite"""
    print(f"\n{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     AUTHENTICATION SYSTEM - INTEGRATION TEST SUITE                 ║")
    print(f"║     Testing: {API_BASE_URL}")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    try:
        # Check server is running
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"{Colors.RED}✗ Server not responding properly{Colors.END}")
            return
        
        print(f"{Colors.GREEN}✓ Server is running{Colors.END}\n")
        
        # Run test suites
        test_signup()
        tokens = test_login()
        
        if tokens:
            test_token_refresh(tokens)
            test_profile_without_auth()
            test_user_profile(tokens)
            test_role_based_access(tokens)
            test_admin_operations(tokens)
            test_logout(tokens)
        
        # Summary
        print_section("TEST SUITE COMPLETE")
        print(f"{Colors.GREEN}All tests completed!{Colors.END}\n")
        
    except requests.exceptions.ConnectionError:
        print(f"\n{Colors.RED}✗ ERROR: Cannot connect to API server{Colors.END}")
        print(f"   Make sure server is running at {API_BASE_URL}")
        print(f"   Run: python main.py{Colors.END}\n")
    except Exception as e:
        print(f"\n{Colors.RED}✗ ERROR: {str(e)}{Colors.END}\n")


if __name__ == "__main__":
    run_all_tests()
