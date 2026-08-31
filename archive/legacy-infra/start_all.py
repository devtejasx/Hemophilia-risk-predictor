#!/usr/bin/env python3
"""
Start both FastAPI backend and Streamlit frontend
Usage: python start_all.py
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def start_backend():
    """Start FastAPI backend"""
    print("\n" + "="*70)
    print("🏥 Starting FastAPI Backend...")
    print("="*70)
    
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000"
    ]
    
    try:
        subprocess.Popen(cmd)
        print("✅ Backend started on http://localhost:8000")
        print("📚 Docs available at http://localhost:8000/docs")
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False
    
    return True


def start_frontend():
    """Start Streamlit frontend"""
    print("\n" + "="*70)
    print("🎨 Starting Streamlit Frontend...")
    print("="*70)
    
    # Change back to project root
    os.chdir(Path(__file__).parent)
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port=8501"
    ]
    
    try:
        subprocess.Popen(cmd)
        print("✅ Frontend started on http://localhost:8501")
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return False
    
    return True


def main():
    """Start both services"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "🏥 Hemophilia AI Platform - Start All" + " "*16 + "║")
    print("╚" + "="*68 + "╝\n")
    
    # Start backend
    if not start_backend():
        print("Failed to start backend. Exiting.")
        sys.exit(1)
    
    # Wait for backend to be ready
    print("\n⏳ Waiting 3 seconds for backend to initialize...")
    time.sleep(3)
    
    # Start frontend
    if not start_frontend():
        print("Failed to start frontend. Backend is running but frontend failed.")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ SUCCESS! Both services are running:\n")
    print("   🔧 Backend API:      http://localhost:8000")
    print("   📚 API Documentation: http://localhost:8000/docs")
    print("   🎨 Frontend:          http://localhost:8501")
    print("\n⏹️  Press CTRL+C in either terminal to stop\n")
    print("="*70 + "\n")
    
    # Keep script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        sys.exit(0)


if __name__ == "__main__":
    main()
