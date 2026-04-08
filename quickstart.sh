#!/bin/bash

# Hemophilia Clinical AI - Quick Start Script for Linux/Mac
# Windows users: Use PowerShell or follow docker-compose command manually

set -e  # Exit on error

echo "🏥 Hemophilia Clinical AI - Quick Start Setup"
echo "=============================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker found"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed."
    exit 1
fi

echo "✅ docker-compose found"
echo ""

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your configuration"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🚀 Starting services..."
docker-compose down || true
docker-compose up -d

echo ""
echo "⏳ Waiting for services to start (15 seconds)..."
sleep 15

# Check if services are running
echo ""
echo "📊 Checking service status..."
docker-compose ps

echo ""
echo "✅ All services started successfully!"
echo ""
echo "🌐 Access the system:"
echo "   Frontend (Streamlit): http://localhost:8501"
echo "   Backend API:          http://localhost:8000"
echo "   API Docs:             http://localhost:8000/docs"
echo ""
echo "👤 Test Account:"
echo "   Email:    test@local.com"
echo "   Password: Test12345"
echo ""
echo "📚 Documentation:"
echo "   - FULLSTACK_README.md for overview"
echo "   - DEPLOYMENT.md for detailed setup"
echo ""
echo "🛑 To stop services:"
echo "   docker-compose stop"
echo ""
echo "🔍 To view logs:"
echo "   docker-compose logs -f"
echo ""
