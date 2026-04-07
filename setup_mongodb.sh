#!/bin/bash
# or for Windows: setup_mongodb.bat

# MongoDB Quick Setup Script
# Detects system and guides installation

echo "===================================="
echo "MongoDB Setup Helper"
echo "===================================="
echo ""

# Check if MongoDB is already installed
if command -v mongod &> /dev/null; then
    echo "✅ MongoDB is already installed!"
    mongod --version
    echo ""
    read -p "Start MongoDB? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mongod
    fi
else
    echo "❌ MongoDB not found"
    echo ""
    echo "Choose installation option:"
    echo "1) Local MongoDB (Community Edition)"
    echo "2) MongoDB Atlas (Cloud - Free)"
    echo "3) Skip for now (use in-memory storage)"
    echo ""
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            echo ""
            echo "Local MongoDB Installation:"
            echo "1. Go to: https://www.mongodb.com/try/download/community"
            echo "2. Download for your OS"
            echo "3. Run installer"
            echo "4. After installation, run: mongod"
            echo ""
            read -p "Opened browser? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    open "https://www.mongodb.com/try/download/community"
                elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                    xdg-open "https://www.mongodb.com/try/download/community"
                elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
                    start "https://www.mongodb.com/try/download/community"
                fi
            fi
            ;;
        2)
            echo ""
            echo "MongoDB Atlas (Cloud) Setup:"
            echo "1. Go to: https://www.mongodb.com/cloud/atlas"
            echo "2. Create free account"
            echo "3. Create M0 cluster"
            echo "4. Get connection string"
            echo "5. Update chatbot-service/.env"
            echo ""
            read -p "Opened browser? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    open "https://www.mongodb.com/cloud/atlas"
                elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                    xdg-open "https://www.mongodb.com/cloud/atlas"
                elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
                    start "https://www.mongodb.com/cloud/atlas"
                fi
            fi
            ;;
        3)
            echo "Skipping MongoDB. Using in-memory storage (chatbot will work, but conversations won't persist)."
            ;;
    esac
fi

echo ""
echo "===================================="
echo "After setup complete:"
echo "1. cd chatbot-service"
echo "2. npm start"
echo "3. Go to http://localhost:8502"
echo "===================================="
