@echo off
REM MongoDB Setup Helper for Windows
REM Checks if MongoDB is installed and guides setup

setlocal enabledelayedexpansion
title MongoDB Setup Helper

cls
echo.
echo ================================================================================
echo    MongoDB Setup Helper for Windows
echo ================================================================================
echo.

REM Check if MongoDB is installed
mongod --version >nul 2>&1
if errorlevel 1 (
    echo ❌ MongoDB not found on this system
    echo.
    echo Choose installation option:
    echo.
    echo   1) Download Local MongoDB Community Edition
    echo   2) Use MongoDB Atlas Cloud (Free)
    echo   3) Skip for now ^(use in-memory storage^)
    echo.
    
    set /p choice="Enter choice (1-3): "
    
    if "!choice!"=="1" (
        echo.
        echo Downloading MongoDB Community Edition...
        echo https://www.mongodb.com/try/download/community
        echo.
        echo Opening browser...
        start https://www.mongodb.com/try/download/community
        echo.
        echo After installation:
        echo   1. Restart this script or terminal
        echo   2. MongoDB should be in System PATH
        echo   3. Run: mongod
        echo.
        pause
    ) else if "!choice!"=="2" (
        echo.
        echo MongoDB Atlas Cloud Setup:
        echo https://www.mongodb.com/cloud/atlas
        echo.
        echo Opening browser...
        start https://www.mongodb.com/cloud/atlas
        echo.
        echo Steps:
        echo   1. Create free account
        echo   2. Create M0 cluster
        echo   3. Get connection string
        echo   4. Update chatbot-service\.env
        echo.
        pause
    ) else if "!choice!"=="3" (
        echo.
        echo Continuing with in-memory storage.
        echo Note: Conversations will not persist after server restart.
        echo.
        pause
    ) else (
        echo Invalid choice
        pause
        exit /b 1
    )
) else (
    echo ✅ MongoDB is installed!
    echo.
    mongod --version
    echo.
    set /p start_mongo="Start MongoDB service? (Y/N): "
    
    if /i "!start_mongo!"=="Y" (
        echo.
        echo Starting MongoDB...
        echo.
        net start MongoDB
        if errorlevel 1 (
            echo.
            echo Service not found. Starting mongod directly...
            echo.
            mongod
        ) else (
            echo.
            echo ✅ MongoDB started successfully
            echo.
            echo Your conversations will now be saved to mongoDB when using the chatbot!
            echo.
            pause
        )
    )
)

echo.
echo ================================================================================
echo    Next Steps:
echo ================================================================================
echo.
echo 1. Edit chatbot-service\.env with MongoDB connection string:
echo    - Local: mongodb://localhost:27017/hemophilia-chatbot
echo    - Atlas: mongodb+srv://user:pass@cluster.mongodb.net/hemophilia-chatbot
echo.
echo 2. Restart chatbot backend:
echo    cd chatbot-service
echo    npm start
echo.
echo 3. Go to http://localhost:8502 and test Quick Chat
echo.
echo 4. Verify data saved:
echo    mongosh hemophilia-chatbot
echo    db.conversations.find()
echo.
echo ================================================================================
echo.
pause
