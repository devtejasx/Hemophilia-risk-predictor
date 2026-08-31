# Hemophilia Clinical AI - Quick Start Script for Windows
# Run this in PowerShell as Administrator

Write-Host "🏥 Hemophilia Clinical AI - Quick Start Setup" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
try {
    docker version | Out-Null
    Write-Host "✅ Docker found" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    Write-Host "   Visit: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Check if docker-compose is installed
try {
    docker-compose version | Out-Null
    Write-Host "✅ docker-compose found" -ForegroundColor Green
} catch {
    Write-Host "❌ docker-compose is not installed." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Create .env if it doesn't exist
if (!(Test-Path ".env")) {
    Write-Host "📝 Creating .env file from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "⚠️  Please edit .env with your configuration" -ForegroundColor Yellow
} else {
    Write-Host "✅ .env file already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "🚀 Starting services..." -ForegroundColor Cyan
docker-compose down 2>$null
docker-compose up -d

Write-Host ""
Write-Host "⏳ Waiting for services to start (20 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 20

# Check if services are running
Write-Host ""
Write-Host "📊 Checking service status..." -ForegroundColor Cyan
docker-compose ps

Write-Host ""
Write-Host "✅ All services started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access the system:" -ForegroundColor Cyan
Write-Host "   Frontend (Streamlit): http://localhost:8501" -ForegroundColor White
Write-Host "   Backend API:          http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs:             http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "👤 Test Account:" -ForegroundColor Cyan
Write-Host "   Email:    test@local.com" -ForegroundColor White
Write-Host "   Password: Test12345" -ForegroundColor White
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "   - FULLSTACK_README.md for overview" -ForegroundColor White
Write-Host "   - DEPLOYMENT.md for detailed setup" -ForegroundColor White
Write-Host ""
Write-Host "🛑 To stop services:" -ForegroundColor Cyan
Write-Host "   docker-compose stop" -ForegroundColor Gray
Write-Host ""
Write-Host "🔍 To view logs:" -ForegroundColor Cyan
Write-Host "   docker-compose logs -f" -ForegroundColor Gray
Write-Host ""

# Open browser (Windows)
Write-Host "Opening browser..." -ForegroundColor Cyan
Start-Process "http://localhost:8501"
