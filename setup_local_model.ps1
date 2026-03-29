# PowerShell Script to Setup Local Pre-trained GPT Model
# Run this to install transformers and download the model

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   Local GPT Model Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "❌ Virtual environment not activated!" -ForegroundColor Red
    Write-Host "Please run: & .\.venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host "✅ Virtual environment detected`n" -ForegroundColor Green

# Install requirements
Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt --timeout 1000

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Installation failed" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host "`n✅ Dependencies installed successfully!`n" -ForegroundColor Green

# Download pre-trained model
Write-Host "🤖 Downloading pre-trained model (DistilGPT-2)..." -ForegroundColor Cyan
Write-Host "   This may take a few minutes on first run...`n" -ForegroundColor Yellow

try {
    python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('distilgpt2'); AutoModelForCausalLM.from_pretrained('distilgpt2'); print('✅ Model downloaded successfully!')"
} catch {
    Write-Host "⚠️  Model download will happen automatically on first use" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   ✅ Setup Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Your chatbot is ready with:" -ForegroundColor Green
Write-Host "  • Local pre-trained GPT model" -ForegroundColor White
Write-Host "  • Offline Q&A capability" -ForegroundColor White
Write-Host "  • Knowledge base integration" -ForegroundColor White
Write-Host "  • API fallback support`n" -ForegroundColor White

Write-Host "Next: Run 'streamlit run app.py' to start!`n" -ForegroundColor Cyan

Write-Host "Testing chatbot..." -ForegroundColor Cyan
python local_model.py

Read-Host "Press Enter to exit"
