# Setup script for Cogni-Mamba Chatbot
# Run this script to set up your environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Cogni-Mamba Chatbot Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
Write-Host ""
Write-Host "Checking for virtual environment..." -ForegroundColor Yellow
if (Test-Path "master") {
    Write-Host "✓ Virtual environment 'master' found" -ForegroundColor Green
    $useExisting = Read-Host "Use existing virtual environment? (y/n)"
    if ($useExisting -eq "n") {
        Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force master
        python -m venv master
        Write-Host "✓ New virtual environment created" -ForegroundColor Green
    }
} else {
    Write-Host "Creating virtual environment 'master'..." -ForegroundColor Yellow
    python -m venv master
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "master\Scripts\Activate.ps1"
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "✓ pip upgraded" -ForegroundColor Green

# Install PyTorch
Write-Host ""
Write-Host "Installing PyTorch..." -ForegroundColor Yellow
$cudaAvailable = Read-Host "Do you have NVIDIA GPU with CUDA? (y/n)"
if ($cudaAvailable -eq "y") {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Cyan
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "Installing PyTorch (CPU only)..." -ForegroundColor Cyan
    pip install torch torchvision torchaudio
}
Write-Host "✓ PyTorch installed" -ForegroundColor Green

# Install other dependencies
Write-Host ""
Write-Host "Installing other dependencies..." -ForegroundColor Yellow
pip install transformers datasets tqdm accelerate
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Create checkpoints directory
Write-Host ""
Write-Host "Creating checkpoints directory..." -ForegroundColor Yellow
if (-not (Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints" | Out-Null
    Write-Host "✓ Checkpoints directory created" -ForegroundColor Green
} else {
    Write-Host "✓ Checkpoints directory already exists" -ForegroundColor Green
}

# Test imports
Write-Host ""
Write-Host "Testing imports..." -ForegroundColor Yellow
$testScript = @"
import torch
import transformers
import datasets
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"@
$testScript | python
Write-Host "✓ All imports successful" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review config.json to adjust model settings" -ForegroundColor White
Write-Host "2. Run training: python train.py" -ForegroundColor White
Write-Host "3. Start chatbot: python chatbot.py" -ForegroundColor White
Write-Host ""
Write-Host "For detailed instructions, see quick_start.md" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to start training
$startTraining = Read-Host "Would you like to start training now? (y/n)"
if ($startTraining -eq "y") {
    Write-Host ""
    Write-Host "Starting training..." -ForegroundColor Green
    Write-Host "This may take 30-60 minutes on GPU, several hours on CPU" -ForegroundColor Yellow
    Write-Host ""
    python train.py
}
