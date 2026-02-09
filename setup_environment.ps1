# Quantum Biodiversity Project - Windows Setup Script
# Author: Nicolas Mendes de Araújo
# Date: February 2026

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Quantum Biodiversity Project - Windows Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Verificando Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion encontrado" -ForegroundColor Green
} catch {
    Write-Host "✗ Python não encontrado!" -ForegroundColor Red
    Write-Host "  → Instale Python 3.9+ de: https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`nCriando ambiente virtual..." -ForegroundColor Yellow
python -m venv quantum_env

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Erro ao criar ambiente virtual" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Ambiente virtual criado" -ForegroundColor Green

# Activate virtual environment
Write-Host "`nAtivando ambiente virtual..." -ForegroundColor Yellow
.\quantum_env\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nAtualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip | Out-Null

# Install dependencies
Write-Host "`nInstalando dependências (isso pode demorar alguns minutos)..." -ForegroundColor Yellow

Write-Host "  → Quantum Computing (Qiskit)..." -ForegroundColor Cyan
pip install qiskit==1.0.0 qiskit-aer==0.13.3 qiskit-ibm-runtime==0.18.0 --quiet

Write-Host "  → Chemistry (RDKit)..." -ForegroundColor Cyan
pip install rdkit==2023.9.4 --quiet

Write-Host "  → Data Science..." -ForegroundColor Cyan
pip install numpy==1.26.3 pandas==2.1.4 scipy==1.11.4 scikit-learn==1.4.0 --quiet

Write-Host "  → Visualization..." -ForegroundColor Cyan
pip install matplotlib==3.8.2 seaborn==0.13.0 plotly==5.18.0 --quiet

Write-Host "  → Utilities..." -ForegroundColor Cyan
pip install networkx==3.2.1 tqdm==4.66.1 python-dotenv==1.0.0 requests==2.31.0 --quiet

Write-Host "  → Jupyter (opcional)..." -ForegroundColor Cyan
pip install jupyter==1.0.0 ipywidgets==8.1.1 --quiet

Write-Host "✓ Todas as dependências instaladas" -ForegroundColor Green

# Create project structure
Write-Host "`nCriando estrutura de diretórios..." -ForegroundColor Yellow

$directories = @(
    "data\raw",
    "data\processed",
    "data\results",
    "src\classical",
    "src\quantum",
    "src\utils",
    "src\analysis",
    "notebooks",
    "figures",
    "papers"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

Write-Host "✓ Estrutura de diretórios criada" -ForegroundColor Green

# Move existing files to proper locations
Write-Host "`nOrganizando arquivos..." -ForegroundColor Yellow

if (Test-Path "classical_molecular_selection.py") {
    Move-Item -Path "classical_molecular_selection.py" -Destination "src\classical\" -Force
    Write-Host "  → classical_molecular_selection.py → src\classical\" -ForegroundColor Cyan
}

if (Test-Path "quantum_molecular_selection.py") {
    Move-Item -Path "quantum_molecular_selection.py" -Destination "src\quantum\" -Force
    Write-Host "  → quantum_molecular_selection.py → src\quantum\" -ForegroundColor Cyan
}

# Create .env template
Write-Host "`nCriando template de configuração (.env)..." -ForegroundColor Yellow

$envTemplate = @"
# IBM Quantum Credentials
# Get your token from: https://quantum.ibm.com/
IBM_QUANTUM_TOKEN=your_token_here
IBM_QUANTUM_CHANNEL=ibm_quantum
IBM_QUANTUM_INSTANCE=ibm_qasm_simulator
"@

$envTemplate | Out-File -FilePath ".env.template" -Encoding utf8
Write-Host "✓ Template .env.template criado" -ForegroundColor Green

# Create requirements.txt
Write-Host "`nGerando requirements.txt..." -ForegroundColor Yellow
pip freeze > requirements.txt
Write-Host "✓ requirements.txt gerado" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Completo!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Próximos Passos:" -ForegroundColor Yellow
Write-Host "  1. Obter token IBM Quantum: https://quantum.ibm.com/" -ForegroundColor White
Write-Host "  2. Copiar .env.template para .env e adicionar seu token" -ForegroundColor White
Write-Host "  3. Ativar ambiente: .\quantum_env\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  4. Verificar setup: python verify_setup.py" -ForegroundColor White
Write-Host ""
Write-Host "Estrutura criada:" -ForegroundColor Yellow
Write-Host "  data\          - Datasets (raw, processed, results)" -ForegroundColor White
Write-Host "  src\           - Código-fonte (classical, quantum, utils)" -ForegroundColor White
Write-Host "  notebooks\     - Jupyter notebooks" -ForegroundColor White
Write-Host "  figures\       - Visualizações" -ForegroundColor White
Write-Host "  papers\        - Rascunhos de paper" -ForegroundColor White
Write-Host ""
Write-Host "Para desativar o ambiente virtual: deactivate" -ForegroundColor Cyan
Write-Host ""
