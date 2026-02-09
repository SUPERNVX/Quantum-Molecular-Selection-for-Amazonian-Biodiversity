#!/bin/bash

# Quantum Biodiversity Project - Environment Setup
# Author: Nicolas Mendes de Araújo
# Date: February 2026

echo "========================================"
echo "Quantum Biodiversity Project Setup"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv quantum_biodiversity_env

# Activate virtual environment
echo "Activating virtual environment..."
source quantum_biodiversity_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "Installing Qiskit (IBM Quantum SDK)..."
pip install qiskit==1.0.0
pip install qiskit-aer==0.13.3
pip install qiskit-ibm-runtime==0.18.0

# Install chemistry/molecular tools
echo "Installing chemistry libraries..."
pip install rdkit==2023.9.4
pip install pandas==2.1.4
pip install numpy==1.26.3
pip install scipy==1.11.4

# Install visualization tools
echo "Installing visualization libraries..."
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install plotly==5.18.0

# Install optimization tools
echo "Installing optimization libraries..."
pip install networkx==3.2.1
pip install scikit-learn==1.4.0

# Install notebook support (optional)
echo "Installing Jupyter support..."
pip install jupyter==1.0.0
pip install ipywidgets==8.1.1

# Install utilities
echo "Installing utilities..."
pip install tqdm==4.66.1
pip install python-dotenv==1.0.0

# Create project structure
echo "Creating project structure..."
mkdir -p data/{raw,processed,results}
mkdir -p src/{classical,quantum,utils}
mkdir -p notebooks
mkdir -p figures
mkdir -p papers

# Create .env template for IBM Quantum credentials
echo "Creating .env template..."
cat > .env.template << 'EOF'
# IBM Quantum Credentials
# Get your token from: https://quantum.ibm.com/
IBM_QUANTUM_TOKEN=your_token_here
IBM_QUANTUM_CHANNEL=ibm_quantum
IBM_QUANTUM_INSTANCE=ibm_qasm_simulator
EOF

# Create requirements.txt
echo "Creating requirements.txt..."
pip freeze > requirements.txt

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Get your IBM Quantum token: https://quantum.ibm.com/"
echo "2. Copy .env.template to .env and add your token"
echo "3. Activate environment: source quantum_biodiversity_env/bin/activate"
echo "4. Run verification: python src/utils/verify_setup.py"
echo ""
echo "Project structure created:"
echo "  data/          - Raw and processed datasets"
echo "  src/           - Source code (classical, quantum, utilities)"
echo "  notebooks/     - Jupyter notebooks for exploration"
echo "  figures/       - Generated plots and visualizations"
echo "  papers/        - Paper drafts and LaTeX files"
echo ""
