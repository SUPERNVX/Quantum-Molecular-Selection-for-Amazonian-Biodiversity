# Instalação de Dependências - Execute este arquivo após criar o ambiente virtual
#
# USO:
#   .\quantum_env\Scripts\Activate.ps1
#   python install_dependencies.py

import subprocess
import sys

def install_package(package):
    """Instala um pacote via pip"""
    print(f"Instalando {package}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package, "--quiet"
        ])
        print(f"  ✓ {package} instalado")
        return True
    except subprocess.CalledProcessError:
        print(f"  ✗ Erro ao instalar {package}")
        return False

def main():
    print("=" * 60)
    print("INSTALAÇÃO DE DEPENDÊNCIAS")
    print("=" * 60)
    print()
    
    # Atualizar pip
    print("Atualizando pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    print("  ✓ pip atualizado\n")
    
    # Lista de pacotes
    packages = [
        # Quantum Computing
        ("Qiskit", "qiskit==1.0.0"),
        ("Qiskit Aer", "qiskit-aer==0.13.3"),
        ("Qiskit IBM Runtime", "qiskit-ibm-runtime==0.18.0"),
        
        # Chemistry
        ("RDKit", "rdkit"),
        
        # Data Science
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("SciPy", "scipy"),
        ("Scikit-learn", "scikit-learn"),
        
        # Visualization
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("Plotly", "plotly"),
        
        # Utilities
        ("NetworkX", "networkx"),
        ("TQDM", "tqdm"),
        ("Python-dotenv", "python-dotenv"),
        ("Requests", "requests"),
    ]
    
    failed = []
    
    for name, package in packages:
        if not install_package(package):
            failed.append(name)
    
    print()
    print("=" * 60)
    
    if not failed:
        print("✓ Todas as dependências instaladas com sucesso!")
        print()
        print("Próximos passos:")
        print("  1. Executar: python verify_setup.py")
        print("  2. Coletar dados: python src\\utils\\data_collection.py")
    else:
        print(f"✗ {len(failed)} pacotes falharam:")
        for name in failed:
            print(f"  - {name}")
        print()
        print("Tente instalar manualmente:")
        for name in failed:
            pkg = next(p for n, p in packages if n == name)
            print(f"  pip install {pkg}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
