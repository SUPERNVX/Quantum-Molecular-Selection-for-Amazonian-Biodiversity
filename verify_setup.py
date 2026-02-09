"""
Verification script for Quantum Biodiversity Project
Tests all dependencies and IBM Quantum connection
"""

import sys
from typing import Dict, Tuple

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        return True, f"✓ {package_name or module_name}"
    except ImportError as e:
        return False, f"✗ {package_name or module_name}: {str(e)}"

def verify_dependencies() -> Dict[str, bool]:
    """Verify all required dependencies"""
    print("=" * 50)
    print("DEPENDENCY VERIFICATION")
    print("=" * 50)
    
    dependencies = {
        # Quantum computing
        'qiskit': 'Qiskit',
        'qiskit_aer': 'Qiskit Aer',
        'qiskit_ibm_runtime': 'Qiskit IBM Runtime',
        
        # Chemistry
        'rdkit': 'RDKit',
        
        # Data science
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        
        # Visualization
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'plotly': 'Plotly',
        
        # Graph/Network
        'networkx': 'NetworkX',
        
        # Utilities
        'tqdm': 'TQDM',
        'dotenv': 'Python-dotenv',
    }
    
    results = {}
    for module, name in dependencies.items():
        success, message = test_import(module, name)
        print(message)
        results[name] = success
    
    return results

def verify_qiskit_version():
    """Check Qiskit version and capabilities"""
    print("\n" + "=" * 50)
    print("QISKIT VERSION CHECK")
    print("=" * 50)
    
    try:
        import qiskit
        print(f"✓ Qiskit version: {qiskit.__version__}")
        
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        print("✓ Can create quantum circuits")
        
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        print("✓ Local simulator available")
        
        return True
    except Exception as e:
        print(f"✗ Qiskit verification failed: {e}")
        return False

def verify_ibm_connection():
    """Verify IBM Quantum connection"""
    print("\n" + "=" * 50)
    print("IBM QUANTUM CONNECTION")
    print("=" * 50)
    
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        token = os.getenv('IBM_QUANTUM_TOKEN')
        
        if not token or token == 'your_token_here':
            print("⚠ IBM Quantum token not configured")
            print("  → Get your token: https://quantum.ibm.com/")
            print("  → Add to .env file: IBM_QUANTUM_TOKEN=your_token")
            return False
        
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        # Try to authenticate
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        print("✓ IBM Quantum authentication successful")
        
        # List available backends
        backends = service.backends()
        print(f"✓ Available backends: {len(backends)}")
        
        # Check for real quantum hardware
        real_backends = [b for b in backends if not b.simulator]
        if real_backends:
            print(f"✓ Real quantum computers available: {len(real_backends)}")
            print(f"  Example: {real_backends[0].name}")
        
        return True
        
    except Exception as e:
        print(f"✗ IBM Quantum connection failed: {e}")
        print("  → Check your token in .env file")
        return False

def verify_rdkit():
    """Verify RDKit chemistry toolkit"""
    print("\n" + "=" * 50)
    print("RDKIT CHEMISTRY TOOLKIT")
    print("=" * 50)
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        # Test molecule creation
        mol = Chem.MolFromSmiles('CCO')  # Ethanol
        if mol:
            print("✓ Can create molecules from SMILES")
            
            # Test descriptor calculation
            mw = Descriptors.MolWt(mol)
            print(f"✓ Can calculate molecular descriptors (MW: {mw:.2f})")
            
            return True
        else:
            print("✗ Failed to create test molecule")
            return False
            
    except Exception as e:
        print(f"✗ RDKit verification failed: {e}")
        return False

def main():
    """Run all verifications"""
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║  QUANTUM BIODIVERSITY PROJECT - SETUP VERIFY  ║")
    print("╚" + "=" * 48 + "╝")
    print("\n")
    
    results = []
    
    # Test 1: Dependencies
    dep_results = verify_dependencies()
    results.append(all(dep_results.values()))
    
    # Test 2: Qiskit
    results.append(verify_qiskit_version())
    
    # Test 3: RDKit
    results.append(verify_rdkit())
    
    # Test 4: IBM Connection (optional)
    ibm_ok = verify_ibm_connection()
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    if all(results):
        print("✓ All core dependencies verified successfully!")
        if ibm_ok:
            print("✓ IBM Quantum connection verified!")
            print("\n🎉 You're ready to start the project!")
        else:
            print("⚠ IBM Quantum not configured (optional for now)")
            print("  You can work with local simulators")
            print("\n✓ You're ready to start development!")
    else:
        print("✗ Some dependencies failed verification")
        print("  → Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n")

if __name__ == "__main__":
    main()
