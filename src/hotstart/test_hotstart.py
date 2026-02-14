import unittest
import os
import pandas as pd
from src.hotstart.classical import MolecularDiversitySelector
from src.hotstart.quantum import build_molecular_hamiltonian, build_biased_initial_state

class TestHotstartSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup paths
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.data_path = os.path.join(cls.base_dir, 'data/processed/brnpdb.csv')
        cls.cache_dir = os.path.join(cls.base_dir, 'data/processed/hotstart_test')
        
        # Load small data for testing
        if os.path.exists(cls.data_path):
            cls.df = pd.read_csv(cls.data_path).head(5)
        else:
            cls.df = pd.DataFrame({'smiles': ['C', 'CC', 'CCC', 'CCCC', 'CCCCC'], 'id': range(5)})

    def test_classical_selector(self):
        selector = MolecularDiversitySelector(self.df, cache_dir=self.cache_dir)
        indices, score, _, bitstring = selector.greedy_selection(k=2)
        
        self.assertEqual(len(indices), 2)
        self.assertEqual(len(bitstring), 5)
        self.assertIn('1', bitstring)

    def test_quantum_operators(self):
        selector = MolecularDiversitySelector(self.df, cache_dir=self.cache_dir)
        hamiltonian = build_molecular_hamiltonian(selector.similarity_matrix, k=2)
        
        # Check if hamiltonian is a SparsePauliOp
        from qiskit.quantum_info import SparsePauliOp
        self.assertIsInstance(hamiltonian, SparsePauliOp)
        
        # Check initial state
        initial_state = build_biased_initial_state(5, "10001", alpha=0.8)
        self.assertEqual(initial_state.num_qubits, 5)

if __name__ == '__main__':
    unittest.main()
