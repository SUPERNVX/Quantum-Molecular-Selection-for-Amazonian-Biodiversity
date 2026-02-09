"""
quantum_molecular_selection.py

Quantum Approximate Optimization Algorithm (QAOA) for molecular diversity
Uses IBM Quantum to solve MaxCut formulation of diversity problem
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import time

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler as IBMSampler

# Optimization
from scipy.optimize import minimize

# Chemistry
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


class QuantumMolecularSelector:
    """
    QAOA-based molecular diversity optimization
    
    Key Idea:
    - Convert diversity problem to QUBO (Quadratic Unconstrained Binary Optimization)
    - Encode as Ising Hamiltonian
    - Use QAOA to find ground state
    - Decode solution to molecular selection
    
    QAOA Advantage:
    - Can escape local optima (unlike greedy)
    - Explores solution space efficiently
    - Scalable to larger problems
    """
    
    def __init__(self, 
                 molecules_df: pd.DataFrame,
                 use_real_quantum: bool = False,
                 ibm_token: str = None):
        """
        Args:
            molecules_df: DataFrame with molecular data
            use_real_quantum: Use IBM Quantum hardware (vs. simulator)
            ibm_token: IBM Quantum API token
        """
        self.molecules_df = molecules_df
        self.use_real_quantum = use_real_quantum
        
        # Compute fingerprints and similarity
        self.fingerprints = self._compute_fingerprints()
        self.similarity_matrix = self._compute_similarity_matrix()
        
        # IBM Quantum setup
        if use_real_quantum:
            assert ibm_token is not None, "IBM token required for real quantum"
            self.service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=ibm_token
            )
            self.backend = self._select_backend()
        else:
            self.backend = AerSimulator()
    
    def _compute_fingerprints(self) -> List:
        """Compute Morgan fingerprints"""
        print("Computing molecular fingerprints...")
        fingerprints = []
        
        for smiles in self.molecules_df['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fingerprints.append(fp)
        
        return fingerprints
    
    def _compute_similarity_matrix(self) -> np.ndarray:
        """Compute pairwise Tanimoto similarities"""
        print("Computing similarity matrix...")
        n = len(self.fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                sim = DataStructs.TanimotoSimilarity(
                    self.fingerprints[i],
                    self.fingerprints[j]
                )
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
        
        return similarity_matrix
    
    def _select_backend(self):
        """Select least busy IBM Quantum backend"""
        print("Selecting IBM Quantum backend...")
        backends = self.service.backends(
            filters=lambda x: x.configuration().n_qubits >= 20 and 
                            not x.configuration().simulator and
                            x.status().operational
        )
        
        # Sort by queue size
        backends_sorted = sorted(backends, 
                                key=lambda x: x.status().pending_jobs)
        
        selected = backends_sorted[0]
        print(f"Selected backend: {selected.name}")
        print(f"  Qubits: {selected.configuration().n_qubits}")
        print(f"  Pending jobs: {selected.status().pending_jobs}")
        
        return selected
    
    def formulate_qubo(self, k: int) -> Tuple[np.ndarray, float]:
        """
        Convert diversity maximization to QUBO
        
        Diversity Maximization:
            maximize: sum_{i<j} (1 - S_ij) * x_i * x_j
            subject to: sum_i x_i = k
        
        Where:
            x_i ∈ {0,1} = molecule i is selected
            S_ij = Tanimoto similarity between i and j
        
        QUBO Reformulation:
            minimize: -sum_{i<j} (1 - S_ij) * x_i * x_j + penalty * (sum_i x_i - k)^2
        
        Args:
            k: Number of molecules to select
            
        Returns:
            Q: QUBO matrix (n x n)
            penalty: Penalty coefficient for constraint
        """
        n = len(self.fingerprints)
        Q = np.zeros((n, n))
        
        # Objective: maximize diversity = minimize negative diversity
        for i in range(n):
            for j in range(i+1, n):
                distance = 1.0 - self.similarity_matrix[i][j]
                Q[i][j] = -distance  # Negative because we're minimizing
                Q[j][i] = -distance
        
        # Constraint penalty: (sum x_i - k)^2
        penalty = 10.0  # Tunable parameter
        
        # Expand: (sum x_i)^2 - 2k * sum x_i + k^2
        # Add to diagonal and off-diagonal
        for i in range(n):
            Q[i][i] += penalty  # Coefficient of x_i^2
            Q[i][i] -= 2 * penalty * k  # Linear term
        
        for i in range(n):
            for j in range(i+1, n):
                Q[i][j] += 2 * penalty  # Coefficient of x_i * x_j
                Q[j][i] += 2 * penalty
        
        return Q, penalty
    
    def qubo_to_ising(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert QUBO to Ising Hamiltonian
        
        QUBO: minimize x^T Q x, x ∈ {0,1}^n
        Ising: minimize z^T J z + h^T z, z ∈ {-1,+1}^n
        
        Transformation: x_i = (1 + z_i) / 2
        
        Returns:
            J: Coupling matrix
            h: External field
            offset: Constant offset
        """
        n = Q.shape[0]
        
        J = Q / 4.0
        h = np.sum(Q, axis=1) / 2.0
        offset = np.sum(Q) / 4.0
        
        return J, h, offset
    
    def create_qaoa_circuit(self, 
                           J: np.ndarray, 
                           h: np.ndarray, 
                           p: int = 1) -> QuantumCircuit:
        """
        Create QAOA circuit for Ising Hamiltonian
        
        QAOA Ansatz (p layers):
        1. Initial state: |+>^n (equal superposition)
        2. For each layer l=1..p:
            a. Problem Hamiltonian: exp(-i * gamma_l * H_problem)
            b. Mixer Hamiltonian: exp(-i * beta_l * H_mixer)
        3. Measure in computational basis
        
        Args:
            J: Coupling matrix
            h: External field
            p: Number of QAOA layers (depth)
            
        Returns:
            QuantumCircuit with parametrized QAOA ansatz
        """
        n = J.shape[0]
        
        # Create quantum and classical registers
        qr = QuantumRegister(n, 'q')
        cr = ClassicalRegister(n, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Parameters for QAOA
        gamma = [Parameter(f'γ_{i}') for i in range(p)]
        beta = [Parameter(f'β_{i}') for i in range(p)]
        
        # Initial state: Hadamard on all qubits
        qc.h(range(n))
        qc.barrier()
        
        # QAOA layers
        for layer in range(p):
            # Problem Hamiltonian: exp(-i * gamma * H_problem)
            # H_problem = sum_ij J_ij Z_i Z_j + sum_i h_i Z_i
            
            # ZZ interactions
            for i in range(n):
                for j in range(i+1, n):
                    if abs(J[i][j]) > 1e-6:
                        qc.rzz(2 * gamma[layer] * J[i][j], qr[i], qr[j])
            
            # Z rotations
            for i in range(n):
                if abs(h[i]) > 1e-6:
                    qc.rz(2 * gamma[layer] * h[i], qr[i])
            
            qc.barrier()
            
            # Mixer Hamiltonian: exp(-i * beta * H_mixer)
            # H_mixer = sum_i X_i
            for i in range(n):
                qc.rx(2 * beta[layer], qr[i])
            
            qc.barrier()
        
        # Measurement
        qc.measure(qr, cr)
        
        return qc
    
    def compute_expectation(self, 
                           counts: Dict[str, int], 
                           J: np.ndarray, 
                           h: np.ndarray, 
                           offset: float) -> float:
        """
        Compute expectation value of Ising Hamiltonian from measurement counts
        
        E = <z^T J z + h^T z> + offset
        
        Args:
            counts: Measurement counts {bitstring: count}
            J: Coupling matrix
            h: External field
            offset: Constant offset
            
        Returns:
            Expectation value
        """
        n = J.shape[0]
        expectation = 0.0
        total_shots = sum(counts.values())
        
        for bit_key, count in counts.items():
            # Handle both bitstring (str) and integer (int) keys from different Samplers
            if isinstance(bit_key, str):
                bitstring = bit_key
            else:
                bitstring = format(bit_key, f'0{n}b')
                
            # Convert bitstring to spin vector (-1, +1)
            # Reverse bitstring because qubit 0 is usually the rightmost bit in binary representation
            z = np.array([1 if bit == '0' else -1 
                         for bit in bitstring[::-1]])
            
            # Compute energy: z^T J z + h^T z
            energy = np.dot(z, np.dot(J, z)) + np.dot(h, z) + offset
            
            # Weighted by probability
            expectation += energy * (count / total_shots)
        
        return expectation
    
    def qaoa_optimize(self, 
                     k: int = 20, 
                     p: int = 1, 
                     shots: int = 1024,
                     maxiter: int = 100) -> Tuple[List[int], float, float]:
        """
        Run QAOA optimization
        
        Workflow:
        1. Formulate QUBO
        2. Convert to Ising
        3. Create QAOA circuit
        4. Classical optimization loop:
            a. Bind parameters
            b. Run circuit on quantum backend
            c. Compute expectation
            d. Update parameters
        5. Extract solution from best measurement
        
        Args:
            k: Number of molecules to select
            p: QAOA depth
            shots: Number of measurements per evaluation
            maxiter: Maximum optimization iterations
            
        Returns:
            (selected_indices, diversity_score, execution_time)
        """
        print(f"\n{'='*60}")
        print(f"QUANTUM APPROXIMATE OPTIMIZATION (QAOA)")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"  k (selection size): {k}")
        print(f"  p (QAOA depth): {p}")
        print(f"  shots: {shots}")
        print(f"  backend: {self.backend.name if hasattr(self.backend, 'name') else 'simulator'}")
        
        start_time = time.time()
        
        # Step 1: Formulate QUBO
        print("\n[1/5] Formulating QUBO...")
        Q, penalty = self.formulate_qubo(k)
        
        # Step 2: Convert to Ising
        print("[2/5] Converting to Ising Hamiltonian...")
        J, h, offset = self.qubo_to_ising(Q)
        
        # Step 3: Create QAOA circuit
        print("[3/5] Creating QAOA circuit...")
        qaoa_circuit = self.create_qaoa_circuit(J, h, p)
        print(f"  Circuit depth: {qaoa_circuit.depth()}")
        print(f"  Number of gates: {len(qaoa_circuit.data)}")
        
        # Step 4: Classical optimization
        print("[4/5] Running classical optimization...")
        
        best_counts = None
        best_expectation = np.inf
        
        def objective_function(params):
            """Objective: minimize expectation value"""
            nonlocal best_counts, best_expectation
            
            # Bind parameters to circuit
            param_dict = {}
            for i in range(p):
                param_dict[f'γ_{i}'] = params[i]
                param_dict[f'β_{i}'] = params[p + i]
            
            bound_circuit = qaoa_circuit.assign_parameters(param_dict)
            
            # Execute circuit
            if self.use_real_quantum:
                # Use IBM Quantum hardware
                with Session(service=self.service, backend=self.backend) as session:
                    sampler = IBMSampler(session=session)
                    job = sampler.run(bound_circuit, shots=shots)
                    result = job.result()
                    counts = result.quasi_dists[0].binary_probabilities()
            else:
                # Use local simulator
                sampler = AerSampler()
                job = sampler.run(bound_circuit, shots=shots)
                result = job.result()
                counts = result.quasi_dists[0]
            
            # Compute expectation
            expectation = self.compute_expectation(counts, J, h, offset)
            
            if expectation < best_expectation:
                best_expectation = expectation
                best_counts = counts
            
            return expectation
        
        # Initial parameters (random)
        initial_params = np.random.uniform(0, 2*np.pi, 2*p)
        
        # Optimize
        result = minimize(
            objective_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': maxiter}
        )
        
        print(f"  Optimization converged: {result.success}")
        print(f"  Final expectation: {result.fun:.4f}")
        print(f"  Function evaluations: {result.nfev}")
        
        # Step 5: Extract solution
        print("[5/5] Extracting solution...")
        
        # Find most common bitstring (that satisfies constraint)
        valid_solutions = []
        
        for bit_key, count in best_counts.items():
            # Handle both bitstring (str) and integer (int) keys
            if isinstance(bit_key, str):
                bitstring = bit_key
            else:
                bitstring = format(bit_key, f'0{len(J)}b')
                
            # Count selected molecules
            selected = [i for i, bit in enumerate(bitstring[::-1]) if bit == '0']
            
            if len(selected) == k:  # Satisfies constraint
                valid_solutions.append((selected, count))
        
        if valid_solutions:
            # Sort by frequency
            valid_solutions.sort(key=lambda x: x[1], reverse=True)
            best_solution = valid_solutions[0][0]
        else:
            # No valid solution - take closest
            print("  Warning: No exact solution found, using approximation")
            all_solutions = [(bitstring, count) for bitstring, count in best_counts.items()]
            all_solutions.sort(key=lambda x: x[1], reverse=True)
            bitstring = all_solutions[0][0]
            selected = [i for i, bit in enumerate(bitstring[::-1]) if bit == '0']
            
            # Adjust to exactly k molecules
            if len(selected) < k:
                remaining = set(range(len(J))) - set(selected)
                selected.extend(list(remaining)[:k - len(selected)])
            elif len(selected) > k:
                selected = selected[:k]
            
            best_solution = selected
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity(best_solution)
        
        execution_time = time.time() - start_time
        
        print(f"\nQAOA Result:")
        print(f"  Selected molecules: {len(best_solution)}")
        print(f"  Diversity score: {diversity_score:.4f}")
        print(f"  Execution time: {execution_time:.2f}s")
        
        return best_solution, diversity_score, execution_time
    
    def _calculate_diversity(self, selected_indices: List[int]) -> float:
        """Calculate diversity score for solution"""
        diversity = 0.0
        for i in range(len(selected_indices)):
            for j in range(i+1, len(selected_indices)):
                idx_i = selected_indices[i]
                idx_j = selected_indices[j]
                distance = 1.0 - self.similarity_matrix[idx_i][idx_j]
                diversity += distance
        
        return diversity


def main():
    """Example usage"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load dataset
    print("Loading molecular dataset...")
    df = pd.read_csv('data/processed/amazonian_molecules.csv')
    print(f"Loaded {len(df)} molecules")
    
    # FIRST: Test with simulator (Limit to 15 qubits for local simulation)
    print("\n" + "="*60)
    print("TESTING WITH SIMULATOR (Sub-amostragem para 15 qubits)")
    print("="*60)
    
    df_subset = df.head(15)
    selector_sim = QuantumMolecularSelector(df_subset, use_real_quantum=False)
    qaoa_selection_sim, qaoa_diversity_sim, qaoa_time_sim = selector_sim.qaoa_optimize(
        k=5, # Reduzido proporcionalmente
        p=1, 
        shots=1024
    )
    
    # THEN: If simulator works, use real quantum
    use_real = os.getenv('RUN_REAL_QUANTUM', 'n').lower() == 'y'
    
    if use_real:
        token = os.getenv('IBM_QUANTUM_TOKEN')
        
        if not token or token == 'your_token_here':
            print("ERROR: IBM Quantum token not configured in .env")
            return
        
        print("\n" + "="*60)
        print("RUNNING ON REAL IBM QUANTUM HARDWARE")
        print("="*60)
        
        selector_real = QuantumMolecularSelector(df, use_real_quantum=True, ibm_token=token)
        qaoa_selection_real, qaoa_diversity_real, qaoa_time_real = selector_real.qaoa_optimize(
            k=20, 
            p=1,  # Keep p=1 to save quantum time
            shots=512  # Reduce shots to save time
        )
        
        # Save results
        results = {
            'qaoa_simulator_selection': qaoa_selection_sim,
            'qaoa_simulator_diversity': qaoa_diversity_sim,
            'qaoa_simulator_time': qaoa_time_sim,
            'qaoa_real_selection': qaoa_selection_real,
            'qaoa_real_diversity': qaoa_diversity_real,
            'qaoa_real_time': qaoa_time_real,
        }
    else:
        results = {
            'qaoa_simulator_selection': qaoa_selection_sim,
            'qaoa_simulator_diversity': qaoa_diversity_sim,
            'qaoa_simulator_time': qaoa_time_sim,
        }
    
    pd.DataFrame([results]).to_csv('data/results/quantum_results.csv', index=False)
    print("\nResults saved to data/results/quantum_results.csv")


if __name__ == "__main__":
    main()
