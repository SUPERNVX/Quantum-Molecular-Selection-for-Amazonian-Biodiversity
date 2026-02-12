"""
lite_selector.py

Versão Otimizada para SIMULAÇÃO LOCAL (CPU/GPU).
Remove overhead de transpilação e primitives V2 para máxima velocidade.
Ideal para N < 20 qubits.
"""

import numpy as np
import pandas as pd
import time
from typing import List, Tuple

# Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

# Otimização
from scipy.optimize import minimize

# Química
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs

class QuantumMolecularSelectorLite:
    def __init__(self, molecules_df: pd.DataFrame):
        self.molecules_df = molecules_df.copy()
        
        # 1. Pré-processamento Rápido
        self.mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.fingerprints, self.valid_indices = self._compute_fingerprints()
        self.molecules_df = self.molecules_df.iloc[self.valid_indices].reset_index(drop=True)
        self.n_qubits = len(self.fingerprints)
        
        self.similarity_matrix = self._compute_similarity_matrix()
        print(f"[OK] Inicializado para {self.n_qubits} moleculas.")

    def _compute_fingerprints(self) -> Tuple[List, List[int]]:
        fps = []
        valid_idx = []
        for idx, smiles in enumerate(self.molecules_df['smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = self.mfgen.GetFingerprint(mol)
                fps.append(fp)
                valid_idx.append(idx)
        return fps, valid_idx

    def _compute_similarity_matrix(self) -> np.ndarray:
        n = len(self.fingerprints)
        matrix = np.zeros((n, n))
        for i in range(n):
            if i < n - 1:
                sims = DataStructs.BulkTanimotoSimilarity(
                    self.fingerprints[i], self.fingerprints[i+1:]
                )
                for k_offset, sim in enumerate(sims):
                    j = i + 1 + k_offset
                    matrix[i][j] = matrix[j][i] = sim
        return matrix

    def formulate_problem(self, k: int) -> Tuple[SparsePauliOp, float]:
        n = self.n_qubits
        penalty = 2.0 * k 
        pauli_list = []
        offset = 0.0
        
        h_coeffs = np.zeros(n)
        J_coeffs = {} 
        
        for i in range(n):
            Q_ii = penalty * (1 - 2 * k)
            h_coeffs[i] += Q_ii * 0.5
            offset += Q_ii * 0.5

        for i in range(n):
            for j in range(i + 1, n):
                Q_ij = (self.similarity_matrix[i][j] - 1.0) + (2 * penalty)
                term_val = Q_ij / 4.0
                offset += term_val
                h_coeffs[i] -= term_val
                h_coeffs[j] -= term_val
                J_coeffs[(i,j)] = term_val

        for i in range(n):
            if abs(h_coeffs[i]) > 1e-6:
                label = ['I'] * n
                label[n - 1 - i] = 'Z'
                pauli_list.append(("".join(label), h_coeffs[i]))
        
        for (i, j), val in J_coeffs.items():
            if abs(val) > 1e-6:
                label = ['I'] * n
                label[n - 1 - i] = 'Z'
                label[n - 1 - j] = 'Z'
                pauli_list.append(("".join(label), val))

        H = SparsePauliOp.from_list(pauli_list)
        return H, offset

    def build_circuit(self, params: List[float], p: int):
        n = self.n_qubits
        qc = QuantumCircuit(n)
        qc.h(range(n))
        gammas = params[:p]
        betas = params[p:]
        
        # Hamiltoniano de Custo (H_P) - Gerar novamente ou passar via cache
        # Para velocidade no loop de otimização, evitamos reconstruir o hamiltoniano.
        # Mas para o build_circuit puro, usamos os parâmetros.
        return qc # Este método será usado internamente de forma otimizada

    def qaoa_optimize_lite(self, k: int, p: int = 1) -> Tuple[List[int], float]:
        print(f"\n[OK] QAOA Turbo Lite | N={self.n_qubits}, k={k}, p={p}")
        start_time = time.time()
        H, offset = self.formulate_problem(k)
        
        def build_fast_qc(params):
            qc = QuantumCircuit(self.n_qubits)
            qc.h(range(self.n_qubits))
            gammas = params[:p]
            betas = params[p:]
            for layer in range(p):
                gamma = gammas[layer]
                beta = betas[layer]
                for pauli, coeff in zip(H.paulis, H.coeffs):
                    label = pauli.to_label()
                    indices = [idx for idx, char in enumerate(reversed(label)) if char == 'Z']
                    if len(indices) == 1:
                        qc.rz(2 * gamma * coeff.real, indices[0])
                    elif len(indices) == 2:
                        qc.rzz(2 * gamma * coeff.real, indices[0], indices[1])
                qc.rx(2 * beta, range(self.n_qubits))
            return qc

        def objective_function(params):
            qc = build_fast_qc(params)
            sv = Statevector(qc)
            return sv.expectation_value(H).real

        # Início TQA
        initial_params = [0.1, 0.4] * p
        res = minimize(objective_function, initial_params, method='COBYLA', options={'maxiter': 100})
        
        final_qc = build_fast_qc(res.x)
        sv = Statevector(final_qc)
        probs = sv.probabilities_dict()
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        best_solution = None
        for bitstring, prob in sorted_probs[:500]:
            selected = [i for i, b in enumerate(reversed(bitstring)) if b == '0']
            if len(indices := selected) and len(indices) == k:
                best_solution = indices
                break
        
        if not best_solution:
            best_solution = list(range(k))
            
        elapsed = time.time() - start_time
        return best_solution, elapsed

if __name__ == "__main__":
    df = pd.DataFrame({'smiles': ['C', 'CC', 'CCC', 'CCCC']})
    selector = QuantumMolecularSelectorLite(df)
    res, t = selector.qaoa_optimize_lite(k=2)
    print(f"Resultado: {res}")
