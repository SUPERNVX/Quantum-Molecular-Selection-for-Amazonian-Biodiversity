"""
quantum_molecular_selection_lite.py

Versão Otimizada para SIMULAÇÃO LOCAL (CPU/GPU).
Remove overhead de transpilação e primitives V2 para máxima velocidade.
Ideal para N < 20 qubits.
"""

import numpy as np
import pandas as pd
import time
from typing import List, Tuple

# Qiskit Minimalista (Focado em Performance Local)
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit import ParameterVector

# Otimização
from scipy.optimize import minimize

# Química
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

class QuantumMolecularSelectorLite:
    def __init__(self, molecules_df: pd.DataFrame):
        self.molecules_df = molecules_df.copy()
        
        # 1. Pré-processamento Rápido
        self.fingerprints, self.valid_indices = self._compute_fingerprints()
        self.molecules_df = self.molecules_df.iloc[self.valid_indices].reset_index(drop=True)
        self.similarity_matrix = self._compute_similarity_matrix()
        
        n_qubits = len(self.fingerprints)
        print(f"✅ Inicializado para {n_qubits} moléculas.")
        if n_qubits > 24:
            print("⚠️ AVISO: N > 24 pode ser lento para simulação de Statevector local.")

    def _compute_fingerprints(self) -> Tuple[List, List[int]]:
        fps = []
        valid_idx = []
        for idx, smiles in enumerate(self.molecules_df['smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
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
                    matrix[i][j] = sim
                    matrix[j][i] = sim
        return matrix

    def formulate_problem(self, k: int) -> Tuple[SparsePauliOp, float]:
        """
        Gera Hamiltoniano Ising e Offset.
        Usa penalidade calibrada para simulação exata.
        """
        n = len(self.fingerprints)
        
        # Heurística de Penalidade:
        # Deve ser maior que o ganho máximo de diversidade possível.
        # Max diversidade ganha por 1 molécula ≈ k (pois dist <= 1)
        penalty = 2.0 * k 
        
        pauli_list = []
        offset = 0.0
        
        # Construção Direta dos Coeficientes (Sem matriz Q intermédia para economizar memória)
        
        # 1. Termos Lineares (Z_i)
        # Vêm da restrição: P(1-2k)x_i -> coeff de Z_i é -0.5 * P(1-2k)
        # E da diversidade (Q_ij off-diagonal contribui para Z_i e Z_j)
        
        h_coeffs = np.zeros(n)
        J_coeffs = {} # (i,j) -> val
        
        # Diagonal (Linear) contributions from Constraint
        # Q_ii = P * (1 - 2k)
        # Ising h_i += Q_ii / 2
        for i in range(n):
            Q_ii = penalty * (1 - 2 * k)
            h_coeffs[i] += Q_ii * 0.5
            offset += Q_ii * 0.5 # from identity part

        # Off-diagonal contributions
        for i in range(n):
            for j in range(i + 1, n):
                # Q_ij = (sim_ij - 1) + 2*P
                Q_ij = (self.similarity_matrix[i][j] - 1.0) + (2 * penalty)
                
                # Ising Map: x_i x_j -> (I - Z_i - Z_j + Z_i Z_j) / 4
                term_val = Q_ij / 4.0
                
                offset += term_val
                h_coeffs[i] -= term_val # contribui para Z_i
                h_coeffs[j] -= term_val # contribui para Z_j
                J_coeffs[(i,j)] = term_val # contribui para Z_i Z_j

        # Construir SparsePauliOp
        for i in range(n):
            if abs(h_coeffs[i]) > 1e-6:
                label = ['I'] * n
                label[n - 1 - i] = 'Z' # Little Endian
                pauli_list.append(("".join(label), h_coeffs[i]))
        
        for (i, j), val in J_coeffs.items():
            if abs(val) > 1e-6:
                label = ['I'] * n
                label[n - 1 - i] = 'Z'
                label[n - 1 - j] = 'Z'
                pauli_list.append(("".join(label), val))

        H = SparsePauliOp.from_list(pauli_list)
        return H, offset

    def qaoa_optimize_lite(self, k: int, p: int = 1) -> Tuple[List[int], float]:
        """
        QAOA Otimizado para CPU Local (Statevector).
        """
        print(f"\n⚡ QAOA Turbo Lite | N={len(self.fingerprints)}, k={k}, p={p}")
        start_time = time.time()
        
        n_qubits = len(self.fingerprints)
        H, offset = self.formulate_problem(k)
        
        # --- Construção Manual do Circuito (Mais leve que QAOAAnsatz) ---
        def build_circuit(params):
            qc = QuantumCircuit(n_qubits)
            # Camada Inicial |+>
            qc.h(range(n_qubits))
            
            # Extrair gammas e betas
            gammas = params[:p]
            betas = params[p:]
            
            for layer in range(p):
                gamma = gammas[layer]
                beta = betas[layer]
                
                # 1. Cost Hamiltonian (Evolução H_P)
                # Como H é diagonal (apenas Z e ZZ), podemos aplicar portas diretamente
                # RZ(theta) = exp(-i * theta/2 * Z)
                # Queremos exp(-i * gamma * H)
                # Se termo é a*Z, aplicamos RZ(2*gamma*a)
                # Se termo é a*ZZ, aplicamos RZZ(2*gamma*a)
                
                # Iterar sobre os Paulis do Hamiltoniano para aplicar portas
                # (Isso é muito mais rápido que transpilar para statevector)
                for pauli, coeff in zip(H.paulis, H.coeffs):
                    coeff_real = coeff.real
                    s_str = pauli.to_label() # string tipo "IZZ..."
                    
                    # Identificar índices ativos (onde não é 'I')
                    indices = [k for k, char in enumerate(reversed(s_str)) if char == 'Z']
                    
                    if len(indices) == 1:
                        # RZ gate
                        qc.rz(2 * gamma * coeff_real, indices[0])
                    elif len(indices) == 2:
                        # RZZ gate
                        qc.rzz(2 * gamma * coeff_real, indices[0], indices[1])
                        
                # 2. Mixer Hamiltonian (Evolução H_M = sum X)
                qc.rx(2 * beta, range(n_qubits))
                
            return qc

        # --- Função Objetivo Ultra-Rápida ---
        # Cache para evitar reconstruir hamiltoniano
        H_matrix = H.to_matrix() # Pré-calcula matriz densa (OK para N < 16)
        # Se N > 16, usar H apenas como operador esparso (SparsePauliOp trata isso)
        
        def objective_function(params):
            qc = build_circuit(params)
            
            # A mágica da velocidade: Statevector exato
            # Não faz sampling, calcula <psi|H|psi> via álgebra linear
            sv = Statevector(qc)
            
            # Expectation value
            # Para N < 16, matrix multiplication é instantânea
            energy = sv.expectation_value(H).real
            return energy

        # Inicialização TQA
        initial_params = []
        dt = 0.7
        for i in range(p): initial_params.append((i+1)/p * dt) # Gammas
        for i in range(p): initial_params.append((1-(i+1)/p) * dt) # Betas

        print("   Otimizando (COBYLA via Statevector)...")
        res = minimize(
            objective_function, 
            initial_params, 
            method='COBYLA', 
            options={'maxiter': 100, 'tol': 1e-4} # Tol ajuda a parar se convergir
        )
        
        print(f"   Convergiu. Energia: {res.fun + offset:.4f}")
        
        # Extrair Resultado Final
        final_qc = build_circuit(res.x)
        sv = Statevector(final_qc)
        probs = sv.probabilities_dict() # Retorna {bitstring: probabilidade} exata
        
        # Selecionar melhor bitstring válido
        # Ordenar por probabilidade
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        best_solution = None
        
        for bitstring, prob in sorted_probs:
            # Reverter bitstring (Little Endian)
            rev_bits = bitstring[::-1]
            selected = [i for i, bit in enumerate(rev_bits) if bit == '1']
            
            if len(selected) == k:
                best_solution = selected
                print(f"✅ Solução Exata encontrada: {bitstring} (Prob: {prob:.4f})")
                break
        
        if not best_solution:
            print("⚠️ Nenhuma solução exata no topo. Pegando a mais provável e ajustando.")
            top_str = sorted_probs[0][0][::-1]
            best_solution = [i for i, bit in enumerate(top_str) if bit == '1']
            # Ajuste bruto
            best_solution = best_solution[:k]
            while len(best_solution) < k:
                for i in range(n_qubits):
                    if i not in best_solution:
                        best_solution.append(i)
                        break
                        
        elapsed = time.time() - start_time
        print(f"⏱️ Tempo: {elapsed:.2f}s")
        return best_solution, elapsed

if __name__ == "__main__":
    # Teste de Stress
    print("Gerando dataset N=12...")
    # Criar moléculas dummy para testar carga
    dummy_smiles = [
        'C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC', 
        'O', 'CO', 'CCO', 'CCCO', 'CCCCO', 'CCCCCO'
    ]
    df = pd.DataFrame({'smiles': dummy_smiles})
    
    # Rodar versão Lite
    selector = QuantumMolecularSelectorLite(df)
    
    # Deve rodar instantaneamente
    indices, t = selector.qaoa_optimize_lite(k=4, p=1)
    print("Índices:", indices)