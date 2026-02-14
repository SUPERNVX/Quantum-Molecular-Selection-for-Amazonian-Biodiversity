"""
hybrid_selector.py

Hibrido de alto desempenho: Greedy (Warm Start) + QAOA (Refinamento).
Otimizado para simulacao local com Statevector.

Adaptado para usar o MolecularDiversitySelector da pasta hotstart.
"""

import os
import sys
import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict, Optional

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

# Otimizacao Classica
from scipy.optimize import minimize

# Importar o seletor classico da pasta hotstart
from src.hotstart.classical import MolecularDiversitySelector


class HybridQuantumSelector:
    """
    Seletor molecular hibrido: Greedy (Warm Start) + QAOA (Refinamento).
    Otimizado para GPU via AerSimulator.
    """
    
    def __init__(self, molecules_df: pd.DataFrame, cache_dir: str = "data/temp_hybrid", use_gpu: bool = True):
        """
        Inicializa o seletor hibrido.
        
        Args:
            molecules_df: DataFrame com coluna 'smiles'
            cache_dir: Diretorio para cache da matriz de similaridade
            use_gpu: Ativar aceleracao por GPU
        """
        self.molecules_df = molecules_df.copy()
        self.cache_dir = cache_dir
        
        # Usar o MolecularDiversitySelector para fingerprints e similaridade
        print(f"[Hybrid] Inicializando para {len(molecules_df)} moleculas...")
        self.classical_selector = MolecularDiversitySelector(molecules_df, cache_dir=cache_dir)
        
        # Copiar dados relevantes
        self.n_qubits = self.classical_selector.n_molecules
        self.similarity_matrix = self.classical_selector.similarity_matrix
        
        # Configurar Backend Aer
        backend_name = "Hybrid"
        try:
            if use_gpu:
                # Criar simulator com suporte explícito a NVIDIA cuStateVec conforme teste manual
                # cuStateVec_enable=True ativa a biblioteca de aceleração da NVIDIA
                try:
                    self.backend = AerSimulator(
                        method='statevector',
                        device='GPU',
                        cuStateVec_enable=True
                    )
                    # Testar funcionalidade com mini-circuito
                    test_circ = QuantumCircuit(1)
                    test_circ.id(0)
                    self.backend.run(test_circ).result()
                    print(f"[{backend_name}] GPU VALIDADA! Usando NVIDIA cuStateVec (RTX 4060).")
                except Exception as e:
                    print(f"[{backend_name}] AVISO: GPU aceita mas falhou no teste de run: {e}. Usando CPU...")
                    self.backend = AerSimulator(method='statevector', device='CPU')
            else:
                self.backend = AerSimulator(method='statevector', device='CPU')
        except Exception as e:
            print(f"[{backend_name}] AVISO: Erro ao configurar backend, usando CPU padrao. ({e})")
            self.backend = AerSimulator(method='statevector', device='CPU')
            
        print(f"[{backend_name}] Backend: {self.backend.options.get('device', 'CPU')}")

    def formulate_problem(self, k: int) -> SparsePauliOp:
        """
        Formula o problema QUBO como Hamiltoniano.
        Objetivo: Maximizar Diversidade = Minimizar -Somatorio(1 - S_ij) x_i x_j.
        Isso equivale a: Minimizar Somatorio(S_ij - 1) x_i x_j + lambda(Sum x_i - k)^2.
        """
        n = self.n_qubits
        # Aumentar penalty para N=25 para garantir validade (k=8)
        penalty = 15.0 if n >= 25 else 8.0
        
        pauli_list = []
        
        # Coeficientes QUBO: Q = Sum Q_ii x_i + Sum Q_ij x_i x_j
        # Da expansao de lambda(Sum x_i - k)^2:
        # Diagonal: lambda(1 - 2k)
        Q_ii = np.full(n, penalty * (1 - 2 * k))
        
        # Off-diagonal: (Similarity - 1) + 2*lambda
        # O -1.0 eh essencial para que S_ij=0 (maxima diversidade) seja o menor valor.
        Q_ij = (self.similarity_matrix - 1.0) + (2 * penalty)
        
        # Conversao para Ising (Hamiltoniano de Z):
        # H = Sum h_i Z_i + Sum J_ij Z_i Z_j
        h_coeffs = -Q_ii / 2.0
        J_coeffs = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                val = Q_ij[i][j] / 4.0
                h_coeffs[i] -= val
                h_coeffs[j] -= val
                J_coeffs[(i, j)] = val
        
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
        
        return SparsePauliOp.from_list(pauli_list)

    def build_biased_initial_state(self, greedy_indices: List[int], alpha: float = 0.85) -> QuantumCircuit:
        """
        Cria estado inicial enviesado pelo Greedy.
        """
        qc = QuantumCircuit(self.n_qubits)
        theta_idle = 2 * np.arcsin(np.sqrt(1 - alpha))
        theta_active = 2 * np.arcsin(np.sqrt(alpha))
        
        for i in range(self.n_qubits):
            if i in greedy_indices:
                qc.ry(theta_active, i)
            else:
                qc.ry(theta_idle, i)
        return qc

    def run_hybrid_optimize(self, k: int, p: int = 1, alpha: float = 0.85, maxiter: int = 50, initial_indices: Optional[List[int]] = None) -> Tuple[List[int], float, float]:
        """
        Executa otimizacao hibrida otimizada.
        """
        print(f"\n[Hybrid] Iniciando | N={self.n_qubits}, k={k}, p={p}, alpha={alpha}")
        start_time = time.time()
        
        # 1. Warm Start: Executar Greedy ou usar fornecido
        if initial_indices is not None:
            print(f"[Hybrid] Step 1: Using provided Warm Start indices...")
            greedy_indices = initial_indices
            greedy_score = self.classical_selector.calculate_diversity_score(greedy_indices)
        else:
            print(f"[Hybrid] Step 1: Greedy Warm Start...")
            greedy_indices, greedy_score, _, _ = self.classical_selector.greedy_selection(k=k)
            
        print(f"[Hybrid] Initial score (Warm Start): {greedy_score:.4f}")
        
        # 2. Preparar Ansatz
        H_op = self.formulate_problem(k)
        initial_state = self.build_biased_initial_state(greedy_indices, alpha=alpha)
        
        from qiskit.circuit import ParameterVector
        gammas = ParameterVector('gamma', p)
        betas = ParameterVector('beta', p)
        
        ansatz = QuantumCircuit(self.n_qubits)
        ansatz.compose(initial_state, inplace=True)
        
        for layer in range(p):
            for pauli, coeff in zip(H_op.paulis, H_op.coeffs):
                label = pauli.to_label()
                active = [i for i, char in enumerate(reversed(label)) if char == 'Z']
                if len(active) == 1:
                    ansatz.rz(2 * gammas[layer] * coeff.real, active[0])
                elif len(active) == 2:
                    ansatz.rzz(2 * gammas[layer] * coeff.real, active[0], active[1])
            ansatz.rx(2 * betas[layer], range(self.n_qubits))
            
        # 3. Otimizacao
        from qiskit_aer.library import save_expectation_value
        print(f"[Hybrid] Step 2: QAOA Refinement (Aer {self.backend.options.get('device', 'CPU')})...")
        
        # Acompanhamento de progresso
        self._iter_count = 0
        self._last_print_time = time.time()
        
        def objective(params_values):
            self._iter_count += 1
            current_time = time.time()
            
            if self._iter_count == 1 or current_time - self._last_print_time > 5:
                print(f"[Hybrid] Iteracao {self._iter_count}... (Tempo decorrido: {current_time - start_time:.1f}s)")
                self._last_print_time = current_time

            bound_qc = ansatz.assign_parameters({gammas: params_values[:p], betas: params_values[p:]})
            tmp_qc = bound_qc.copy()
            tmp_qc.save_expectation_value(H_op, range(self.n_qubits))
            
            try:
                job = self.backend.run(tmp_qc, shots=None)
                result = job.result()
            except Exception as e:
                # Fallback para CPU
                if 'GPU' in str(e) or 'gpu' in str(e).lower():
                    print(f"[Hybrid] AVISO: Falha na GPU durante execucao. Mudando para CPU... ({e})")
                    self.backend = AerSimulator(method='statevector', device='CPU')
                    job = self.backend.run(tmp_qc, shots=None)
                    result = job.result()
                else:
                    raise e
                    
            return result.data()['expectation_value'].real

        initial_params = [0.05] * p + [0.2] * p
        
        res = minimize(
            objective, 
            initial_params, 
            method='COBYLA', 
            options={'maxiter': maxiter, 'disp': True}
        )
        
        # 4. Amostragem final
        print("[Hybrid] Analisando resultados...")
        final_qc = ansatz.assign_parameters({gammas: res.x[:p], betas: res.x[p:]})
        final_qc.measure_all()
        
        job = self.backend.run(final_qc, shots=2048)
        counts = job.result().get_counts()
        
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        best_solution = None
        best_diversity = -1.0
        
        valid_count = 0
        for bitstring, _ in sorted_counts[:100]:
            indices = [i for i, b in enumerate(reversed(bitstring)) if b == '1']
            if len(indices) == k:
                valid_count += 1
                diversity = self.classical_selector.calculate_diversity_score(indices)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_solution = indices
        
        print(f"[Hybrid] Amostragem concluida: {valid_count} solucoes validas (k={k}) encontradas no Top 100.")
        
        # 5. Comparar com Greedy e escolher melhor
        if best_solution is None or best_diversity < greedy_score:
            print(f"[Hybrid] QAOA ({best_diversity if best_solution else 'N/A'}) nao superou Greedy ({greedy_score:.4f}). Usando solucao Greedy.")
            best_solution = list(greedy_indices)
            best_diversity = greedy_score
        else:
            improvement = ((best_diversity - greedy_score) / greedy_score) * 100
            print(f"[Hybrid] QAOA SUPEROU Greedy! +{improvement:.2f}% (Score: {best_diversity:.4f})")
        
        exec_time = time.time() - start_time
        print(f"[Hybrid] Concluido em {exec_time:.2f}s | Diversidade: {best_diversity:.4f}")
        
        return best_solution, best_diversity, exec_time


def run_hybrid_on_trap(trap_name: str, p: int = 1, alpha: float = 0.85, maxiter: int = 50) -> Dict:
    """
    Executa QAOA Hibrido em uma trap.
    
    Args:
        trap_name: Nome da trap (ex: 'trap_N25_K5')
        p: Numero de camadas QAOA
        alpha: Confianca no Greedy
        
    Returns:
        Dicionario com resultados
    """
    import json
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    trap_path = os.path.join(base_dir, 'data/traps', trap_name)
    
    # Carregar dados
    molecules_path = os.path.join(trap_path, 'molecules.csv')
    metadata_path = os.path.join(trap_path, 'metadata.json')
    
    if not os.path.exists(molecules_path):
        print(f"Erro: Trap nao encontrada em {trap_path}")
        return None
    
    df = pd.read_csv(molecules_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    n = metadata['n']
    k = metadata['k']
    
    print(f"\n{'='*60}")
    print(f"HYBRID QAOA: {trap_name}")
    print(f"N={n}, k={k}, alpha={alpha}")
    print(f"{'='*60}")
    
    # Executar
    selector = HybridQuantumSelector(df, cache_dir="data/temp_hybrid")
    indices, score, exec_time = selector.run_hybrid_optimize(k=k, p=p, alpha=alpha, maxiter=maxiter)
    
    return {
        'trap': trap_name,
        'n': n,
        'k': k,
        'algorithm': 'Hybrid',
        'score': score,
        'time': exec_time,
        'indices': indices
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='QAOA Hibrido para selecao molecular')
    parser.add_argument('--trap', type=str, help='Nome da trap (ex: trap_N25_K5)')
    parser.add_argument('--p', type=int, default=1, help='Numero de camadas QAOA')
    parser.add_argument('--alpha', type=float, default=0.85, help='Confianca no Greedy (0.5-0.95)')
    parser.add_argument('--maxiter', type=int, default=50, help='Maximo de iteracoes')
    
    args = parser.parse_args()
    
    if args.trap:
        result = run_hybrid_on_trap(args.trap, p=args.p, alpha=args.alpha, maxiter=args.maxiter)
        if result:
            print(f"\nResultado: {result['score']:.4f} em {result['time']:.2f}s")
    else:
        print("Uso: python hybrid_selector.py --trap trap_N25_K5")
