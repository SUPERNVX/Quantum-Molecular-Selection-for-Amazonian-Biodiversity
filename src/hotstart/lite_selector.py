"""
lite_selector.py

Versao otimizada para SIMULACAO LOCAL (CPU).
Remove overhead de transpilacao e primitives V2 para maxima velocidade.
Ideal para N < 30 qubits.

Adaptado para usar o MolecularDiversitySelector da pasta hotstart.
"""

import os
import sys
import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

# Otimizacao
from scipy.optimize import minimize

# Importar o seletor classico da pasta hotstart
from src.hotstart.classical import MolecularDiversitySelector


class QuantumMolecularSelectorLite:
    """
    Seletor molecular quantico otimizado para simulacao local via Aer (GPU).
    
    Usa AerSimulator para simulacao acelerada.
    Ideal para problemas com N < 32 qubits no simulador.
    """
    
    def __init__(self, molecules_df: pd.DataFrame, cache_dir: str = "data/temp_lite", use_gpu: bool = True):
        """
        Inicializa o seletor.
        
        Args:
            molecules_df: DataFrame com coluna 'smiles'
            cache_dir: Diretorio para cache da matriz de similaridade
            use_gpu: Se deve tentar usar GPU (NVIDIA)
        """
        self.molecules_df = molecules_df.copy()
        self.cache_dir = cache_dir
        
        # Usar o MolecularDiversitySelector para fingerprints e similaridade
        print(f"[Lite] Inicializando para {len(molecules_df)} moleculas...")
        self.classical_selector = MolecularDiversitySelector(molecules_df, cache_dir=cache_dir)
        
        # Copiar dados relevantes
        self.n_qubits = self.classical_selector.n_molecules
        self.similarity_matrix = self.classical_selector.similarity_matrix
        
        # Configurar Backend Aer
        backend_name = "Lite" if "Lite" in self.__class__.__name__ else "Hybrid"
        try:
            available_devices = AerSimulator().available_devices()
            if use_gpu and 'GPU' in available_devices:
                print(f"[{backend_name}] Configurando AerSimulator com suporte a GPU...")
                self.backend = AerSimulator(method='statevector', device='GPU')
            else:
                if use_gpu:
                    print(f"[{backend_name}] GPU solicitada mas não disponível. Usando CPU...")
                self.backend = AerSimulator(method='statevector', device='CPU')
        except Exception as e:
            print(f"[{backend_name}] AVISO: Erro ao configurar backend, usando CPU padrao. ({e})")
            self.backend = AerSimulator(method='statevector', device='CPU')
            
        print(f"[{backend_name}] Pronto com {self.n_qubits} qubits no backend {self.backend.name} ({self.backend.options.get('device', 'CPU')}).")

    def formulate_problem(self, k: int) -> Tuple[SparsePauliOp, float]:
        """
        Formula o problema QUBO como Hamiltoniano.
        H = penalty * (sum(x_i) - k)^2 + sum(S_ij * x_i * x_j)
        """
        n = self.n_qubits
        # Penalidade balanceada: deve ser maior que a similaridade media para manter k
        penalty = 5.0 
        
        pauli_list = []
        offset = 0.0
        
        # Coeficientes QUBO: Q_ii * x_i + Q_ij * x_i * x_j
        Q_ii = np.full(n, penalty * (1 - 2 * k))
        # Q_ij para diversidade (min S_ij) e penalidade (2 * penalty)
        Q_ij = self.similarity_matrix * 1.0 + (2 * penalty)
        
        # Mapeamento Ising: x_i = (1 - Z_i) / 2
        # h_i = -Q_ii/2 - sum(Q_ij/4)
        # J_ij = Q_ij / 4
        
        h_coeffs = -Q_ii / 2.0
        J_coeffs = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                val = Q_ij[i][j] / 4.0
                h_coeffs[i] -= val
                h_coeffs[j] -= val
                J_coeffs[(i, j)] = val
                offset += val
                
        offset += np.sum(Q_ii) / 2.0
        
        # Adicionar termos extras para offset fixo do Hamiltoniano (penalty*k^2)
        offset += penalty * (k**2)

        # Construir SparsePauliOp
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

    def qaoa_optimize(self, k: int, p: int = 1, maxiter: int = 50) -> Tuple[List[int], float, float]:
        """
        Executa otimizacao QAOA otimizada para Aer.
        """
        print(f"\n[Lite] QAOA | N={self.n_qubits}, k={k}, p={p} | Backend: {self.backend.options.get('device', 'CPU')}")
        start_time = time.time()
        
        # 1. Formular Hamiltoniano
        H, offset = self.formulate_problem(k)
        
        # 2. Pre-construir parte do circuito (Estado Inicial + Mixer)
        # Para otimizacao, vamos construir o ansatz dinamicamente mas de forma limpa
        from qiskit.circuit import ParameterVector
        
        gammas = ParameterVector('gamma', p)
        betas = ParameterVector('beta', p)
        
        ansatz = QuantumCircuit(self.n_qubits)
        ansatz.h(range(self.n_qubits))
        
        for layer in range(p):
            # Custo
            # Nota: Para velocidade max, poderiamos usar o evolution gate de Qiskit
            # mas o rz/rzz manual no Aer ja e excelente se nao repetirmos transpilacao
            for pauli, coeff in zip(H.paulis, H.coeffs):
                label = pauli.to_label()
                active = [i for i, char in enumerate(reversed(label)) if char == 'Z']
                if len(active) == 1:
                    ansatz.rz(2 * gammas[layer] * coeff.real, active[0])
                elif len(active) == 2:
                    ansatz.rzz(2 * gammas[layer] * coeff.real, active[0], active[1])
            
            # Mixer
            ansatz.rx(2 * betas[layer], range(self.n_qubits))
            
        # 3. Transpilacao unica (se possivel para Aer)
        # O AerSimulator nao precisa de transpilation pesada se os portais forem padrao
        
        print(f"[Lite] Iniciando minimizacao (maxiter={maxiter})...")
        
        # Variaveis para acompanhamento
        self._iter_count = 0
        self._last_print_time = time.time()
        
        def objective(params_values):
            self._iter_count += 1
            current_time = time.time()
            
            # Print de progresso a cada iteracao ou 5 segundos
            if self._iter_count == 1 or current_time - self._last_print_time > 5:
                print(f"[Lite] Iteracao {self._iter_count}... (Tempo decorrido: {current_time - start_time:.1f}s)")
                self._last_print_time = current_time

            # Vincular parametros
            bound_qc = ansatz.assign_parameters({gammas: params_values[:p], betas: params_values[p:]})
            
            # Executar simulacao Rapida
            from qiskit_aer.library import save_expectation_value
            tmp_qc = bound_qc.copy()
            tmp_qc.save_expectation_value(H, range(self.n_qubits))
            
            try:
                job = self.backend.run(tmp_qc, shots=None)
                result = job.result()
            except Exception as e:
                # Se falhar no GPU, tenta fallback imediato para CPU
                if 'GPU' in str(e) or 'gpu' in str(e).lower():
                    print(f"[Lite] AVISO: Falha na GPU durante execucao. Mudando para CPU... ({e})")
                    self.backend = AerSimulator(method='statevector', device='CPU')
                    job = self.backend.run(tmp_qc, shots=None)
                    result = job.result()
                else:
                    raise e
                    
            exp_val = result.data()['expectation_value'].real
            return exp_val

        # Inicializacao
        initial_params = [0.1] * p + [0.3] * p
        
        res = minimize(
            objective, 
            initial_params, 
            method='COBYLA', 
            options={'maxiter': maxiter, 'disp': True}
        )
        
        # 4. Amostragem final
        print("[Lite] Gerando resultado final...")
        final_qc = ansatz.assign_parameters({gammas: res.x[:p], betas: res.x[p:]})
        final_qc.measure_all()
        
        # Se N > 25, Statevector.probabilities_dict() pode ser lento, 
        # melhor usar sampling do Aer
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
        
        print(f"[Lite] Amostragem concluida: {valid_count} solucoes validas (k={k}) encontradas no Top 100.")
                    
        if best_solution is None:
            print("[Lite] Nenhuma solucao valida encontrada na amostragem. Fallback para solucao gulosa...")
            greedy_indices, greedy_score, _, _ = self.classical_selector.greedy_selection(k)
            best_solution = list(greedy_indices)
            best_diversity = greedy_score
        else:
            # Comparar com Greedy
            greedy_indices, greedy_score, _, _ = self.classical_selector.greedy_selection(k)
            if best_diversity < greedy_score:
                print(f"[Lite] QAOA ({best_diversity:.4f}) nao superou Greedy ({greedy_score:.4f}).")
            else:
                improvement = ((best_diversity - greedy_score) / greedy_score) * 100
                print(f"[Lite] QAOA SUPEROU Greedy! Score: {best_diversity:.4f} (+{improvement:.2f}%)")
            
        exec_time = time.time() - start_time
        print(f"[Lite] Finalizado em {exec_time:.2f}s | Score: {best_diversity:.4f}")
        
        return best_solution, best_diversity, exec_time


def run_lite_on_trap(trap_name: str, p: int = 1, maxiter: int = 50) -> Dict:
    """
    Executa QAOA Lite em uma trap.
    
    Args:
        trap_name: Nome da trap (ex: 'trap_N25_K5')
        p: Numero de camadas QAOA
        
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
    print(f"QAOA LITE: {trap_name}")
    print(f"N={n}, k={k}")
    print(f"{'='*60}")
    
    # Executar
    selector = QuantumMolecularSelectorLite(df, cache_dir="data/temp_lite")
    indices, score, exec_time = selector.qaoa_optimize(k=k, p=p, maxiter=maxiter)
    
    return {
        'trap': trap_name,
        'n': n,
        'k': k,
        'algorithm': 'QAOA_Lite',
        'score': score,
        'time': exec_time,
        'indices': indices
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='QAOA Lite para selecao molecular')
    parser.add_argument('--trap', type=str, help='Nome da trap (ex: trap_N25_K5)')
    parser.add_argument('--p', type=int, default=1, help='Numero de camadas QAOA')
    parser.add_argument('--maxiter', type=int, default=50, help='Maximo de iteracoes')
    
    args = parser.parse_args()
    
    if args.trap:
        result = run_lite_on_trap(args.trap, p=args.p, maxiter=args.maxiter)
        if result:
            print(f"\nResultado: {result['score']:.4f} em {result['time']:.2f}s")
    else:
        print("Uso: python lite_selector.py --trap trap_N25_K5")
