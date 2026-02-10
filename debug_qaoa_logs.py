"""
debug_qaoa_logs.py

Script de diagnóstico para validar os bugs identificados em quantum_molecular_selection.py
Adicione estas funções de log ao código original para validar as hipóteses.
"""

import numpy as np


def validate_qubo_to_ising_conversion(Q):
    """
    Diagnóstico Bug #1: Verificar se matriz J tem diagonal não-nula
    
    Execute após formulate_qubo() e antes de qubo_to_ising()
    """
    print("\n" + "="*60)
    print("DIAGNÓSTICO: Conversão QUBO → Ising")
    print("="*60)
    
    n = Q.shape[0]
    
    # Conversão ATUAL (com bug)
    J_buggy = Q / 4.0
    h_buggy = np.sum(Q, axis=1) / 2.0
    offset_buggy = np.sum(Q) / 4.0
    
    # Conversão CORRETA
    J_correct = Q.copy() / 4.0
    np.fill_diagonal(J_correct, 0)  # Diagonal deve ser zero!
    
    # Offset correto: (1/4)sum(Q) + (1/4)trace(Q)
    offset_correct = np.sum(Q) / 4.0 + np.trace(Q) / 4.0
    
    # h correto: (1/2) * sum_j Q[i,j] para cada i
    h_correct = np.sum(Q, axis=1) / 2.0  # Este está correto
    
    print(f"\n[BUG #1] Diagonal de J (deve ser zero):")
    print(f"  Diagonal atual: {np.diag(J_buggy)[:5]}... (mostrando 5 primeiros)")
    print(f"  Max diagonal: {np.max(np.abs(np.diag(J_buggy))):.6f}")
    print(f"  ❌ PROBLEMA: J tem diagonal não-nula!" if np.max(np.abs(np.diag(J_buggy))) > 1e-10 else "  ✅ OK")
    
    print(f"\n[BUG #2] Diferença no offset:")
    print(f"  Offset atual: {offset_buggy:.6f}")
    print(f"  Offset correto: {offset_correct:.6f}")
    print(f"  Diferença: {abs(offset_buggy - offset_correct):.6f}")
    print(f"  ❌ PROBLEMA: Offset incorreto!" if abs(offset_buggy - offset_correct) > 1e-10 else "  ✅ OK")
    
    # Impacto no circuito
    n_rzz_gates_buggy = np.sum(np.abs(J_buggy) > 1e-6)
    n_rzz_gates_correct = np.sum(np.abs(J_correct) > 1e-6)
    
    print(f"\n[IMPACTO] Número de gates RZZ no circuito:")
    print(f"  Com bug: {n_rzz_gates_buggy}")
    print(f"  Corrigido: {n_rzz_gates_correct}")
    print(f"  Gates extras (incorretos): {n_rzz_gates_buggy - n_rzz_gates_correct}")
    
    return J_buggy, J_correct, offset_buggy, offset_correct


def validate_penalty_weight(Q, k, penalty):
    """
    Diagnóstico Issue #3: Verificar se penalidade é adequada
    
    Execute após formulate_qubo()
    """
    print("\n" + "="*60)
    print("DIAGNÓSTICO: Peso da Penalidade")
    print("="*60)
    
    n = Q.shape[0]
    
    # Valores de diversidade (off-diagonal, antes da penalidade)
    diversity_values = []
    for i in range(n):
        for j in range(i+1, n):
            diversity_values.append(-Q[i][j])  # Negativo porque Q tem -distância
    
    diversity_values = np.array(diversity_values)
    
    print(f"\n[Estatísticas da Diversidade]")
    print(f"  Min: {np.min(diversity_values):.4f}")
    print(f"  Max: {np.max(diversity_values):.4f}")
    print(f"  Mean: {np.mean(diversity_values):.4f}")
    print(f"  Std: {np.std(diversity_values):.4f}")
    
    # Penalidade atual
    print(f"\n[Penalidade Atual]")
    print(f"  Valor: {penalty:.4f}")
    
    # Custo de violar restrição (selecionar k+1 moléculas)
    # (sum x_i - k)^2 = 1^2 = 1, multiplicado pela penalidade
    violation_cost = penalty
    
    # Ganho máximo de diversidade ao adicionar uma molécula extra
    max_diversity_gain = (k) * np.max(diversity_values)  # k pares adicionais no máximo
    
    print(f"\n[Análise de Trade-off]")
    print(f"  Custo de violar restrição (k+1 moléculas): {violation_cost:.4f}")
    print(f"  Ganho máx. de diversidade com molécula extra: {max_diversity_gain:.4f}")
    
    if violation_cost <= max_diversity_gain:
        print(f"  ⚠️ ATENÇÃO: Penalidade pode ser muito baixa!")
        print(f"     Otimizador pode preferir violar restrição para ganhar diversidade")
        print(f"     Recomendação: penalty >= {max_diversity_gain * 1.5:.4f}")
    else:
        print(f"  ✅ Penalidade adequada para evitar violações")
    
    # Penalidade recomendada
    recommended_penalty = max(max_diversity_gain * 2, np.max(diversity_values) * k)
    print(f"\n[Recomendação]")
    print(f"  Penalidade recomendada: {recommended_penalty:.4f}")
    
    return recommended_penalty


def validate_bitstring_convention(n_qubits, test_bitstring="0101"):
    """
    Diagnóstico Issue #4: Verificar convenção de bitstring
    
    Testa se a conversão está correta
    """
    print("\n" + "="*60)
    print("DIAGNÓSTICO: Convenção de Bitstring")
    print("="*60)
    
    # Pad test bitstring to correct length
    test_bitstring = test_bitstring.zfill(n_qubits)
    if len(test_bitstring) > n_qubits:
        test_bitstring = test_bitstring[-n_qubits:]
    
    print(f"\n[Teste com bitstring: '{test_bitstring}' ({n_qubits} qubits)]")
    
    # Conversão atual do código
    z_current = np.array([1 if bit == '0' else -1 for bit in test_bitstring[::-1]])
    x_current = (1 + z_current) / 2  # x_i = (1 + z_i) / 2
    selected_current = [i for i, bit in enumerate(test_bitstring[::-1]) if bit == '0']
    
    print(f"\n[Conversão Atual]")
    print(f"  Bitstring original: {test_bitstring}")
    print(f"  Bitstring revertido: {test_bitstring[::-1]}")
    print(f"  z (spins): {z_current}")
    print(f"  x (seleção): {x_current.astype(int)}")
    print(f"  Índices selecionados: {selected_current}")
    
    # Verificação: qubit 0 é o bit mais à direita no bitstring
    print(f"\n[Verificação Qiskit]")
    print(f"  Em Qiskit, qubit 0 é o bit menos significativo (LSB)")
    print(f"  bitstring[{-1}] = '{test_bitstring[-1]}' → qubit 0")
    print(f"  Se bit='0', z=+1, x=1 (selecionado) ✓")
    
    return selected_current


def suggest_improved_qubo_to_ising(Q):
    """
    Retorna a versão corrigida da conversão QUBO → Ising
    """
    n = Q.shape[0]
    
    # CORREÇÃO: Diagonal de J deve ser zero
    J = Q.copy() / 4.0
    np.fill_diagonal(J, 0)
    
    # h está correto
    h = np.sum(Q, axis=1) / 2.0
    
    # CORREÇÃO: Offset deve incluir contribuição da diagonal
    offset = np.sum(Q) / 4.0 + np.trace(Q) / 4.0
    
    return J, h, offset


def suggest_adaptive_penalty(Q, k, safety_factor=2.0):
    """
    Calcula penalidade adaptativa melhorada
    
    Args:
        Q: Matriz QUBO (antes de adicionar penalidade)
        k: Número de moléculas a selecionar
        safety_factor: Fator de segurança para garantir que violação seja pior
    
    Returns:
        penalty: Penalidade recomendada
    """
    n = Q.shape[0]
    
    # Encontrar diversidade máxima (off-diagonal)
    max_diversity = 0
    for i in range(n):
        for j in range(i+1, n):
            if abs(Q[i][j]) > max_diversity:
                max_diversity = abs(Q[i][j])
    
    # Penalidade deve ser maior que o ganho de diversidade de adicionar k moléculas
    # No pior caso, adicionar 1 molécula adiciona até k-1 pares
    max_gain_from_violation = max_diversity * k
    
    penalty = max_gain_from_violation * safety_factor
    
    # Garantir valor mínimo razoável
    if penalty < 1.0:
        penalty = 1.0
    
    return penalty


# Exemplo de uso:
if __name__ == "__main__":
    print("="*70)
    print("DIAGNÓSTICO QAOA - QUANTUM MOLECULAR SELECTION")
    print("="*70)
    
    # Criar QUBO de exemplo (simulado)
    np.random.seed(42)
    n = 10
    k = 3
    
    # Simular matriz de similaridade (valores entre 0 e 1)
    similarity = np.random.rand(n, n)
    similarity = (similarity + similarity.T) / 2  # Tornar simétrica
    np.fill_diagonal(similarity, 1)  # Diagonal = 1 (auto-similaridade)
    
    # Criar QUBO de diversidade
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance = 1.0 - similarity[i][j]
            Q[i][j] = -distance
            Q[j][i] = -distance
    
    # Testar penalidade
    penalty_current = abs(np.sum(Q)) / (n * k) * 2.0
    if penalty_current < 0.1:
        penalty_current = 1.0
    
    # Adicionar penalidade ao QUBO
    Q_with_penalty = Q.copy()
    for i in range(n):
        Q_with_penalty[i][i] += penalty_current
        Q_with_penalty[i][i] -= 2 * penalty_current * k
    for i in range(n):
        for j in range(i+1, n):
            Q_with_penalty[i][j] += 2 * penalty_current
            Q_with_penalty[j][i] += 2 * penalty_current
    
    # Executar diagnósticos
    validate_qubo_to_ising_conversion(Q_with_penalty)
    validate_penalty_weight(Q, k, penalty_current)
    validate_bitstring_convention(n, "0101010101")
    
    print("\n" + "="*70)
    print("CORREÇÕES SUGERIDAS")
    print("="*70)
    
    J_correct, h_correct, offset_correct = suggest_improved_qubo_to_ising(Q_with_penalty)
    penalty_recommended = suggest_adaptive_penalty(Q, k)
    
    print(f"\n[Penalidade Recomendada]: {penalty_recommended:.4f}")
    print(f"[Offset Corrigido]: {offset_correct:.4f}")
    print(f"[J diagonal (deve ser zero)]: {np.diag(J_correct)}")
