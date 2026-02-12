"""
run_on_hardware.py

Script para executar a seleção molecular em hardware REAL da IBM Quantum.
Permite configurar o tamanho do subset (N) e o número de moléculas a selecionar (k).
"""

import pandas as pd
import os
from dotenv import load_dotenv
from src.quantum.hybrid_selector import HybridQuantumSelector

def main():
    # 1. Carregar váriáveis de ambiente (IBM Token)
    load_dotenv()
    token = os.getenv("IBM_QUANTUM_TOKEN")
    
    if not token:
        print("[ERRO] IBM_QUANTUM_TOKEN não encontrado no arquivo .env")
        return

    # 2. Configurações do Usuário
    SUBSET_SIZE = 20    # N: Total de moléculas no subset quântico
    K_SELECTION = 4     # k: Quantas moléculas selecionar
    SMART_SAMPLING = True # Se True, usa clusterização inteligente. Se False, usa aleatório.
    BACKEND = None      # Backend IBM (ex: 'ibm_kyoto') ou None

    print(f"--- Iniciando Execução em Hardware Real ---")
    print(f"Configuração: N={SUBSET_SIZE}, k={K_SELECTION}")

    # 3. Carregar dados reais
    df = pd.read_csv("data/processed/brnpdb.csv")
    
    if SMART_SAMPLING:
        print(f"Gerando subset inteligente (N={SUBSET_SIZE}) via Clusterização...")
        from scripts.ibm_prep import prepare_quantum_subset
        df_subset, _ = prepare_quantum_subset(df, target_size=SUBSET_SIZE, method='clustering')
    else:
        print(f"Gerando subset aleatório (N={SUBSET_SIZE})...")
        df_subset = df.sample(n=SUBSET_SIZE, random_state=42)

    # 4. Inicializar Seletor Híbrido em modo Hardware Real
    selector = HybridQuantumSelector(
        molecules_df=df_subset, 
        use_real_quantum=True, 
        ibm_token=token,
        backend_name=BACKEND
    )

    # 5. Executar Otimização
    print(f"Enviando job para a IBM Quantum (pode demorar devido à fila)...")
    indices, elapsed = selector.run_hybrid_optimize(k=K_SELECTION)

    # 6. Exibir Resultados
    print("\n" + "="*40)
    print("RESULTADO DA IBM QUANTUM")
    print("="*40)
    print(f"Índices Selecionados: {indices}")
    print(f"Tempo Total: {elapsed:.2f}s")
    
    selected_mols = df_subset.iloc[indices]
    print("\nMoléculas Escolhidas:")
    print(selected_mols[['Compound_Name', 'smiles']])
    print("="*40)

if __name__ == "__main__":
    main()
