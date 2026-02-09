"""
data_preprocessing.py

Limpeza e preparação do dataset molecular
Calcula fingerprints e matriz de similaridade
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pickle
import os
from typing import List, Tuple


def load_and_clean_data() -> pd.DataFrame:
    """
    Carrega e limpa datasets moleculares
    
    Returns:
        DataFrame limpo e validado
    """
    print("=" * 60)
    print("PRÉ-PROCESSAMENTO DE DADOS MOLECULARES")
    print("=" * 60)
    
    # Carregar dataset PubChem
    print("\n[1/5] Carregando dados...")
    df_pubchem = pd.read_csv('data/raw/amazonian_molecules.csv')
    print(f"  → PubChem: {len(df_pubchem)} moléculas")
    
    # Tentar carregar NuBBE (opcional)
    try:
        df_nubbe = pd.read_csv('data/raw/nubbe_database.csv')
        print(f"  → NuBBE: {len(df_nubbe)} moléculas")
        df = pd.concat([df_pubchem, df_nubbe], ignore_index=True)
        print(f"  → Total combinado: {len(df)} moléculas")
    except FileNotFoundError:
        print("  ⚠ NuBBE database não encontrado (ok, usando apenas PubChem)")
        df = df_pubchem
    
    # Remover duplicatas por SMILES
    print("\n[2/5] Removendo duplicatas...")
    original_size = len(df)
    df = df.drop_duplicates(subset=['smiles'])
    print(f"  → Removidas {original_size - len(df)} duplicatas")
    
    # Validar SMILES
    print("\n[3/5] Validando estruturas moleculares...")
    valid_smiles = []
    invalid_count = 0
    
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(True)
        else:
            valid_smiles.append(False)
            invalid_count += 1
    
    df = df[valid_smiles].reset_index(drop=True)
    print(f"  → {invalid_count} SMILES inválidos removidos")
    print(f"  → Dataset final: {len(df)} moléculas válidas")
    
    return df


def compute_fingerprints(df: pd.DataFrame) -> List:
    """
    Calcula Morgan fingerprints para todas as moléculas
    
    Args:
        df: DataFrame com coluna 'smiles'
        
    Returns:
        Lista de fingerprints (rdkit.DataStructs.ExplicitBitVect)
    """
    print("\n[4/5] Calculando fingerprints moleculares...")
    print("  Tipo: Morgan Fingerprint (ECFP4)")
    print("  Raio: 2")
    print("  Bits: 2048")
    
    fingerprints = []
    
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fingerprints.append(fp)
        
        if (idx + 1) % 50 == 0:
            print(f"  → Processadas {idx + 1}/{len(df)} moléculas")
    
    print(f"  ✓ {len(fingerprints)} fingerprints calculados")
    
    return fingerprints


def compute_similarity_matrix(fingerprints: List) -> np.ndarray:
    """
    Calcula matriz de similaridade Tanimoto pairwise
    
    Args:
        fingerprints: Lista de fingerprints moleculares
        
    Returns:
        Matriz de similaridade (n x n)
    """
    print("\n[5/5] Calculando matriz de similaridade Tanimoto...")
    
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))
    
    total_comparisons = n * (n - 1) // 2
    completed = 0
    
    for i in range(n):
        # Diagonal: similaridade consigo mesmo = 1.0
        similarity_matrix[i][i] = 1.0
        
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim
            
            completed += 1
            if completed % 10000 == 0:
                progress = (completed / total_comparisons) * 100
                print(f"  → Progresso: {progress:.1f}% ({completed}/{total_comparisons})")
    
    print(f"  ✓ Matriz {n}x{n} calculada")
    
    # Estatísticas de similaridade
    upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]
    print(f"\nEstatísticas de similaridade:")
    print(f"  Média: {upper_triangle.mean():.4f}")
    print(f"  Mediana: {np.median(upper_triangle):.4f}")
    print(f"  Mínimo: {upper_triangle.min():.4f}")
    print(f"  Máximo: {upper_triangle.max():.4f}")
    
    return similarity_matrix


def analyze_diversity(df: pd.DataFrame, similarity_matrix: np.ndarray):
    """
    Análise exploratória de diversidade estrutural
    
    Args:
        df: DataFrame com moléculas
        similarity_matrix: Matriz de similaridade
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE DIVERSIDADE ESTRUTURAL")
    print("=" * 60)
    
    n = len(df)
    
    # Calcular diversidade média por fonte/espécie
    print("\nDiversidade por fonte/espécie:")
    source_col = 'source' if 'source' in df.columns else 'species'
    for source in df[source_col].unique()[:5]:  # Top 5
        source_indices = df[df[source_col] == source].index.tolist()
        
        if len(source_indices) > 1:
            source_similarities = []
            for i in source_indices:
                for j in source_indices:
                    if i < j:
                        source_similarities.append(similarity_matrix[i][j])
            
            avg_similarity = np.mean(source_similarities)
            diversity = 1 - avg_similarity
            print(f"  {source[:30]:30} → Diversidade: {diversity:.4f}")
    
    # Moléculas mais "centrais" (alta similaridade média)
    print("\nMoléculas mais representativas (centrais):")
    avg_similarities = similarity_matrix.mean(axis=1)
    central_indices = np.argsort(avg_similarities)[::-1][:3]
    
    for idx in central_indices:
        source = df.loc[idx, source_col]
        avg_sim = avg_similarities[idx]
        print(f"  ID {idx}: {source[:30]:30} (Sim média: {avg_sim:.4f})")
    
    # Moléculas mais "únicas" (baixa similaridade média)
    print("\nMoléculas mais únicas (periféricas):")
    unique_indices = np.argsort(avg_similarities)[:3]
    
    for idx in unique_indices:
        source = df.loc[idx, source_col]
        avg_sim = avg_similarities[idx]
        print(f"  ID {idx}: {source[:30]:30} (Sim média: {avg_sim:.4f})")


def save_processed_data(df: pd.DataFrame, 
                       fingerprints: List, 
                       similarity_matrix: np.ndarray):
    """
    Salva dados processados
    
    Args:
        df: DataFrame limpo
        fingerprints: Lista de fingerprints
        similarity_matrix: Matriz de similaridade
    """
    print("\n" + "=" * 60)
    print("SALVANDO DADOS PROCESSADOS")
    print("=" * 60)
    
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar DataFrame
    df_path = os.path.join(output_dir, 'amazonian_molecules.csv')
    df.to_csv(df_path, index=False)
    print(f"✓ Dataset: {df_path}")
    
    # Salvar fingerprints
    fp_path = os.path.join(output_dir, 'fingerprints.pkl')
    with open(fp_path, 'wb') as f:
        pickle.dump(fingerprints, f)
    print(f"✓ Fingerprints: {fp_path}")
    
    # Salvar matriz de similaridade
    sim_path = os.path.join(output_dir, 'similarity_matrix.npy')
    np.save(sim_path, similarity_matrix)
    print(f"✓ Matriz de similaridade: {sim_path}")
    
    # Criar metadados
    source_col = 'source' if 'source' in df.columns else 'species'
    metadata = {
        'n_molecules': len(df),
        'n_sources': df[source_col].nunique(),
        'fingerprint_type': 'Morgan (ECFP4)',
        'fingerprint_radius': 2,
        'fingerprint_bits': 2048,
        'similarity_metric': 'Tanimoto',
        'avg_molecular_weight': df['mw'].mean(),
        'avg_logp': df['logp'].mean(),
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Metadados: {metadata_path}")


def main():
    """Pipeline completo de pré-processamento"""
    
    # 1. Carregar e limpar
    df = load_and_clean_data()
    
    # 2. Calcular fingerprints
    fingerprints = compute_fingerprints(df)
    
    # 3. Calcular similaridade
    similarity_matrix = compute_similarity_matrix(fingerprints)
    
    # 4. Análise exploratória
    analyze_diversity(df, similarity_matrix)
    
    # 5. Salvar dados processados
    save_processed_data(df, fingerprints, similarity_matrix)
    
    print("\n" + "=" * 60)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)
    print("\nPróximos passos:")
    print("  1. Executar baseline clássico:")
    print("     python src/classical/classical_molecular_selection.py")
    print("  2. Executar QAOA (simulador):")
    print("     python src/quantum/quantum_molecular_selection.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
