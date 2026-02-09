"""
data_collection.py

Coleta moléculas amazônicas de bancos de dados públicos
Foca em PubChem API (gratuita, sem necessidade de chave)
"""

import requests
import pandas as pd
import time
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Dict, Optional
import os

# Espécies amazônicas de interesse
AMAZONIAN_SPECIES = [
    "Uncaria tomentosa",        # Unha de gato
    "Copaifera langsdorffii",   # Copaíba
    "Bertholletia excelsa",     # Castanha-do-Pará
    "Paullinia cupana",         # Guaraná
    "Euterpe oleracea",         # Açaí
    "Hevea brasiliensis",       # Seringueira
    "Phyllanthus niruri",       # Quebra-pedra
    "Carapa guianensis",        # Andiroba
    "Dipteryx odorata",         # Cumaru
    "Virola surinamensis",      # Ucuuba
    "Tabebuia impetiginosa",    # Ipê-roxo
    "Mauritia flexuosa",        # Buriti
    "Astrocaryum aculeatum",    # Tucumã
    "Oenocarpus bataua",        # Patauá
    "Theobroma grandiflorum",   # Cupuaçu
]


def search_pubchem_species(species_name: str, max_molecules: int = 10) -> List[int]:
    """
    Busca moléculas no PubChem por nome de espécie
    
    Args:
        species_name: Nome científico da espécie
        max_molecules: Número máximo de moléculas por espécie
        
    Returns:
        Lista de CIDs (Compound IDs) do PubChem
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    search_url = f"{base_url}/compound/name/{species_name}/cids/JSON"
    
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cids = data.get('IdentifierList', {}).get('CID', [])
            return cids[:max_molecules]
        else:
            return []
    except Exception as e:
        print(f"  ⚠ Erro ao buscar {species_name}: {e}")
        return []


def get_molecule_properties(cid: int) -> Optional[Dict]:
    """
    Obtém propriedades moleculares do PubChem
    
    Args:
        cid: Compound ID do PubChem
        
    Returns:
        Dicionário com propriedades ou None se falhar
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    prop_url = f"{base_url}/compound/cid/{cid}/property/CanonicalSMILES,MolecularFormula,MolecularWeight,IUPACName/JSON"
    
    try:
        response = requests.get(prop_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['PropertyTable']['Properties'][0]
        else:
            return None
    except Exception as e:
        print(f"  ⚠ Erro ao obter CID {cid}: {e}")
        return None


def calculate_molecular_descriptors(smiles: str) -> Optional[Dict]:
    """
    Calcula descritores moleculares usando RDKit
    
    Args:
        smiles: String SMILES da molécula
        
    Returns:
        Dicionário com descritores ou None se SMILES inválido
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol),
    }
    
    return descriptors


def is_druglike(descriptors: Dict) -> bool:
    """
    Aplica Regra dos Cinco de Lipinski + filtros adicionais
    
    Critérios drug-like:
    - Peso molecular: 150-500 Da
    - LogP: ≤ 5
    - Doadores de ligação H: ≤ 5
    - Aceptores de ligação H: ≤ 10
    - Ligações rotacionáveis: ≤ 10
    
    Args:
        descriptors: Dicionário com descritores moleculares
        
    Returns:
        True se molécula é drug-like
    """
    if descriptors is None:
        return False
    
    return (
        150 <= descriptors['mw'] <= 500 and
        descriptors['logp'] <= 5 and
        descriptors['hbd'] <= 5 and
        descriptors['hba'] <= 10 and
        descriptors['rotatable_bonds'] <= 10
    )


def collect_dataset(output_path: str = 'data/raw/amazonian_molecules.csv') -> pd.DataFrame:
    """
    Coleta dataset completo de moléculas amazônicas
    
    Args:
        output_path: Caminho para salvar CSV
        
    Returns:
        DataFrame com moléculas coletadas
    """
    print("=" * 60)
    print("COLETA DE MOLÉCULAS AMAZÔNICAS - PubChem")
    print("=" * 60)
    print(f"\nEspécies alvo: {len(AMAZONIAN_SPECIES)}")
    print(f"Meta: ~300-500 moléculas drug-like\n")
    
    all_molecules = []
    
    for idx, species in enumerate(AMAZONIAN_SPECIES, 1):
        print(f"[{idx}/{len(AMAZONIAN_SPECIES)}] Processando: {species}")
        
        # Buscar CIDs
        cids = search_pubchem_species(species, max_molecules=15)
        print(f"  → Encontrados {len(cids)} compostos")
        
        # Processar cada molécula
        for cid in cids:
            mol_data = get_molecule_properties(cid)
            
            if mol_data:
                smiles = mol_data.get('CanonicalSMILES')
                descriptors = calculate_molecular_descriptors(smiles)
                
                # Filtrar por drug-likeness
                if is_druglike(descriptors):
                    all_molecules.append({
                        'cid': cid,
                        'species': species,
                        'smiles': smiles,
                        'formula': mol_data.get('MolecularFormula'),
                        'name': mol_data.get('IUPACName', 'Unknown'),
                        **descriptors
                    })
            
            # Rate limiting (PubChem: max 5 requests/second)
            time.sleep(0.2)
        
        print(f"  ✓ Coletadas {len([m for m in all_molecules if m['species'] == species])} moléculas drug-like")
    
    # Criar DataFrame
    df = pd.DataFrame(all_molecules)
    
    # Remover duplicatas por SMILES
    original_size = len(df)
    df = df.drop_duplicates(subset=['smiles'])
    duplicates_removed = original_size - len(df)
    
    # Salvar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Sumário
    print("\n" + "=" * 60)
    print("COLETA CONCLUÍDA")
    print("=" * 60)
    print(f"Total de moléculas coletadas: {original_size}")
    print(f"Duplicatas removidas: {duplicates_removed}")
    print(f"Dataset final: {len(df)} moléculas únicas")
    print(f"\nSalvo em: {output_path}")
    
    # Estatísticas
    print("\nEstatísticas do Dataset:")
    print(f"  Peso molecular médio: {df['mw'].mean():.2f} Da")
    print(f"  LogP médio: {df['logp'].mean():.2f}")
    print(f"  Espécies representadas: {df['species'].nunique()}")
    
    return df


if __name__ == "__main__":
    dataset = collect_dataset()
    
    print("\n" + "=" * 60)
    print("Próximos passos:")
    print("  1. Revisar dataset em: data/raw/amazonian_molecules.csv")
    print("  2. Executar pré-processamento: python src/utils/data_preprocessing.py")
    print("=" * 60)
