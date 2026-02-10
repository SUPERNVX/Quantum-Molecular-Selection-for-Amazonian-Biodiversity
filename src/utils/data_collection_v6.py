"""
data_collection_v6.py - Dataset Expansion (Target: 1000+ molecules)

Estratégia:
1. Busca ampliada por Famílias e Gêneros amazônicos (NCBI Entrez).
2. Extração direta de fontes especializadas (NuBBE via PubChem PUG REST).
3. Processamento em lote para otimizar taxa de requisição.
"""

import requests
import pandas as pd
import time
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Dict, Optional

# Taxonomia Amazônica de Alto Rendimento (Baseado em pesquisa)
AMAZONIAN_FAMILIES = [
    "Leguminosae", "Rubiaceae", "Lauraceae", "Annonaceae", 
    "Euphorbiaceae", "Burseraceae", "Rutaceae", "Myrtaceae", 
    "Melastomataceae", "Malvaceae", "Sapotaceae", "Apocynaceae",
    "Piperaceae", "Bignoniaceae", "Fabaceae"
]

AMAZONIAN_GENERA = [
    "Protium", "Theobroma", "Hevea", "Cecropia", "Aniba", 
    "Vismia", "Aspidosperma", "Piper", "Psychotria", "Tabebuia",
    "Copaifera", "Bixa", "Euterpe", "Paullinia"
]

def search_entrez_sids(term: str, max_results: int = 200) -> List[str]:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pcsubstance",
        "term": f"{term}[All Fields]",
        "retmode": "json",
        "retmax": max_results
    }
    try:
        response = requests.get(base_url, params=params, timeout=20)
        if response.status_code == 200:
            return response.json().get('esearchresult', {}).get('idlist', [])
        return []
    except Exception:
        return []

def get_nubbe_sids(max_results: int = 500) -> List[str]:
    """Extrai SIDs diretamente da fonte NuBBE no PubChem"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceall/NuBBE/sids/JSON?retmax={max_results}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json().get('InformationList', {}).get('Information', [{}])[0].get('SID', [])
        return []
    except Exception:
        return []

def sids_to_cids_batch(sids: List[str]) -> List[int]:
    if not sids: return []
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    # Dividir em lotes de 100 para evitar URLs muito longas
    all_cids = []
    for i in range(0, len(sids), 100):
        batch = sids[i:i+100]
        sid_string = ",".join(batch)
        url = f"{base_url}/substance/sid/{sid_string}/cids/JSON"
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('InformationList', {}).get('Information', []):
                    cid_list = item.get('CID')
                    if cid_list:
                        if isinstance(cid_list, list): all_cids.extend(cid_list)
                        else: all_cids.append(cid_list)
            time.sleep(0.2)
        except Exception:
            continue
    return list(set(all_cids))

def get_compound_properties_batch(cids: List[int]) -> List[Dict]:
    """Busca propriedades para um lote de CIDs em uma única requisição"""
    if not cids: return []
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    cid_string = ",".join(map(str, cids))
    url = f"{base_url}/compound/cid/{cid_string}/property/SMILES,CanonicalSMILES,IsomericSMILES,MolecularWeight,IUPACName,MolecularFormula/JSON"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json().get('PropertyTable', {}).get('Properties', [])
        return []
    except Exception:
        return []

def calculate_and_filter(prop: Dict) -> Optional[Dict]:
    smiles = prop.get('SMILES') or prop.get('CanonicalSMILES') or prop.get('IsomericSMILES')
    if not smiles: return None
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    # Filtro relaxado para biodiversidade
    if 80 <= mw <= 1200 and -5 <= logp <= 12 and hbd <= 20 and hba <= 30:
        return {
            'cid': prop['CID'],
            'smiles': smiles,
            'mw': mw,
            'logp': logp,
            'hbd': hbd,
            'hba': hba,
            'formula': prop.get('MolecularFormula'),
            'name': prop.get('IUPACName', 'Unknown')
        }
    return None

def collect_massive_dataset(output_path: str = 'data/raw/amazonian_molecules_v6.csv'):
    print("=" * 60)
    print("EXPANSÃO DE DATASET AMAZÔNICO - v6 (Alvo: 1000+)")
    print("=" * 60)
    
    all_results = []
    seen_cids = set()
    
    # 1. Coleta NuBBE (Fonte de alta fidelidade)
    print("\n[1/3] Extraindo base NuBBE...")
    nubbe_sids = get_nubbe_sids(max_results=800)
    nubbe_cids = sids_to_cids_batch(nubbe_sids)
    print(f"  → Encontrados {len(nubbe_cids)} CIDs no NuBBE")
    
    # 2. Coleta Taxonômica
    print("\n[2/3] Buscando famílias e gêneros amazônicos...")
    taxa_to_search = AMAZONIAN_FAMILIES + AMAZONIAN_GENERA
    taxa_cids = []
    for taxon in taxa_to_search:
        print(f"  Buscando {taxon}...", end=" ", flush=True)
        sids = search_entrez_sids(taxon, max_results=150)
        cids = sids_to_cids_batch(sids)
        taxa_cids.extend(cids)
        print(f"({len(cids)} CIDs)")
        time.sleep(0.3)
    
    combined_cids = list(set(nubbe_cids + taxa_cids))
    print(f"\n[3/3] Processando {len(combined_cids)} CIDs únicos totais...")
    
    # Processar em lotes de 50 para propriedades
    for i in range(0, len(combined_cids), 50):
        batch = combined_cids[i:i+50]
        batch = [c for c in batch if c not in seen_cids]
        if not batch: continue
        
        for c in batch: seen_cids.add(c)
        
        props = get_compound_properties_batch(batch)
        for p in props:
            res = calculate_and_filter(p)
            if res:
                all_results.append(res)
        
        if len(all_results) % 100 == 0:
            print(f"  → Progresso: {len(all_results)} moléculas válidas coletadas...")
        
        time.sleep(0.5) # Respeitando limites do PubChem
        
        if len(all_results) >= 1500: # Margem de segurança
            break

    df = pd.DataFrame(all_results)
    df = df.drop_duplicates(subset=['smiles'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"CONCLUÍDO: {len(df)} moléculas salvas em {output_path}")
    print("=" * 60)
    return df

if __name__ == "__main__":
    collect_massive_dataset()
