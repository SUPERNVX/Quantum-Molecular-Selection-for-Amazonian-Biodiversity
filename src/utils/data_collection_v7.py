"""
data_collection_v7.py - High Throughput Dataset Expansion (Target: 1000+)

Estratégia:
1. Limites massivos: 3000 NuBBE SIDs + 500 per Taxon.
2. Busca por classes químicas (Alcaloides, Terpenos, Flavonoides) combinadas com termos regionais.
3. Paralelismo simulado via lotes maiores e processamento otimizado.
"""

import requests
import pandas as pd
import time
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Dict, Optional

# Expansão Taxonômica Massiva
TAXA = [
    "Leguminosae", "Rubiaceae", "Lauraceae", "Annonaceae", 
    "Euphorbiaceae", "Burseraceae", "Rutaceae", "Myrtaceae", 
    "Melastomataceae", "Malvaceae", "Sapotaceae", "Apocynaceae",
    "Piperaceae", "Bignoniaceae", "Fabaceae", "Solanaceae",
    "Araceae", "Arecaceae", "Bromeliaceae", "Clusiaceae",
    "Malpighiaceae", "Sapindaceae", "Menispermaceae"
]

CHEMICAL_CLASSES = [
    "Alkaloids", "Terpenoids", "Flavonoids", "Quinones", 
    "Coumarins", "Saponins", "Tannins", "Iridoids"
]

REGION_TERMS = ["Amazon", "Brazil", "Tropical", "Neotropical"]

def search_entrez_sids(term: str, max_results: int = 500) -> List[str]:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pcsubstance",
        "term": f"{term}",
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

def get_nubbe_sids(max_results: int = 3000) -> List[str]:
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
    if not cids: return []
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    cid_string = ",".join(map(str, cids))
    url = f"{base_url}/compound/cid/{cid_string}/property/SMILES,CanonicalSMILES,IsomericSMILES,MolecularWeight,IUPACName,MolecularFormula/JSON"
    try:
        response = requests.get(url, timeout=40)
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
    if 80 <= mw <= 1500 and -5 <= logp <= 15:
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

def collect_v7():
    print("=" * 60)
    print("COLETA EM ALTA ESCALA - v7 (Alvo: 1000+)")
    print("=" * 60)
    
    all_results = []
    seen_cids = set()
    
    # 1. NuBBE (Fonte Principal)
    print("\n[1/3] Extraindo NuBBE (3000 SIDs)...")
    nubbe_sids = get_nubbe_sids(max_results=3000)
    nubbe_cids = sids_to_cids_batch(nubbe_sids)
    print(f"  → NuBBE: {len(nubbe_cids)} CIDs")
    
    # 2. Busca Combinatória (Classe + Região)
    print("\n[2/3] Buscas combinatórias (Classe Química + Região)...")
    combined_queries = []
    for c in CHEMICAL_CLASSES:
        for r in REGION_TERMS:
            combined_queries.append(f"{c} {r}")
    
    search_terms = TAXA + combined_queries
    extra_cids = []
    for term in search_terms:
        print(f"  Buscando {term}...", end=" ", flush=True)
        sids = search_entrez_sids(term, max_results=300)
        cids = sids_to_cids_batch(sids)
        extra_cids.extend(cids)
        print(f"({len(cids)} CIDs)")
        time.sleep(0.3)
        if len(set(nubbe_cids + extra_cids)) > 15000: break # Limite de memória/tempo
    
    final_cids = list(set(nubbe_cids + extra_cids))
    print(f"\n[3/3] Processando {len(final_cids)} CIDs únicos totais...")
    
    for i in range(0, len(final_cids), 80): # Lotes maiores
        batch = [c for c in final_cids[i:i+80] if c not in seen_cids]
        if not batch: continue
        for c in batch: seen_cids.add(c)
        
        props = get_compound_properties_batch(batch)
        for p in props:
            res = calculate_and_filter(p)
            if res:
                all_results.append(res)
        
        if len(all_results) % 200 == 0 or len(all_results) < 200:
            print(f"  → Válidas: {len(all_results)} | Processadas: {i+80}")
        
        time.sleep(0.5)
        if len(all_results) >= 2000: break

    df = pd.DataFrame(all_results)
    df = df.drop_duplicates(subset=['smiles'])
    
    output_path = 'data/raw/amazonian_molecules_v7.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"CONCLUÍDO: {len(df)} moléculas salvas em {output_path}")
    print("=" * 60)
    return df

if __name__ == "__main__":
    collect_v7()
