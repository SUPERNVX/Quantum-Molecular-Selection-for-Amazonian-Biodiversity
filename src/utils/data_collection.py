"""
data_collection.py - v5 (Key Fix)

Coleta moléculas da biodiversidade amazônica usando NCBI Entrez e PubChem PUG REST.
Consertado erro de chaves de SMILES (SMILES vs CanonicalSMILES).
"""

import requests
import pandas as pd
import time
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Dict, Optional

# Espécies amazônicas de interesse (Expandido)
AMAZONIAN_SPECIES = [
    "Uncaria tomentosa", "Paullinia cupana", "Euterpe oleracea", 
    "Bixa orellana", "Bertholletia excelsa", "Theobroma grandiflorum",
    "Aniba rosaeodora", "Copaifera langsdorffii", "Carapa guianensis",
    "Phyllanthus niruri", "Physalis angulata", "Ptychopetalum olacoides",
    "Banisteriopsis caapi", "Psychotria viridis", "Tabebuia impetiginosa"
]

def search_entrez_sids(term: str, max_results: int = 50) -> List[str]:
    """Busca SIDs (Substance IDs) no NCBI via Entrez"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pcsubstance",
        "term": term,
        "retmode": "json",
        "retmax": max_results
    }
    try:
        response = requests.get(base_url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return data.get('esearchresult', {}).get('idlist', [])
        return []
    except Exception:
        return []

def sids_to_cids(sids: List[str]) -> List[int]:
    """Converte SIDs para CIDs usando PubChem PUG REST"""
    if not sids: return []
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    sid_string = ",".join(sids)
    url = f"{base_url}/substance/sid/{sid_string}/cids/JSON"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            cids = []
            for item in data.get('InformationList', {}).get('Information', []):
                cid_list = item.get('CID')
                if cid_list:
                    if isinstance(cid_list, list):
                        cids.extend(cid_list)
                    else:
                        cids.append(cid_list)
            return list(set(cids))
        return []
    except Exception:
        return []

def get_molecule_properties(cid: int) -> Optional[Dict]:
    """Obtém propriedades moleculares do PubChem"""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    # Adicionando todas as variantes de SMILES e propriedades úteis
    prop_url = f"{base_url}/compound/cid/{cid}/property/SMILES,CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,IUPACName/JSON"
    try:
        response = requests.get(prop_url, timeout=12)
        if response.status_code == 200:
            props = response.json()['PropertyTable']['Properties'][0]
            # Tentar várias chaves de SMILES
            smiles = props.get('SMILES') or props.get('CanonicalSMILES') or props.get('IsomericSMILES')
            if smiles:
                props['smiles_to_use'] = smiles
                return props
        return None
    except Exception:
        return None

def calculate_molecular_descriptors(smiles: str) -> Optional[Dict]:
    """Calcula descritores moleculares (RDKit)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol),
    }

def is_druglike(descriptors: Dict) -> bool:
    """Filtro drug-like (Generalizado para Biodiversidade)"""
    if descriptors is None: return False
    # Filtros amplos para capturar diversidade natural
    return (
        80 <= descriptors['mw'] <= 1500 and 
        -5 <= descriptors['logp'] <= 15 and
        descriptors['hbd'] <= 25 and
        descriptors['hba'] <= 35
    )

def collect_dataset(output_path: str = 'data/raw/amazonian_molecules.csv') -> pd.DataFrame:
    """Coleta dataset utilizando Entrez e PubChem PUG REST"""
    print("=" * 60)
    print("COLETA DE MOLÉCULAS AMAZÔNICAS - PubChem (v5 - Final)")
    print("=" * 60)
    
    all_molecules = []
    seen_cids = set()
    
    # Adicionando termos genéricos fortes
    search_queries = AMAZONIAN_SPECIES + [
        "Amazonian plant natural product",
        "Brazilian biodiversity",
        "Amazon rainforest chemical",
        "South American medicinal plant"
    ]
    
    for idx, query in enumerate(search_queries, 1):
        print(f"[{idx}/{len(search_queries)}] Buscando: {query}")
        sids = search_entrez_sids(query, max_results=50)
        
        if not sids and " " in query:
            sids = search_entrez_sids(query.split()[0])
            
        if not sids: continue
            
        cids = sids_to_cids(sids)
        new_cids = [c for c in cids if c not in seen_cids]
        
        if not new_cids: continue
            
        print(f"  → Processando {len(new_cids)} novos CIDs")
        
        count = 0
        for cid in new_cids[:25]:
            seen_cids.add(cid)
            mol_data = get_molecule_properties(cid)
            if mol_data:
                smiles = mol_data.get('smiles_to_use')
                if not smiles: continue
                
                descriptors = calculate_molecular_descriptors(smiles)
                if is_druglike(descriptors):
                    all_molecules.append({
                        'cid': cid,
                        'source': query,
                        'smiles': smiles,
                        'formula': mol_data.get('MolecularFormula'),
                        'name': mol_data.get('IUPACName', 'Unknown'),
                        **descriptors
                    })
                    count += 1
            # Rate limiting (PubChem PUG REST)
            time.sleep(0.4)
        print(f"  ✓ {count} moléculas adicionadas")

    if not all_molecules:
        print("\nFATAL: Nenhuma molécula encontrada!")
        return pd.DataFrame()

    df = pd.DataFrame(all_molecules)
    df = df.drop_duplicates(subset=['smiles'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"CONCLUÍDO: {len(df)} moléculas únicas salvas")
    if not df.empty:
        print(f"Estatísticas:")
        print(f"  - Média MW: {df['mw'].mean():.2f}")
        print(f"  - Espécies/Teremos representados: {df['source'].nunique()}")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    dataset = collect_dataset()
    print("\nExecução finalizada com sucesso!" if not dataset.empty else "Falha na coleta.")
