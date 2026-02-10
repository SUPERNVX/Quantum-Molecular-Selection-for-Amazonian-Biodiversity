"""
data_collection_v8.py - High Intensity Genera Search (Target: 1000+)

Estratégia:
1. Lista massiva de 120+ gêneros botânicos amazônicos.
2. Busca profunda: 300 SIDs por gênero.
3. Processamento de até 5000 CIDs únicos.
"""

import requests
import pandas as pd
import time
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Dict, Optional

# Lista de 120 Gêneros Amazônicos (Crítico para diversidade e volume)
AMAZONIAN_GENERA = [
    "Uncaria", "Copaifera", "Euterpe", "Bixa", "Bertholletia", "Paullinia", 
    "Theobroma", "Aniba", "Carapa", "Phyllanthus", "Psychotria", "Ptychopetalum", 
    "Banisteriopsis", "Tabebuia", "Physalis", "Cecropia", "Vismia", "Aspidosperma", 
    "Piper", "Miconia", "Virola", "Myristica", "Jatropha", "Croton", "Manihot", 
    "Mimosa", "Bauhinia", "Senna", "Cassia", "Swartzia", "Ocotea", "Nectandra", 
    "Endlicheria", "Persea", "Guatteria", "Xylopia", "Annona", "Rollinia", 
    "Duguetia", "Bactris", "Astrocaryum", "Mauritia", "Oenocarpus", "Socratea", 
    "Iriartea", "Licania", "Couepia", "Parinari", "Hirtella", "Eschweilera", 
    "Lecythis", "Couratari", "Gustavia", "Pouteria", "Chrysophyllum", "Micropholis", 
    "Syzygium", "Eugenia", "Myrcia", "Calyptranthes", "Psidium", "Campomanesia", 
    "Clidemia", "Leandra", "Tibouchina", "Bellucia", "Clusia", "Symphonia", 
    "Garcinia", "Calophyllum", "Byrsonima", "Bunchosia", "Hiraea", "Stigmaphyllon", 
    "Diplopterys", "Tetrapterys", "Serjania", "Cardiospermum", "Cupania", 
    "Talisia", "Matayba", "Sapindus", "Allophylus", "Hevea", "Alchornea", 
    "Maprounea", "Mabea", "Sapium", "Euphorbia", "Drypetes", "Cleistanthus", 
    "Bridelia", "Antidesma", "Baccaurea", "Macaranga", "Mallotus", "Acalypha", 
    "Tragia", "Dalechampia", "Cnidoscolus", "Micrandra", "Pausandra", "Goupia", 
    "Protium", "Tetragastris", "Trattinnickia", "Crepidospermum", "Dacryodes", 
    "Zanthoxylum", "Esenbeckia", "Pilocarpus", "Galipea", "Metrodorea", 
    "Sigmatanthus", "Spathelia", "Dictyoloma", "Hortia", "Cedrela", "Swietenia", 
    "Guarea", "Trichilia", "Cabralea", "Anacardium", "Mangifera", "Spondias",
    "Maytenus", "Salacia", "Calliandra", "Inga", "Parkia", "Parkinsonia"
]

def search_entrez_sids(term: str, max_results: int = 300) -> List[str]:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pcsubstance",
        "term": f"{term}",
        "retmode": "json",
        "retmax": max_results
    }
    try:
        response = requests.get(base_url, params=params, timeout=15)
        if response.status_code == 200:
            return response.json().get('esearchresult', {}).get('idlist', [])
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
    if 80 <= mw <= 1800 and -5 <= logp <= 20: # Filtro muito amplo para maximizar volume
        return {
            'cid': prop['CID'],
            'smiles': smiles,
            'mw': mw,
            'logp': logp,
            'formula': prop.get('MolecularFormula'),
            'name': prop.get('IUPACName', 'Unknown')
        }
    return None

def collect_v8():
    print("=" * 60)
    print("COLETA INTENSIVA POR GÊNEROS - v8 (Alvo: 1000+)")
    print("=" * 60)
    
    all_results = []
    seen_cids = set()
    total_cids_found = 0
    
    for idx, genus in enumerate(AMAZONIAN_GENERA, 1):
        print(f"[{idx}/{len(AMAZONIAN_GENERA)}] Processando {genus}...", end=" ", flush=True)
        sids = search_entrez_sids(genus, max_results=300)
        cids = sids_to_cids_batch(sids)
        
        new_cids = [c for c in cids if c not in seen_cids]
        total_cids_found += len(new_cids)
        
        # Processar lote de propriedades para este gênero
        if new_cids:
            # Limitar a 80 CIDs por gênero para diversidade e velocidade
            batch_to_process = new_cids[:80]
            props = get_compound_properties_batch(batch_to_process)
            
            genus_count = 0
            for p in props:
                res = calculate_and_filter(p)
                if res:
                    res['source'] = genus
                    all_results.append(res)
                    seen_cids.add(p['CID'])
                    genus_count += 1
            print(f"({len(new_cids)} novos CIDs, {genus_count} adicionados)")
        else:
            print("(0 novos CIDs)")
            
        if len(all_results) >= 2000:
            print("\nMeta atingida (2000+ moléculas). Encerrando coleta.")
            break
            
        time.sleep(0.3)

    df = pd.DataFrame(all_results)
    df = df.drop_duplicates(subset=['smiles'])
    
    output_path = 'data/raw/amazonian_molecules_v8.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"CONCLUÍDO: {len(df)} moléculas salvas em {output_path}")
    print(f"Total de CIDs únicos explorados: {len(seen_cids)}")
    print("=" * 60)
    return df

if __name__ == "__main__":
    collect_v8()
