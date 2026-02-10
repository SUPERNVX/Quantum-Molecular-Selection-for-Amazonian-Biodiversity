"""
data_collection_v9.py - Final Push to 1000+

Alvo: Gêneros de altíssimo rendimento para completar o dataset.
"""

import requests
import pandas as pd
import time
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Dict, Optional

GENERA_FINAL_PUSH = ["Solanum", "Psychotria", "Piper", "Miconia", "Virola", "Hevea", "Cecropia"]

def search_entrez_sids(term: str, max_results: int = 500) -> List[str]:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pcsubstance", "term": term, "retmode": "json", "retmax": max_results}
    try:
        r = requests.get(base_url, params=params, timeout=15)
        return r.json().get('esearchresult', {}).get('idlist', []) if r.status_code == 200 else []
    except: return []

def sids_to_cids_batch(sids: List[str]) -> List[int]:
    if not sids: return []
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    all_cids = []
    for i in range(0, len(sids), 100):
        batch = sids[i:i+100]
        url = f"{base_url}/substance/sid/{','.join(batch)}/cids/JSON"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                for item in r.json().get('InformationList', {}).get('Information', []):
                    c = item.get('CID')
                    if c:
                        if isinstance(c, list): all_cids.extend(c)
                        else: all_cids.append(c)
            time.sleep(0.2)
        except: continue
    return list(set(all_cids))

def get_prop_batch(cids: List[int]) -> List[Dict]:
    if not cids: return []
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(map(str, cids))}/property/SMILES,CanonicalSMILES,MolecularWeight,IUPACName,MolecularFormula/JSON"
    try:
        r = requests.get(url, timeout=40)
        return r.json().get('PropertyTable', {}).get('Properties', []) if r.status_code == 200 else []
    except: return []

def collect_v9():
    print("FINAL PUSH...")
    all_results = []
    seen_cid_df = pd.read_csv('data/raw/amazonian_molecules_final_1000.csv')['cid'].tolist()
    seen_cids = set(seen_cid_df)
    
    for g in GENERA_FINAL_PUSH:
        print(f"Buscando {g}...", end=" ")
        sids = search_entrez_sids(g, max_results=500)
        cids = sids_to_cids_batch(sids)
        new = [c for c in cids if c not in seen_cids]
        print(f"({len(new)} novos CIDs)")
        
        if new:
            props = get_prop_batch(new[:100]) # Processar até 100 novos por gênero
            for p in props:
                sm = p.get('SMILES') or p.get('CanonicalSMILES')
                if sm:
                    all_results.append({
                        'cid': p['CID'], 'smiles': sm, 
                        'mw': p.get('MolecularWeight'), 'source': g,
                        'name': p.get('IUPACName', 'Unknown'),
                        'formula': p.get('MolecularFormula')
                    })
                    seen_cids.add(p['CID'])
        time.sleep(0.5)

    if all_results:
        df_new = pd.DataFrame(all_results)
        df_old = pd.read_csv('data/raw/amazonian_molecules_final_1000.csv')
        df_final = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=['smiles'])
        df_final.to_csv('data/raw/amazonian_molecules_final_1000.csv', index=False)
        print(f"Total Final: {len(df_final)} moléculas.")

if __name__ == "__main__":
    collect_v9()
