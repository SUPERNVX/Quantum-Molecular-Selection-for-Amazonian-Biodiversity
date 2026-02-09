import requests
import json

sids = ['135339936', '135339935', '481101440']
base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

for sid in sids:
    print(f"\nTesting SID: {sid}")
    
    # Try CID conversion
    url = f"{base_url}/substance/sid/{sid}/cids/JSON"
    resp = requests.get(url)
    print(f"  CID conversion status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"  CID data: {resp.json()}")
    
    # Try direct substance properties
    # Note: Substance property endpoint is different: /substance/sid/{sid}/JSON
    url_sub = f"{base_url}/substance/sid/{sid}/JSON"
    resp_sub = requests.get(url_sub)
    print(f"  Substance data status: {resp_sub.status_code}")
    if resp_sub.status_code == 200:
        # Check for SMILES in the substance record (often in PC-Compounds -> PC-Compound_structure)
        # or in PC-Substance_synonyms
        print(f"  Substance JSON length: {len(resp_sub.text)}")
