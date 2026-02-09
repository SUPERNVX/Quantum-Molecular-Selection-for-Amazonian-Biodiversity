import requests
import json

def search_entrez(term, db="pcsubstance", retmax=20):
    """Search NCBI databases via E-Utilities"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": db,
        "term": term,
        "retmode": "json",
        "retmax": retmax
    }
    resp = requests.get(base_url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        return data.get('esearchresult', {}).get('idlist', [])
    return []

# Test Amazon AND plant in pcsubstance
print("Searching Amazon AND plant in pcsubstance...")
sids = search_entrez("Amazon AND plant", db="pcsubstance", retmax=50)
print(f"SIDs found: {len(sids)}")
print(f"Sample SIDs: {sids[:10]}")

# Test SID to CID conversion via PUG REST
if sids:
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    sid_batch = ",".join(sids[:20])
    url = f"{base_url}/substance/sid/{sid_batch}/cids/JSON"
    print(f"\nConverting SIDs to CIDs: {url}")
    resp = requests.get(url)
    if resp.status_code == 200:
        cids_data = resp.json()
        cids = [item.get('CID') for item in cids_data['InformationList']['Information'] if item.get('CID')]
        print(f"Converted to {len(cids)} unique CIDs")
        print(f"Sample CIDs: {cids[:10]}")
