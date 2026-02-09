import requests
import json

def search_entrez(term, db="pcsubstance"):
    """Search NCBI databases via E-Utilities"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": db,
        "term": term,
        "retmode": "json",
        "retmax": 20
    }
    print(f"Searching {db} for '{term}'...")
    resp = requests.get(base_url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        ids = data.get('esearchresult', {}).get('idlist', [])
        print(f"  Found {len(ids)} IDs")
        return ids
    else:
        print(f"  Error: {resp.status_code}")
        return []

# Test for aspirin (should work)
print("--- Test: Aspirin ---")
search_entrez("aspirin", db="pccompound")

# Test for species in substances
print("\n--- Test: Uncaria tomentosa in pcsubstance ---")
ids = search_entrez("Uncaria tomentosa", db="pcsubstance")
print(f"SIDs for Uncaria tomentosa: {ids}")

# Test for general Amazonian in substances
print("\n--- Test: Amazonian in pcsubstance ---")
ids_amazon = search_entrez("Amazonia", db="pcsubstance")
print(f"SIDs for Amazonia query: {ids_amazon}")
