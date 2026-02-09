import requests
import json
import time

base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Test source-based search for NuBBE
print("Testing source-based search for 'NuBBE'...")
url = f"{base_url}/substance/source/NuBBE/cids/JSON"
print(f"URL: {url}")
resp = requests.get(url)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:500]}")

if resp.status_code == 200:
    data = resp.json()
    if 'IdentifierList' in data:
        cids = data['IdentifierList']['CID']
        print(f"Found {len(cids)} CIDs via NuBBE source")
    elif 'Waiting' in data or 'ListKey' in data:
        lk = data.get('Waiting', {}).get('ListKey') or data.get('ListKey')
        print(f"ListKey: {lk}")
        time.sleep(5)
        poll_url = f"{base_url}/substance/listkey/{lk}/cids/JSON"
        poll_resp = requests.get(poll_url)
        print(f"Poll Status: {poll_resp.status_code}")
        print(f"Poll Response: {poll_resp.text[:500]}")

# Test search by all for Amazonian
print("\nTesting 'search/all' for 'Amazonian'...")
url = f"{base_url}/compound/search/all/Amazonian/cids/JSON"
resp = requests.get(url)
print(f"Search all Status: {resp.status_code}")
print(f"Search all Response: {resp.text[:500]}")
