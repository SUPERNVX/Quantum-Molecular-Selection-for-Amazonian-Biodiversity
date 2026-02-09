import requests
import json

base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

url = f"{base_url}/sources/substance/JSON"
resp = requests.get(url)

if resp.status_code == 200:
    sources = resp.json().get('InformationList', {}).get('SourceName', [])
    
    # Case-insensitive search
    matches = [s for s in sources if "nubbe" in s.lower() or "brazil" in s.lower()]
    print(f"Matches for 'nubbe' or 'brazil': {matches}")
    
    # Also search for 'Amazon'
    amazon_matches = [s for s in sources if "amazon" in s.lower()]
    print(f"Matches for 'amazon': {amazon_matches}")
    
    # Search for anything related to 'natural product'
    np_matches = [s for s in sources if "natural product" in s.lower()]
    print(f"Matches for 'natural product' (first 10): {np_matches[:10]}")
else:
    print(f"Error: {resp.status_code}")
