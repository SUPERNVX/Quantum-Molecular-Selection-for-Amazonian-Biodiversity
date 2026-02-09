import requests
import json

cid = 3564 # CID derived from Banisteriopsis caapi SID
base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Try to get ALL properties
url = f"{base_url}/compound/cid/{cid}/property/CanonicalSMILES,IsomericSMILES,InChI,InChIKey,MolecularFormula,MolecularWeight,IUPACName/JSON"
print(f"URL: {url}")
resp = requests.get(url)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    print(json.dumps(resp.json(), indent=2))
else:
    print(f"Error: {resp.text}")

# Try to check if it's a CID or if it's actually an SID?
# Wait, my conversion logic said it found CID 3564.
