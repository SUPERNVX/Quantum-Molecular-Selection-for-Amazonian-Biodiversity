import requests
import json

cid = 702
base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
prop_url = f"{base_url}/compound/cid/{cid}/property/CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,IUPACName/JSON"

print(f"Testing URL: {prop_url}")
resp = requests.get(prop_url)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    print(json.dumps(resp.json(), indent=2))
else:
    print(f"Error: {resp.text}")
