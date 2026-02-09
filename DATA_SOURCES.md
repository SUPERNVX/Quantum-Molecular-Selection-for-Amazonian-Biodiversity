# DATA SOURCES FOR AMAZONIAN MOLECULAR DIVERSITY

## CRITICAL: You DON'T need privileged data. Public databases are sufficient.

---

## PRIMARY DATA SOURCES (Recommended)

### 1. **PubChem** (FREE, NO API KEY NEEDED)
- **What**: 100+ million chemical structures
- **Amazonian subset**: Search "Amazonian plant" OR specific species
- **API**: https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
- **How to get**:
  ```python
  import requests
  
  # Search for Amazonian compounds
  search_term = "Amazonian AND plant"
  url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{search_term}/cids/JSON"
  response = requests.get(url)
  cids = response.json()['IdentifierList']['CID']
  
  # Get molecular data
  for cid in cids[:100]:  # First 100
      mol_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
      mol_data = requests.get(mol_url).json()
  ```

### 2. **ZINC Database** (FREE)
- **What**: Drug-like molecules, 230+ million
- **Subset**: Natural products
- **URL**: https://zinc.docking.org/
- **Download**: Pre-filtered natural products dataset
  ```bash
  # Download natural products subset
  wget https://zinc.docking.org/substances/subsets/natural-products.smi
  ```

### 3. **ChEMBL** (FREE, Medicinal Chemistry)
- **What**: Bioactive molecules with drug-like properties
- **API**: https://www.ebi.ac.uk/chembl/api/data/docs
- **Filter**: Natural products from plants
  ```python
  from chembl_webresource_client.new_client import new_client
  
  molecule = new_client.molecule
  # Search for natural products
  natural_products = molecule.filter(
      natural_product='1',
      molecule_properties__mw_freebase__lte=500
  )
  ```

### 4. **NuBBE Database** (Brazilian Natural Products) ⭐ BEST FIT
- **What**: Natural products from Brazilian biodiversity
- **URL**: https://nubbe.iq.unesp.br/portal/nubbedb.html
- **Size**: 2,000+ compounds from Brazilian flora/fauna
- **Format**: SMILES, MOL files
- **Download**: Direct from website (free registration)

### 5. **Brazilian Flora 2020** (Species List)
- **What**: Complete list of Brazilian plant species
- **URL**: http://floradobrasil.jbrj.gov.br/
- **Use**: Get species names → search in PubChem
- **Amazonian filter**: Filter by "Amazônia" biome

---

## SECONDARY SOURCES (For enrichment)

### 6. **LOTUS Initiative** (Natural Products)
- **What**: 750,000+ natural products with biological sources
- **URL**: https://lotus.naturalproducts.net/
- **GitHub**: https://github.com/lotusnprod/lotus-processor
- **Download**: Full dataset as CSV

### 7. **COCONUT** (Collection of Open Natural Products)
- **What**: 400,000+ natural products
- **URL**: https://coconut.naturalproducts.net/
- **Download**: Bulk download available

### 8. **NAPRALERT** (Natural Products Alert)
- **What**: 200,000+ natural products from ethnobotany
- **URL**: https://www.napralert.org/
- **Access**: Free for academic use

---

## MOLECULAR PROPERTIES DATABASES

### 9. **PubChem BioAssay**
- **What**: Biological activity data
- **Use**: Get known activities for validation
- **API**: Same as PubChem

### 10. **DrugBank** (FREE tier)
- **What**: Drug molecules with targets
- **URL**: https://go.drugbank.com/
- **Use**: Identify therapeutic targets

---

## RECOMMENDED WORKFLOW

### Step 1: Get Core Dataset (NuBBE + PubChem)
```python
# 1. Download NuBBE database (manual from website)
# 2. Search PubChem for Amazonian species
# 3. Combine into single dataset

species_list = [
    "Uncaria tomentosa",  # Cat's claw
    "Copaifera",          # Copaiba
    "Bertholletia excelsa",  # Brazil nut
    "Paullinia cupana",   # Guaraná
    "Euterpe oleracea",   # Açaí
    # ... add 50+ more
]

for species in species_list:
    search_pubchem(species)
    extract_smiles()
    calculate_properties()
```

### Step 2: Filter to Relevant Molecules
```python
# Criteria for "interesting" molecules:
- Molecular weight: 150-500 Da (drug-like)
- LogP: 0-5 (cell permeability)
- H-bond donors/acceptors: reasonable
- No toxic substructures
- Structural diversity (Tanimoto similarity < 0.8)
```

### Step 3: Calculate Diversity Metrics
```python
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def calculate_diversity_score(molecules):
    """
    Maximize structural diversity in selected set
    This is your optimization objective
    """
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(m, 2) 
                    for m in molecules]
    
    diversity = 0
    for i in range(len(fingerprints)):
        for j in range(i+1, len(fingerprints)):
            similarity = DataStructs.TanimotoSimilarity(
                fingerprints[i], 
                fingerprints[j]
            )
            diversity += (1 - similarity)  # Higher = more diverse
    
    return diversity
```

---

## DATASET SIZE RECOMMENDATIONS

For your paper to be credible:

- **Minimum**: 100 molecules (too small, but publishable as proof-of-concept)
- **Good**: 500 molecules (solid undergraduate research)
- **Excellent**: 1,000+ molecules (competitive with grad student work)

**Target: 500 molecules from NuBBE + PubChem Amazonian search**

---

## DATA PREPARATION SCRIPT

```python
"""
data_collection.py
Downloads and prepares Amazonian molecular dataset
"""

import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import time

def search_pubchem_species(species_name):
    """Search PubChem for molecules from a species"""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # Search for species
    search_url = f"{base_url}/compound/name/{species_name}/cids/JSON"
    
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            cids = response.json()['IdentifierList']['CID']
            return cids[:10]  # Limit to 10 per species
        else:
            return []
    except Exception as e:
        print(f"Error searching {species_name}: {e}")
        return []

def get_molecule_data(cid):
    """Get molecular properties from PubChem CID"""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # Get SMILES and properties
    prop_url = f"{base_url}/compound/cid/{cid}/property/CanonicalSMILES,MolecularFormula,MolecularWeight,IUPACName/JSON"
    
    try:
        response = requests.get(prop_url, timeout=10)
        if response.status_code == 200:
            return response.json()['PropertyTable']['Properties'][0]
        else:
            return None
    except Exception as e:
        print(f"Error getting CID {cid}: {e}")
        return None

def calculate_molecular_descriptors(smiles):
    """Calculate RDKit descriptors"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol),
    }
    
    return descriptors

def is_druglike(descriptors):
    """Lipinski's Rule of Five + extra filters"""
    if descriptors is None:
        return False
    
    return (
        150 <= descriptors['mw'] <= 500 and
        descriptors['logp'] <= 5 and
        descriptors['hbd'] <= 5 and
        descriptors['hba'] <= 10 and
        descriptors['rotatable_bonds'] <= 10
    )

# AMAZONIAN SPECIES LIST (add more from Flora do Brasil)
AMAZONIAN_SPECIES = [
    "Uncaria tomentosa",
    "Copaifera langsdorffii",
    "Bertholletia excelsa",
    "Paullinia cupana",
    "Euterpe oleracea",
    "Hevea brasiliensis",
    "Phyllanthus niruri",
    "Carapa guianensis",
    "Dipteryx odorata",
    "Virola surinamensis",
    # Add 40+ more species here
]

def main():
    """Collect Amazonian molecular dataset"""
    all_molecules = []
    
    print("Collecting molecules from PubChem...")
    for species in AMAZONIAN_SPECIES:
        print(f"Processing: {species}")
        cids = search_pubchem_species(species)
        
        for cid in cids:
            mol_data = get_molecule_data(cid)
            if mol_data:
                smiles = mol_data.get('CanonicalSMILES')
                descriptors = calculate_molecular_descriptors(smiles)
                
                if is_druglike(descriptors):
                    all_molecules.append({
                        'cid': cid,
                        'species': species,
                        'smiles': smiles,
                        'formula': mol_data.get('MolecularFormula'),
                        'name': mol_data.get('IUPACName', 'Unknown'),
                        **descriptors
                    })
            
            time.sleep(0.2)  # Rate limiting
    
    # Save dataset
    df = pd.DataFrame(all_molecules)
    df.to_csv('data/raw/amazonian_molecules.csv', index=False)
    print(f"Collected {len(df)} molecules")
    
    return df

if __name__ == "__main__":
    dataset = main()
```

---

## VERIFICATION

After collecting data, verify:
1. ✓ At least 100 molecules
2. ✓ SMILES strings are valid (RDKit can parse)
3. ✓ No duplicates (use InChI keys)
4. ✓ Structural diversity (Tanimoto matrix)
5. ✓ Source attribution (which species/database)

---

## NEXT STEPS

Once you have the dataset:
1. Run diversity analysis (classical baseline)
2. Formulate optimization problem
3. Implement QAOA solution
4. Compare results

**You don't need "privileged data". You need a smart question about public data.**
