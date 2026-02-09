# Quantum-Accelerated Molecular Selection for Amazonian Biodiversity

**Author**: Nicolas Mendes de Araújo  
**Affiliation**: Colégio de Santa Inês, São Paulo, Brazil | LACQ Feynman (Liga Acadêmica de Computação Quântica)  
**Date**: February 2026  
**Status**: In Development

---

## 🎯 Project Overview

This research investigates the application of quantum computing to optimize molecular selection from Amazonian biodiversity. Using the Quantum Approximate Optimization Algorithm (QAOA) on IBM Quantum hardware, we address the computational challenge of selecting maximally diverse molecular subsets for drug discovery applications.

**Research Question**: Can quantum algorithms provide computational advantages over classical methods in selecting structurally diverse molecules from large chemical databases?

**Significance**: 
- The Amazon rainforest contains an estimated 10 million species, with < 1% chemically characterized
- Traditional drug discovery costs $10k-50k per molecule to synthesize and test
- Intelligent molecular selection can accelerate bioprospecting while preserving biodiversity

---

## 📊 Key Results (Preliminary)

| Algorithm | Diversity Score | Execution Time | Notes |
|-----------|----------------|----------------|-------|
| Greedy | TBD | TBD | Classical baseline |
| Genetic Algorithm | TBD | TBD | Optimized heuristic |
| QAOA (Simulator) | TBD | TBD | Quantum algorithm |
| QAOA (Real Quantum) | TBD | TBD | IBM Quantum hardware |

*Results will be updated as experiments complete*

---

## 🔬 Methodology

### Dataset
- **Source**: PubChem, NuBBE Database (Brazilian Natural Products)
- **Size**: 300-500 Amazonian plant-derived molecules
- **Filters**: Drug-like properties (Lipinski's Rule of Five)
- **Representation**: Morgan fingerprints (radius=2, 2048 bits)

### Problem Formulation
**Objective**: Maximize structural diversity in selected molecular subset

```
maximize: sum_{i<j} (1 - Tanimoto_similarity(i,j)) * x_i * x_j
subject to: sum_i x_i = k
where: x_i ∈ {0,1} indicates molecule selection
```

**Optimization Approaches**:
1. **Classical Greedy**: O(kn) complexity, fast but suboptimal
2. **Genetic Algorithm**: Evolutionary optimization, better solutions
3. **QAOA**: Quantum variational algorithm, potential quantum advantage

### QAOA Implementation
- **Depth**: p = 1-3 layers
- **Backend**: IBM Quantum (ibmq_manila, ibmq_quito)
- **Shots**: 512-1024 measurements
- **Optimizer**: COBYLA (constrained optimization)

---

## 📁 Project Structure

```
quantum_biodiversity_project/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned, preprocessed data
│   └── results/                # Experimental results
├── src/
│   ├── classical/              # Classical algorithms
│   │   ├── greedy_selector.py
│   │   └── genetic_selector.py
│   ├── quantum/                # Quantum algorithms
│   │   ├── qaoa_selector.py
│   │   └── qubo_formulation.py
│   ├── utils/                  # Utilities
│   │   ├── data_collection.py
│   │   └── visualization.py
│   └── analysis/               # Analysis scripts
├── notebooks/                  # Jupyter notebooks
├── figures/                    # Generated plots
├── papers/                     # Paper drafts
├── setup_environment.sh        # Setup script
├── verify_setup.py             # Verification script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- IBM Quantum account (free): https://quantum.ibm.com/

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/quantum-biodiversity.git
cd quantum-biodiversity

# 2. Run setup
bash setup_environment.sh

# 3. Verify installation
python verify_setup.py

# 4. Configure IBM Quantum
# Copy .env.template to .env and add your token
cp .env.template .env
# Edit .env and add: IBM_QUANTUM_TOKEN=your_token_here
```

### Quick Start

```python
# Collect dataset
python src/utils/data_collection.py

# Run classical baseline
python src/classical/greedy_selector.py --k 20

# Run QAOA (simulator first)
python src/quantum/qaoa_selector.py --backend simulator --k 20

# Compare results
python src/analysis/compare_algorithms.py
```

---

## 📖 Documentation

- **[Execution Guide](EXECUTION_GUIDE.md)**: Complete step-by-step workflow
- **[Data Sources](DATA_SOURCES.md)**: Where to get molecular data
- **[API Documentation](docs/api.md)**: Code reference (coming soon)
- **[Lab Notebook](docs/lab_notebook.md)**: Daily experimental log

---

## 🧪 Experiments

### Completed
- [x] Environment setup
- [ ] Data collection (in progress)
- [ ] Classical baseline
- [ ] QAOA implementation
- [ ] Quantum hardware execution
- [ ] Comparative analysis

### Planned
- [ ] Error mitigation techniques
- [ ] Larger problem sizes (n=1000+)
- [ ] Different molecular properties (solubility, toxicity)
- [ ] Integration with experimental validation

---

## 📝 Publications

### Preprints
- [ ] arXiv submission (Target: April 2026)

### Conference Submissions
- [ ] IEEE Quantum Week Student Paper Competition (Target: May 2026)
- [ ] LAWQC (Latin American Workshop on Quantum Computing)

### Journal Submissions (Future)
- [ ] Quantum Information Processing
- [ ] Scientific Reports

---

## 🤝 Acknowledgments

- **LACQ Feynman**: Gabriel Albuquerque and team for mentorship and quantum computing resources
- **IBM Quantum**: Quantum computing access through IBM Quantum Experience
- **NuBBE Database**: Prof. Vanderlan Bolzani for Brazilian natural product data

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Data Attribution**:
- PubChem: Public domain
- NuBBE Database: Academic use with attribution

---

## 📧 Contact

**Nicolas Mendes de Araújo**
- Email: nicolasmda09@gmail.com
- GitHub: [@supernvx](https://github.com/supernvx)

**Project Updates**: Follow development at [GitHub Project Board](link)

---

## 🌟 Impact

This research aims to demonstrate:
1. **Scientific**: Quantum computing applied to real-world biodiversity challenges
2. **Educational**: Open-source framework for quantum algorithm research
3. **Social**: Accelerating drug discovery from endangered ecosystems

**Broader Impact**: Methods developed here can be applied to:
- Other biodiversity hotspots globally
- Materials discovery
- Chemical library optimization
- Personalized medicine

---

## 🔗 Related Work

**Quantum Optimization**:
- Farhi et al. (2014): "A Quantum Approximate Optimization Algorithm"
- Cerezo et al. (2021): "Variational Quantum Algorithms"

**Molecular Diversity**:
- Willett (1999): "Dissimilarity-Based Algorithms for Selecting Structurally Diverse Sets"
- Medina-Franco et al. (2008): "Characterization of Activity Landscapes"

**Biodiversity Informatics**:
- ter Steege et al. (2013): "Hyperdominance in the Amazonian Tree Flora"
- Sousa-Baena et al. (2014): "Brazilian Flora 2020 Database"

---

## 📊 Citation

If you use this code or methodology, please cite:

```bibtex
@misc{araujo2026quantum,
  author = {Araújo, Nicolas Mendes de},
  title = {Quantum-Accelerated Molecular Selection for Amazonian Biodiversity},
  year = {2026},
  publisher = {arXiv},
  note = {In preparation}
}
```

---

## 🛠️ Development Status

| Component | Status | Last Updated |
|-----------|--------|--------------|
| Environment Setup | ✅ Complete | 2026-02-08 |
| Data Collection | 🔄 In Progress | - |
| Classical Baseline | ⏳ Pending | - |
| QAOA Implementation | ⏳ Pending | - |
| Quantum Execution | ⏳ Pending | - |
| Analysis | ⏳ Pending | - |
| Paper Writing | ⏳ Pending | - |

**Legend**: ✅ Complete | 🔄 In Progress | ⏳ Pending | ❌ Blocked

---

## 🎓 Educational Resources

**Included Tutorials**:
- [ ] Introduction to QAOA (notebook)
- [ ] Molecular fingerprints explained (notebook)
- [ ] QUBO formulation guide (notebook)
- [ ] IBM Quantum basics (notebook)

**External Resources**:
- Qiskit Textbook: https://qiskit.org/learn/
- RDKit Tutorials: https://www.rdkit.org/docs/
- Quantum Computing for the Very Curious: https://quantum.country/

---

## 🐛 Known Issues

- [ ] Large similarity matrices (n>500) cause memory issues
  - **Workaround**: Sparse matrix representation
- [ ] QAOA convergence sensitive to initial parameters
  - **Workaround**: Multiple random initializations
- [ ] IBM Quantum queue times variable
  - **Workaround**: Submit jobs during off-peak hours (US night time)

**Report Issues**: [GitHub Issues](your_repo/issues)

---

## 🚧 Future Directions

### Short-term (3 months)
- [ ] Implement error mitigation (readout error, ZNE)
- [ ] Test on larger datasets (n=1000+)
- [ ] Benchmarking on different quantum backends

### Medium-term (6 months)
- [ ] Integration with molecular dynamics simulations
- [ ] Multi-objective optimization (diversity + drug-likeness)
- [ ] Web interface for molecular selection

### Long-term (1 year+)
- [ ] Experimental validation with synthesis partners
- [ ] Application to other biodiversity hotspots
- [ ] Production-ready pipeline for drug discovery

---

**Last Updated**: February 8, 2026  
**Version**: 0.1.0-alpha  
**Contributors**: 1

---

*This project combines quantum computing, biodiversity conservation, and drug discovery to address real-world challenges. We believe in open science, reproducible research, and using technology for social impact.*

**Join us in protecting the Amazon, one molecule at a time. 🌳🔬💊**
