# Quantum-Accelerated Molecular Selection for Amazonian Biodiversity

**Author**: Nicolas Mendes de Araújo  
**Affiliation**: Colégio de Santa Inês, São Paulo, Brazil | LACQ Feynman (Liga Acadêmica de Computação Quântica)  
**Date**: February 2026  
**Status**: 0.3.0 - Quantum Advantage Verified ($N=25$)

---

## 🎯 Project Overview

This research investigates the application of quantum computing to optimize molecular selection from Amazonian biodiversity. Using the **Hybrid Warm-Start QAOA (Quantum Approximate Optimization Algorithm)**, we address the computational challenge of selecting maximally diverse molecular subsets for drug discovery.

**Key Innovation**: The use of a "Hybrid" architecture where a classical greedy algorithm initializes the quantum state ("Warm-Start"), allowing QAOA to perform a high-fidelity local search for global optima that classical heuristics miss.

---

## 🏆 Current Benchmark Results

| Scale (Qubits) | Configuration | Greedy Baseline | GA (Genetic) | **QAOA Hybrid** | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **N=15** | K=6 | 13.7625 | 13.8344 | **14.0627** | ✅ Superior (+2.2%) |
| **N=25** | K=8 | 26.1803 | 26.1969 | **26.3147** | ✅ Superior (Refined) |

*Results confirm that QAOA successfully navigates non-convex diversity landscapes where Greedy and GA (100 gen) settle in local minima.*

---

## 🔬 Methodology

### Dataset
- **Source**: [BrNPDB](https://brnpdb.shinyapps.io/BrNPDB/) (Brazilian Natural Product Database)
- **Selection**: 810 Amazonian molecules refined by drug-likeness (MW: 150-600, LogP: -1 to 6).
- **Representation**: Morgan fingerprints (radius=2, 2048 bits).

### Hybrid Architecture
1. **Classical Warm-Start**: $O(N^2)$ Greedy algorithm identifies a high-quality initial cluster.
2. **Quantum Refinement**: QAOA (p=2) explores the Hilbert space around the classical solution.
3. **Simulation**: Optimized `AerSimulator` with GPU/CPU fallback for $N < 30$ stability.

---

## 📁 Project Structure

```
quantum_biodiversity_project/
├── data/
│   ├── processed/              # High-quality Amazonian datasets
│   ├── traps/                  # Hard instances for benchmark (N=15, 25, 30+)
│   └── results/                # Optimization logs and scores
├── src/
│   ├── hotstart/               # Core Hybrid/Lite Pipeline
│   │   ├── hybrid_selector.py  # Main refining solver (V2 Sampler)
│   │   ├── lite_selector.py    # High-speed local simulator
│   │   ├── classical.py        # Classical heuristics (Greedy/GA)
│   │   └── find_greedy_traps.py # Benchmark generator
│   └── utils/                  # Chemical and hardware utilities
├── demo_refinement.py          # Command center for demonstrations
├── SCIENTIFIC_CHANGELOG.md     # Detailed scientific verification log
├── setup_environment.ps1       # Environment recovery for Windows
└── README.md                   # This file
```

---

## 🚀 Getting Started

### 1. Environment Setup
```powershell
.\setup_environment.ps1
.venv\Scripts\Activate.ps1
```

### 2. Run Demonstration (N=15 or N=25)
```powershell
# Demonstrates QAOA beating Greedy and GA
python demo_refinement.py
```

### 3. Run Custom Optimization
```powershell
python src/hotstart/hybrid_selector.py --trap trap_N25_K8 --p 2 --maxiter 100
```

---

## 📖 Key Documentation
- **[Scientific Changelog](SCIENTIFIC_CHANGELOG.md)**: Proof of Quantum Advantage and Technical Evolution.
- **[Walkthrough](docs/walkthrough.md)**: Detailed instructions for each component.
- **[Project Status](PROJECT_STATUS.md)**: Current development milestones.

---

## 🤝 Acknowledgments
- **LACQ Feynman**: Gabriel Albuquerque and team for mentorship.
- **IBM Quantum**: Access to real hardware for future N=127+ scale.

**Nicolas Mendes de Araújo** | [@supernvx](https://github.com/supernvx)
