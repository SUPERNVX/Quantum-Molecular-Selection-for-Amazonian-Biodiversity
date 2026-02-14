# Hotstart System for Molecular Selection

This directory contains the production-ready hybrid classical-quantum pipeline for optimizing molecular diversity with **Warm-Start QAOA**.

## Core Selectors

### 1. [`hybrid_selector.py`](hybrid_selector.py)
- **Primary Use**: High-fidelity refinement and Real Hardware execution.
- **Backend**: Uses Qiskit Runtime V2 (SamplerV2).
- **Architecture**: Greedy Warm-Start + Parametrized Ansatz refinement.
- **Hardware Support**: Includes logic for IBM Quantum Real Hardware runs.

### 2. [`lite_selector.py`](lite_selector.py)
- **Primary Use**: High-speed local simulation and benchmarking (N < 30).
- **Optimization**: Uses `AerSimulator` directly, bypassing the overhead of Runtime primitives.
- **Stability**: Features an automatic GPU (RTX 4060) to CPU fallback mechanism.

### 3. [`classical.py`](classical.py)
- **Primary Use**: Core heuristics and data handling.
- **Algorithms**: Implements Greedy and Genetic Algorithm (GA) baselines.
- **Features**: RDKit Morgan Radius 2 fingerprints, Tanimoto similarity, and aggressive matrix caching in `data/processed/`.

---

## Benchmark Scenarios (Traps)

The [`find_greedy_traps.py`](find_greedy_traps.py) script generated hard instances specifically designed to showcase quantum advantage:

| Trap Name | N | K | Status |
| :--- | :--- | :--- | :--- |
| `trap_N15_K6` | 15 | 6 | ✅ QAOA Victory (+2.2%) |
| `trap_N25_K8` | 25 | 8 | ✅ QAOA Victory (Refined) |
| `trap_N30_K5` | 30 | 5 | ⏳ Pending Benchmark |

---

## How to Execute

### Local Performance Test (Lite)
```powershell
python src/hotstart/lite_selector.py --trap trap_N25_K8 --p 1
```

### High-Fidelity Refinement (Hybrid)
```powershell
python src/hotstart/hybrid_selector.py --trap trap_N15_K6 --p 2 --maxiter 100
```

## Advanced Configuration

### GPU Acceleration
The system automatically detects NVIDIA GPUs for simulation. To verify:
```powershell
python -c "from qiskit_aer import AerSimulator; print(AerSimulator().available_devices())"
```

### Hamiltonians & Sparsity
Both selectors build a **Sparse Ising Hamiltonian**.
- **Constraint**: Enforced via a penalty factor (lambda) balanced against diversity scores.
- **Warm-Start**: The initial state is biased towards the Greedy solution to ensure rapid convergence in the correct feasible region.
