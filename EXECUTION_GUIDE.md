# EXECUTION GUIDE: Quantum Molecular Selection Project

**Author**: Nicolas Mendes de Araújo  
**Timeline**: 10-12 weeks  
**Goal**: Publishable research paper + WOW college application component

---

## 📋 COMPLETE CHECKLIST

### Week 1-2: Setup & Data Collection

**Day 1-2: Environment Setup**
```bash
# Clone/setup project
mkdir quantum_biodiversity_project
cd quantum_biodiversity_project

# Run setup script
bash setup_environment.sh

# Verify installation
python verify_setup.py

# Get IBM Quantum token
# Go to: https://quantum.ibm.com/
# Account → API Token → Copy
# Paste into .env file
```

✅ **Checkpoint 1**:
- [ ] Python environment working
- [ ] All dependencies installed
- [ ] IBM Quantum account created
- [ ] Token configured in .env
- [ ] Can run verify_setup.py successfully

**Day 3-7: Data Collection**
```python
# Run data collection
python src/utils/data_collection.py

# This will:
# 1. Search PubChem for Amazonian species
# 2. Download NuBBE database
# 3. Filter drug-like molecules
# 4. Calculate descriptors
# 5. Save to data/raw/amazonian_molecules.csv
```

**Manual Steps**:
1. Visit NuBBE Database: https://nubbe.iq.unesp.br/
2. Register (free)
3. Download full database
4. Move to `data/raw/nubbe_database.csv`

✅ **Checkpoint 2**:
- [ ] Collected 300-500 molecules
- [ ] SMILES strings validated
- [ ] No duplicates (check InChI keys)
- [ ] Descriptors calculated
- [ ] Dataset saved in data/raw/

**Day 8-14: Data Processing**
```python
# Clean and prepare dataset
python src/utils/data_preprocessing.py

# This will:
# 1. Remove invalid molecules
# 2. Calculate fingerprints
# 3. Compute similarity matrix
# 4. Analyze diversity distribution
# 5. Save to data/processed/
```

✅ **Checkpoint 3**:
- [ ] Clean dataset: 200-400 valid molecules
- [ ] Fingerprints computed
- [ ] Similarity matrix saved
- [ ] Initial EDA plots generated
- [ ] Dataset documented

---

### Week 3-4: Classical Baseline

**Day 15-21: Implement Greedy Algorithm**
```python
# Run greedy selection
python src/classical/greedy_selector.py

# Test different values of k
for k in [10, 15, 20, 25, 30]:
    python src/classical/greedy_selector.py --k $k
```

**Expected Results**:
- Execution time: < 1 second for k=20
- Diversity score: baseline reference
- Solution quality: local optimum

✅ **Checkpoint 4**:
- [ ] Greedy algorithm working
- [ ] Tested multiple k values
- [ ] Results saved
- [ ] Visualizations created

**Day 22-28: Implement Genetic Algorithm**
```python
# Run genetic algorithm
python src/classical/genetic_selector.py

# Tune hyperparameters
python src/classical/genetic_selector.py --population 100 --generations 200
```

**Expected Results**:
- Better than greedy (5-15% improvement)
- Execution time: 10-60 seconds
- Reproducible with seed

✅ **Checkpoint 5**:
- [ ] GA implementation working
- [ ] Outperforms greedy
- [ ] Hyperparameters tuned
- [ ] Convergence plots generated
- [ ] Results documented

---

### Week 5-6: Quantum Algorithm (Simulator)

**Day 29-35: QAOA Implementation**
```python
# Test QAOA with simulator (FREE, unlimited)
python src/quantum/qaoa_selector.py --backend simulator --p 1

# This is where you spend most time debugging
# DO NOT touch real quantum hardware yet
```

**Critical Steps**:
1. **Verify QUBO formulation**:
   ```python
   python src/quantum/verify_qubo.py
   # Should output: "QUBO formulation correct ✓"
   ```

2. **Test circuit creation**:
   ```python
   python src/quantum/test_circuit.py
   # Check: circuit compiles, depth reasonable
   ```

3. **Run small test** (n=10 molecules, k=3):
   ```python
   python src/quantum/qaoa_selector.py --n 10 --k 3 --p 1
   # Should complete in < 5 minutes
   ```

4. **Scale up gradually**:
   ```python
   # n=20, k=5
   python src/quantum/qaoa_selector.py --n 20 --k 5 --p 1
   
   # n=50, k=10
   python src/quantum/qaoa_selector.py --n 50 --k 10 --p 1
   ```

✅ **Checkpoint 6**:
- [ ] QAOA circuit compiles
- [ ] Small tests pass
- [ ] Simulator results make sense
- [ ] Parameter optimization working
- [ ] No errors in logs

**Day 36-42: Optimize QAOA**
```python
# Try different p values
for p in [1, 2, 3]:
    python src/quantum/qaoa_selector.py --p $p --shots 2048

# Try different optimizers
for optimizer in ['COBYLA', 'NELDER-MEAD', 'POWELL']:
    python src/quantum/qaoa_selector.py --optimizer $optimizer
```

**Goal**: Find best hyperparameters before using real quantum hardware

✅ **Checkpoint 7**:
- [ ] Tested p=1,2,3 (p=1 usually best)
- [ ] Tested different optimizers
- [ ] Found optimal parameters
- [ ] Documented why these parameters work
- [ ] Simulation results saved

---

### Week 7-8: Real Quantum Hardware

**CRITICAL: You have ~10 minutes of quantum time. Don't waste it.**

**Before Running on Real Hardware**:
```bash
# Final verification checklist
python src/quantum/pre_flight_check.py

# This checks:
# ✓ Code has no bugs
# ✓ Circuit compiles
# ✓ Simulator works
# ✓ IBM account active
# ✓ Backend selected
```

**Day 43-49: Execute on IBM Quantum**

**Strategy**: Run multiple small experiments, not one big one

```python
# Experiment 1: Baseline (p=1, k=20)
python src/quantum/run_real_quantum.py \
    --k 20 \
    --p 1 \
    --shots 512 \
    --backend ibmq_manila
# Estimated time: ~2 minutes

# Experiment 2: Higher depth (p=2, k=20)
python src/quantum/run_real_quantum.py \
    --k 20 \
    --p 2 \
    --shots 512 \
    --backend ibmq_manila
# Estimated time: ~3 minutes

# Experiment 3: Different problem size (k=15)
python src/quantum/run_real_quantum.py \
    --k 15 \
    --p 1 \
    --shots 512 \
    --backend ibmq_manila
# Estimated time: ~2 minutes

# Total: ~7 minutes (leaves 3 minutes buffer)
```

**What to Expect**:
- Queue time: 5 minutes - 2 hours (depends on backend)
- Execution time: 1-3 minutes per job
- Results: noisier than simulator
- Success rate: 70-90% (some jobs may fail)

✅ **Checkpoint 8**:
- [ ] At least 3 successful quantum runs
- [ ] Results saved with metadata
- [ ] Error rates documented
- [ ] Comparison to simulator documented
- [ ] Quantum advantage demonstrated (or honestly reported lack thereof)

**Day 50-56: Analysis & Visualization**
```python
# Generate comparison plots
python src/analysis/compare_algorithms.py

# Creates:
# 1. Diversity vs. Time plot
# 2. Solution quality comparison
# 3. Convergence plots
# 4. Error analysis (quantum noise)
```

✅ **Checkpoint 9**:
- [ ] All algorithms compared fairly
- [ ] Figures publication-quality
- [ ] Statistics computed (p-values, effect sizes)
- [ ] Tables created
- [ ] Results folder organized

---

### Week 9-10: Paper Writing

**Structure** (IEEE Quantum Week format):
```
Title: Quantum-Accelerated Molecular Selection for Amazonian Biodiversity Conservation

Abstract (200 words)
1. Introduction (1.5 pages)
   - Biodiversity crisis in Amazon
   - Cost of traditional drug discovery
   - Quantum computing opportunity
2. Related Work (1 page)
   - QAOA applications
   - Molecular diversity optimization
   - Biodiversity informatics
3. Methods (2 pages)
   - Dataset preparation
   - QUBO formulation
   - Classical algorithms (baseline)
   - QAOA implementation
4. Results (2 pages)
   - Algorithm comparison
   - Quantum vs. classical
   - Noise analysis
   - Selected molecules
5. Discussion (1 page)
   - Quantum advantage (or lack thereof, be honest)
   - Limitations (noise, scalability)
   - Future work (error mitigation, larger systems)
6. Conclusion (0.5 pages)
References (20-30 papers)
```

**Day 57-63: Write Draft 1**
```latex
# Use template
cp papers/templates/ieee_quantum.tex papers/draft_v1.tex

# LaTeX structure
- main.tex (main document)
- sections/introduction.tex
- sections/methods.tex
- sections/results.tex
- sections/discussion.tex
- figures/ (all plots)
- bibliography.bib
```

✅ **Checkpoint 10**:
- [ ] All sections drafted
- [ ] Figures inserted
- [ ] Tables formatted
- [ ] Bibliography complete
- [ ] Citations correct

**Day 64-70: Revisions**
```bash
# Self-review checklist
python src/analysis/paper_checks.py

# Checks for:
# - Missing citations
# - Inconsistent notation
# - Unclear figures
# - Grammatical errors
```

**Get Feedback**:
1. LACQ Feynman colleagues
2. School science teacher
3. USP professor (if you can connect)

✅ **Checkpoint 11**:
- [ ] Draft reviewed by 2+ people
- [ ] Feedback incorporated
- [ ] Figures polished
- [ ] Writing clear and concise
- [ ] No obvious errors

---

### Week 11-12: Publication & Presentation

**Day 71-77: Finalize Paper**
```bash
# Generate final PDF
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Upload to arXiv
# Category: quant-ph (Quantum Physics) or cs.ET (Emerging Technologies)
```

**Submission Options** (in priority order):
1. **arXiv preprint** (IMMEDIATE - do this first)
   - Free, instant credibility
   - Citable DOI
   - Shows seriousness

2. **IEEE Quantum Week Student Paper Competition**
   - Deadline: ~May (check website)
   - Prestigious venue
   - Travel grant if accepted

3. **Brazilian conferences**:
   - LAWQC (Latin American Workshop on Quantum Computing)
   - SEMISH (Seminário Integrado de Software e Hardware)
   - ERCEMAPI (Meeting on Quantum Information)

4. **Journals** (higher barrier, longer timeline):
   - Quantum Information Processing
   - Scientific Reports
   - Brazilian Journal of Physics

✅ **Checkpoint 12**:
- [ ] Paper on arXiv
- [ ] Submitted to conference/journal
- [ ] Code on GitHub (public)
- [ ] Dataset released (if possible)

**Day 78-84: Create Presentation**
```python
# Generate presentation figures
python src/analysis/presentation_plots.py

# Creates simplified, high-impact visuals
```

**Presentation Structure** (15 minutes):
1. Hook (1 min): "The Amazon has 10M species, we've studied < 1%"
2. Problem (2 min): Cost of traditional screening
3. Solution (2 min): Quantum optimization overview
4. Methods (3 min): QAOA, implementation
5. Results (4 min): Comparison, quantum advantage
6. Impact (2 min): Real-world applications
7. Q&A (variable)

✅ **Checkpoint 13**:
- [ ] Slides created (PowerPoint/Beamer)
- [ ] Practiced presentation
- [ ] Timing correct
- [ ] Answers to common questions prepared

---

## 🎯 SUCCESS CRITERIA

**Minimum Viable Project** (publishable):
- ✓ 100+ molecules dataset
- ✓ Classical baseline implemented
- ✓ QAOA working on simulator
- ✓ At least 1 real quantum run
- ✓ Honest results (even if quantum doesn't win)
- ✓ Paper on arXiv

**Strong Project** (competitive):
- ✓ 300+ molecules dataset
- ✓ Multiple classical baselines
- ✓ 3+ real quantum experiments
- ✓ Thorough noise analysis
- ✓ Conference submission
- ✓ GitHub repo with documentation

**Exceptional Project** (tier 1 material):
- ✓ 500+ molecules (curated, documented)
- ✓ Novel QUBO formulation
- ✓ Comprehensive quantum vs. classical comparison
- ✓ Error mitigation attempted
- ✓ Published in conference proceedings
- ✓ Cited by others / Community impact

---

## ⚠️ COMMON PITFALLS TO AVOID

1. **Starting with real quantum too early**
   → Fix: Perfect your code on simulator first

2. **Overpromising results**
   → Fix: Be honest if quantum doesn't outperform classical

3. **Poor documentation**
   → Fix: Comment code, keep lab notebook, document decisions

4. **Trying to use all 10 minutes at once**
   → Fix: Multiple small experiments >> one big failure

5. **Ignoring noise**
   → Fix: Analyze error rates, be realistic about limitations

6. **Writing paper at the end**
   → Fix: Draft introduction/methods while coding

7. **Isolation**
   → Fix: Share progress with LACQ, get feedback early

---

## 📊 TIME BUDGET

**Total**: 10-12 weeks = 70-84 days

**Breakdown**:
- Setup & Data: 14 days (20%)
- Classical: 14 days (20%)
- Quantum (sim): 14 days (20%)
- Quantum (real): 14 days (20%)
- Paper: 14 days (20%)

**Recommended Schedule**:
- **Weekdays**: 2-3 hours/day (coding, experiments)
- **Weekends**: 4-6 hours/day (analysis, writing)
- **Total effort**: ~200 hours

**Quantum Hardware Usage**:
- Budget: 10 minutes total
- Strategy: 5-7 experiments × 1-2 minutes each
- Buffer: Keep 2-3 minutes for failures

---

## 🚀 GETTING STARTED NOW

**Today (Day 0)**:
```bash
# 1. Run setup
bash setup_environment.sh

# 2. Verify
python verify_setup.py

# 3. Get IBM token
# Visit: https://quantum.ibm.com/

# 4. Read documentation
cat DATA_SOURCES.md
```

**This Week (Days 1-7)**:
- [ ] Environment fully set up
- [ ] IBM Quantum account created
- [ ] Start data collection from PubChem
- [ ] Register at NuBBE database

**Next Week (Days 8-14)**:
- [ ] Complete dataset collection (target: 300+ molecules)
- [ ] Data cleaning and preprocessing
- [ ] Generate initial visualizations

---

## 📝 DOCUMENTATION CHECKLIST

Throughout the project, maintain:

- [ ] **Lab Notebook**: Daily log of experiments, decisions, failures
- [ ] **Code Comments**: Every function documented
- [ ] **README**: Clear instructions for reproducing results
- [ ] **Data Provenance**: Where each molecule came from
- [ ] **Results Log**: Every experiment recorded with parameters
- [ ] **Git Commits**: Meaningful commit messages

---

## 🎓 LEARNING RESOURCES

**Qiskit Tutorials**:
- https://qiskit.org/learn/
- Specifically: QAOA tutorial, VQE tutorial

**QAOA Papers**:
- Original QAOA paper: Farhi et al. (2014)
- Recent review: Cerezo et al. (2021) - "Variational Quantum Algorithms"

**Molecular Similarity**:
- RDKit documentation: https://www.rdkit.org/docs/
- Fingerprints tutorial

**Paper Writing**:
- "How to Write a Scientific Paper" - George Whitesides
- IEEE Quantum Week author guidelines

---

**Ready to start? Run the setup script and verify everything works!**

```bash
bash setup_environment.sh && python verify_setup.py
```

**Questions during execution? Document them in issues.md and we'll resolve together.**

Good luck, Nicolas. This project can genuinely be tier 1 material if executed well. 
The key is honesty, rigor, and persistence. You've got this.
