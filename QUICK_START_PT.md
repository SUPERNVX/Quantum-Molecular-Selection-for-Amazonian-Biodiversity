# Guia Rápido de Setup - Quantum Molecular Selection

## 🚀 Início Rápido (2 minutos)

### Passo 1: Recuperar o Ambiente
Se você estiver em uma nova sessão, certifique-se de que o ambiente virtual está ativado:

```powershell
# Ativar ambiente virtual oficial
.venv\Scripts\Activate.ps1
```

### Passo 2: Rodar a Demonstração de Vitória (N=25)
Para ver o QAOA superando os benchmarks clássicos em tempo real:

```powershell
python demo_refinement.py
```

---

## 🧪 Executando Experimentos Customizados

### 1. Seletor Lite (Simulação Ultra-Rápida)
Ideal para testes rápidos em seu computador local (até 25-28 qubits).
```powershell
python src/hotstart/lite_selector.py --trap trap_N25_K8 --p 1
```

### 2. Seletor Hybrid (Refinamento de Alta Fidelidade)
O seletor oficial que bateu o Algoritmo Genético.
```powershell
python src/hotstart/hybrid_selector.py --trap trap_N25_K8 --p 2 --maxiter 100
```

---

## 📁 Estrutura de Pastas Úteis

- **`data/traps/`**: Contém os cenários de "Armadilha" onde o Greedy falha.
- **`src/hotstart/`**: Código-fonte dos seletores modernos.
- **`SCIENTIFIC_CHANGELOG.md`**: Detalhes técnicos de cada vitória e benchmark.

## 🔧 Solução de Problemas

### Erro: "ModuleNotFoundError"
Isso geralmente significa que o `.venv` não foi ativado ou está corrompido.
**Solução**: Rode `.\setup_environment.ps1` e depois ative o ambiente.

### Erro: "Out of Memory" (N > 28)
Simulações locais acima de 28 qubits exigem muita RAM.
**Solução**: Para escalas maiores, utilize o seletor `hybrid` configurado para hardware real da IBM ou use o `sparsity_threshold` se disponível.

---

**Dúvidas?** Consulte o `README.md` principal ou o `SCIENTIFIC_CHANGELOG.md`.
