# Guia Rápido de Setup - Quantum Molecular Selection

## 🚀 Início Rápido (2 minutos)

### Passo 1: Recuperar o Ambiente
Se você estiver no **Windows Nativo** (CPU):
```powershell
.venv12\Scripts\activate
```

Se você estiver no **WSL2 (Linux/GPU)**:
```bash
source ~/venv_linux/bin/activate
```

### Passo 2: Rodar a Demonstração de Vitória (N=25)
No WSL2 para usar a **GPU (RTX 4060)** e ganhar a aceleração de 7.3x:
```bash
wsl bash -c "source ~/venv_linux/bin/activate && cd /mnt/c/Users/super/Projetos/Quantum-Molecular-Selection-for-Amazonian-Biodiversity && python3 demo_refinement.py --trap trap_N25_K8"
```

---

## 🏎️ Guia de Performance GPU (RTX 4060)

*   **Por que o uso da GPU é baixo (~10%)?**
    O problema de 25 qubits ocupa apenas ~512MB de VRAM. A GPU é tão rápida que ela passa mais tempo esperando o processador (CPU) enviar os parâmetros do que calculando. O ganho real está na **latência por iteração** (1.7s vs 12s).
*   **Quando a GPU brilha?**
    Acima de 26 qubits (`N=26`) ou com circuitos profundos (`p > 2`).

---

## 🧪 Estratégia Científica: Caça às Traps

### 1. Caça Clássica (CPU)
A busca por novas "Armadilhas" (`refine_heavyweight_trap.py`) é **puramente clássica** e usa multiprocessing na CPU.
```powershell
# No Windows Nativo, tente achar traps mais difíceis
python refine_heavyweight_trap.py --n 25 --k 8 --trials 5000
```

### 2. Ataque Quântico (GPU)
Use a GPU para tentar fechar o gap em traps onde o Greedy falhou.
*   **Prioridade:** Traps onde o GA Goal é muito maior que o Greedy.
*   **Alvos Recomendados:** `trap_N25_K8` (Gap atual de ~1.6%).
*   **Comando:** `python demo_refinement.py --trap trap_N25_K8 --p 2` (dentro do WSL).

---

## 📁 Estrutura de Pastas Úteis

- **`data/traps/`**: Cenários de "Armadilha" científicos.
- **`src/hotstart/`**: Código-fonte dos seletores modernos.
- **`SCIENTIFIC_CHANGELOG.md`**: Registro histórico de vitórias e setups.

---

**Dúvidas?** Consulte o `README.md` principal ou o arquivo `migration_guide.md` nos artefatos.
