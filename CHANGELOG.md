# Changelog

Todas as mudanças notáveis neste projeto estão documentadas abaixo, organizadas por etapas de desenvolvimento.

---

## 🏆 Etapa 5: Consolidação e Cleanup (V0.3.0) - 2026-02-13
**Foco:** Refatoração, remoção de legados e documentação final de alta fidelidade.

### Removido (Cleanup)
- **Diretórios Legados:** Remoção completa de `src/classical` e `src/quantum` (supersedidos pela pasta `hotstart`).
- **Scripts de Diagnóstico:** Deletados scripts temporários como `reproduce_lite_freeze.py`, `debug_sim.py`, `test_fp.py`, etc.
- **Documentação Obsoleta:** Remoção de `DATA_SOURCES.md` e `EXECUTION_GUIDE.md` (informações consolidadas no README).

### Modificado (Refatoração)
- **Documentação Central:** Atualização massiva do `README.md`, `PROJECT_STATUS.md` e `QUICK_START_PT.md` para refletir a nova arquitetura.
- **Hotstart README:** Documentação específica para as ferramentas de produção `lite_selector.py` e `hybrid_selector.py`.

---

## 🚀 Etapa 4: Escalabilidade e Vitória Quântica (N=25) - 2026-02-12
**Foco:** Superação da barreira dos 25 qubits e demonstração de vantagem quântica escalada.

### Adicionado
- **Vitória N=25, K=8:** QAOA Hybrid superou o Greedy e o Algoritmo Genético em um espaço de busca de $2^{25}$ estados.
- **Refinamento de Alta Fidelidade:** Implementação de $p=2$ camadas com 100 iterações de otimização COBYLA.
- **Demo de Refinamento:** Criação do script `demo_refinement.py` para demonstração rápida das vitórias científicas.

### Modificado
- **Otimização de Ansatz:** Introdução de `ParameterVector` para evitar a reconstrução do circuito a cada iteração, reduzindo o overhead.
- **Estabilidade de Simulação:** Integração total com `AerSimulator` e sistema de fallback automático GPU -> CPU.

---

## 🔬 Etapa 3: Pivot Algorítmico e Arquitetura Hybrid - 2026-02-11
**Foco:** Transição para o paradigma Hybrid (Warm-Start) e correção do "Hamiltoniano Cego".

### Adicionado
- **Hybrid Selector:** Integração oficial entre o Warm-Start (Greedy) e o refinamento quântico.
- **Sparse Hamiltonians:** Substituição de matrizes densas por `SparsePauliOp` para evitar erros de OOM em sistemas grandes.
- **SCIENTIFIC_CHANGELOG.md:** Criação do diário de bordo científico para registro de hipóteses e provas matemáticas.

### Corrigido
- **Alinhamento Ising-QUBO:** Correção na leitura das Pauli Strings para garantir que a energia quântica seja 100% equivalente à diversidade estrutural.
- **Little-Endian Logic:** Sincronização da ordem dos bits entre seletores clássicos e quânticos.

---

## 📊 Etapa 2: Coleta e Baselines (V0.2.0) - 2026-02-09
**Foco:** Validação do dataset Amazônico e estabelecimento das metas clássicas.

### Adicionado
- **Sistema de Coleta Robusta (v5):** Integração com NCBI Entrez para busca taxonômica.
- **Dataset Refinado:** Criação do subconjunto de 810 moléculas com propriedades fármaco-tópicas (Lipinski-like).
- **Find Greedy Traps:** Desenvolvimento de scripts para localizar instâncias onde a heurística guloza falha.

---

## 🏗️ Etapa 1: Ambiente e Setup (V0.1.0) - 2026-02-08
**Foco:** Construção da fundação técnica e infraestrutura.

### Adicionado
- **Setup Automatizado:** Criação de scripts para Windows PowerShell e Linux.
- **Filtros RDKit:** Implementação inicial de fingerprints Morgan Radius 2.
- **Infraestrutura:** Configuração do ambiente virtual e verificação de dependências.

---

**Mantenedor:** Nicolas Mendes de Araújo (@supernvx)
**Última Atualização:** 13 de Fevereiro de 2026
