# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Não Lançado]

### Planejado
- Análise comparativa (Greedy vs GA vs QAOA)
- Visualizações de diversidade molecular
- Paper draft (LaTeX)
- Submissão arXiv

---

## [0.1.0] - 2026-02-09

### Adicionado
- Setup inicial do projeto
- Ambiente virtual Python (`quantum_env`)
- Estrutura de diretórios completa
- Scripts de setup automatizado:
  - `setup_environment.ps1` - Setup Windows PowerShell
  - `QUICK_START_PT.md` - Guia rápido em português
- Pipeline de extração de dados:
  - `src/utils/data_collection.py` - Coleta PubChem API
  - `src/utils/data_preprocessing.py` - Fingerprints e similaridade
- Scripts de verificação:
  - `verify_setup.py` - Validação de dependências
- Algoritmos implementados:
  - `src/classical/classical_molecular_selection.py` - Greedy + Genetic
  - `src/quantum/quantum_molecular_selection.py` - QAOA
- Documentação:
  - `README.md` - Visão geral do projeto
  - `DATA_SOURCES.md` - Guia de fontes de dados
  - `EXECUTION_GUIDE.md` - Plano de 10-12 semanas
  - `implementation_plan.md` - Plano técnico detalhado
  - `walkthrough.md` - Guia completo de setup

### Dependências
- Qiskit 1.0.0 (computação quântica)
- Qiskit Aer 0.13.3 (simulador)
- Qiskit IBM Runtime 0.18.0 (acesso a hardware real)
- RDKit 2023.9.4 (química computacional)
- NumPy 1.26.3
- Pandas 2.1.4
- SciPy 1.11.4
- Scikit-learn 1.4.0
- Matplotlib 3.8.2
- Seaborn 0.13.0
- Plotly 5.18.0
- NetworkX 3.2.1
- TQDM 4.66.1
- Python-dotenv 1.0.0
- Requests 2.31.0
- Jupyter 1.0.0 (opcional)

### Configuração
- Python 3.9+ requerido (testado com Python 3.14.0)
- Ambiente virtual isolado
- IBM Quantum Account configurável (opcional para início)

---

## Formato de Versionamento

### [MAJOR.MINOR.PATCH]

**MAJOR**: Mudanças incompatíveis na API  
**MINOR**: Novas funcionalidades compatíveis  
**PATCH**: Correções de bugs compatíveis  

### Categorias de Mudanças

- **Adicionado**: Novas funcionalidades
- **Modificado**: Mudanças em funcionalidades existentes
- **Descontinuado**: Funcionalidades que serão removidas
- **Removido**: Funcionalidades removidas
- **Corrigido**: Correções de bugs
- **Segurança**: Vulnerabilidades corrigidas

---

## Links

- [Repositório GitHub](https://github.com/supernvx/quantum-biodiversity)
- [Issues](https://github.com/supernvx/quantum-biodiversity/issues)
- [Documentação](README.md)
- [LACQ Feynman](https://lacq.com.br/)
- [IBM Quantum](https://quantum.ibm.com/)

---

**Data da última atualização**: 2026-02-09  
**Mantenedor**: Nicolas Mendes de Araújo (@supernvx)
