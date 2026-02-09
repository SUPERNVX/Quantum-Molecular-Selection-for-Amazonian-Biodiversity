# Guia Rápido de Setup - Quantum Molecular Selection

## 🚀 Início Rápido (5 minutos)

### Passo 1: Executar Setup
```powershell
# No PowerShell, navegue até a pasta do projeto
cd C:\Users\super\Projetos\quantum

# Execute o script de setup
.\setup_environment.ps1

# Se houver erro de execução de scripts, execute primeiro:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Passo 2: Verificar Instalação
```powershell
# Ativar ambiente virtual
.\quantum_env\Scripts\Activate.ps1

# Verificar setup
python verify_setup.py
```

**Saída esperada**: Todos os ✓ verdes (exceto IBM Quantum, que é opcional por enquanto)

### Passo 3: Configurar IBM Quantum (Opcional)
```powershell
# 1. Criar conta em: https://quantum.ibm.com/
# 2. Copiar seu token da página Account → API Token

# 3. Criar arquivo .env (copiar do template)
Copy-Item .env.template .env

# 4. Editar .env e adicionar seu token:
notepad .env
# Substituir "your_token_here" pelo seu token real
```

---

## 📊 Passo 4: Coletar Dados

```powershell
# Executar coleta de dados do PubChem (demora ~10-15 minutos)
python src\utils\data_collection.py
```

**Resultado**: Arquivo `data/raw/amazonian_molecules.csv` com ~300-500 moléculas

### Opcional: Adicionar NuBBE Database
1. Acessar: https://nubbe.iq.unesp.br/portal/nubbedb.html
2. Registrar (grátis)
3. Download do database completo
4. Salvar como: `data\raw\nubbe_database.csv`

---

## 🧪 Passo 5: Pré-processar Dados

```powershell
# Calcular fingerprints e matriz de similaridade (~5 minutos)
python src\utils\data_preprocessing.py
```

**Resultado**: 
- `data/processed/amazonian_molecules.csv` (limpo)
- `data/processed/fingerprints.pkl`
- `data/processed/similarity_matrix.npy`

---

## 🎯 Passo 6: Executar Algoritmos

### Baseline Clássico (Greedy + Genetic)
```powershell
python src\classical\classical_molecular_selection.py
```

**Tempo**: ~1-2 minutos  
**Resultado**: `data/results/classical_baseline.csv`

### QAOA Quântico (Simulador)
```powershell
python src\quantum\quantum_molecular_selection.py
# Quando perguntado sobre hardware real, responder: n
```

**Tempo**: ~5-10 minutos (simulador local)  
**Resultado**: `data/results/quantum_results.csv`

### QAOA em Hardware Real (Opcional - USE COM CUIDADO!)
```powershell
python src\quantum\quantum_molecular_selection.py
# Quando perguntado sobre hardware real, responder: y
```

> ⚠️ **ATENÇÃO**: Você tem apenas ~10 minutos de tempo quântico gratuito!

---

## 📁 Estrutura Final

```
quantum/
├── data/
│   ├── raw/
│   │   └── amazonian_molecules.csv      ← Dados brutos (~500 moléculas)
│   ├── processed/
│   │   ├── amazonian_molecules.csv      ← Limpo e validado
│   │   ├── fingerprints.pkl             ← Morgan fingerprints
│   │   └── similarity_matrix.npy        ← Matriz Tanimoto
│   └── results/
│       ├── classical_baseline.csv       ← Resultados Greedy + GA
│       └── quantum_results.csv          ← Resultados QAOA
├── src/
│   ├── utils/
│   │   ├── data_collection.py          ← Extração PubChem
│   │   └── data_preprocessing.py        ← Fingerprints + Similaridade
│   ├── classical/
│   │   └── classical_molecular_selection.py
│   └── quantum/
│       └── quantum_molecular_selection.py
├── quantum_env/                         ← Ambiente Python virtual
├── .env                                 ← Suas credenciais IBM
└── README.md                            ← Documentação completa
```

---

## 🔧 Solução de Problemas

### Erro: "RDKit não encontrado"
```powershell
# Reinstalar RDKit
pip uninstall rdkit
pip install rdkit==2023.9.4
```

### Erro: "IBM Quantum token inválido"
```powershell
# Verificar arquivo .env
notepad .env

# Certificar-se de que o token está correto (sem espaços)
# Token deve começar com caracteres alfanuméricos
```

### Erro: "Memória insuficiente" (matriz de similaridade)
- **Solução**: Reduzir número de moléculas
- Editar `data/raw/amazonian_molecules.csv` e manter apenas primeiras 200 linhas

### Erro PowerShell: "Execução de scripts desabilitada"
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📈 Próximos Passos

### Curto Prazo (Esta Semana)
- [ ] Completar setup do ambiente
- [ ] Coletar dataset (300+ moléculas)
- [ ] Executar baseline clássico
- [ ] Testar QAOA no simulador

### Médio Prazo (Próximas 2-3 Semanas)
- [ ] Executar QAOA em hardware real (1-2 experimentos)
- [ ] Análise comparativa (Greedy vs GA vs QAOA)
- [ ] Criar visualizações (matplotlib/seaborn)

### Longo Prazo (1-2 Meses)
- [ ] Escrever paper (draft)
- [ ] Submeter para arXiv
- [ ] Submeter para conferência (IEEE Quantum Week, LAWQC)

---

## 📚 Recursos Adicionais

### Documentação do Projeto
- `README.md` - Visão geral completa
- `DATA_SOURCES.md` - Guia de fontes de dados
- `EXECUTION_GUIDE.md` - Plano de 10-12 semanas

### Links Úteis
- **IBM Quantum**: https://quantum.ibm.com/
- **Qiskit Docs**: https://qiskit.org/documentation/
- **RDKit Docs**: https://www.rdkit.org/docs/
- **PubChem API**: https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest

### Comunidade
- **LACQ Feynman**: Grupo de computação quântica
- **GitHub Issues**: Para reportar problemas
- **Stack Overflow**: Tag `qiskit` para dúvidas

---

## ✅ Checklist de Validação

- [ ] Ambiente Python ativado (`quantum_env`)
- [ ] Todas dependências instaladas (`verify_setup.py` ✓)
- [ ] Dataset coletado (≥100 moléculas)
- [ ] Dados pré-processados (fingerprints + matriz)
- [ ] Algoritmo clássico executado com sucesso
- [ ] QAOA funciona no simulador
- [ ] (Opcional) IBM Quantum configurado

---

**Tempo total estimado para setup completo**: 30-45 minutos

**Dúvidas?** Consulte `README.md` ou `EXECUTION_GUIDE.md`
