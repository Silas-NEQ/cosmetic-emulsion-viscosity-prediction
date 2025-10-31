# Previsão da viscosidade de uma emulsão para produção cosmética

![Python](https://img.shields.io/badge/Python-3.11.4-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-orange)
![Status](https://img.shields.io/badge/Status-Concluído-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

🧑🏻‍🔬 Projeto desenvolvido por Silas Nascimento
LinkedIn: [www.linkedin.com/in/engquim-silas-nascimento](www.linkedin.com/in/engquim-silas-nascimento)

🏭 Objetivo: Desenvolver um modelo preditivo capaz de estimar a viscosidade final de
emulsões cosméticas, antecipando desvios de qualidade e apoiando a tomada de decisão
sobre ajustes operacionais durante a produção 

## 📁 Estrutura do Projeto

```
cosmetic_emulsion_viscosity_predict/
│
├── data/
│   ├── raw/                    # Dados brutos originais
│   ├── processed/              # Dados processados
│   └── results/                # Resultados e métricas
│
├── models/                     # Modelos treinados
├── src/                        # Código fonte
│   ├── 01_EDA.py              # Análise exploratória
│   ├── 02_preprocessing.py    # Pré-processamento
│   ├── 03_modeling.py         # Treinamento de modelos
│   ├── 04_cross_validation.py # Validação cruzada
│   ├── 05_dimensionality_analyse.py # Análise dimensional
│   ├── 06_final_models.py     # Treinamento dos modelos finais
│   ├── 07_conclusion.py       # Testes finais
│   └── util.py                # Funções auxiliares
│
├── README.md                  # Apresentação do projeto (Versão em inglês)
├── README(PT-BR).md           # Este arquivo
├── REPORT.md                  # Relatório técnico (Versão em inglês)
├── REPORT(PT-BR).md           # Relatório técnico
└── requirements.txt           # Dependências do projeto
```

## 🚀 Como Executar

### Pré-requisitos
- Python 3.11.4 
- Gerenciador de pacotes pip
- Git (para clonar o repositório)

### Instalação

- **Clone o repositório**
git clone https://github.com/Silas-NEQ/cosmetic-emulsion-viscosity-prediction.git

- **Entre na pasta do projeto**
cd cosmetic-emulsion-viscosity-prediction

- **Instale as dependências**
pip install -r requirements.txt

- **Execute os scripts na ordem da pasta**
python src/01_EDA.py

## Metodologia
🔍 Análise Exploratória (EDA)
Análise de distribuição das variáveis
Identificação de correlações
Detecção de outliers e valores missing

⚙️ Pré-processamento
Codificação de variáveis categóricas (One-Hot Encoding, Label Encoding)
Normalização e padronização de features
Divisão treino/teste (80/20)

🤖 Modelos Implementados
Regressão Polinomial - Modelo escolhido
Redes Neurais Artificiais Regressor
XGBoost Regressor
Random Forest Regressor
SVM Regressor
Ensemble Methods (Combinação dos 3 primeiros modelos dessa lista)

📊 Métricas de Avaliação
R² Score: Variância explicada pelo modelo
MAE (Mean Absolute Error): Erro médio


## Resultados
O modelo de Regressão Polinomial foi selecionado como definitivo devido ao:
🎯 Melhor performance (R² = 0.903)
⚡ Alta eficiência computacional (0.08s/run)
🔧 Simplicidade de implementação e manutenção
📈 Melhor custo-benefício operacional

## Aplicação Prática
O modelo permite:
✅ Antecipar desvios de qualidade na produção
✅ Reduzir retrabalho e desperdícios
✅ Otimizar parâmetros de processo
✅ Melhorar consistência do produto final


💡 Nota: Este projeto utiliza um dataset sintético gerado por inteligência artificial para demonstrar a aplicação prática de machine learning na indústria cosmética. O foco está na metodologia e na aplicabilidade das técnicas em contextos industriais reais.