## RELATÓRIO TÉCNICO FINAL

# Projeto de Previsão de Viscosidade em Emulsões Cosméticas

## 💡 Nota Metodológica
**Importante**: Este projeto utiliza um **dataset sintético** gerado por inteligência artificial (IA) para simular dados de produção de emulsões cosméticas. O objetivo principal é demonstrar a aplicação prática de algoritmos de machine learning em um contexto industrial, servindo como portfólio técnico e estudo de caso educacional.

## 1. Resumo
Este projeto desenvolveu um sistema preditivo para estimar a viscosidade final de uma emulsão cosmética durante o processo de produção. Foram implementados e comparados cinco algoritmos de machine learning, sendo a Regressão Polinomial selecionada como modelo final devido ao seu equilíbrio ideal entre performance preditiva e eficiência computacional.

## 2. Introdução

### 2.1 Contexto Industrial
A viscosidade representa uma das propriedades mais críticas em formulações de emulsões cosméticas, exercendo influência direta em múltiplos aspectos do produto final:

- **Estabilidade físico-química**: Valores inadequados de viscosidade podem acelerar fenômenos de separação de fases como creaming e sedimentação (TADROS, 2013)
- **Características sensoriais**: A viscosidade determina parâmetros como espalhabilidade, absorção cutânea e sensação tátil, fatores cruciais para aceitação do consumidor (SCHRAMM, 2005)
- **Processabilidade industrial**: Durante a manufatura, a viscosidade afeta diretamente a eficiência de homogeneização, bombeamento e enchimento (MYERS, 2005)

### 2.2 Problema Negocial
Variações nos parâmetros de processo (temperatura, agitação, concentração de emulsificantes) frequentemente resultam em inconsistências na viscosidade final, acarretando:

- Retrabalho de lotes fora de especificação
- Desperdício de matéria-prima e insumos
- Inconsistência na qualidade do produto final
- Impacto negativo na eficiência produtiva

A capacidade de prever a viscosidade final com base em variáveis de processo monitoráveis permitiria intervenções proativas, otimizando o controle de qualidade e reduzindo perdas operacionais.

## 3. Metodologia

### 3.1 Coleta e Preparação de Dados
- **Fonte**: Dataset sintético gerado por IA para replicar padrões reais de produção
- **Variáveis simuladas**: Parâmetros típicos de processo industrial temperatura (°C), velocidade de agitação(RPM), concentração(%), tempo de reação(min), pH, umidade(%) e viscosidade(cP)
- **Objetivo pedagógico**: Desenvolver e validar metodologias de machine learning aplicáveis a contextos industriais 

### 3.2 Pré-processamento
- Codificação de variáveis categóricas
- Normalização de features numéricas
- Divisão estratificada: 80% treino, 20% teste

### 3.3 Modelos Implementados

**Regressão Polinomial**
- **Complexidade**: Grau 2

**Redes Neurais Artificiais**
- **Parâmetros**: activation='relu', alpha=0.01, hidden_layer_sizes= (50,), learning_rate_init=0.001, solver='lbfgs', max_iter=2000

**XGBoost**
- **Parâmetros**: n_estimators=300, max_depth=3, learning_rate=0.05

**Ensemble**
- **Composição**: Regressão Polinomial, Redes Neurais e XGBoost
- **Método**: Média ponderada das previsões

**Random Forest**
- **Parâmetros**: n_estimators=500, max_depth=20, max_features=0.5,min_samples_leaf=1, min_samples_split=2

**SVM**
- **Parâmetros**: Valores default. Não foram feitas análises mais profundas, pois na primeira análise o algoritmo obteve um valor muito baixo de R² (0.07) e foi descartado de imediato

## 4. Resultados e Análise

### 4.1 Performance Comparativa

| Modelo | R² Score |    MAE   | Ranking Performance |
|--------|----------|----------|---------------------|
| Polynomial Regression | 0.903 | 242.61 | 🥇 **1º Lugar** |
| Ensemble | 0.899 | 246.98 | 🥈 2º Lugar |
| Neural Network | 0.891 | 253.26 | 🥉 3º Lugar |
| XGBoost | 0.885 | 263.96 | 4º Lugar |
| Random Forest | 0.868 | 273.80 | - |
| SVM | 0.07 | 759.49 | - |

### 4.2 Análise Detalhada

#### 4.2.1 Regressão Polinomial (Modelo Escolhido)
- **R² = 0.903**: Explica 90.3% da variância na viscosidade
- **MAE = 242.61**: Erro médio aceitável para o conjunto de dados
- **Velocidade de Processamento**: Tempo de execução foi de 0.08s/run

#### 4.2.2 Performance vs. Eficiência
- **Ensemble**: Performance ligeiramente inferior (R² = 0.899), com custo computacional significativamente maior, devido ser uma composição dos 3 modelos finais. 
- **Neural Network**: Performance similar ao ensemble (R² = 0.891) com o maior tempo de processamento dentre os modelos final (27.66s/run). Uma complexidade desnecessária.
- **XGBoost**: Performance inferior dentre os 3 modelos finais (R² = 0.885), nesta aplicação específica, porém segundo melhor tempo de processamento (1.17s/run)
- **Random Forest**: Modelo descartado após o processo de validação cruzada devido ter o pior desempenho. Apresentou o menor valor de R² (R² = 0.868) e o maior valor de MAE (MAE = 273.80), além do maior tempo de execução (38.41s/run).

#### 4.2.3 Análise Dimensional
Uma análise de componentes principais (PCA) foi conduzida para avaliar o impacto da redução dimensional na performance dos modelos. Os resultados demonstraram que a técnica não apresentou benefícios significativos para esse conjunto de dados específico:

- **Regressão Polinomial**: O PCA resultou em aumento do tempo de processamento sem melhorias perceptíveis nas métricas de accuracy (R²) ou erro (MAE)
- **Redes Neurais**: Observou-se discreta melhoria no R² e redução marginal no MAE, contudo, estas melhorias foram consideradas insignificantes face ao acréscimo computacional incorrido
- **XGBoost**: Verificou-se redução no tempo de execução, porém acompanhada de deterioração na performance preditiva, com decréscimo do R² e aumento do MAE

## 5. Conclusão
- Os 3 modelos finais demonstraram capacidade preditiva relevante (R² > 0.85)
- A relação não-linear entre variáveis de processo e viscosidade foi adequadamente capturada
- A análise dimensional foi descartada por não agregar valor preditivo substancial, mantendo-se assim as features originais para todos os modelos finais.
- A simplicidade da Regressão Polinomial mostrou-se vantajosa frente a modelos mais complexos

**Decisão de Modelo Final**
A Regressão Polinomial foi selecionada como modelo definitivo devido a:
- ✅ Performance líder (R² = 0.903)
- ✅ Eficiência computacional superior (1.17 s/run)
- ✅ Facilidade de implementação e manutenção
- ✅ Melhor custo-benefício operacional


## 6. Referências Técnicas
- 1. **TADROS, T. F.** *Emulsion Science and Technology: A General Introduction*. Wiley-VCH, 2013.
2. **SCHRAMM, L. L.** *Emulsions, Foams, and Suspensions: Fundamentals and Applications*. Wiley-VCH, 2005.
3. **MYERS, D.** *Surfactant Science and Technology*. 3rd ed. Wiley, 2005.
4. **HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J.** *The Elements of Statistical Learning*. 2nd ed. Springer, 2009.
5. **GÉRON, A.** *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. 2nd ed. O'Reilly, 2019.
6. Documentação técnica: Scikit-Learn, XGBoost, 2023.

---

**Data do Relatório**: 28/10/2025  
**Elaborado por**: Silas Nascimento 
**Área**: Engenheiria Química / Inteligência Artificial