#  Aprendizado Auto-Supervisionado com SwAV em Séries Temporais

##  Descrição Geral
Este projeto explora o algoritmo **SwAV (Swapping Assignments Between Views)** aplicado ao conjunto de dados **PAMAP2**, que contém séries temporais associadas à classificação de atividades humanas. A abordagem busca avaliar o desempenho do aprendizado auto-supervisionado adaptado para dados temporais, utilizando transformações específicas e técnicas de aumento de dados.

---

##  Metodologia

### **1. O Algoritmo SwAV**
O SwAV é um algoritmo de aprendizado auto-supervisionado que utiliza:
- **Clusterização por Protótipos**: Criação de protótipos que representam clusters das representações aprendidas.
- **Predição Trocada**: O modelo prediz os clusters de uma visualização aumentada com base em outra visualização da mesma série temporal.

A função perda é baseada em entropia cruzada, promovendo consistência entre as representações.

---

### **2. Adaptação para Séries Temporais**
- **Transformações Aumentadas**:
  - Deslocamento aleatório no tempo.
  - Adição de ruído gaussiano.
  - Escalonamento da amplitude.
  - Permutação aleatória.
- **Arquitetura do Encoder**:
  - Camadas convolucionais 1D para extrair padrões temporais.
  - Estrutura MLP (Perceptron Multicamadas) para abstrações mais altas.

---

### **3. Configuração Experimental**
- **Hiperparâmetros**:
  - Camadas do encoder: [768, 128, 64, 32].
  - Número de protótipos: {16, 32, 64}.
  - Learning rate: {0.001, 0.0001}.
- **Treinamento**:
  - Utilizou *early stopping*.
  - Testes realizados com diferentes proporções de dados rotulados (10%, 20%, ... até 100%).

---

##  Resultados

### **Melhor Configuração**:
- Encoder: MLP com camadas [768, 128, 64, 32].
- Número de Protótipos: 64.
- Learning Rate: 0.0001.
- Estratégia: Freeze + Fine Tuning.

### **Métricas de Desempenho**:
| Métrica         | Valor    |
|-----------------|----------|
| **Acurácia Treino**   | ~80%     |
| **Acurácia Validação**| ~70%     |

### **Visualização das Representações**
As representações aprendidas foram projetadas em 2D com **t-SNE**, mostrando boa separação entre as classes.

---

##  Limitações e Pontos de Melhoria
- Explorar outras configurações de hiperparâmetros (número de camadas, regularização).
- Testar diferentes conjuntos de dados para melhorar a generalização.
- Implementar mais transformações específicas para séries temporais.

