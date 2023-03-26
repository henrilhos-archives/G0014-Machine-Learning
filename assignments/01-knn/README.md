# Machine Learning

Prof. Claudinei Dias (Ney)

## Atividade 01 - *k*-Nearest Neighbors (*k*-NN)

Classificar as instâncias do dataset Iris usando a abordagem de aprendizado supervisionado.

[Repositório de dataset para Machine Learning](http://archive.ics.uci.edu/ml/index.php)

O Dataset contém as estimativas obtidas a partir do conjunto de dados da íris com 150 instâncias (50 para cada classe), o qual contem três classes: *Iris setosa,*, *Iris versicolor* e *Iris virginica*. Cada classe contém os quatro atributos a seguir, em centímetros: *sepal length*, *sepal width*, *petal length* e *petal width*. A figura abaixo exibe as fotos de amostras das três classes da íris, observa-se o quão sutil são as diferenças em relação aos atributos de comprimento e largura da sépala e da pétala de cada uma das flores. Apresente a métrica de avaliação acurácia.

<p align="center" style="text-align: center">
  <b>Figuras (da esquerda para direita):</b> <i>Iris setosa</i>,
  <i>Iris versicolor</i> e <i>Iris virginica</i>
</p>

<p align="center" style="text-align: center;">
    <img height="150px" alt="Iris setosa" src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/900px-Kosaciec_szczecinkowaty_Iris_setosa.jpg" />
    <img height="150px" alt="Iris versicolor" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/1600px-Iris_versicolor_3.jpg" />
    <img height="150px" alt="Iris virginica" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/1472px-Iris_virginica.jpg" />
</p>

<p align="center" style="text-align: center">
  <b>Dataset:</b>
  <a href="http://archive.ics.uci.edu/ml/index.php"
    >UC Irvine Machine Learning Repository</a
  >
</p>

### Opções para desenvolvimento em Python

#### Online

- Replit - https://replit.com/languages/python
- Google Colab - https://colab.research.google.com/

#### Stand-alone

- Python - https://www.python.org/downloads/
- Pycharm - https://www.jetbrains.com/pt-br/products/#type=ide
- Anaconda - https://www.anaconda.com/

#### Bibliotecas

- scikit-learn - https://scikit-learn.org/
- pandas - https://pandas.pydata.org/
- Keras - https://keras.io/

### Exemplos

- [Versão em Jupyter Notebook](https://colab.research.google.com/drive/1KyxUp5zVq38JLiwSXvGCcY2KXECOmz0v?usp=sharing)
- Versão em Python:

```python
import csv
import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt

def load_training_dataset(filename, training_set=[], test_set=[]):
    with open(filename, 'r') as csv_file:
        lines = csv.reader(csv_file)
        dataset = list(lines)
        for x in range(0, len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
                if random.random() < 0.66:
                    training_set.append(dataset[x])
                else:
                    test_set.append(dataset[x])

    print('Train set: ' + repr(len(training_set)))
    print('Test set: ' + repr(len(test_set)))

def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def get_response(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
        sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]

def main():
    # iteration for learning
    for i in range(1):
        training_set = []
        test_set = []
        load_training_dataset('iris.data', training_set, test_set)
        predictions = []
        for x in range(len(test_set)):
            neighbors = get_neighbors(training_set, test_set[x], 9)# k = 9
            result = get_response(neighbors)
            predictions.append(result)
            print('> actual=' + repr(test_set[x][-1]) + ', actual=' + repr(result))

main()
```

### Solução complementar usando `scikit-learn`

- Usar a biblioteca `scikit-learn` de aprendizado de máquina de código aberto para a linguagem de programação Python
- Ver modelo KNN em [scikit-learn](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html)
- Inserir código para computar a matriz de confusão, ver protótipo abaixo:
    - `sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)`
- Imprimir métricas de avaliação: acurácia, precisão, revocação e F1-score.

### Solução complementar com Notebook na plataforma do Kaggel

- Analisar, estudar e comparar os Modelos de Classificadores KNN, SVM, Logistic Regression, DecisionTree
- [ML from Scratch with IRIS](https://www.kaggle.com/ash316/ml-from-scratch-with-iris)
