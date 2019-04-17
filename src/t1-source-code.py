# Trabalho 1

# O problema será identificar se o sinal de sonar obtido (60 valores reais,
# correspondentes a energia em diferentes bandas de frequência e ângulos de retorno) representa
# uma rocha (“R”) ou uma mina (“M”)

# Bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Neuron import Neuron
from sklearn.metrics import confusion_matrix

# Import dos datasets
train_df = pd.read_csv('dados/sonar.train-data', header=None)
test_df  = pd.read_csv('dados/sonar.test-data', header=None)

# Dimensões dos datasets importados
#train_df.head()
train_df.shape

#test_df.head()
test_df.shape

# Substituir as Strings no dataset por valores numéricos
cleanup_nums = {60: {"M": 1.0, "R": 0.0}}

train_df.replace(cleanup_nums, inplace=True)
test_df.replace(cleanup_nums, inplace=True)

print (train_df.head())
print (test_df.head())

# Describing the database
#train_df[60].value_counts()
print(train_df.iloc[:,:-1].describe())
print(test_df.iloc[:,:-1].describe())

# 145 samples from train/validation dataframe(~70%) 
# + 63 samples from test dataframe (~30%) =  208 samples
X_train = train_df.iloc[:,:-1].values   # Independent variables
y_train = train_df.iloc[:, -1].values   # Dependent variables (classes)

X_test = test_df.iloc[:,:-1].values     # Independent variables
y_test = test_df.iloc[:, -1].values     # Dependent variables (classes)
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)
print(type(X_train))
print(type(y_train))
print(X_train)
print(y_train)

# Single neuron Perceptron

lRate   = 0.04
nEpoch  = 30000
n_inputs = len(X_train[0])
perceptron = Neuron(n_inputs, lRate)
weights = perceptron.learn(X_train, y_train, nEpoch)
print ('Weights:\n{}'.format(weights))
y_pred = []
for row in X_test:
    y_pred.append(perceptron.predict(row))

cm      = confusion_matrix(y_test, y_pred) 
print('Confusion matrix on test database:\n{}'.format(cm))

####################################### Visualização da Matriz de Confusão ######################################################

import itertools
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Matriz de Confusão',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão Normalizada")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe Desejada')
    plt.xlabel('Classe Obtida')

# Classification Report and Confusion Matrix
from sklearn import metrics
classes = ['Rocks', 'Mines']
plot_confusion_matrix(cm, classes, normalize=True)
print (metrics.classification_report(y_test, y_pred))