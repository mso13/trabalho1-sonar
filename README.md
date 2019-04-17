## Trabalho 1 - Tópicos em Engenharia (IA)
**Primeiro trabalho de implementação computacional 1/2019** 

#### ***Rede "Perceptron" e base de dados "Sonar"***

1. ##### Introdução

   O estudante deverá demonstrar conhecer a Rede Neural Artificial conhecida como
   “*Perceptron*”, aplicando-a na solução de um problema de reconhecimento de padrões.
   Usaremos a base de dados "Sonar", disponível em: [https://archive.ics.uci.edu/ml/machine-learningdatabases/undocumented/connectionist-bench/sonar/](https://archive.ics.uci.edu/ml/machine-learningdatabases/undocumented/connectionist-bench/sonar/).

   No entanto, para efeito de comparação, uma partição específica entre dados de treinamento e teste será fornecida. O problema é identificar se o sinal de sonar obtido (60 valores reais, correspondentes a energia em diferentes bandas de frequência e ângulos de retorno) representa uma rocha (“R”) ou uma mina (“M”).

2. ##### Requisitos Básicos

   A) Demonstrar um código computacional capaz de:
   A.1) Ler os arquivos de entrada (ou algum outro que tenha sido preparado a partir deste para facilitar a montagem da rede). Os dados propriamente ditos não devem ser modificados, para efeito de comparação, mas pré-processamento adicional dos dados é permitido, embora provavelmente desnecessário neste problema.

   A.2) Treinar um *Perceptron* com os dados de treinamento.

   A.3) Testar o *Perceptron* com os dados de teste.

   B) Fazer uma análise dos resultados obtidos, verificando a evolução da taxa de acerto (acurácia) e o erro quadrático no arquivo de treinamento, a acurácia final no arquivo de teste, as dificuldades encontradas, as soluções propostas, os valores usados para os parâmetros de treinamento. O arquivo de teste não deve ser usado para nenhuma otimização da rede. Apenas para testar uma rede já treinada.

3. ##### Regras Gerais e Observações

   A) Não será pré-definida uma linguagem de programação. 
   
   B) O algoritmo deve ser implementado pelo estudante. Não se deve simplesmente utilizar um simulador ou outro software já disponível (incluindo o Toolbox do MATLAB). Podem ser utilizadas bibliotecas matemáticas para as operações necessárias (operações com matrizes, leituras de arquivos, etc).
   
   C) Apresentar o código, os resultados obtidos e a análise.

------

