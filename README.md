# Celero - Desafio para desenvolvedor de AI
## ENGLISH
## Objetivo:
In this project we will use a movie review database from IMDB comentary section made available in an open source way by [Stanford university](https://ai.stanford.edu/~amaas/data/sentiment/).

The main goal is to train an artificial intelligence that has two modes: 
* The train mode, in which the model will be trained with the dataset above, being inputed the path of the train dataset folder.
```
python3 Celero.py --modo treino --diretorio <dataset path>
```
* And the execution mode, in which when inputed a random text file from a movie review, the AI should return if this comentary had a positive or negative feeling.
```
python3 Celero.py --modo execucao --diretorio <file path>
```
## Initial considerations:
* The problem uses a NLP (Natural Language Processing) artificial intelligence. For AI's of this type, either Machine Learning or Neural Networks can be done.
  * In bigger datasets, neural networks perform better, but they have a higher complexity to be developed. So, we will use machine learning.
* As NLP algorithms transform sentences in vectors, a classification algorithm that takes a large number of features is recommended, like  Naive Bayes or Support Vector Machines.

![](https://raw.githubusercontent.com/leoprasel/Celero/main/plots/wordcloud.png)

## Words Vetorization:
### Bag of Words and TF-IDF:
To build the model, we will use the Bag of Words model, function that transforms inputed sentences into a vector of words, with every word of the dataset being an index of this vector.

But only the vectorization doesn't produce interessant results, because there are way too many common words like "is","the", "a", and other non-relevant words. To solve this problem, we can use the "stop words" that exclude this words from the dataset, and we can also use TF-IDF.

TF-IDF is an acronym of Term frequency - inverse document frequency, and it's a tool in order to calculate an weighted average of the words that are most frequent in relation to the others. In this project we will use the TfidfVectorizer lib, that automatically makes the vectorization and the tf-idf simultaneously.

### Stemming vs Lemmatization:
In order to properly clean the data, we can't consider similar words into two different indexes, so for example verbs and its conjugations like "love" and "loved" are words that should express the same idea.

For that, we can use Stemming to cut the words suffixes, or Lemmatization, to transform the words to its lexical form.

In theory, Lemmatization should perform better, but in my practical tests, it had 86.9% precision against 87.1% of Stemming's, who got a better result in this test.
### max_features:
The bigger the number in this vector, more complex is the model, increasing the probability of bias in the model. Without restrictions, the model can get up to 70 thousand different words, given that there are around 171 thousand words in the english dictiory. Although there are many words, it is estimated that 3 thousand words cover 95% of the coloquial written vocabulary.

In this model, we can see the model accuracy based on the amount of words:
![](https://raw.githubusercontent.com/leoprasel/Celero/main/plots/features_tfidf.jpg)

As shown in the chart, around 2000-2500 words gives us more precision, being 2000 the number we chose in this project.
## Choosing the machine learning model:
### Naive Bayes:
To the gaussian Naive Bayes model, in which doesn't have any important ajustable parameters, we got an **accuracy result of 86,5%**

### SVC Radial basis function (rbf)
This model took way longer to train that the other ones, longer than half and hour. The results also wasn't so great, with **only 70,32% precision**
### Linear SVC:
For the support vector machines model, we used the linear kernel (LinearSVC), this model has a hyperparameter called C, which is the intensity of the plane in classifying wrongly some data. The bigger the C param, bigger the overfit.
Plotting the model accuracy in relation to C, we see the model works better around 0.05-0.1, which is a low and acceptable value.

![](https://raw.githubusercontent.com/leoprasel/Celero/main/plots/c_param_linear_svc.jpg)

In short, the **better precision was from the SVC model with 87,1%**. The Linear SVC was the chosen one, using C = 0.1 and it took  **1.7 minutes** to train the model.

## Final considerations:
The moedel accutacy got to satisfatory numbersin the way it is programmed, but it is always possible to improve its performance beyond what we have today. The main improvement point is on the data cleaning of the text files, in which the Stemming and Lemmatization functions should be better than the one observed superficially at the time of the model construction. 

I tried cleaning the stop-words outside the vectorization function to have control of the deleted words, but it increased a lot the program run time, so I chose to use the built-in function.

## PORTUGUESE
## Objetivo:
Neste projeto utilizaremos uma base de dados de comentários de filmes feitos no site de reviews IMDB dispobilizado pela [universidade de Stanford](https://ai.stanford.edu/~amaas/data/sentiment/).

O objetivo do desafio é treinar uma inteligência artificial que possui dois modos: 
* O modo treino, no qual ela irá ser treinada com o dataset acima, sendo inputado o diretório da pasta do dataset de treino.
```
python3 Celero.py --modo treino --diretorio <dataset path>
```
* E o modo execução, onde ao ser inputado um arquivo texto com uma review de um filme, a AI deve retornar se este comentário é um comentário de cunho positivo ou negativo.
```
python3 Celero.py --modo execucao --diretorio <file path>
```
## Considerações iniciais:
* O problema se trata de uma inteligência artificial NLP (Natural Language Processing). Para AI's deste tipo, tanto Machine Learning como Redes neurais são possíveis.
  * Em datasets maiores, redes neurais performam melhor, porém estas possuem uma complexidade muito maior para serem desenvolvidas. Portanto, iremos utilizar machine learning.
* Como os algoritmos de machine learning de NLP transformam as frases em vetores, é ideal um algoritmo de classificação que seja amigável com um numero grande de features, como o Naive Bayes ou Support Vector Machines.

![](https://raw.githubusercontent.com/leoprasel/Celero/main/plots/wordcloud.png)

## Vetorização de palavras:
### Bag of Words e TF-IDF:
Para montar o modelo, utilizaremos o Bag od Words, função que transforma as frases que são inputadas em um vetor de palavras, com todas as palavras do dataset ocupando uma posição do vetor.

Mas somente a vetorização não produz resultados tão interessantes, pois muitas palavras comuns como "is","the", "a", que não são relevantes para o modelo são as mais utilizadas. Para resolver este problema, podemos usar as "stop words" que excluem estas palavras comuns do dataset, e também podemos utilizar o TF-IDF.

TF-IDF termo para Term frequency - inverse document frequency, é uma ferramenta para fazer uma média ponderada das palavras que aparecem muito em relação às outras. Neste projeto utilizaremos o TfidfVectorizer, que já faz a vetorização e tf-idf simultaneamente.

### Stemming vs Lemmatization:
Para a limpeza efetiva dos dados, não podemos considerar que palavras similares entrem em vetores diferentes, portanto verbos e suas conjugações como "love" e "loved" são palavras que deveriam expressar a mesma ideia.

Para isso, pode-se utilizar o Stemming para cortar os sufixos das palavras, ou o Lemmatization, para transformar a palavra em sua forma lexical.

Em teoria o Lemmatization deveria performar melhor, mas em meus testes práticos, ele teve precisão de 86.9% contra os 87.1% do Stemming, que obteve melhor resultado nesta experiência.
### max_features:
Quanto mais palavras existem nesse vetor, mais complexo é o modelo, aumentando a probabilidade de bias no modelo. Irrestrito, o modelo chega a mais de 70 mil palavras diferentes, sendo que existem 171 mil palavras no dicionário inglês. Embora existam muitas palavras, estima-se que 3 mil palavras encobrem 95% das palavras escritas coloquialmente.

No modelo, podemos ver a acuracidade do modelo conforme aumentamos o numero de palavras:
![](https://raw.githubusercontent.com/leoprasel/Celero/main/plots/features_tfidf.jpg)

Conforme o gráfico, em torno de 2000-2500 palavras retorna a maior precisão, sendo 2000 o numero escolhido nesse projeto.
## Escolha do modelo de machine learning:
### Naive Bayes:
Para o modelo gaussiano da Naive Bayes, o qual não possui parametros de ajuste importantes, obtemos um **resultado de acurácia de 86,5%**

### SVC Radial basis function (rbf)
O modelo demorou muito mais que os outros para treinar, ultrapassando meia hora. Os resultados também não foram muito bons, com **apenas 70,32% de precisão**
### SVC Linear:
Para o modelo de support vector machines, utilizando o kernel linear (LinearSVC), possuimos de hiperparametro desse modelo a variável C, que é a intensidade de regularização do plano em classificar erroneamente um dado. Quanto maior o C, mais propenso a overfit.
Plotando a acurácia do modelo em relação ao C, vemos que esta desempenha melhor em torno de 0,05-0,1, que é um numero baixo e aceitável.

![](https://raw.githubusercontent.com/leoprasel/Celero/main/plots/c_param_linear_svc.jpg)

Portanto, a **melhor precisão do modelo SVC foi de 87,1%**. O modelo SVC Linear foi o escolhido como o modelo final, utilizando C = 0,1 e demorou **1.7 minutos** para terminar de treinar.

## Considerações finais:
A acurácia do modelo alcançou patamares satisfatórios da maneira que está programado, porém é possivel melhorar a performance do modelo além do que temos hoje. O principal ponto de melhoria está na limpeza dos arquivos texto, em que as funções de Stemming e Lemmatization deveriam desempenhar melhor do que o que foi observado superficialmente por mim na hora da construção do modelo. 

Tentei fazer a limpeza das stop-words fora da função vetorizante para ter controle das palavras que seriam excluidas, mas isto aumentava drasticamente o runtime do programa, portanto optei por utilizar a função built-in.
