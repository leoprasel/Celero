# Celero - Desafio para desenvolvedor de AI

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
