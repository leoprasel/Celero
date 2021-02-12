# Celero
## Desafio para desenvolvedor de AI
### Objetivo:
Neste projeto utilizaremos uma base de dados de comentários de filmes feitos no site de reviews IMDB dispobilizado pela [universidade de Stanford](https://ai.stanford.edu/~amaas/data/sentiment/).

O objetivo do desafio é treinar uma inteligência artificial que possui dois modos: 
* O modo treino, no qual ela irá ser treinada com o dataset acima, sendo inputado o diretório da pasta do dataset de treino.
* E o modo execução, onde ao ser inputado um arquivo texto com uma review de um filme, a AI deve retornar se este comentário é um comentário de cunho positivo ou negativo.

### Considerações iniciais:
* O problema se trata de uma inteligência artificial NLP (Natural Language Processing). Para AI's deste tipo, tanto Machine Learning como Redes neurais são possíveis.
  * Em datasets maiores, redes neurais performam melhor, porém estas possuem uma comlpexidade muito maior para serem desenvolvidas. Portanto, iremos utilizar machine learning.
* Como os algoritmos de machine learning de NLP transformam as frases em vetores, é ideal um algoritmo de classificação que seja amigável com um numero grande de features, como o Naive Bayes ou Support Vector Machines.

### Vetorização de palavras:
Para montar o modelo, as frases são inputadas por meio de um vetor de palavras, com todas as palavras do dataset ocupando uma posição do vetor.
#### max_features:
Quanto mais palavras existem nesse vetor, mais complexo é o modelo, aumentando a probabilidade de bias no modelo. Irrestrito, o modelo chega a mais de 70 mil palavras diferentes, sendo que existem 171 mil palavras no dicionário inglês.

Estima-se que 3 mil palavras encobrem 95% das palavras escritas coloquialmente.


### Naive Bayes vs SVC:
#### SVC:
