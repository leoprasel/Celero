from datetime import datetime
import numpy as np
import pandas as pd
import click
import os
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

@click.command()
@click.option('--modo', '-m')
@click.option('--diretorio', '-d')
def main(modo, diretorio):
    if modo=='treino':
        train_path_neg = r'D:\Leo\Área de Trabalho\Python\Celero\Celero\Database\train\neg\\'
        train_path_pos = r'D:\Leo\Área de Trabalho\Python\Celero\Celero\Database\train\pos\\'
        test_path_neg = r'D:\Leo\Área de Trabalho\Python\Celero\Celero\Database\test\neg\\'
        test_path_pos = r'D:\Leo\Área de Trabalho\Python\Celero\Celero\Database\test\pos\\'

        print(datetime.now())

        #Function to read the text files and concatenate them into a pandas dataframe
        def read_files(files_path, evaluation, mode):
            text_list = []
            for file in os.listdir(files_path):
                file_review = open(files_path + file,'r', encoding='utf8').read()
                text_list.append(file_review)
            
            df = pd.DataFrame(np.array(text_list),columns=['text'])
            df['evaluation'] = evaluation
            df['mode'] = mode
            return df

        train_neg_df = read_files(train_path_neg, -1, "train")
        train_pos_df = read_files(train_path_pos, 1, "train")
        train_df = pd.concat([train_neg_df, train_pos_df]).reset_index(drop=True)

        test_neg_df = read_files(test_path_neg, -1, 'test')
        test_pos_df = read_files(test_path_pos, 1, 'test')
        test_df = pd.concat([test_neg_df, test_pos_df]).reset_index(drop=True)
        
        #Function to replace HTML tags, punctuations and other undesirable characters
        #This function will also get the root of the word in order to get a better accuracy
        def cleaning(text):
            text = re.sub(r'<.*?>', '', text)
            text = re.sub('[^a-zA-Z]',' ', text)
            text = text.strip().lower()
            
            lmtzr = WordNetLemmatizer()
            text = lmtzr.lemmatize(text)
            return text

        #Creating the Bag of Words vector
        vectorizer = CountVectorizer(stop_words="english", preprocessor=cleaning)

        training_features = vectorizer.fit_transform(train_df["text"])    
        test_features = vectorizer.transform(test_df["text"])

        # Training the support vector classifier
        model = LinearSVC()
        model.fit(training_features, train_df["evaluation"])
        y_pred = model.predict(test_features)

        # Results
        acc = accuracy_score(test_df["evaluation"], y_pred)

        print("Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
        print(datetime.now())









if __name__ == "__main__":
    main()