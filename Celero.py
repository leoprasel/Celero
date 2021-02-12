#Importing global libraries
from datetime import datetime
import numpy as np
import pandas as pd
import click
import os
import re
import pickle
import nltk
nltk.download('wordnet', quiet=True) 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

#Python CLI options
@click.command()
@click.option('--modo', '-m')
@click.option('--diretorio', '-d')

def main(modo, diretorio):
    #Function to clean the text, replacing HTML tags, punctuations and other undesirable characters
    #This function will also get the root of the word in order to get a better accuracy(Lemmatization)
    def cleaning(text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub('[^a-zA-Z]',' ', text)
        text = text.strip().lower().split()
        
        lmtzr = WordNetLemmatizer()
        text = [lmtzr.lemmatize(word) for word in text]
        text = ' '.join(text)
        return text
    
    if modo=='treino':
        
        #Importing training mode exclusive libraries for performance
        
        print('Started at: ',datetime.now())

        from sklearn.svm import LinearSVC
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score
        
        #Relative paths
        train_path_neg = diretorio + '/train/neg//'
        train_path_pos = diretorio +  '/train/pos//'
        test_path_neg = diretorio + '/test/neg//'
        test_path_pos = diretorio + '/test/pos//'

        

        #Function to read the text files and concatenate them into a pandas dataframe
        def read_files(files_path, evaluation, mode):
            review_list = []
            for file in os.listdir(files_path):
                file_review = open(files_path + file,'r', encoding='utf8').read()
                review_list.append(file_review)
            
            df = pd.DataFrame(np.array(review_list),columns=['text'])
            df['evaluation'] = evaluation
            df['mode'] = mode
            return df

        print('Reading files...')
        train_neg_df = read_files(train_path_neg, -1, "train")
        train_pos_df = read_files(train_path_pos, 1, "train")
        train_df = pd.concat([train_neg_df, train_pos_df]).reset_index(drop=True)

        test_neg_df = read_files(test_path_neg, -1, 'test')
        test_pos_df = read_files(test_path_pos, 1, 'test')
        test_df = pd.concat([test_neg_df, test_pos_df]).reset_index(drop=True)
        
        #Cleaning the data
        print('Cleaning files...')
        train_df['cleaned_text'] = train_df.text.apply(lambda x: cleaning(x))
        test_df['cleaned_text'] = test_df.text.apply(lambda x: cleaning(x))

        #Creating the Bag of Words vector
        vectorizer = CountVectorizer(stop_words="english", max_features=500)

        training_features = vectorizer.fit_transform(train_df["cleaned_text"])
        test_features = vectorizer.transform(test_df["cleaned_text"])

        # Training the support vector classifier
        print('Training model...')
        model = LinearSVC()
        model.fit(training_features, train_df["evaluation"])
        y_pred = model.predict(test_features)

        # Results
        accuracy = accuracy_score(test_df["evaluation"], y_pred)

        print("Precisao do modelo: {:.2f}".format(accuracy*100))

        print('Saving the machine learning model...')
        with open('celero_model.pickle','wb') as file:
            pickle.dump(model, file)
        with open('celero_vectorizer.pickle','wb') as file:
            pickle.dump(vectorizer, file)

        
        print('Ended at: ',datetime.now())
    
    elif modo == 'execucao':
        with open('celero_model.pickle','rb') as file:
            model = pickle.load(file)
        with open('celero_vectorizer.pickle','rb') as file:
            vectorizer = pickle.load(file)


        review_list = []
        review = open(diretorio,'r', encoding='utf8').read()
        review = cleaning(review)
        review_list.append(review)
        df = pd.DataFrame(np.array(review_list),columns=['text'])

        review_vec = vectorizer.transform(df['text'])

        y_pred = model.predict(review_vec)
        if int(y_pred) == 1:
            print('Positive Review!')
        else:
            print('Negative Review!')


    else:
        print('modo errado!')








if __name__ == "__main__":
    main()