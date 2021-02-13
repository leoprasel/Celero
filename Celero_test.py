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
from nltk.stem.porter import PorterStemmer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Python CLI options configuration
@click.command()
@click.option('--modo', '-m')
@click.option('--diretorio', '-d')

def main(modo, diretorio):
    #Function to clean the text, replacing HTML tags, punctuations and other undesirable characters
    #This function will also get the root of the word in order to get a better accuracy
    def cleaning(text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub("n't",' not', text)
        text = re.sub("'ve",' have', text)
        text = re.sub("'m",' am', text)
        text = re.sub("'re",' are', text)
        text = re.sub("'s",' is', text)
        text = re.sub('[^a-zA-Z]',' ', text)
        text = text.strip().lower().split()
        
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text]

        #An alternative is to use a word Lemmatizer instead of the porterstemmer:
        #The results were supposed to be better, but in my tests the porterstemmer did better.
        #lmtzr = WordNetLemmatizer()
        #text = [lmtzr.lemmatize(word) for word in text]
        
        text = ' '.join(text)
        return text
    
    if modo=='treino':
        print('Started at: ',datetime.now())
        time_start = datetime.now()

        #Importing training mode exclusive libraries for performance
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
        
        time_read = datetime.now()
        print('Reading files took: ', (time_read - time_start).total_seconds(), 'seconds')

        #Cleaning the data
        print('Cleaning files...')
        train_df['cleaned_text'] = train_df.text.apply(lambda x: cleaning(x))
        test_df['cleaned_text'] = test_df.text.apply(lambda x: cleaning(x))

        #Creating the Bag of Words vector
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)

        train_bow_vector = vectorizer.fit_transform(train_df["cleaned_text"])
        test_bow_vector = vectorizer.transform(test_df["cleaned_text"])

        time_clean = datetime.now()
        print('Cleaning files took: ', (time_clean - time_read).total_seconds(), 'seconds')

        # Training the support vector classifier
        print('Training model...')
        model = LinearSVC(C=0.01)
        model.fit(train_bow_vector, train_df["evaluation"])
        pred = model.predict(test_bow_vector)

        time_model = datetime.now()
        print('Traning the model took: ', (time_model - time_clean).total_seconds(), 'seconds')

        # Results
        accuracy = accuracy_score(test_df["evaluation"], pred)

        print("Precisao do modelo: {:.2f}".format(accuracy*100))

        print('Saving the machine learning model...')
        with open('celero_model.pickle','wb') as file:
            pickle.dump(model, file)
        with open('celero_vectorizer.pickle','wb') as file:
            pickle.dump(vectorizer, file)

        
        print('Ended at: ',datetime.now())
        time_end = datetime.now()
        print('The entire program took: ', (time_end - time_start).total_seconds(), 'seconds')

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

        pred = model.predict(review_vec)
        if int(pred) == 1:
            print('Positive Review!')
        else:
            print('Negative Review!')


    else:
        print('modo errado, tente novamente!')








if __name__ == "__main__":
    main()