import re
import sys
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from cleantext import clean

file_content = []
file_label = []

csv_file = open('Lab05/train_stage.csv', 'r')
reader = csv.reader(csv_file, delimiter=',')

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

# Wczytanie pliku do file_content
for row in reader:
    new_row = row[1].lower()
    file_label.append(row[0])
    new_row = deEmojify(new_row)
    file_content.append(new_row)

csv_file.close()

file_content.remove('review_full')
file_label.remove('rating_review')

# Function to delete emojisP
original = [TreebankWordTokenizer().tokenize(x) for x in file_content]

stop_words = stopwords.words('english')
other_words = ['<', 'br', '/', '>', '(', ')', ',', '.', '..', '...', '?', '!', '``', "'", "''", '-', ';', ':', '--']
stop_words.extend(other_words)

without_stop_words = []
lemmatization, new_original = [], []

for row in original:
    new_row = [x for x in row if x not in stop_words]
    lemmatization.append(' '.join([WordNetLemmatizer().lemmatize(x) for x in new_row]))


def learning(method, array):
    scores = []
    tvectorizer = TfidfVectorizer(max_features=1000)    
    y = np.array(file_label)
    X = tvectorizer.fit_transform(array)

    clf = MLPClassifier(hidden_layer_sizes=(500,250,100,25,5), max_iter=10)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    for train_index, test_index in tqdm(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        scores.append(balanced_accuracy_score(y_test, predict))
        print(balanced_accuracy_score(y_test, predict))

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print("Method: %s, score: %.10f (%.10f)" % (method, mean_score, std_score))

print('\nMachine learning:')
learning('lemmatization', lemmatization)
