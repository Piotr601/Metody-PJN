import csv
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

file_content = []
file_label = []

# Zadanie 2.1

csv_file = open('imbd.csv', 'r')
reader = csv.reader(csv_file, delimiter=',')
    
# Wczytanie pliku do file_content
for row in reader:
    new_row = row[0].lower()
    file_content.append(new_row)
    file_label.append(row[1])
    
csv_file.close()

file_content.remove('content')
file_label.remove('label')

# Tokenizacja
original = [TreebankWordTokenizer().tokenize(x) for x in file_content]

stop_words = stopwords.words('english')
other_words = ['<', 'br', '/', '>', '(', ')', ',', '.', '..', '...', '?', '!', '``', "'", "''", '-', ';', ':', '--']
stop_words.extend(other_words)

without_stop_words = []
stemming, lemmatization, new_original = [], [], []

token_wsw, token_lemm = [],[]
token_stem, token_org = [],[]


for row in original:
    new_row = [x for x in row if x not in stop_words]
    without_stop_words.append(' '.join(new_row))
    stemming.append(' '.join([PorterStemmer().stem(x) for x in new_row]))
    lemmatization.append(' '.join([WordNetLemmatizer().lemmatize(x) for x in new_row]))
    new_original.append(' '.join(row))
    
# print(without_stop_words[0])
# print(stemming[0])
# print(new_original[0])
# print(lemmatization[0])

# Zadanie 2.2
def classification(method, array):
    X_train, X_test, y_train, y_test = train_test_split(array, file_label, test_size=0.3, random_state=42)

    vectorize = CountVectorizer(max_features=10000)
    
    X_train_cv = vectorize.fit_transform(X_train)
    X_test_cv = vectorize.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_cv, y_train)
    
    predict = clf.predict(X_test_cv)
    score = balanced_accuracy_score(y_test, predict)
    
    print(f"Method: {method}, score: {score}")

print('\nClassification:')
classification('original', new_original)
classification('stop_words', without_stop_words)
classification('stemming', stemming)
classification('lemmatization', lemmatization)

# Zadanie 2.3
def learning(method, array):
    scores = []
    tvectorizer = TfidfVectorizer(max_features=300)
    
    y = np.array(file_label)

    X = tvectorizer.fit_transform(array)

    clf = MLPClassifier(max_iter=1000)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        scores.append(balanced_accuracy_score(y_test, predict))

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print("Method: %s, score: %.10f (%.10f)" % (method, mean_score, std_score))

print('\nMachine learning:')
learning('original', new_original)
learning('stop_words', without_stop_words)
learning('stemming', stemming)
learning('lemmatization', lemmatization)