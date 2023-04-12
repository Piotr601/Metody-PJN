import os
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Variables
path = 'Lab03/poezja'
files_path_list = []
files_name_list = []
temp = []
path_temp = []
new_files_name_list = []

# Zadanie 3.1
# Scanning files section
for folder in os.listdir(path):
    if folder != 'stopwordsPL.txt':
        for item in os.listdir(f'{path}/{folder}'):
            if item != '.DS_Store':
                path_temp.append(f'{path}/{folder}/{item}')
                new_files_name_list.append(f'{path}/{folder}/{item}')
                temp.append(item)
        files_name_list.append(temp)
        files_path_list.append(path_temp)
        temp, path_temp = [], []

# Stopwords
stopwords_list = []
with open('Lab03/poezja/stopwordsPL.txt', 'r') as stopwords_file:
    for line in stopwords_file:
        stopwords_list.append(line[:-1])

# Function to check that string is alhpanumeric
def check_alpha_num(string):
    if string.isalnum():
        return True
    return False

# Function to do operation on file
def file_operations_tokenize_stopowrods_filter(path):
    f = open(f'{path}', 'r')
    r = f.read().lower()
    tck = word_tokenize(r)
    new_tck = [x for x in tck if x not in stopwords_list]
    filtered_tck = filter(check_alpha_num, new_tck)
    return list(filtered_tck)

# Opening each file and do file operations
filtered_list = []
for folder in files_path_list:
    for item in folder:
        
        filtered = file_operations_tokenize_stopowrods_filter(item)
        temp.append(filtered)
        
    filtered_list.append(temp)
    temp = []

# Creating each list
janusz = filtered_list[0]
jan = filtered_list[1]  
adam = filtered_list[2]

extended_list = janusz + jan + adam

EPOCHS = 16

# Creating model Word2Vec
model = Word2Vec(sentences=extended_list, vector_size=16, window=5, min_count=1)
model.train(janusz, total_examples=len(janusz) , epochs=EPOCHS)
model.train(jan, total_examples=len(jan), epochs=EPOCHS)
model.train(adam, total_examples=len(adam), epochs=EPOCHS)

word_similarity_list = [
    ['wiatr', 'fale'], 
    ['trawie', 'zioła'],
    ['zbroja', 'szalonych'],
    ['cichym', 'szeptem']
]

for item in word_similarity_list:
    print(f"Slowa: {item[0]}-{item[1]}\t\t{model.wv.similarity(item[0], item[1])}")

'''
Wyniki nieskie jak sie spodziewano (dla epoch 16)

Slowa: wiatr-fale               0.49287110567092896
Slowa: trawie-zioła             -0.48758548498153687
Slowa: zbroja-szalonych         -0.07089608907699585
Slowa: cichym-szeptem           -0.06894713640213013

Zaskakujące jest to, ze ostatnia para wyrazow jest niepodobna, mimo ze jest powiązana.
Oczekiwałem tez rezultatow, ze zbroja-szalonych będzie miała gorsze podobieństwo.

'''

# Zadanie 3.2
doc_vector = []


def cosine_similarity(a, b):
    k = np.dot(a,b)
    nrm_a = np.linalg.norm(a)
    nrm_b = np.linalg.norm(b)
    return k / (nrm_a * nrm_b)
    
for file in extended_list:
    file_vector = []
    for content in file:
        file_vector.append(model.wv[str(content)])
        
    doc_vector.append(np.mean(file_vector, axis=0))

matrix_shape = len(extended_list)
similarity_matrix = np.zeros((matrix_shape, matrix_shape))
for i in range(matrix_shape):
    for j in range(matrix_shape):
        if i == j:
            similarity_matrix[i][j] = 0
        else:
            similarity_matrix[i][j] = cosine_similarity(doc_vector[i], doc_vector[j])
            
print(similarity_matrix)
sim_min = np.where(similarity_matrix == similarity_matrix.min())
sim_max = np.where(similarity_matrix == similarity_matrix.max())

print(f"\nMacierz prawdopodobieństwa: \n{similarity_matrix}")
print(f"\nNajmniej podobne dokumenty: {new_files_name_list[sim_min[0][0]]} oraz {new_files_name_list[sim_min[0][1]]}")
print(f"Najbardziej podobne dokumenty: {new_files_name_list[sim_max[0][0]]} oraz {new_files_name_list[sim_max[0][1]]}\n")

'''
Rezultat ukazany wyzej,
Najbardziej podobne sa od tego samego autora, a najmniej od roznych

Najmniej podobne dokumenty: Lab03/poezja/juliusz/juliusz_słowacki_rozłączenie.txt oraz Lab03/poezja/jan/jan_kochanowski_na_łakome.txt
Najbardziej podobne dokumenty: Lab03/poezja/adam/adam_mickiewicz_reduta_ordona.txt oraz Lab03/poezja/adam/adam_mickiewicz_świtezianka.txt

'''

#Zadanie 3.2
VECTOR_SIZE = 33

documents = [TaggedDocument(doc, [i]) for i, doc in zip(new_files_name_list, extended_list)]
model_doc2vec = Doc2Vec(documents, vector_size=VECTOR_SIZE, window=2, min_count=1)

doc2vec_matrix_shape = len(documents)
similarity_matrix = np.zeros((matrix_shape, VECTOR_SIZE))
for i in range(matrix_shape):
    for j in range(VECTOR_SIZE):
        if i == j:
            similarity_matrix[i][j] = 0
        else:
            similarity_matrix[i][j] = cosine_similarity(model_doc2vec[i], model_doc2vec[j])
            
sim_min = np.where(similarity_matrix == similarity_matrix.min())
sim_max = np.where(similarity_matrix == similarity_matrix.max())

print(f"\nMacierz prawdopodobieństwa: \n{similarity_matrix}")
print(f"\nNajmniej podobne dokumenty: {new_files_name_list[sim_min[0][0]]} oraz {new_files_name_list[sim_min[0][1]]}")
print(f"Najbardziej podobne dokumenty: {new_files_name_list[sim_max[0][0]]} oraz {new_files_name_list[sim_max[0][1]]}\n")

'''
Uzyskany rezultat:

Najmniej podobne dokumenty: Lab03/poezja/jan/jan_kochanowski_o_żywocie_ludzkim.txt oraz Lab03/poezja/adam/adam_mickiewicz_żegluga.txt
Najbardziej podobne dokumenty: Lab03/poezja/jan/jan_kochanowski_o_żywocie_ludzkim.txt oraz Lab03/poezja/adam/adam_mickiewicz_oda_do_młodości.txt

'''
