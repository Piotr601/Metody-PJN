import re
import json
from PIL import Image
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

FILE_TO_OPEN = "text.txt"
FILE_TO_SAVE = "tekst_nieparzyste.txt"
FILE_STOPWORDS = "stopwordsPL.txt"
buffor_to_save_in_file = ''
buffor_file = ''

# Zadanie 1.1
# Odczyt
with open(FILE_TO_OPEN, 'r') as read_file:
    for number, line in enumerate(read_file):
        #print(number, line)
        if number % 2 != 0:
            buffor_to_save_in_file += line 
        buffor_file += line

# Zapis 
with open(FILE_TO_SAVE, 'w') as write_file:
    write_file.write(buffor_to_save_in_file)
    
    
# Zadanie 1.2
# Dzielenie na zdania
sentences_list = re.findall(r"\w[^—\n.!?]+\w", buffor_file)

for index, item in enumerate(sentences_list):
    sentences_list[index] = re.sub(r",", '', item)
    
# Usuwanie spacji
sentences_string = ' '.join([str(elem) for elem in sentences_list])
sentences_counter = len(sentences_list)

# Usuwanie znaków specjalnych
sentences_str = sentences_string
cleared_sentences = re.sub(r",'", '', sentences_str)

# Dzielenie na słowa
words_list = re.split(r"\s", sentences_string)
words_counter = len(words_list)

print(f"Words: {words_counter}, Sentences {sentences_counter}")

# Zapis do JSON
json_dict = {
    "ilosc zdan" : sentences_counter,
    "lista oczyszczonych zdan" : sentences_list,
    "ilosc slow" : words_counter,
    "lista oczyszczonych slow" : words_list
}

json_object = json.dumps(json_dict, indent=4, ensure_ascii=False)
 
with open("result.json", "w", encoding='utf8') as outfile:
    outfile.write(json_object)
    


# Zadanie 1.3
# Pierwsze drzewo
json_file = open('result.json')
data = json.load(json_file)

text = data['lista oczyszczonych slow']

mask_img = np.array(Image.open("maska.jpg"))


wc = WordCloud(background_color="white", max_words=2000, mask=mask_img,
               contour_width=3, contour_color='steelblue')

wc.generate(cleared_sentences)
wc.to_file("chmura.png")

# Drugie drzewo
mask_img = np.array(Image.open("maska.jpg"))

stopwords = set(STOPWORDS)
with open("stopwordsPL.txt", "r") as filee:
    for line in filee:
        stopwords.add(line[:-1])

wc = WordCloud(background_color="white", max_words=2000, mask=mask_img,
               stopwords=stopwords, contour_width=3, contour_color='steelblue')

wc.generate(cleared_sentences)
wc.to_file("lepsza_chmura.png")

# Czym rozni sie drzewo bez i z stopwordsami.
# Mozna zauwazyc, ze drzewo ze stopwordsami blokuje slowa, ktore mogą byc bardzo czesto
# uzywanie i dzieki temu zajmuja one bardzo duzo miejsca. Dzieki uzyciu tej funkcjonalnosci
# mozna ignorowac konkretne slowa, dzieki temu dostaniemy bardziej 'uzyteczne' drzewo

# Zamkniecie pliku
json_file.close()