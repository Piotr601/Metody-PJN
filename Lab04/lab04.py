import requests
import json
import random
import time
from bs4 import BeautifulSoup

# Zadanie 5.1
url = "https://wolnelektury.pl/api/books"
response = requests.get(url)

with open(f"Lab04/wolne_lektury.json", "w", encoding="utf-8") as file:
    json.dump(response.json(), file, ensure_ascii=False ,indent=1)
    
with open(f"Lab04/wolne_lektury.json", "r", encoding="utf-8") as json_file:
    content = json.load(json_file)

random_books = random.sample(content, 20)

with open(f"Lab04/wolne_lektury_20.json", "w", encoding="utf-8") as books20:
    json.dump(random_books, books20, ensure_ascii=False, indent=1)

for item in random_books:
    print(f"{item['title']}\t{item['author']}\t{item['genre']}\t{item['kind']}\t{item['epoch']}")
    
    print(f"----------------------------------------  TRESC  --------------------------------------------------------------")
    try:
        item_response = requests.get(item['href']).json()
        txt_item_response = requests.get(item_response['txt']).text
        print(txt_item_response[:500])
    except:
        print("THE TXT IS INVALID")
    print(f"\n")
    
# Inny serwis wykorzystujący API to na przykład: IMDb

# Zadanie 5.2
url = "https://gazetawroclawska.pl/wiadomosci"
response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')
title_element = soup.find('h2')

article_title = title_element.text
title = article_title.replace('NOWE','').replace('[CENY]', '')[1:]
print(f"Article title: {title}")

robots_url = "https://gazetawroclawska.pl/robots.txt"
robot_response = requests.get(robots_url)
print(f"\n-------- ROBOT.TXT --------\n{robot_response.text}")

'''
Cytując z https://developers.google.com/search/docs/crawling-indexing/robots/intro?hl=pl
Służy on głównie do zarządzania ruchem robotów indeksujących w witrynie i zazwyczaj 
stosuje się go do wykluczenia pliku z indeksu Google w zależności od jego typu:
'''

# # Zadanie 5.3
# # Opóźnienie jest stosowane by zapobiegać dostaniu bana - algorytm
# # moze potraktowac takie zapytania jako ataki

url = "https://gazetawroclawska.pl/wiadomosci"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
links = []

for link in soup.find_all(class_ = "atomsListingArticleTileWithSeparatedLink__titleLink"): #soup.findAll("a"):
    article_href = link.get('href')
    links.append(link.get('href'))

links = [x for x in links if x != '#']

for l in links:
    
    print(f"{url}{l}\n")
    link_response = requests.get(f"{url}{l}")
    link_soup = BeautifulSoup(link_response.text, 'html.parser')
    
    title = link_soup.find('h1')
    acapit = link_soup.find(class_='md')
    print(title.text)
    print(acapit.text)
    
    time.sleep(5)
    print('\n\n')