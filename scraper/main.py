import os
import urllib
import urllib.parse
import requests
from threading import Thread
from bs4 import *
import shutil

def FileProcessorThread(file):
    song_name = file[:-19]
    
    predicted_lastfm_url = "https://www.last.fm/search/tracks?q=" + str(urllib.parse.quote(song_name, safe=''))
    response = requests.get(predicted_lastfm_url)
    if response.status_code != 200:
        raise ConnectionError("Search Page Status code is not 200")

    body = response.text
    
    soup = BeautifulSoup(body, 'html.parser')
    results = soup.findAll('tbody')
    for result in results:
        if 'data-playlisting-add-entries' in result.attrs:
            rowsoup = result.findAll('tr')
            rowresult = rowsoup[0]
            if 'class' in rowresult.attrs:
                if 'chartlist-row--with-artist' in rowresult.attrs['class']:
                    infosoup = rowresult.findAll('td')
                    for infores in infosoup:
                        if 'class' in infores.attrs:
                            if 'chartlist' in infores.attrs['class'][0]:
                                if ['chartlist-name'] == infores.attrs['class']:
                                    a_tag = infores.select_one('a')
                                    href = a_tag['href']
                            else:
                                print("Couldn't find the chartlist: ", infores.attrs['class'])
        else:
            print("Failed!")
    
    if 'href' in locals() is False:
        raise Exception("Href is none? Snippet:", rowresult)
    
    tags_url = 'https://last.fm' + href
    response = requests.get(tags_url)
    if response.status_code != 200:
        raise ConnectionError("Status code is not 200")
        
    tags = []

    body = response.text
    soup = BeautifulSoup(body, 'html.parser')
    results = soup.findAll('section')
    for result in results:
        if 'class' in result.attrs:
            if 'catalogue-tags' in result.attrs['class']:
                if 'about-artist-tags' not in result.attrs['class']:
                    ul_element = result.select_one('ul')
                    if 'class' in ul_element.attrs:
                        if 'tags-list--global' in ul_element.attrs['class']:
                            for tag in ul_element.find_all(class_='tag'):
                                tag_text = tag.a.text
                                tags.append(tag_text)

    print(song_name, "Has the tags:", tags)

    song_name = song_name.replace(" ", "_")
    song_name = song_name.replace("'", "_")
    song_name = song_name.replace("-", "_")
    song_name = song_name.replace("*", "_")
    song_name = song_name.replace('"', "_")
    actual_file_location = os.path.join(PATH, file)
    new_file_location = os.path.join(PATH, song_name + ".webm")
    shutil.move(actual_file_location, new_file_location)


    tags_str = ', '.join(tags)
    tags_str = 'Composed by Snails House, ' + tags_str
    txt_name = song_name + ".txt"
    txt_file_location = os.path.join(PATH, txt_name)
    with open(txt_file_location, 'w') as f:
        f.write(tags_str)

thread_list = []

PATH = "videos"
list_dir = os.listdir(PATH)

print("number of files in dir:", len(list_dir))

# Thread(target=FileProcessorThread, args=(list_dir[0],)).start()

for file in list_dir:
    Thread(target=FileProcessorThread, args=(file,)).start()