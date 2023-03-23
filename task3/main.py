import re
import nltk
import json
from os import listdir
from bs4 import BeautifulSoup as bs
from natsort import natsorted
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('words')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
words = set(words.words())
en_stopwords = stopwords.words('english')


def dir_reader(directory):
    files = natsorted([file for file in listdir(directory) if file.endswith('txt')])
    files_content = []
    for file in files:
        files_content.append(file_reader(f'{directory}/{file}'))
    return files_content


def file_reader(filename):
    with open(filename, 'r') as file:
        return ''.join(file.readlines())


def extract_text(htmls):
    print('Extracting text')
    texts = []
    for html in htmls:
        texts.append(clean_html(html))
    return texts


def tokenize_collection(texts):
    print('Tokenizing collection')
    tokens = dict()
    for i, text in enumerate(texts):
        token_set = tokenize(text)
        for token in token_set:
            if token not in tokens:
                tokens[token] = {i + 1}
            else:
                tokens[token].add(i + 1)
    return tokens


def tokenize(text):
    text = text.lower()
    t = re.sub(r'[^A-Za-z-]', ' ', text)
    t = re.sub(r'\d', '', t)
    t = t.split(' ')
    t = filter(lambda chunk: chunk in words and len(chunk) > 1, t)
    return set(t)


def remove_stopwords(tokens):
    print('Removing stopwords')
    for word in en_stopwords:
        if word in tokens:
            del tokens[word]
    return tokens


def clean_html(html):
    soup = bs(html, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()

    if soup.find('body') is not None:
        text = soup.find('body').get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text


def save_tokens(tokens):
    print('Saving tokens')
    with open('tokens.txt', 'w') as file:
        for token in tokens:
            file.write(token + "\n")


def get_lemmas(tokens):
    print('Getting lemmas')
    parts_of_speech = ["a", "s", "r", "n", "v"]
    lemmas = {}
    lemma_pages = {}
    for token, token_pages in tokens.items():
        for part_of_speech in parts_of_speech:
            token_lem = lemmatizer.lemmatize(token, part_of_speech)
            if token_lem not in lemmas:
                lemmas[token_lem] = {token}
                lemma_pages[token_lem] = token_pages
            else:
                lemmas[token_lem].add(token)
                lemma_pages[token_lem].union(token_pages)
    return dict(sorted(lemmas.items(), key=lambda i: -len(i[1]))), lemma_pages


def save_lemmas(lemmas):
    print('Saving lemmas')
    with open('lemmas.txt', 'w') as file:
        for lemma, forms in lemmas.items():
            file.write(f'{lemma}: {" ".join(forms)}\n')


def create_index(lemma_pages):
    print('Creating index')
    with open('inverted_index.txt', 'w') as index_file:
        json_obj = []
        for key, value in lemma_pages.items():
            json_obj.append({
                "count": len(value),
                "inverted_array": list(value),
                "word": key
            })
        json_str = json.dumps(json_obj)
        json_str = json_str.replace("}, ", "},\n")
        index_file.write(json_str)


# Перед выполнением нужно запустить task1/main.py для скачивания файлов
if __name__ == '__main__':
    contents = dir_reader('../task1/downloads')
    texts = extract_text(contents)
    tokens = tokenize_collection(texts)
    tokens = remove_stopwords(tokens)
    save_tokens(tokens)
    lemmas, lemma_pages = get_lemmas(tokens)
    save_lemmas(lemmas)
    create_index(lemma_pages)
