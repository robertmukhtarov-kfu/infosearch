import re
from os import listdir

from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

import nltk

nltk.download('wordnet')
nltk.download('words')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
words = set(words.words())
en_stopwords = stopwords.words('english')


def dir_reader(directory):
    files = [file for file in listdir(directory) if file.endswith('txt')]
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
    tokens = set()
    for text in texts:
        if text is not None:
            tokens.update(tokenize(text))
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
    return [word for word in tokens if word not in en_stopwords and len(word) > 1]


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
    for token in tokens:
        for part_of_speech in parts_of_speech:
            token_lem = lemmatizer.lemmatize(token, part_of_speech)
            if not token_lem in lemmas:
                lemmas[token_lem] = [token]
            else:
                if token not in lemmas[token_lem]:
                    lemmas[token_lem].append(token)
    return dict(sorted(lemmas.items(), key=lambda i: -len(i[1])))


def save_lemmas(lemmas):
    print('Saving lemmas')
    with open('lemmas.txt', 'w') as file:
        for lemma, forms in lemmas.items():
            file.write(f'{lemma}: {" ".join(forms)}\n')


if __name__ == '__main__':
    # Перед выполнением нужно запустить task1/main.py для скачивания файлов
    contents = dir_reader('../task1/downloads')
    texts = extract_text(contents)
    tokens = tokenize_collection(texts)
    tokens = remove_stopwords(tokens)
    save_tokens(tokens)
    lemmas = get_lemmas(tokens)
    save_lemmas(lemmas)
