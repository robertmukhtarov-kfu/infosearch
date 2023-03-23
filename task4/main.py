import math
import os
import re
import nltk
from pathlib import Path
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


def save_cleaned_htmls_if_needed(htmls_directory, save_directory):
    cleaned_htmls_dir = save_directory
    if Path(cleaned_htmls_dir).is_dir() and len(listdir(cleaned_htmls_dir)) > 0:
        return
    print('Cleaning htmls and saving texts')
    htmls = dir_reader(htmls_directory)
    texts = extract_text(htmls)
    Path(cleaned_htmls_dir).mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts):
        with open(f'{cleaned_htmls_dir}/{i + 1}.txt', 'w') as file:
            file.write(text)


def tokenize_with_count(text):
    text = text.lower()
    t = re.sub(r'[^A-Za-z-]', ' ', text)
    t = re.sub(r'\d', '', t)
    t = t.split(' ')
    t = filter(lambda chunk: chunk in words and len(chunk) > 1, t)
    token_count_dict = {}
    for token in t:
        if token in token_count_dict:
            token_count_dict[token] += 1
        else:
            token_count_dict[token] = 1
    return token_count_dict


def count_tokens(texts):
    print('Counting termins')
    token_count_dicts = []
    for i, text in enumerate(texts):
        token_count_dicts.append(tokenize_with_count(text))
    return token_count_dicts


def load_termins(termins_file):
    print('Loading termins')
    termins_set = set()
    with open(termins_file, 'r') as file:
        for line in file:
            termins_set.add(line.strip())
    return termins_set


def load_lemmas(lemmas_file):
    print('Loading lemmas')
    lemma_form_dict = {}
    with open(lemmas_file, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            word = parts[0].strip()
            forms = [form.strip() for form in parts[1].split()]
            lemma_form_dict[word] = forms
    return lemma_form_dict


def count_lemma_forms(lemma_form_dict, token_count_dicts):
    print('Counting lemma forms')
    lemma_count_dicts = []
    for token_count_dict in token_count_dicts:
        lemma_count_dict = {}
        for lemma, forms in lemma_form_dict.items():
            count = 0
            for form in forms:
                if form in token_count_dict:
                    count += token_count_dict[form]
            if count != 0:
                lemma_count_dict[lemma] = count
        lemma_count_dicts.append(lemma_count_dict)
    return lemma_count_dicts


def calculate_doc_word_sums(token_count_dicts):
    print('Counting total word number per doc')
    sums = []
    for token_count_dict in token_count_dicts:
        sums.append(sum(token_count_dict.values()))
    return sums


def calculate_tf_idf(result_dir, word_count_dicts, termins_set, doc_word_sums):
    print(f'Calculating tf-idf for {result_dir}')
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    for file in listdir(result_dir):
        os.remove(os.path.join(result_dir, file))
    doc_count = len(word_count_dicts)
    for i, word_count_dict in enumerate(word_count_dicts):
        with open(f'{result_dir}/{i + 1}.txt', 'a') as file:
            for word, count in word_count_dict.items():
                if word not in termins_set:
                    continue
                tf = count / doc_word_sums[i]
                idf = math.log(doc_count / sum(1 for d in word_count_dicts if word in d))
                tf_idf = tf * idf
                if idf == 0.0:
                    file.write(f'{word} 0.0 0.0\n')
                else:
                    file.write(f'{word} {idf:.20f} {tf_idf:.20f}\n')


# Перед выполнением нужно запустить task1/main.py и task2/main.py
if __name__ == '__main__':
    save_cleaned_htmls_if_needed('../task1/downloads', 'cleaned_htmls')
    texts = dir_reader('cleaned_htmls')
    termins_set = load_termins('../task2/tokens.txt')
    lemma_form_dict = load_lemmas('../task2/lemmas.txt')
    token_count_dicts = count_tokens(texts)
    lemma_count_dicts = count_lemma_forms(lemma_form_dict, token_count_dicts)
    doc_word_sums = calculate_doc_word_sums(token_count_dicts)
    calculate_tf_idf('termins', token_count_dicts, termins_set, doc_word_sums)
    calculate_tf_idf('lemmas', lemma_count_dicts, termins_set, doc_word_sums)
