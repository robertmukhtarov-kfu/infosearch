import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify

app = Flask(__name__)


def load_index(index_file):
    index = []
    with open(index_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            index.append(parts[1])
    return index


def load_lemmas(lemmas_file):
    lemmas = []
    with open(lemmas_file, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            lemma = parts[0].strip()
            lemmas.append(lemma)
    return lemmas


def load_lemmas_in_docs_list(lemmas_dir):
    lemmas_in_docs = []
    for i in range(100):
        with open(f'{lemmas_dir}/{i + 1}.txt', 'r') as file:
            doc_lemmas = set()
            for line in file:
                doc_lemmas.add(line.strip().split()[0])
            lemmas_in_docs.append(doc_lemmas)
    return lemmas_in_docs


def lemmatize_query(query, lemmatizer, lemma_vocabulary):
    lemma_set = set(lemma_vocabulary)
    lemmatized_query = []
    parts_of_speech = ["a", "s", "r", "n", "v"]
    for token in query:
        for part_of_speech in parts_of_speech:
            lemma = lemmatizer.lemmatize(token, part_of_speech)
            if lemma in lemma_set:
                lemmatized_query.append(lemma)
                break
    return lemmatized_query


def calculate_query_tfidf(query, lemmas_in_docs):
    query_lemma_count_dict = {}
    for lemma in query:
        if lemma in query_lemma_count_dict:
            query_lemma_count_dict[lemma] += 1
        else:
            query_lemma_count_dict[lemma] = 1
    query_lemma_tfidf_dict = {}
    for lemma, count in query_lemma_count_dict.items():
        tf = count / len(query)
        idf = math.log(100 / sum(1 for s in lemmas_in_docs if lemma in s))
        query_lemma_tfidf_dict[lemma] = tf * idf
    return query_lemma_tfidf_dict


def convert_query_to_vector(query_tfidf, lemma_vocabulary):
    vector = np.zeros(len(lemma_vocabulary))
    for lemma, tfidf in query_tfidf.items():
        if lemma in lemma_vocabulary:
            i = lemma_vocabulary.index(lemma)
            vector[i] = tfidf
    vector_normalized = vector / np.linalg.norm(vector)
    return vector_normalized


def search(query_vector, doc_lemma_matrix_normalized, index):
    cos_similarities = cosine_similarity([query_vector], doc_lemma_matrix_normalized)[0]
    indices = cos_similarities.argsort()[::-1]
    docs = []
    for i in range(10):
        docs.append(index[indices[i]])
    return docs


@app.route('/')
def main_page():
    with open('search.html', 'r') as f:
        html = f.read()
    return html


@app.route('/search', methods=['POST'])
def search_query():
    query = request.form['query']
    try:
        query_lemmatized = lemmatize_query(query.lower().strip().split(), lemmatizer, lemma_vocabulary)
        query_tfidf = calculate_query_tfidf(query_lemmatized, lemmas_in_docs)
        query_vector = convert_query_to_vector(query_tfidf, lemma_vocabulary)
        search_results = search(query_vector, doc_lemma_matrix_normalized, index)
        return jsonify(search_results)
    except Exception as e:
        return e.args[0], 500


if __name__ == '__main__':
    lemmatizer = WordNetLemmatizer()
    index = load_index('../task1/index.txt')
    lemma_vocabulary = load_lemmas('../task2/lemmas.txt')
    doc_lemma_matrix_normalized = np.load('vector_matrix_normalized.npy')
    lemmas_in_docs = load_lemmas_in_docs_list('../task4/lemmas')
    app.run()
