import numpy as np


def load_lemmas(lemmas_file):
    print('Loading lemmas')
    lemmas = []
    with open(lemmas_file, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            lemma = parts[0].strip()
            lemmas.append(lemma)
    return lemmas


def calculate_vector_matrix(lemma_vocabulary, tfidf_directory):
    print('Calculating vector matrix')
    doc_lemma_matrix = np.zeros((100, len(lemma_vocabulary)))
    for i in range(100):
        with open(f'{tfidf_directory}/{i + 1}.txt', 'r') as f:
            for line in f:
                values = line.strip().split()
                lemma, tfidf = values[0], values[2]
                if lemma in lemma_vocabulary:
                    j = lemma_vocabulary.index(lemma)
                    doc_lemma_matrix[i, j] = tfidf
    return doc_lemma_matrix


if __name__ == '__main__':
    lemma_vocabulary = load_lemmas('../task2/lemmas.txt')
    doc_lemma_matrix = calculate_vector_matrix(lemma_vocabulary, '../task4/lemmas')
    doc_lemma_matrix_normalized = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, doc_lemma_matrix)
    np.save('vector_matrix_normalized.npy', doc_lemma_matrix_normalized)
