import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
# global vars
stopwords = nltk.corpus.stopwords.words('english')

def read_doc_file():
    '''
    ler todos os textos-fonte com titulo
    '''
    all_docs = []
    all_files = os.listdir('TextoFonteComTitulo')
    for fl in all_files:
        f = open('TextoFonteComTitulo/' + fl, 'r')
        content = f.read()
        all_docs.append(content)
    return all_docs

def read_documents_into_sentence_tokens(all_docs):
    '''
    divide cada doc em lista de frases
    lista com listas(estas listas sao tokens de frase)
    '''
    all_docs_sentences = []
    for i in range(len(all_docs)):
        sentence_tokens = all_docs[i].split('.')
        for i in range(len(sentence_tokens)):
            sentence_tokens[i] = sentence_tokens[i].replace('\n', '')
            sentence_tokens[i] = sentence_tokens[i].replace('\r', '')
            sentence_tokens[i] = sentence_tokens[i].replace('\t', '')
        sentence_tokens = sentence_tokens[:-1]
        all_docs_sentences.append(sentence_tokens)
    return all_docs_sentences

def cos_sims(x, x2, thresholdCS=0):
    '''
    :param x: sparseMatrix with all sentences transform
    :param x2: sparseM with 1 sentence transform
    :param thresholdCS: if value is provided returns only cosine_sims above the given threshold value
    :return: returns cosine_sims, all if no thresholdCS argument is provided
    filters cosine_sims list
    '''

    conected = []
    cosine_sims = cosine_similarity(x, x2)
    for i in range(len(cosine_sims)):
        if cosine_sims[i] < thresholdCS:
            conected.append(i) # apends index of the sentence in the sentence list
    return conected

def addToGraph(graph, id, sim_indexes):
    '''
    :param graph:
    :param id:
    :param sim_indexes:
    :return:

    adds an element to the graph
    '''

    graph[str(id)] = sim_indexes
    return


def createGraph(elements, thresholdCS, vectorizer):
    '''
    :param elements: sentences in all docs
    :param thresholdCS:
    :param vectorizer: vectorizer with fitted vocabulary
    :return: graph
    '''
    '''
    the graph is created by the following kind of element:
    where Si is sentence i
    id is index of the sentence in all_docs_sentences

    each graph element is:
        key: id of element Si, value: set that contains the sentences that link to sentence Si
        sentences link only if cosine_sim between them is higher than a certain threshold
    '''

    graph = {}
    for sentences in elements: # sentences is senteces in a document
        x = vectorizer.transform(sentences)
        for i in range(len(sentences)):
            x2 = vectorizer.transform([sentences[i]])
            indexes = cos_sims(x, x2, thresholdCS)
            addToGraph(graph, i, indexes)
    return graph