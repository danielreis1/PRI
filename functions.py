import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
        sentence_tokens = sentence_tokens[:-1]
        all_docs_sentences.append(sentence_tokens)
    return all_docs_sentences

def cosine_similarities(x, x2, thresholdCS = 0):
    '''
    :param x: sparseMatrix with all sentences transform
    :param x2: sparseM with 1 sentence transform
    :param thresholdCS: if value is provided returns only cosine_sims above the given threshold value
    :return: returns cosine_sims, all if no thresholdCS argument is provided
    '''
    cosine_sims = cosine_similarity(x,x2)
    for i in range(len(cosine_sims)):
        if i < thresholdCS:
            del cosine_sims[i]
    return cosine_sims

def addToGraph(graph, element, thresholdCS):
    '''
    args:
        graph where the element is added
        element to be added
        thresholdCS threshold cosine similarity
    adds an element to the graph
    when an element X is added the cosine similarity between it and all other elements is computed (elements yi),
    if the value of the cosine similarity is higher than the thresholdCS, then X is connected to Yi
    if element similarity is not above threshold with any element, it's dropped and not added to the graph
    '''


    return


