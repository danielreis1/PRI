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

def read_doc_file(path):
    '''
    ler todos os textos-fonte com titulo
    '''
    all_docs = []
    all_files = os.listdir(path)
    for fl in all_files:
        f = open(path + '/' + fl, 'r')
        content = f.readline().decode('cp1252') + '.'  # primeira frase e' o titulo
        content += f.read().decode('cp1252')
        all_docs.append(content)
    return all_docs

def read_documents_into_sentence_tokens(all_docs):
    '''
    divide cada doc em lista de frases
    lista com listas(estas listas sao tokens de frase)
    '''
    all_sentences = []
    for i in range(len(all_docs)): # i is a doc
        sentence_tokens = all_docs[i].split('.')
        sentence_tokens = clean_sentences(sentence_tokens)
        all_sentences.append(sentence_tokens)
    return all_sentences

def clean_sentences(sentences):
    ret_sentences = []
    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace('\n', '')
        sentences[i] = sentences[i].replace('\r', '')
        sentences[i] = sentences[i].replace('\t', '')
        if sentences[i] == '':  # checks for empty string
            continue;
        ret_sentences.append(sentences[i])
    return ret_sentences


def cos_sims(x, x2, thresholdCS=0):
    '''
    :param x: sparseMatrix with all sentences transform
    :param x2: sparseM with 1 sentence transform
    :param thresholdCS: if value is provided returns only cosine_sims above the given threshold value
    :return: returns list with indexes of the phrases with a cosine similarity above threshold
    filters cosine_sims list
    '''

    conected = []
    cosine_sims = cosine_similarity(x, x2)
    for i in range(len(cosine_sims)):
        if cosine_sims[i] > thresholdCS:
            conected.append(i) # apends index of the sentence in the sentence list
    return conected

def addToGraph(graph, id, vals):
    '''
    :param graph:
    :param id: index da frase na lista de frases
    :param vals: valores do grafo, pode ser uma lista com features ou uma feature so:
      ex: indexes na lista de frases cujas frases ultrapassam a threshold de semelhanca
    :return:
    adds an element to the graph
    '''

    graph[str(id)] = vals
    return

def createGraph(elements, thresholdCS, vectorizer):
    '''
    :param elements: sentences in all docs
    :param thresholdCS:
    :param vectorizer: vectorizer with fitted vocabulary
    :return: list with graphs - each graph 1 document
    '''
    '''
    the graph is created by the following kind of element:
    where Si is sentence i
    id is index of the sentence in all_docs_sentences

    each graph element is:
        key: id of element Si, value: set that contains the sentences that link to sentence Si
        sentences link only if cosine_sim between them is higher than a certain threshold
    '''
    graphs = []
    graph = {}
    for sentences in elements: # sentences variable is senteces in a document
        graph = {}
        x = vectorizer.transform(sentences)
        for i in range(len(sentences)):
            x2 = vectorizer.transform([sentences[i]])
            indexes = cos_sims(x, x2, thresholdCS)
            addToGraph(graph, i, indexes)
        graphs.append(graph.copy())
    return graphs

def get_top5_from_dict(D):
    sort = sorted(D, key = D.get, reverse=True)[:5]
    #print(sort)
    return sort

def PR_sentences_pos(doc_senteces):
    '''
    :param doc_senteces: sentences in a document
    :return: {'index': value}
    '''
    ret = []
    N = len(doc_senteces)
    for i in range(len(doc_senteces)):
        ret[str(i)] = 1/N
    return  ret

def PR_EW_TFIDF(vectorizer):
    '''
    :param x: vectorizer transform for element 1 
    :param x2: same as x for element 2
    :return: dict with 'index': TF - IDF
    '''
    '''
    works for edges and prior
    '''

    x = vectorizer.transform()
    x2 = vectorizer.transform()


    return cosine_similarity(x, x2)

def PR_prob_NaiveBayes():


    return

def EW_nounPhrases():
    return

def EW_SVD():
    '''

    :return:
    '''
    '''
        
    '''


    return


