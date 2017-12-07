
from __future__ import division
from functions import *
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Perceptron
from scipy import sparse
from sklearn.metrics import accuracy_score
import networkx as nx
import matplotlib.pyplot as plt
import pickle

vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, analyzer = 'word')

def set_test_graph(doc):
    G = nx.Graph()
    for i in range(len(all_docs_test_sentences[doc])):
        G.add_node(i)
    return G

def set_train_graph(key, doc):
    G = nx.Graph()
    for i in range(len(all_docs_sentences[key][doc])):
        G.add_node(i)
    return G


#########################################################################
X_train = []
y_train = []
X_test = []
y_test = []

#########################################################################
def check_summaries(grupo, doc, frase):
    if all_docs_sentences[grupo][doc][frase] in all_summaries_sentences[grupo][doc]:
        return 1
    else:
        return 0

def check_summaries_test(doc, frase):
    if all_docs_test_sentences[doc][frase] in all_summaries_test_sentences[doc]:
        return 1
    else:
        return 0

#########################################################################
def get_map(values):
    count = 0
    soma = 0
    for i in range(len(values)):
        if values[i] == 1:
            count = count + 1
            soma = soma + (count/(i+1))
    if soma == 0:
        return 0
    return(soma/count) 

#########################################################################
def read_doc_files(categoria):
    all_docs[categoria] = []
    all_summaries[categoria] = []
    all_files = os.listdir('Textos/Originais/' + categoria)
    for fl in all_files:
        f1 = open('Textos/Originais/' + categoria + '/' + fl, 'r')
        f2 = open('Textos/Sumarios/' + categoria + '/Sum-' + fl, 'r')
        content1 = f1.read().decode('cp1252')
        content2 = f2.read().decode('cp1252')
        all_docs[categoria].append(content1)
        all_summaries[categoria].append(content2)

def read_doc_test_files():
    all_files = os.listdir('TextoFonteComTitulo')
    for fl in all_files:
        f1 = open('TextoFonteComTitulo/' + fl, 'r')
        f2 = open('Summaries/' + 'Ext-' + fl, 'r')
        content1 = f1.read().decode('cp1252')
        content2 = f2.read().decode('cp1252')
        all_docs_test.append(content1)
        all_summaries_test.append(content2)
#########################################################################
def read_documents_into_sentences(all_docs):
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

def read_summaries_into_sentences(all_docs):
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

#########################################################################
def get_cosine(x,x2):
    return cosine_similarity(x, x2)[0][0]

#########################################################################
def get_train_data():
    global X_train
    print 'getting train data'
    '''
    Cria a train data e a target data para treinar o perceptron (so usa posicao no texto e coseno)
    '''
    for key in all_docs_sentences:
        print 'on key: ' + str(key)
        all_docs_sentences[key] = read_documents_into_sentences(all_docs[key])
        for i in range(len(all_docs[key])):
            print 'doc: ' + str(i)
            x = vectorizer.fit_transform(all_docs_sentences[key][i])
            G = set_train_graph(key, i)
            ValsX = [0 for a in range(len(all_docs_sentences[key][i]))]
            for j in range(len(all_docs_sentences[key][i])):
                sentence = all_docs_sentences[key][i][j]
                x2 = vectorizer.transform([sentence])
                nb = bayes_pr(all_docs[key][i], all_docs_sentences[key][i][j])
                cosine = get_cosine(x,x2)
                G.add_node(j)
                if cosine > 0.2:
                    for k in range(len(all_docs_sentences[key][i])):
                        G.add_edge(j, k)
                y_train.append(check_summaries(key, i, j))
                ValsX[j] = [j, cosine, nb]
            for j in range(len(all_docs_sentences[key][i])):
                bc = nx.betweenness_centrality(G)
                ValsX[j] = ValsX[j] + [bc[j]]
            X_train = X_train + ValsX


def get_test_data():
    global X_test
    print 'getting test data'
    '''
    Cria a test data e a target data
    '''
    for i in range(len(all_docs_test)):
        print 'doc: ' + str(i)
        all_docs_test_sentences = read_documents_into_sentences(all_docs_test)
        x = vectorizer.fit_transform(all_docs_test_sentences[i])
        G = set_test_graph(i)
        ValsX = [[] for a in range(len(all_docs_test_sentences[i]))]
        for j in range(len(all_docs_test_sentences[i])):
            sentence = all_docs_test_sentences[i][j]
            x2 = vectorizer.transform([sentence])
            nb = bayes_pr(all_docs_test[i], all_docs_test_sentences[i][j])
            cosine = get_cosine(x,x2)
            G.add_node(j)
            if cosine > 0.2:
                for k in range(len(all_docs_test_sentences[i])):
                    G.add_edge(j, k)
            y_test.append(check_summaries_test(i, j))
            ValsX[j] = [j, cosine, nb]
        for j in range(len(all_docs_test_sentences[i])):
            bc = nx.betweenness_centrality(G)
            ValsX[j] = ValsX[j] + [bc[j]]
        X_test = X_test + ValsX


#########################################################################
all_docs_test = []
all_summaries_test = []
read_doc_test_files() 
all_docs_test_sentences = read_documents_into_sentences(all_docs_test)
all_summaries_test_sentences = read_summaries_into_sentences(all_summaries_test)

#########################################################################
groups = ['Brasil' ,'Cotidiano', 'Dinheiro', 'Especial', 'Mundo', 'Opiniao', 'Tudo']
all_docs = {}
all_summaries = {}

#########################################################################
def get_top_sentences():
    print 'getting to sentences'
    first = 0
    for i in range(len(all_docs_test_sentences)):
        avg_prec = []
        this_doc = y_pred[first:(first + len(all_docs_test_sentences[i]))]
        top5 = sorted(range(len(this_doc)), key=lambda i: this_doc[i])[-5:]
        print 'doc: ' + str(i)
        for j in top5:
            print '    ' + all_docs_test_sentences[i][j]
            avg_prec.append(check_summaries_test(i,j))
        first = first + len(all_docs_test_sentences[i])
        print 'map: ' + str(get_map(avg_prec))
        print'--------------------------------------------'

#########################################################################
for i in range(len(groups)):
    read_doc_files(groups[i])

all_docs_sentences = copy.deepcopy(all_docs)
all_summaries_sentences = copy.deepcopy(all_summaries)

for key in all_summaries:
    all_summaries_sentences[key] = read_summaries_into_sentences(all_summaries[key])


#print X_train
get_test_data()


try:
    print ('loading pickle')
    ppn = pickle.load(open("perceptron.p", "rb"))
except (OSError, IOError) as e:
    get_train_data()
    print 'creating perceptron'
    ppn = Perceptron(max_iter=40, random_state=0)
    ppn.fit(X_train,y_train)
    print ('dumping pickle')
    pickle.dump( ppn, open( "perceptron.p", "wb" ) )

print ('predicting...')
y_pred = ppn.decision_function(X_test)

get_top_sentences()





