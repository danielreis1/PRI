from functions import *


def rank(graph, sentences_length, iterations=50): # rank for ex.1
    '''
    :param graph:
    :param iterations: number of iterations when ranking
    :param sentences_length: length of sentences in the document being ranked
    :return: dictionary with: 'sentence_number': rank , for all items in the graph
    '''

    rank_dict = {}
    page_rank = 0
    N = sentences_length

    # initialize rank_dict
    for i in graph:
        rank_dict[str(i)] = float(1/N)

    for i in range(iterations):
        for key in graph: # keys are sentences index in document
            rank_function(N, rank_dict, graph, key, d)

    return rank_dict

def rank_function(N, rank_dict, graph, key, d=0.15):
    '''
    :param N: number sentences
    :param rank_dict: dictionary with 'index': rank
    :param d:
    :return: return rank_dict
    '''

    sum = 0
    for i in graph[key][0]: # i are keys of dictionary - i is sentence number in a doc
        iter_key = str(i)
        pr = rank_dict[iter_key] #prev rank
        links = len(graph[str(iter_key)][0])
        sum += float(pr)/links

    rank_dict[key] = float(d)/N + (1-d) * sum

    return rank_dict


graph = {}

d = 0.15
all_docs = [] #lista com os documentos todos
all_docs_sentences = [] #lista com lista de frases de cada doc

all_docs, all_summaries = read_doc_file('ex1')

print ('all_docs')
print (all_docs)
print('')

all_docs_sentences = read_documents_into_sentence_tokens(all_docs)

print ('all_docs_sentences')
print (all_docs_sentences)
print('')

vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words=stopwords)
vectorizer.fit(all_docs)
graphs = createGraph(all_docs_sentences, all_docs, vectorizer, 0.2)
graph = graphs[0]

print 'graph'
print (graph)
print

maxD = 0
rankz = 0
best_ranked_sent = None
rank_dict = rank(graph, len(all_docs_sentences[0]), iterations=50)

for i in rank_dict:
    if (rank_dict[i] > rankz):
        rankz = rank_dict[i]
        best_ranked_sent = i

rankz = 0
print 'best_ranked_sent'
print best_ranked_sent
print
for di in range(1, 101, 1):
    d = float(di)/100
    #print d
    rank_dict = rank(graph, len(all_docs_sentences[0]), iterations=50)
    #print (rank_dict)
    if rank_dict[best_ranked_sent] > rankz:
        rankz = rank_dict[best_ranked_sent] # utilizando a frase melhor cotada
        maxD = d

print 'maxD'
print maxD
print

print 'comparing d=0.15 to maxD'
print
d = maxD
rank_dict = rank(graph, len(all_docs_sentences[0]), iterations=50)
print rank_dict

indexes = get_top5_from_dict(rank_dict)
indexes = [int(i) for i in indexes]
indexes.sort()
print(indexes)
print

d = 0.15
rank_dict = rank(graph, len(all_docs_sentences[0]), iterations=50)
print rank_dict

print
print

indexes = get_top5_from_dict(rank_dict)
indexes = [int(i) for i in indexes]
indexes.sort()

print(indexes)
for i in indexes:
    print
    print all_docs_sentences[0][i]

