from functions import *

def read_input(): # ex.1
    all_docs = []
    f = open('exercise1.txt', 'r')
    content = f.read()
    all_docs.append(content)

    return all_docs

def rank(graph, sentences_length,  iterations = 50): # rank for ex.1
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
            rank_function(N, rank_dict, graph, key)

    return rank_dict

def rank_function(N, rank_dict, graph, key, d=0.15):
    '''
    :param N:
    :param rank_dict:
    :param d:
    :return: return rank_dict
    '''

    sum = 0
    for i in graph[key]:
        iter_key = str(i)
        pr = rank_dict[iter_key] #prev rank
        links = len(graph[str(iter_key)])
        sum += float(pr/links)

    rank_dict[key] = float(d/N) + (1-d) * sum

    return rank_dict


graph = {}

all_docs = [] #lista com os documentos todos
all_docs_sentences = [] #lista com lista de frases de cada doc

all_docs = read_input()
all_docs_sentences = read_documents_into_sentence_tokens(all_docs)

'''
print ('all_docs')
print (all_docs)
print('')

print ('all_docs_sentences')
print (all_docs_sentences)
print('')
'''

vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words=stopwords)
vectorizer.fit(all_docs)
graphs = createGraph(all_docs_sentences, 0.2, vectorizer)
graph = graphs[0]
#print (graph)
rank_dict = rank(graph, len(all_docs_sentences[0]), iterations=50)
print (rank_dict)


indexes = get_top5_from_dict(rank_dict)
indexes.sort()
indexes = [int(i) for i in indexes]
#print(indexes)

for i in indexes:
    print all_docs_sentences[0][i]
