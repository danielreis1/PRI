from functions import *


graph = {}

d = 0.15
all_docs = [] #lista com os documentos todos
all_docs_sentences = [] #lista com lista de frases de cada doc

all_docs, all_summaries = read_doc_file('ex1')
'''
print ('all_docs')
print (all_docs)
print('')
'''

all_docs_sentences = read_documents_into_sentence_tokens(all_docs)
'''
print ('all_docs_sentences')
print (all_docs_sentences)
print('')
'''

vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words=stopwords)
vectorizer.fit(all_docs)
graphs = createGraph(all_docs_sentences, all_docs, vectorizer, 0.2)
graph = graphs[0].copy()

print 'graph'
print (graph)
print

maxD = 0
rankz = 0
best_ranked_sent = None
rank_dict = rank(graph, len(all_docs_sentences[0]), d, iterations=50)

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
    rank_dict = rank(graph, len(all_docs_sentences[0]), d, iterations=50)
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
rank_dict = rank(graph, len(all_docs_sentences[0]), d, iterations=50)
print rank_dict

indexes = get_top5_from_dict(rank_dict)
indexes = [int(i) for i in indexes]
indexes.sort()
print(indexes)
print

d = 0.15
rank_dict = rank(graph, len(all_docs_sentences[0]), d, iterations=50)
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

