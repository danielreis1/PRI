from functions import *

def read_input():
    all_docs = []
    f = open('exercise1.txt', 'r')
    content = f.read()
    all_docs.append(content)

    return all_docs

graph = {} # grafo com edges igual a cosine similarities

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

vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True,stop_words=stopwords)
vectorizer.fit(all_docs)
graph = createGraph(all_docs_sentences, 0.2, vectorizer)

print (graph)

