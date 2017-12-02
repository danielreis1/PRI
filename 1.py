from functions import *


def read_input():
    all_docs = []
    f = open('exercise1.txt', 'r')
    content = f.read()
    all_docs.append(content)

    return all_docs

nltk.download('punkt')
graph = {} # grafo com edges igual a cosine similarities
'''
the graph is created by the following kind of element:
where Di is document i
id is index of 
each graph element is:
    key: id of element Di, value: set that contains the documents that link to document Di
    document links only if cosine_sim between them is higher than a certain threshold
'''

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
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


vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words = stopwords)

x = vectorizer.fit_transform(all_docs_sentences)

for i in all_docs_sentences:
    x2 = vectorizer.transform(i)

