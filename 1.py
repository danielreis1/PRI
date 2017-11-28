from functions import *
nltk.download('punkt')
nltk.download('stopwords')

graph = {} # grafo com edges igual a cosine similarities
'''
the graph is created by the following kind of element:
where Di is document i
id is index of 
each graph element is:
    key: id of element Di, value: set that contains the documents that link to document Di
    document links only if cosine_sim between them is higher than a certain threshold
'''

stopwords = nltk.corpus.stopwords.words('english')
all_docs = [] #lista com os documentos todos
all_docs_sentences = [] #lista com lista de frases de cada doc


all_docs = read_doc_file()
all_docs_sentences = read_documents_into_sentence_tokens(all_docs)

vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words = stopwords, analyzer = 'word')

x = vectorizer.fit_transform(all_docs_sentences)
x2 = vectorizer.transform()