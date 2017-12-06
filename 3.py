from functions import *
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Perceptron



def check_summaries(grupo, doc, frase):
    if all_docs_sentences[grupo][doc][frase] in all_summaries_sentences[grupo][doc]:
        return 1
    else:
        return 0

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

def read_documents_into_sentences(all_docs):
    '''
    divide cada doc em lista de frases
    lista com listas(estas listas sao tokens de frase)
    '''
    all_sentences = []
    for i in range(len(all_docs)): # i is a doc
        sentence_tokens = all_docs[i].split('. ')
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


vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, analyzer = 'word')
groups = ['Brasil' ,'Cotidiano', 'Dinheiro', 'Especial', 'Mundo', 'Opiniao', 'Tudo']
all_docs = {}
all_summaries = {}

X_train = []
y_train = []



for i in range(len(groups)):
    read_doc_files(groups[i])

all_docs_sentences = copy.deepcopy(all_docs)
all_summaries_sentences = copy.deepcopy(all_summaries)


for key in all_summaries:
    all_summaries_sentences[key] = read_summaries_into_sentences(all_summaries[key])

def get_max_length():
    val = 0
    for key in all_docs_sentences:
        for i in range(len(all_docs_sentences[key])):
            if len(all_docs_sentences[key][i])>val:
                val = len(all_docs_sentences[key][i])
    return val

for key in all_docs:
    cosines = []
    maxLength = get_max_length()
    all_docs_sentences[key] = read_documents_into_sentences(all_docs[key])
    for i in range(len(all_docs[key])):
        x = vectorizer.fit_transform(all_docs_sentences[key][i])
        for j in range(len(all_docs_sentences[key][i])):
            sentence = all_docs_sentences[key][i][j]
            x2 = vectorizer.transform([sentence])
            y_train.append(check_summaries(key, i, j))
            X_train.append([i, cosine_similarity(x, x2)[j]])


ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train,y_train)


y_pred = ppn.predict(X_train)
print('Misclassified samples: %d' % (y_train != y_pred).sum())




