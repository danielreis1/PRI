from functions import *

all_docs = read_doc_file('TextoFonteComTitulo')
all_sentences = read_documents_into_sentence_tokens([all_docs[0]])

print (all_sentences)

# dicionario com as varias funcoes, depois for entre todas elas e representar cada uma diferente, {'nome_func': func}

vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words=stopwords)
vectorizer.fit(all_docs)


