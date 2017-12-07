from functions import *


all_docs, all_summaries = read_doc_file('ex4')
all_docs_sentences = read_documents_into_sentence_tokens(all_docs)
d = 0.15


vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words=stopwordsE)
vectorizer.fit(all_docs)
graphs = createGraph(all_docs_sentences, all_docs, vectorizer, 0.2)

graph = graphs[0].copy()
saveGraph(graph, '4')
#graph = loadtagger('4.p')

rank_dict = rank(graph, len(all_docs_sentences[0]), d, iterations=50)


indexes = get_top5_from_dict(rank_dict)
indexes = [int(i) for i in indexes]
indexes.sort()

print(indexes)
for i in indexes:
    print
    print all_docs_sentences[0][i]

