from functions import *

all_docs, all_summaries = read_doc_file('TextoFonteComTitulo')
all_docs_sentences = read_documents_into_sentence_tokens(all_docs)

#print (all_sentences)

#2 dicionarios com as varias funcoes, depois for entre todas elas e representar cada uma diferente, {'nome_func': func}

PR = {'tfidf': PR_EW_TFIDF} #'sentence_pos':PR_sentences_pos,
EW = {'tfidf':PR_EW_TFIDF}
vectorizer = 0


def getPR(func_name):
    return PR[func_name]


def getEW(func_name):
    return EW[func_name]


def rank(graphs, iterations=50):
    '''
    :param graphs: all graphs
    :param iterations:
    :return: returns list (each index is a document) with lists(each index is a sentence)
    ranks all docs
    '''

    rank_dicts = []
    function_ranks = []

    for prior in PR:
        for eWeight in EW:
            rank_dicts = []
            for doc_number in graphs:
                rank = rank_doc(graphs[doc_number], prior, eWeight, iterations, doc_number)
                rank_dicts.append(rank.copy())
            function_ranks.append(rank_dicts.copy())
    return function_ranks


def rank_doc(graph, prior, eWeight, iterations, doc_number):
    rank_dict = {}
    N = len(all_docs_sentences)

    for i in graph:
        rank_dict[str(i)] = float(1/N)

    for i in range(iterations):
        for key in graph:
            doc = all_docs_sentences[doc_number]
            rank_function(rank_dict, int(key), doc_number, graph, prior, eWeight, doc)
    return PR


def rank_function(rank_dict, sent_number, doc_number, graph, prior, eWeight, doc, d=0.15):
    sent = doc[sent_number]
    prior = getPR(prior)(doc, sent)
    sumPrior = 0

    for i in graph: # sums over all elements in the doc (graph)
        sumPrior += getPR(prior)(doc, doc[int(i)])

    sume = 0
    sumLinks = 0
    sumLinkWeights = 0

    for link in graph[str(sent_number)]: # link is an edge
        PR = rank_dict[sent_number]
        weight = getEW(eWeight)(sent, doc[link])
        for link_of_link in graph[str(link)]:
            sumLinkWeights += getEW(eWeight)(doc[link], )
        sume += float(PR * weight / sumLinkWeights)

    PR = float(d * prior / sumPrior) + (1 - d) * sume
    return PR


vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words=stopwords)
vectorizer.fit(all_docs)

graphs = createGraph(all_docs_sentences, vectorizer)
#print graphs

ranks = rank(graphs)
print ranks

