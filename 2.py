from functions import *


all_docs, all_summaries = read_doc_file('textoFontesTests', path_2='SummariesTests')
all_docs_sentences = read_documents_into_sentence_tokens(all_docs)
all_summaries_sentences = read_documents_into_sentence_tokens(all_summaries)
#print all_docs_sentences
#print (all_sentences)

#2 dicionarios com as varias funcoes, depois for entre todas elas e representar cada uma diferente, {'nome_func': func}

PR = {'tfidf': PR_TFIDF, 'sentences_pos': PR_sentences_pos, 'bayes': PR_prob_NaiveBayes} #'sentence_pos':PR_sentences_pos,
EW = {'tfidf': EW_TFIDF, 'nounP':EW_nounPhrases, 'svd': EW_SVD}
vectorizer = 0
functions = []

#print noun_phrases


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

    function_ranks = []

    for prior in PR:
        for eWeight in EW:
            functions.append((prior, eWeight))
            rank_dicts = []
            cnt = 0
            for doc_number in range(len(graphs)):
                cnt += 1
                print 'doc ' + str(cnt)
                rankD = rank_doc(graphs[doc_number], prior, eWeight, iterations, doc_number)
                rank_dicts.append(rankD.copy())
            function_ranks.append(rank_dicts[:])
    return function_ranks[:]


def rank_doc(graph, prior, eWeight, iterations, doc_number):
    rank_dict = {}
    N = len(all_docs_sentences[doc_number])

    for i in graph:
        rank_dict[str(i)] = float(1)/N
        #print i

    for i in range(iterations):
        for key in graph:
            doc = all_docs[doc_number]
            doc_sentences = all_docs_sentences[doc_number]
            rank_function(rank_dict, int(key), graph, prior, eWeight, doc, doc_sentences)
    return rank_dict.copy()


def rank_function(rank_dict, sent_number, graph, prior_func, eWeight, doc, doc_sentences, d=0.15):

    prior = getPR(prior_func)(graph[str(sent_number)])
    #print prior

    sumPrior = 0
    if prior != 0:
        for i in graph: # sums over all elements in the doc (graph)
            sumPrior += getPR(prior_func)(graph[i])
    else:
        sumPrior = 1

    sume = 0

    for link in graph[str(sent_number)][0]: # link is an edge
        link = int(link)
        PR = rank_dict[str(sent_number)]
        link_sent = doc_sentences[link]
        weight = getEW(eWeight)(graph, link, sent_number)
        sumLinkWeights = 0
        if weight != 0:
            for link_of_link in graph[str(link)][0]:
                link_of_link = int(link_of_link)
                sumLinkWeights += getEW(eWeight)(graph, link, link_of_link)
        else:
            sumLinkWeights = 1

        sume += float(PR) * weight / sumLinkWeights

    rank_dict[sent_number] = float(d) * prior / sumPrior + (1 - d) * sume


vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words=stopwords)
vectorizer.fit(all_docs)

print 'graphs werent save'
print
#graphs = createGraph(all_docs_sentences, all_docs, vectorizer)
#saveGraph(graphs, '2')
#print 'graph saved'


graphs = loadtagger('2.p')

for i in graphs:
    print
    print i

ranks = rank(graphs)
print ranks

mySummary = []

for funcs_index in range(len(ranks)):
    print functions[funcs_index]
    for i_doc in range(len(ranks[funcs_index])):
        #for d in ranks[funcs_index][i_doc]:
        dict = ranks[funcs_index][i_doc]
        print dict
        print
        indexes = get_top5_from_dict(dict)
        indexes = [int(i) for i in indexes]
        indexes.sort()
        myS = []
        for i_sent in indexes:
            print all_docs_sentences[int(i_doc)][int(i_sent)]
            myS.append(all_docs_sentences[int(i_doc)][int(i_sent)])
        mySummary.append(myS[:])

mapValues = []

for mySumDoci in range(len(mySummary)):
    mysum = mySummary[mySumDoci]
    for mS in mySummary[mySumDoci]:
        mapVals = []
        print 'mS'
        print mS
        print
        if mS in all_summaries_sentences: #error aqui
            mapVals.append(1)
        else:
            mapVals.append(0)
    mapValues.append(mapVals[:])

cnt = 0
for mapV in mapValues:
    print 'map for doc number: ' + str(cnt)
    print get_map(mapV)
    print
    cnt += 1

'''
existe um erro ao testar o map
'''