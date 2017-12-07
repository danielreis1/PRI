import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from nltk.corpus import mac_morpho
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import UnigramTagger as ut
from nltk import BigramTagger as bt
from cPickle import dump,load
import numpy as np
import re


nltk.download('punkt')
nltk.download('stopwords')
# global vars
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwordsE = nltk.corpus.stopwords.words('english')
tagger = ''


def read_doc_file(path_1, path_2=None):
    '''
    ler todos os textos-fonte com titulo
    '''

    all_docs = []
    all_summaries = []
    all_files = os.listdir(path_1)
    content = ''
    for fl in all_files:
        with open(path_1 + '/' + fl) as f1:
            for line in f1:
                line = line.decode('cp1252')
                line = line.lower()
                if line.strip() == '':
                    continue
                if '.' not in line:
                    line = line + '.'
                content += line
        all_docs.append(content)
        if path_2 is not None:
            with open(path_2 + '/' + fl) as f2:
                for line in f2:
                    line = line.decode('cp1252')
                    line = line.lower()
                    if line.strip() == '':
                        continue
                    if '.' not in line:
                        line = line + '.'
                    content += line
            all_summaries.append(content)
    return all_docs, all_summaries


def read_documents_into_sentence_tokens(all_docs):
    '''
    divide cada doc em lista de frases
    lista com listas(estas listas sao tokens de frase)
    '''
    all_sentences = []
    for i in range(len(all_docs)): # i is a doc index
        sentence_tokens = re.split(r'[!?.]', all_docs[i])
        sentence_tokens = clean_sentences(sentence_tokens)
        all_sentences.append(sentence_tokens)
    return all_sentences


def clean_sentences(sentences):
    ret_sentences = []
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
        if sentences[i] == '':  # checks for empty string
            continue
        ret_sentences.append(sentences[i])
    return ret_sentences


def cos_sims(x, x2, sentences, doc, self_index, thresholdCS, np):
    '''
    :param x: sparseMatrix with doc transform
    :param x2: sparseM with 1 sentence transform
    :param sentences: sent's in doc
    :param self_index: index of phrase being analyzed
    :param np: noun phrases of sentence being analyzed
    :param thresholdCS: if value is provided returns only cosine_sims above the given threshold value
    :return: {'id':[cosine_sim, noun_phrases, SVD_sims]}
    returns list with indexes of the phrases with a cosine similarity above threshold
    filters cosine_sims list
    '''

    conected = {}
    cosine_sims = cosine_similarity(x, x2)
    for i in range(len(cosine_sims)):
        number_NP = 0
        if i != self_index:
            if cosine_sims[i][0] > thresholdCS:
                #cosine sims
                conected[str(i)] = [cosine_sims[i][0]]

                #noun P
                np2 = getNP_from_sent(sentences[i])
                for n in np2:
                    if n in np:
                        number_NP+=1
                conected[str(i)].append(number_NP)

                #svd
                svd_num = svd_ew(sentences, self_index, i)
                conected[str(i)].append(svd_num)

    return conected.copy()


def svd_ew(sentences, sent1_index, sent2_index):
    '''

    :param sentences: all sentences in doc
    :param sent1_index: sentence being analyzed
    :param sent2_index: sentence(link) of the sentence being analyzed
    :return:
    '''

    vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True,
                                 stop_words=stopwords)
    sparse_m = vectorizer.fit_transform(sentences)  # .toarray() # term frequency in doc

    svd = TruncatedSVD(n_components=100, n_iter=5)
    dense_m = svd.fit_transform(sparse_m)

    sent1_trans = sparse.csr_matrix(dense_m[sent1_index])
    sent2_trans = sparse.csr_matrix(dense_m[sent2_index])

    sim = cosine_similarity(sent1_trans, sent2_trans)
    sent_doc_similarity_prior = sim[0][0]

    return sent_doc_similarity_prior


def addToGraph(graph, id, vals, doc, sentences):
    '''
    :param graph:
    :param id: index da frase na lista de frases
    :param vals: valores do grafo, pode ser uma lista com features ou uma feature so:
      ex: indexes na lista de frases cujas frases ultrapassam a threshold de semelhanca
    :return:
    adds an element to the graph
    '''

    sent = sentences[int(id)]

    #
    vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True,
                                 stop_words=stopwords)
    x = vectorizer.fit_transform([doc])
    x2 = vectorizer.transform([sent])
    sim = cosine_similarity(x, x2)
    sent_doc_similarity_prior = sim[0][0]

    #
    position_doc_prior = float(1) / (int(id) + 1)

    #
    naive_bayes_prior = bayes_pr(doc, sent)

    graph[str(id)] = [vals.copy(), position_doc_prior, sent_doc_similarity_prior, naive_bayes_prior]
    return


def bayes_pr(doc, sent):
    '''

    :param doc: document from which term frequencies are taken
    :param sent: sentence that tells which terms are to be taken
    :return: returns bayes
    '''

    vectorizer = CountVectorizer(analyzer='word', stop_words=stopwords)
    x = vectorizer.fit_transform([doc]).toarray()
    tokenize_doc = vectorizer.get_feature_names()

    vectorizer2 = CountVectorizer(analyzer='word', stop_words=stopwords)
    try:
        vectorizer2.fit([sent])
        tokenize_phrase = vectorizer2.get_feature_names()
    except Exception:
        print 'empty phrase'
        tokenize_phrase = []

    N = len(tokenize_doc)


    bayes = [1]
    tokenize_phrase = [i.strip().lower() for i in tokenize_phrase]
    for term_sent_index in range(len(tokenize_doc)):
        term_sent = tokenize_doc[term_sent_index]
        if term_sent.strip().lower() in tokenize_phrase:
            temp_var = x[0][term_sent_index]
            if temp_var == 0:
                raise Exception('ERROR in bayes')
            bayes.append(float(temp_var)/ N)

    naive_bayes_prior = bayes[0]
    for i in range(len(bayes)):
        naive_bayes_prior = naive_bayes_prior * i
    return naive_bayes_prior


def createGraph(elements, docs, vectorizer, thresholdCS=0.2):
    '''
    :param algs: [0] -> bayes on/off [1] -> svd on/off
    :param elements: sentences in all docs
    :param thresholdCS:
    :param vectorizer: vectorizer with fitted vocabulary
    :return: list with graphs - each graph 1 document
    '''
    '''
    the graph is created by the following kind of element:
    where Si is sentence i
    id is index of the sentence in all_docs_sentences

    each graph element is:
        key: id of element Si, value: set that contains the sentences that link to sentence Si
        sentences link only if cosine_sim between them is higher than a certain threshold
    '''
    graphs = []
    for sentences_index in range(len(elements)): # sentences variable is senteces in a document
        sentences = elements[sentences_index]
        doc = docs[sentences_index]
        graph = {}
        x = vectorizer.transform(sentences)
        for i in range(len(sentences)):
            sent = sentences[i]
            x2 = vectorizer.transform([sent])
            nP = getNP_from_sent(sent)
            indexes = cos_sims(x, x2, sentences, doc, i, thresholdCS, nP)
            addToGraph(graph, i, indexes, doc, sentences)
        graphs.append(graph.copy())
        print 'graph'
    return graphs


def get_top5_from_dict(D):
    '''
    :param D: dictionary
    :return: sorted dictionary
    '''
    sort = sorted(D, key = D.get, reverse=True)[:5]
    #print(sort)
    return sort


def PR_sentences_pos(graph_list):

    indexes, pos, cos_sim, bayes = graph_list

    return pos


def PR_TFIDF(graph_list):

    indexes, pos, cos_sim, bayes = graph_list

    return cos_sim


def PR_prob_NaiveBayes(graph_list):

    indexes, pos, cos_sim, bayes = graph_list

    return bayes


def EW_TFIDF(graph, s1, s2):
    '''
    :param doc: doc in graph corresponds to a graph[int(index)]
    :param s1: index sentence 1
    :param s2: ***
    :return: TFIDF between both sentences in the doc
    '''

    return graph[str(s1)][0][str(s2)][0]


def EW_nounPhrases(graph, s1, s2):
    '''

    :param graph:
    :param s1: sentence_index
    :param s2: sentence_index
    :return:
    '''

    return graph[str(s1)][0][str(s2)][1]


def EW_SVD(graph, s1, s2):
    '''

    :param graph:
    :param s1:
    :param s2:
    :return:
    '''

    return graph[str(s1)][0][str(s2)][2]


def getNPs(tagList):
    voc = []
    nounP= ''
    for i in tagList:
        if(i[2] == 'O'):
            # tira a parte menos interessante da frase
            if(i[1] == '.'):
                if(nounP != ''):
                    voc.append(nounP)
                    nounP = ''
            continue;

        elif (i[2][0] == 'I'):
            nounP += i[0] + ' '

        elif (i[2][0] == 'B'):
            if(nounP != ''):
                voc.append(nounP)
            nounP = ''
            nounP += i[0] + ' '

    voc = set(voc) # remove duplicates
    vocabulary = [i for i in voc]
    return vocabulary

def loadtagger(taggerfilename):
    infile = open(taggerfilename, 'rb')
    tagger = load(infile);
    infile.close()
    return tagger


def processContent(content):
    voc = []
    # load tagger
    tagger = loadtagger('biTagger')
    for sentences in content:
        for sentence in sentences:
            item = sentence
            tokenized = nltk.word_tokenize(item) # word tokens
            tagged = tagger.tag(tokenized)
            chunkGram = r"NP: {((<ADJ>* <N.*>+ <PREP>)? <ADJ>* <N.*>+)+}"
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            col_tags = tree2conlltags(chunked)
            voc += getNPs(col_tags)
    return voc


def getNP_from_sent(item):
    '''

    :param item: sentence
    :return: sentence tags
    '''

    tokenized = nltk.word_tokenize(item)  # word tokens
    tagged = tagger.tag(tokenized)
    chunkGram = r"NP: {((<ADJ>* <N.*>+ <PREP>)? <ADJ>* <N.*>+)+}"
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    col_tags = tree2conlltags(chunked)
    return getNPs(col_tags)

def trainAndSaveTagger():
    mac_morpho_sentences = mac_morpho.tagged_sents()
    taggerNatural = nltk.DefaultTagger('N')
    un_tag = ut(mac_morpho_sentences,backoff=taggerNatural)
    outfile = open('uniTagger','wb')
    dump(un_tag,outfile,-1)
    outfile.close()
    bi_tag = bt(mac_morpho_sentences,backoff=taggerNatural)
    outfile2 = open('biTagger','wb')
    dump(bi_tag,outfile2,-1)
    outfile2.close()


def rank(graph, sentences_length, d, iterations=50): # rank for ex.1
    '''
    :param graph:
    :param iterations: number of iterations when ranking
    :param sentences_length: length of sentences in the document being ranked
    :return: dictionary with: 'sentence_number': rank , for all items in the graph
    '''

    rank_dict = {}
    N = sentences_length

    # initialize rank_dict
    for i in graph:
        rank_dict[str(i)] = float(1)/N

    for i in range(iterations):
        for key in graph: # keys are sentences index in document
            rank_function(N, rank_dict, graph, key, d)

    return rank_dict.copy()

def rank_function(N, rank_dict, graph, key, d=0.15):
    '''
    :param N: number sentences
    :param rank_dict: dictionary with 'index': rank
    :param d:
    :return: return rank_dict
    '''

    sum = 0
    for i in graph[key][0]: # i are keys of dictionary - i is sentence number in a doc
        iter_key = str(i)
        pr = rank_dict[iter_key] #prev rank
        links = len(graph[str(iter_key)][0])
        sum += float(pr)/links

    rank_dict[key] = float(d)/N + (1-d) * sum

    return rank_dict


#trainAndSaveTagger()
tagger = loadtagger('biTagger')