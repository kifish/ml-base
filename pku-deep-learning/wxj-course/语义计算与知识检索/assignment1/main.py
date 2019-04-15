# coding: utf-8
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import word2vec, keyedvectors
from scipy import stats
import numpy as np
from nltk.corpus import wordnet_ic
import nltk
from nltk.corpus import wordnet as wn
import logging
logging.basicConfig(level=logging.INFO)

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

test_data = []
path = './test_data.txt'
with open(path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        word1, word2, sim = line.split(',')  # sim的值为1-5之间
        sim = float(sim)
        test_data.append((word1, word2, sim))

def cal_lch_sim_based_wordnet(test_data,use_max = True):
    result = []
    for sample in test_data:
        sims = []
        sim = -1
        word1, word2, _ = sample
        word1_synsets = wn.synsets(word1)
        word2_synsets = wn.synsets(word2)
        for word1_synset in word1_synsets:
            for word2_synset in word2_synsets:
                if word1_synset.pos() == word2_synset.pos():  # 词性相同
                    try:
                        val = word1_synset.lch_similarity(word2_synset)
                        if val != None:
                            sims.append(val)
                    except Exception as e:
                        pass
#                         print(word1_synset,'\t',word2_synset)
#                         print(e)
        if use_max:
            sim = max(sims)
        else: #use_avg
            sim = sum(sims) / len(sims)
        result.append((word1, word2, sim))
    sims = list(map(lambda x: x[2], result))
    print('lch similarity:')
    print('mean:',np.mean(sims))
    print('std:',np.std(sims))
    return result


def cal_path_sim_based_wordnet(test_data,use_max = True):
    result = []
    for sample in test_data:
        sim = -1
        sims = []
        word1, word2, _ = sample
        word1_synsets = wn.synsets(word1)
        word2_synsets = wn.synsets(word2)
        for word1_synset in word1_synsets:
            for word2_synset in word2_synsets:
                if word1_synset.pos() == word2_synset.pos():  # 词性相同
                    try:
                        val = word1_synset.path_similarity(word2_synset)
                        if val != None:
                            sims.append(val)
                    except Exception as e:
                        pass
#                         print(word1_synset,'\t',word2_synset)
#                         print(e)
        if use_max:
            sim = max(sims)
        else:  # use_avg
            sim = sum(sims) / len(sims)
        result.append((word1, word2, sim))
    sims = list(map(lambda x: x[2], result))
    print('path similarity:')
    print('mean:', np.mean(sims))
    print('std:', np.std(sims))
    return result

def cal_wup_sim_based_wordnet(test_data,use_max = True):
    result = []
    for sample in test_data:
        sim = -1
        sims = []
        word1, word2, _ = sample
        word1_synsets = wn.synsets(word1)
        word2_synsets = wn.synsets(word2)
        for word1_synset in word1_synsets:
            for word2_synset in word2_synsets:
                if word1_synset.pos() == word2_synset.pos():  # 词性相同
                    try:
                        val = word1_synset.wup_similarity(word2_synset)
                        if val != None:
                            sims.append(val)
                    except Exception as e:
                        pass
#                         print(word1_synset,'\t',word2_synset)
#                         print(e)
        if use_max:
            sim = max(sims)
        else:  # use_avg
            sim = sum(sims) / len(sims)
        result.append((word1, word2, sim))
    sims = list(map(lambda x: x[2], result))
    print('wup similarity:')
    print('mean:', np.mean(sims))
    print('std:', np.std(sims))
    return result

def cal_res_sim_based_wordnet(test_data,use_max = True):
    result = []
    for sample in test_data:
        sim = -1
        sims = []
        word1, word2, _ = sample
        word1_synsets = wn.synsets(word1)
        word2_synsets = wn.synsets(word2)
        for word1_synset in word1_synsets:
            for word2_synset in word2_synsets:
                if word1_synset.pos() == word2_synset.pos():  # 词性相同
                    try:
                        val = word1_synset.res_similarity(
                            word2_synset, semcor_ic)
                        if val != None:
                            sims.append(val)
                    except Exception as e:
                        pass
#                         print(word1_synset,'\t',word2_synset)
#                         print(e)
        if len(sims) == 0:
            sims.append(-1) 
        if use_max:
            sim = max(sims)
        else:  # use_avg
            sim = sum(sims) / len(sims)
        result.append((word1, word2, sim))
    sims = list(map(lambda x: x[2], result))
    print('res similarity:')
    print('mean:', np.mean(sims))
    print('std:', np.std(sims))
    return result


def cal_jcn_sim_based_wordnet(test_data,use_max = True):
    result = []
    for sample in test_data:
        sim = -1
        sims = []
        word1, word2, _ = sample
        word1_synsets = wn.synsets(word1)
        word2_synsets = wn.synsets(word2)
        for word1_synset in word1_synsets:
            for word2_synset in word2_synsets:
                if word1_synset.pos() == word2_synset.pos():  # 词性相同
                    try:
                        val = word1_synset.jcn_similarity(
                            word2_synset, brown_ic)
                        if val != None:
                            sims.append(val)
                    except Exception as e:
                        pass
#                         print(word1_synset,'\t',word2_synset)
#                         print(e)
        if len(sims) == 0:
            sims.append(-1)
        if use_max:
            sim = max(sims)
        else:  # use_avg
            sim = sum(sims) / len(sims)
        result.append((word1, word2, sim))
    sims = list(map(lambda x: x[2], result))
    print('jcn similarity:')
    print('mean:', np.mean(sims))
    print('std:', np.std(sims))
    return result


def cal_lin_sim_based_wordnet(test_data, use_max=True):
    result = []
    for sample in test_data:
        sim = -1
        sims = []
        word1, word2, _ = sample
        word1_synsets = wn.synsets(word1)
        word2_synsets = wn.synsets(word2)
        for word1_synset in word1_synsets:
            for word2_synset in word2_synsets:
                if word1_synset.pos() == word2_synset.pos():  # 词性相同
                    try:
                        val = word1_synset.lin_similarity(
                            word2_synset, brown_ic)
                        if val != None:
                            sims.append(val)
                    except Exception as e:
                        pass
#                         print(word1_synset,'\t',word2_synset)
#                         print(e)
        if len(sims) == 0:
            sims.append(-1)
        if use_max:
            sim = max(sims)
        else:  # use_avg
            sim = sum(sims) / len(sims)
        result.append((word1, word2, sim))
    sims = list(map(lambda x: x[2], result))
    print('lin similarity:')
    print('mean:', np.mean(sims))
    print('std:', np.std(sims))
    return result

def cal_sim_based_on_word2vec(test_data):
    sentences = word2vec.Text8Corpus('./text8.txt')  # 经过了分词处理"
    model = word2vec.Word2Vec(
        sentences, size=300, min_count=1, iter = 10, workers=4, window=5)
    #iter = 10 即10个epochs
    result = []
    for sample in test_data:
        word1, word2, _ = sample
        sim = model.similarity(word1, word2)
        result.append((word1, word2, sim))
    sims = list(map(lambda x: x[2], result))
    print('word2vec similarity:')
    print('mean:', np.mean(sims))
    print('std:', np.std(sims))
    return result


def cal_sim_based_on_pretrained_word2vec(test_data):
    EMBEDDING_FILE = './GoogleNews-vectors-negative300.bin'
    word2vecDict = keyedvectors.KeyedVectors.load_word2vec_format(
        EMBEDDING_FILE, binary=True)
    embedding = dict()
    # 词与对应词向量
    for word in word2vecDict.wv.vocab:
        embedding[word] = word2vecDict.word_vec(word)  # 对应的(300,)的词向量
    print('Loaded %s word vectors.' % len(embedding))
    result = []
    for sample in test_data:
        word1, word2, _ = sample
        sim = cosine_similarity([embedding[word1]], [embedding[word2]])[0][0]
        result.append((word1, word2, sim))
    sims = list(map(lambda x: x[2], result))
    print('pretrained word2vec similarity:')
    print('mean:', np.mean(sims))
    print('std:', np.std(sims))
    return result


def main():
    #path_sim
    results = cal_path_sim_based_wordnet(test_data)
    path_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(path_sim_based_on_wordnet, sim_human)
    print('path similarity (max) spearmanr 系数:', rho)
    
    results = cal_path_sim_based_wordnet(test_data,use_max=False)
    path_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(path_sim_based_on_wordnet, sim_human)
    print('path similarity (avg) spearmanr 系数:', rho)

    #lcs_sim
    results = cal_lch_sim_based_wordnet(test_data)
    lch_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(lch_sim_based_on_wordnet, sim_human)
    print('lch similarity (max) spearmanr 系数:', rho)

    results = cal_lch_sim_based_wordnet(test_data, use_max=False)
    lch_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(lch_sim_based_on_wordnet, sim_human)
    print('lch similarity (avg) spearmanr 系数:', rho)

    #wup_sim
    results = cal_wup_sim_based_wordnet(test_data)
    wup_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(wup_sim_based_on_wordnet, sim_human)
    print('wup similarity (max) spearmanr 系数:', rho)

    results = cal_wup_sim_based_wordnet(test_data, use_max=False)
    wup_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(wup_sim_based_on_wordnet, sim_human)
    print('wup similarity (avg) spearmanr 系数:', rho)

    

    #res_sim
    results = cal_res_sim_based_wordnet(test_data)
    res_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(res_sim_based_on_wordnet, sim_human)
    print('res similarity (max) spearmanr 系数:', rho)

    results = cal_res_sim_based_wordnet(test_data, use_max=False)
    res_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(res_sim_based_on_wordnet, sim_human)
    print('res similarity (avg) spearmanr 系数:', rho)

    #jcn_sim
    results = cal_jcn_sim_based_wordnet(test_data)
    jcn_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(jcn_sim_based_on_wordnet, sim_human)
    print('jcn similarity (max) spearmanr 系数:', rho)

    results = cal_jcn_sim_based_wordnet(test_data, use_max=False)
    jcn_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(jcn_sim_based_on_wordnet, sim_human)
    print('jcn similarity (avg) spearmanr 系数:', rho)


    #lin_sim
    results = cal_lin_sim_based_wordnet(test_data)
    lin_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(lin_sim_based_on_wordnet, sim_human)
    print('lin similarity (max) spearmanr 系数:', rho)

    results = cal_lin_sim_based_wordnet(test_data, use_max=False)
    lin_sim_based_on_wordnet = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(lin_sim_based_on_wordnet, sim_human)
    print('lin similarity (avg) spearmanr 系数:', rho)


    #word2vec_sim
    results = cal_sim_based_on_word2vec(test_data)
    sim_based_on_word2vec = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(sim_based_on_word2vec, sim_human)
    print('word2vec similarity spearmanr 系数:', rho)

    #pretrained_word2vec_sim
    results = cal_sim_based_on_pretrained_word2vec(test_data)
    sim_based_on_pretrained_word2vec = list(map(lambda x: x[2], results))
    sim_human = list(map(lambda x: x[2], test_data))
    rho, pval = stats.spearmanr(sim_based_on_pretrained_word2vec, sim_human)
    print('pretrained word2vec similarity spearmanr 系数:', rho)

if __name__ == '__main__':
    main()

