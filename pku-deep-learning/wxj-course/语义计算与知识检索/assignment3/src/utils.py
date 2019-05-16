import os
import re 
from nltk.translate.bleu_score import corpus_bleu
def process_data():
    pattern1 = r'<question id=.*>\t(.*)\n'
    pattern2 = r'<logical form id=.*>\t(.*)\n'
    # train
    with open('../data/MSParS.train',encoding='utf8') as f:
        text = f.read()
    source = re.findall(pattern1,text)
    target = re.findall(pattern2,text)
    assert len(source) == len(target)
    n_sample = len(source)
    print('train n_sample:',n_sample)
    with open('../data/MSParS.simple.train','w',encoding='utf8') as f:
        for idx in range(n_sample):
            f.write(source[idx] + '\t' + target[idx] + '\n')
    # dev
    with open('../data/MSParS.dev',encoding='utf8') as f:
        text = f.read()
    source = re.findall(pattern1,text)
    target = re.findall(pattern2,text)
    assert len(source) == len(target)
    n_sample = len(source)
    print('dev n_sample:',n_sample)
    with open('../data/MSParS.simple.dev','w',encoding='utf8') as f:
        for idx in range(n_sample):
            f.write(source[idx] + '\t' + target[idx] + '\n')


def save_ref():
    with open('../data/MSParS.simple.dev', 'r', encoding='utf8') as f1:
        with open('../data/ref.txt','w',encoding='utf8') as f2:
            for line in f1.readlines():
                s = line.strip().split('\t')[1]
                f2.write(s + '\n')




def cal_acc():
    ref_list = []
    with open('../data/MSParS.simple.dev','r',encoding='utf8') as f:
        for line in f.readlines():
            ref = line.strip().split('\t')[1].split(' ')
            ref_list.append(ref)
    pred_list = []
    with open('../data/pred.txt','r',encoding='utf8') as f:
        for line in f.readlines():
            pred = line.strip().split(' ')
            pred_list.append(pred)
    token_cnt = 0
    true_cnt = 0
    true_line_cnt = 0
    for idx,ref in enumerate(ref_list):
        token_cnt += len(ref)
        pred = pred_list[idx]
        min_len = min(len(ref),len(pred))
        for i in range(min_len):
            if pred[i] == ref[i]:
                true_cnt += 1
        if pred == ref:
            true_line_cnt += 1
    sample_num = len(ref_list)
    print('token num:',token_cnt)
    print('sample num:', sample_num)
    print('token acc',true_cnt / token_cnt)
    print('exact totally match acc',true_line_cnt / sample_num)
    print('calculating bleu...')
    new_ref_list = []
    for ref in ref_list:
        new_ref_list.append([ref])
    score = corpus_bleu(new_ref_list,pred_list)
    print('corpus bleu:',score)
  

def gen_vocab():
    word_dict = {}
    logic_form_dict = {}
    with open('../data/MSParS.simple.train','r',encoding='utf8') as f:
        for line in f.readlines():
            words,logic_forms = line.strip().split('\t')
            words = words.split(' ')
            logic_forms = logic_forms.split(' ')
            for word in words:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
            for logic_form in logic_forms:
                if logic_form not in logic_form_dict:
                    logic_form_dict[logic_form] = 1
                else:
                    logic_form_dict[logic_form] += 1
    with open('../data/train_vocab_q.txt','w',encoding='utf8') as f:
        for k,v in word_dict.items():
            f.write(k+'\t'+str(v)+'\n')
    with open('../data/train_vocab_f.txt', 'w', encoding='utf8') as f:
        for k, v in logic_form_dict.items():
            f.write(k+'\t'+str(v)+'\n')


if __name__ == '__main__':
    #process_data()
    #gen_vocab()
    cal_acc()
    
