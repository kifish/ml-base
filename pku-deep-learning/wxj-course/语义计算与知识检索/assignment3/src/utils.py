import os
import re 
import argparse
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
    cnt = 0
    with open(args.pred_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            pred = line.strip().split(' ')
            pred_list.append(pred)
            cnt += 1
            if cnt == 9000: # skip the acc value
                break
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
def enhance_data():
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
    pattern3 = r'<parameters id=.*>\t(.*)\n'
    pattern4 = r'(?:\|\|\| ){0,1}([^ \|]*?) \(entity\)'
    raw_lines = re.findall(pattern3, text)
    with open('../data/MSParS.entity.train','w',encoding='utf8') as f:
        for idx in range(n_sample):
            raw_line = raw_lines[idx]
            entity_list = re.findall(pattern4,raw_line)
            target_s = target[idx]
            for entity in entity_list:
                target_s = target_s.replace(entity,'<entity>',1)
            f.write(source[idx] + '\t' + target_s + '\n')
    # dev
    with open('../data/MSParS.dev',encoding='utf8') as f:
        text = f.read()
    source = re.findall(pattern1,text)
    target = re.findall(pattern2,text)
    raw_lines = re.findall(pattern3, text)
    assert len(source) == len(target)
    n_sample = len(source)
    print('dev n_sample:',n_sample)
    with open('../data/MSParS.entity.dev','w',encoding='utf8') as f:
        for idx in range(n_sample):
            # print(idx)
            raw_line = raw_lines[idx]
            # print(raw_line)
            entity_list = re.findall(pattern4,raw_line)
            # print(entity_list)
            target_s = target[idx]
            for entity in entity_list:
                target_s = target_s.replace(entity, '<entity>', 1)
            f.write(source[idx] + '\t' + target_s + '\n')

def fill():
    raw_file_path = args.pred_path
    save_path = args.save_path
    idx = 1
    pattern1 = r'<parameters id=.*>\t(.*)\n'
    with open('../data/MSParS.dev', encoding='utf8') as f:
        text = f.read()
    raw_lines = re.findall(pattern1,text)
    del text
    pattern2 = r'(?:\|\|\| ){0,1}([^ \|]*?) \(entity\)'
    with open(raw_file_path,'r',encoding='utf8') as f1:
        with open(save_path,'w',encoding='utf8') as f2:
            for line in f1.readlines():
                print('-----------')
                print(idx)
                raw_line = raw_lines[idx-1]
                entity_list = re.findall(pattern2,raw_line)
                for entity in entity_list:
                    print(entity)
                    # line = line.replace('<U>',entity,1)
                    line = line.replace('<entity>', entity, 1)
                print('-----------')
                f2.write(line)           
                if idx >= 9000:
                    break
                idx += 1

def fill2():
    raw_file_path = args.pred_path
    save_path = args.save_path
    idx = 1
    entity_lines = []
    with open('../data/dev_entity_pred_v2.txt', encoding='utf8') as f:
        for line in f.readlines():
            entity_lines.append(line.strip())
    with open(raw_file_path,'r',encoding='utf8') as f1:
        with open(save_path,'w',encoding='utf8') as f2:
            for line in f1.readlines():
                print('-----------')
                print(idx)
                entity_line = entity_lines[idx-1]
                entity_list = entity_line.split(' ')
                for entity in entity_list:
                    if len(entity) == 0:
                        continue
                    print(entity)
                    # line = line.replace('<U>',entity,1)
                    line = line.replace('<entity>', entity, 1)
                print('-----------')
                f2.write(line)           
                if idx >= 9000:
                    break
                idx += 1


def fill3():
    raw_file_path = args.pred_path
    save_path = args.save_path
    idx = 1
    entity_lines = []
    with open('../data/dev_entity_pred_v2.txt', encoding='utf8') as f:
        for line in f.readlines():
            entity_lines.append(line.strip())
    with open(raw_file_path, 'r', encoding='utf8') as f1:
        with open(save_path, 'w', encoding='utf8') as f2:
            for line in f1.readlines():
                print('-----------')
                print(idx)
                entity_line = entity_lines[idx-1]
                entity_list = entity_line.split(' ')
                for i,entity in enumerate(entity_list):
                    if len(entity) == 0:
                        continue
                    print(entity)
                    # line = line.replace('<U>',entity,1)
                    if i == len(entity_list)-1:
                        line = line.replace('<entity>', entity)
                    else:
                        line = line.replace('<entity>', entity, 1)
                print('-----------')
                f2.write(line)
                if idx >= 9000:
                    break
                idx += 1


def process_test_data():
    pattern = r'<question id=.*>\t(.*)\n'
    # test
    with open('../data/MSParS.test', encoding='utf8') as f:
        text = f.read()
    source = re.findall(pattern, text)
    n_sample = len(source)
    print('test n_sample:', n_sample)
    with open('../data/MSParS.simple.test', 'w', encoding='utf8') as f:
        for idx in range(n_sample):
            f.write(source[idx] + '\n')
   
if __name__ == '__main__':
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-pred_path', type=str, default='../data/pred.txt', help='pred.txt')
    main_arg_parser.add_argument(
        '-save_path', type=str, default='../data/pred_filled.txt', help='pred.txt')
    # parse input params
    args = main_arg_parser.parse_args()
    #process_data()
    #gen_vocab()
    #cal_acc()
    #fill()
    #fill2()
    #fill3()
    #enhance_data()
    process_test_data()