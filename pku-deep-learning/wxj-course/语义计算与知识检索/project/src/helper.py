from collections import defaultdict

def process_data(data_type='train'):
    res = defaultdict(list)
    with open('./data/MSParS.%s'%data_type, encoding='utf8') as f:
        text = f.readlines()
    for i in range(len(text)//5):
        res['question'].append(text[5*i].rstrip().split('\t')[-1].strip().replace('?', ' ?').lower())
        if data_type != 'test':
            res['logical form'].append(text[5*i+1].rstrip().split('\t')[-1])
            res['parameters'].append(get_parameters(text[5*i+2]))
            res['question type'].append(text[5*i+3].rstrip().split('\t')[-1])
        else:
            res['parameters'].append([])

    return res

def get_parameters(input):
    par_lst = input.rstrip().split('\t')[-1].split(' ||| ')
    res = []
    for par in par_lst:
        par_obj, par_type, par_idx = par.split()[:3]
        par_idx = list(map(int, par_idx[1:-1].split(',')))
        res.append([par_obj, par_type[1:-1], par_idx])
    return res

def trans_to_BIO(res_dict, outpath):
    f = open(outpath, 'w', encoding='utf-8')
    for i in range(len(res_dict['question'])):
        token = res_dict['question'][i].split()
        entity = [x for x in res_dict['parameters'][i] if x[1]=='entity' and x[2]!=[-1, -1]]
        tmp = ['O' for _ in range(len(token))]
        for x in entity:
            tmp[x[2][0]] = 'B-E'
            for idx in range(x[2][0]+1, x[2][1]+1):
                tmp[idx] = 'I-E'
        for j in range(len(token)):
            f.write(str(j+1) + '\t' + token[j] + '\t' + tmp[j] + '\n')
        f.write('\n')
    f.close()

def simplify_embedding(emd_file, out_path, words_set):
    fin = open(emd_file, encoding='utf-8')
    fout = open(out_path, 'w', encoding='utf-8')
    while 1:
        line = fin.readline()
        if line:
            if line.split()[0] in words_set:
                fout.write(line)
        else:
            break
    fin.close()
    fout.close()

def get_words_set():
    all_sentences = []
    for data_type in ['train', 'dev', 'test']:
        all_sentences += process_data(data_type)['question']
    words_set = set()
    for x in all_sentences:
        for word in x.split():
            words_set.add(word)
    return words_set



if __name__ == '__main__':
    # for data_type in ['train', 'dev', 'test']:
    #     a = process_data(data_type)
    #     trans_to_BIO(a, './ner/data/mydata/%s.txt'%data_type)
    emd_file = '../hw2/data/worddata/glove.twitter.27B.200d.txt'
    out_path = 'ner/my_embedding_200d.txt'
    simplify_embedding(emd_file, out_path, get_words_set())
    pass