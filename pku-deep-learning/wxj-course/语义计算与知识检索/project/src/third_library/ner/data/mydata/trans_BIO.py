in_path = 'dev_pred.txt'
out_path = 'dev_BIO_pred.txt'
with open(in_path, encoding='utf-8') as f:
    sentence_lst = f.read().split('\n\n')[:-1]

def process_token(string):
    idx, token, tag = string.split('\t')
    return [int(idx)-1, token, tag!='O']
fout = open(out_path, 'w', encoding='utf-8')
for sentence in sentence_lst:
    entity = []
    tmp = ''
    for idx, token, tag in map(process_token, sentence.split('\n')):
        if tag:
            tmp += '_'+token
        else:
            if tmp != '' and (entity==[] or entity[-1]!=tmp[1:]):
                entity.append(tmp[1:])
                tmp = ''
    if tmp != '':
        entity.append(tmp[1:])
    print(entity)
    fout.write('\t'.join(entity)+'\n')
fout.close()




