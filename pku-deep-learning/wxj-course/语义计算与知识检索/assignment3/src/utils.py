import os
import re 
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

if __name__ == '__main__':
    process_data()
