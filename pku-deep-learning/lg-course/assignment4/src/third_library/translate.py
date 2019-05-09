import os, sys
from dataloader import *
from keras.models import load_model
from transformer import Transformer
from keras.optimizers import *
dim_model = 256

itokens, otokens = MakeS2SDict('../../dataset/train_chn2eng_w.txt', dict_file='../../dataset/chn2eng_w_word.txt')
X_test, Y_test = MakeS2SData('../../dataset/test_chn2eng_w.txt', itokens, otokens,
                             h5_file='../../dataset/chn2eng_w.test.h5')
print('test shapes:', X_test.shape, Y_test.shape)

dim_model = 512
s2s = Transformer(itokens, otokens, len_limit=70, dim_model=dim_model, d_inner_hid=2048, \
                  n_head=8, layers=6, dropout=0.1)
s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
save_path = '../../tmp/chn2eng.model.h5'
s2s.model.load_weights(save_path)

if 'eval' in sys.argv:
    while True:
        s = input('> ')
        print(s2s.decode_sequence_fast(s.split(), delimiter=' '))
        rets = s2s.beam_search(s.split(), delimiter=' ')
        for x, y in rets:
            print(x, y)
elif 'debug' in sys.argv:
    with open('../../dataset/test_chn2eng_w.txt') as f:
        for line in f.readlines():
            s = line.split('\t')[0]
            print(s2s.decode_sequence_fast(s.split(), delimiter=' '))
else:
    info = s2s.model.evaluate(x = [X_test,Y_test],y = None)
    print(info)
    rets = s2s.decode_sequence_greedy(X_test)
    sents = s2s.generate_sentence(rets, delimiter=' ')
    with open('../../dataset/tran.txt','w',encoding='utf8') as f:
        for sent in sents:
            f.write(sent + '\n')
        # print(sent)
