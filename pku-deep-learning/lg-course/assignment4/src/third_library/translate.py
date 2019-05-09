import os, sys
from dataloader import *
from keras.optimizers import *
from keras.callbacks import *
from transformer import Transformer, LRSchedulerPerStep

dim_model = 256

itokens, otokens = MakeS2SDict('../../dataset/train_chn2eng.txt', dict_file='../../dataset/chn2eng_word.txt')
X_test, Y_test = MakeS2SData('../../dataset/test_chn2eng.txt', itokens, otokens,
                             h5_file='../../dataset/chn2eng.test.h5')

s2s = Transformer(itokens, otokens, len_limit=70, dim_model=dim_model, d_inner_hid=512, \
                  n_head=8, layers=2, dropout=0.1)
save_path = '../../tmp/chn2eng.model.h5'
s2s.model.load_weights(save_path)

if 'eval' in sys.argv:
    print(s2s.decode_sequence_readout('A black dog eats food .'.split(), delimiter=' '))
    print(s2s.decode_sequence('A black dog eats food .'.split(), delimiter=' '))
    print(s2s.decode_sequence_fast('A black dog eats food .'.split(), delimiter=' '))
    while True:
        quest = input('> ')
        print(s2s.decode_sequence_fast(quest.split(), delimiter=' '))
        rets = s2s.beam_search(quest.split(), delimiter=' ')
        for x, y in rets:
            print(x, y)
else:
    rets = s2s.decode_sequence_greedy(X_test)
    sents = s2s.generate_sentence(rets, delimiter=' ')
    with open('../../dataset/tran.txt','w',encoding='utf8') as f:
        for sent in sents:
            f.write(sent + '\n')
        # print(sent)
