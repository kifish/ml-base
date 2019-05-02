import os, sys
import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *

itokens, otokens = dd.MakeS2SDict('../../dataset/en2de.s2s.txt', dict_file='../../dataset/en2de_word.txt')
Xtrain, Ytrain = dd.MakeS2SData('../../dataset/en2de.s2s.txt', itokens, otokens, h5_file='../../dataset/en2de.h5')
Xvalid, Yvalid = dd.MakeS2SData('../../dataset/en2de.s2s.valid.txt', itokens, otokens, h5_file='../../dataset/en2de.valid.h5')

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)

'''
from rnn_s2s import RNNSeq2Seq
s2s = RNNSeq2Seq(itokens,otokens, 256)
s2s.compile('rmsprop')
s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None))
'''

from transformer import Transformer, LRSchedulerPerStep

d_model = 256
s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
				   n_head=8, layers=2, dropout=0.1)

mfile = 'models/en2de.model.h5'

lr_scheduler = LRSchedulerPerStep(d_model, 4000)
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
try: s2s.model.load_weights(mfile)
except: print('\n\nnew model')

if 'eval' in sys.argv:
	print(s2s.decode_sequence_readout('A black dog eats food .'.split(), delimiter=' '))
	print(s2s.decode_sequence('A black dog eats food .'.split(), delimiter=' '))
	print(s2s.decode_sequence_fast('A black dog eats food .'.split(), delimiter=' '))
	while True:
		quest = input('> ')
		print(s2s.decode_sequence_fast(quest.split(), delimiter=' '))
		rets = s2s.beam_search(quest.split(), delimiter=' ')
		for x, y in rets: print(x, y)
elif 'test' in sys.argv:
	rets = s2s.decode_sequence_greedy(Xvalid[:256])
	sents = s2s.generate_sentence(rets, delimiter=' ')
	for x in sents: print(x)
else:
	s2s.model.summary()
	s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, \
				validation_data=([Xvalid, Yvalid], None), \
				callbacks=[lr_scheduler, model_saver])
	# val_accu @ 30 epoch: 0.7045
