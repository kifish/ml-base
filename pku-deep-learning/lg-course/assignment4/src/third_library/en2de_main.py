import os, sys
from dataloader import *
from keras.optimizers import *
from keras.callbacks import *

itokens, otokens = MakeS2SDict('../../dataset/train_eng2chn.txt', dict_file='../../dataset/eng2chn_word.txt')
Xtrain, Ytrain = MakeS2SData('../../dataset/train_eng2chn.txt', itokens, otokens, h5_file='../../dataset/eng2chn.h5')
Xvalid, Yvalid = MakeS2SData('../../dataset/val_eng2chn.txt', itokens, otokens, h5_file='../../dataset/eng2chn.valid.h5')

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
dim_model = 256
s2s = Transformer(itokens, otokens, len_limit=70, dim_model=dim_model, d_inner_hid=512, \
				   n_head=8, layers=2, dropout=0.1)

if not os.path.exists('../../tmp'):
	os.mkdir('../../tmp')
save_path = '../../tmp/eng2chn.model.h5'
lr_scheduler = LRSchedulerPerStep(dim_model, 4000)
model_saver = ModelCheckpoint(save_path, save_best_only=True, save_weights_only=True)

s2s.compile(Adam(0.0001, 0.9, 0.98, epsilon=1e-9))
try:
	s2s.model.load_weights(save_path)
except:
	print('\n\nnew model')

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
elif 'test' in sys.argv:
	rets = s2s.decode_sequence_greedy(Xvalid[:256])
	sents = s2s.generate_sentence(rets, delimiter=' ')
	for x in sents:
		print(x)
else:
	s2s.model.summary()
	s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, \
				validation_data=([Xvalid, Yvalid], None), \
				callbacks=[lr_scheduler, model_saver])
	# val_accu @ 30 epoch: 0.7045
