import os, sys
from dataloader import *
from keras.optimizers import *
from keras.callbacks import *
from transformer import Transformer, LRSchedulerPerStep
import matplotlib.pyplot as plt

itokens, otokens = MakeS2SDict('../../dataset/train_chn2eng_w.txt', dict_file='../../dataset/chn2eng_w_word.txt')
Xtrain, Ytrain = MakeS2SData('../../dataset/train_chn2eng_w.txt', itokens, otokens,
                             h5_file='../../dataset/chn2eng_w.h5')
Xvalid, Yvalid = MakeS2SData('../../dataset/val_chn2eng_w.txt', itokens, otokens,
                             h5_file='../../dataset/chn2eng_w.valid.h5')

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)

dim_model = 512
s2s = Transformer(itokens, otokens, len_limit=70, dim_model=dim_model, d_inner_hid=2048, \
                  n_head=8, layers=6, dropout=0.1)

if not os.path.exists('../../tmp'):
    os.mkdir('../../tmp')
save_path = '../../tmp/chn2eng.model.h5'
lr_scheduler = LRSchedulerPerStep(dim_model, 4000)
model_saver = ModelCheckpoint(save_path, save_best_only=True, save_weights_only=True) #save_weights_only设置为False会报错： TypeError: can't pickle _thread.RLock objects

s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
s2s.model.summary()

continued = False
if continued:
    s2s.model.load_weights(save_path)
else:
    print('\n\nnew model')
    history = s2s.model.fit([Xtrain, Ytrain], None, batch_size=256, epochs=30, \
                  validation_data=([Xvalid, Yvalid], None), \
                  callbacks=[lr_scheduler, model_saver])
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(0)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('transformer_Training_and_validation_loss.png')



