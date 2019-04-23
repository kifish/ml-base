from keras.layers import Flatten, Dense, Dropout,Input
from keras.applications import resnet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf
import numpy as np
import pickle
base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(64,64,3)), pooling=True, classes=1000)
def cal_seq_acc(out,labels):
    labels = np.asarray(labels).squeeze()
    num_feature,num_sample = labels.shape
    preds = []
    Y_out = []
    for i in range(num_feature):
        val = np.argmax(out[i],axis=1).astype('uint8')
        preds.append(np.count_nonzero(val == labels[i].flatten())/num_sample * 100)
        Y_out.append(val)
    outYarr = np.asarray(Y_out).T
    seq_acc = np.count_nonzero(np.all(outYarr[:,0:4]==labels[0:4,:].T,axis=1))/ np.float(num_sample) * 100
    return preds, Y_out, seq_acc

# input_layer = Input(shape=(None, None, 3))
x = Flatten()(base_model.output)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
pred1 = Dense(11,activation='softmax')(x)
pred2 = Dense(11,activation='softmax')(x)
pred3 = Dense(11,activation='softmax')(x)
pred4 = Dense(11,activation='softmax')(x)
pred5 = Dense(11,activation='softmax')(x)

# X = tf.placeholder(tf.float32, [None, 64, 64, 3])
# Y = tf.placeholder(tf.int32, [None, 5, 11])
# loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 0, :], logits=pred1))
# loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 1, :], logits=pred2))
# loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 2, :], logits=pred3))
# loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 3, :], logits=pred4))
# loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 4, :], logits=pred5))
# loss = loss1 + loss2 + loss3 +loss4 + loss5

# multi_accuracy =
outputs =  [pred1,pred2,pred3,pred4,pred5]
model = Model(input=base_model.input,output = outputs)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

with open('../data/train_rgb.pkl', 'rb') as f:
    X_train, Y_train = pickle.load(f)

with open('../data/test_rgb.pkl', 'rb') as f:
    X_test, Y_test = pickle.load(f)

n_train = X_train.shape[0]
Y_train = [
    Y_train[:, 0, :].reshape(n_train,11),
    Y_train[:, 1, :].reshape(n_train,11),
    Y_train[:, 2, :].reshape(n_train,11),
    Y_train[:, 3, :].reshape(n_train,11),
    Y_train[:, 4, :].reshape(n_train,11)
]

n_test = X_test.shape[0]
Y_test = [
    Y_test[:, 0, :].reshape(n_test,11),
    Y_test[:, 1, :].reshape(n_test,11),
    Y_test[:, 2, :].reshape(n_test,11),
    Y_test[:, 3, :].reshape(n_test,11),
    Y_test[:, 4, :].reshape(n_test,11)
]

#
# print(Y_train.shape)
# print(Y_test.shape)
history = model.fit(x = X_train,y = Y_train,
                                 batch_size = 128,
                                 epochs= 10,
                                 verbose=1,
                                 shuffle = True,
                                 )
yOutr = model.predict(X_train)
scoreTr = model.evaluate(X_train, Y_train, verbose=0)
trainpAcc, outYt, seqTrainAcc = cal_seq_acc(yOutr,Y_train)
print('digit1', 'digit2','digit3','digit4','digit5')
print('Train loss:', scoreTr[0])
print('Train accuracy:', trainpAcc)
print('Train sequence accuracy:', seqTrainAcc)



yOutr = model.predict(X_test)
scoreTe = model.evaluate(X_test, Y_test, verbose=0)
testpAcc, outYt, seqTestAcc = cal_seq_acc(yOutr,X_test)
print('digit1', 'digit2','digit3','digit4','digit5')
print('Test loss:', scoreTe[0])
print('Test accuracy:', testpAcc)
print('Test sequence accuracy:', seqTestAcc)



