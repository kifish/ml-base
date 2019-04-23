from keras.layers import Flatten, Dense, Dropout,Input
from keras.applications import resnet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(64,64,3)), pooling=True, classes=1000)
for layer in base_model.layers:
    layer.trainable=False

def cal_acc(probs,Y):
    probs = np.array(probs)
    Y = np.array(Y)
    single_true_cnt = 0
    multi_true_cnt = 0
    n_samples = Y.shape[0]
    for i in range(n_samples):
        pred_digits = np.argmax(Y[i],axis = 1)
        true_digits = np.argmax(probs[i],axis = 1)
        single_true_cnt += np.sum(np.equal(pred_digits,true_digits).astype('uint8'))
        multi_true_cnt += np.equal(pred_digits,true_digits).all().astype('uint8') #all判断全部相等
    single_digit_acc = single_true_cnt / Y.shape[0] / Y.shape[1]
    seq_acc = multi_true_cnt / Y.shape[0]
    return single_digit_acc,seq_acc

x = Flatten()(base_model.output)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
pred1 = Dense(11,activation='softmax')(x)
pred2 = Dense(11,activation='softmax')(x)
pred3 = Dense(11,activation='softmax')(x)
pred4 = Dense(11,activation='softmax')(x)
pred5 = Dense(11,activation='softmax')(x)

outputs = [pred1,pred2,pred3,pred4,pred5]
model = Model(input=base_model.input,output = outputs)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

with open('../data/train_rgb.pkl', 'rb') as f:
    X_train, Y_train = pickle.load(f)

with open('../data/test_rgb.pkl', 'rb') as f:
    X_test, Y_test = pickle.load(f)



# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.20, random_state=1)
Y_train = list(Y_train)
# X_val = list(X_val) #keras自带的即可
Y_test = list(Y_test)
# 把Y_train转成np array的话,fit会报错

history = model.fit(x = X_train,y = Y_train,
                                 batch_size = 64,
                                 epochs= 5,
                                 verbose=1,
                                 validation_split=0.05,
                                 shuffle = True
                    )
probs = model.predict(X_train)
infos = model.evaluate(X_train, Y_train, verbose=0)
single_acc, seq_acc = cal_acc(probs,Y_train)
print('Train loss:', infos[0])
print('Train single accuracy:', single_acc)
print('Train sequence accuracy:', seq_acc)


probs = model.predict(X_test)
infos = model.evaluate(X_test, Y_test, verbose=0)
single_acc, seq_acc = cal_acc(probs,Y_test)
print('Test loss:', infos[0])
print('Test single accuracy:', single_acc)
print('Test sequence accuracy:', seq_acc)




