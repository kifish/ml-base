from keras.layers import Flatten, Dense, Dropout,Input
from keras.applications import resnet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf
import numpy as np
import pickle
base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(64,64,3)), pooling=True, classes=1000)
#for layer in base_model.layers:
#    layer.trainable=False

def single_acc(Y,prediction):
    #问题在于prediction的shape似乎是(None,11),不是完整的prediction
    # Compute equality vectors
    correct_prediction = tf.equal(tf.argmax(prediction, 2), tf.argmax(Y, 2))
    # Calculate mean accuracy among 1st dimension
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 1)
    # Accuracy of predicting any digit in the images
    accuracy_single = tf.reduce_mean(accuracy)
    return accuracy_single

def multi_acc(Y,prediction):
    # Compute equality vectors
    correct_prediction = tf.equal(tf.argmax(prediction, 2), tf.argmax(Y, 2))
    # Calculate mean accuracy among 1st dimension
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 1)
    # Accuracy of the predicting all numbers in an image
    accuracy_multi = tf.reduce_mean(tf.cast(tf.equal(accuracy, tf.constant(1.0)), tf.float32))
    return accuracy_multi


def cal_acc(probs,Y):
    #似乎还有bug
    probs = np.array(probs)
    Y = np.array(Y)
    print('receive probs:',probs.shape)
    print('receive Y:', Y.shape)
    probs.transpose((1,0,2))
    Y.transpose((1, 0, 2))
    print('transpose...')
    print('probs:',probs.shape)
    print('Y:', Y.shape)
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

# https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
Y_train = [
    Y_train[:,0,:],
    Y_train[:,1,:],
    Y_train[:,2,:],
    Y_train[:,3,:],
    Y_train[:,4,:]
]

Y_test = [
    Y_test[:,0,:],
    Y_test[:,1,:],
    Y_test[:,2,:],
    Y_test[:,3,:],
    Y_test[:,4,:]
]

history = model.fit(x = X_train,y = Y_train,
                                 batch_size = 256,
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




