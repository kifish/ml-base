from keras.layers import Flatten, Dense, Dropout,Input,Activation,MaxPooling2D
from keras.applications import resnet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import numpy as np
import pickle
import matplotlib.pyplot as plt
base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(128,128,3)), pooling=True, classes=1000)
from utils import process_raw_data
def cal_acc(probs,Y):
    probs = np.array(probs)
    Y = np.array(Y)
    print('receive probs:',probs.shape)
    print('receive Y:', Y.shape)
    probs = probs.transpose((1,0,2))
    Y = Y.transpose((1, 0, 2))
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




x = MaxPooling2D(pool_size=(2,2))(base_model.output)
x = Flatten()(x)
x = Dense(128,activation=None)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
pred1 = Dense(11,activation='softmax')(x)
pred2 = Dense(11,activation='softmax')(x)
pred3 = Dense(11,activation='softmax')(x)
pred4 = Dense(11,activation='softmax')(x)
pred5 = Dense(11,activation='softmax')(x)

outputs = [pred1,pred2,pred3,pred4,pred5]
model = Model(input=base_model.input,output = outputs)
from keras import optimizers
adam = optimizers.Adam(lr = 0.01)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.utils import to_categorical
df = pd.read_csv('../data/train_meta_info.csv')
train_datagen = ImageDataGenerator(featurewise_center=False,validation_split=0.2)
train_generator=train_datagen.flow_from_dataframe(
                        dataframe=df,
                        directory="../data/train",
                        x_col="filenames",
                        y_col=["digit1","digit2","digit3","digit4","digit5"],
                        batch_size=32,
                        seed=42,
                        shuffle=True,
                        class_mode='other',
                        target_size=(128,128),
                        subset='training')
validation_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory="../data/train",
    x_col="filenames",
    y_col=["digit1", "digit2", "digit3", "digit4", "digit5"],
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='other',
    target_size=(128, 128),
    subset='validation')

def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        batch_y = to_categorical(batch_y)
        yield (batch_x,[batch_y[:,i,:] for i in range(5)])

history = model.fit_generator(generator_wrapper(train_generator),
                                 batch_size = 12,
                                 epochs= 30,
                                 verbose=1,
                                 validation_data=generator_wrapper(validation_generator),
                                 shuffle = True
                              )
# probs = model.predict(X_train)
# infos = model.evaluate(X_train, Y_train, verbose=0)
# single_acc, seq_acc = cal_acc(probs,Y_train)
# print('Train loss:', infos[0])
# print('Train single accuracy:', single_acc)
# print('Train sequence accuracy:', seq_acc)
#
#
# print('processing data')
# root_path = '../data/test/'
# X_test, Y_test = process_raw_data(root_path)
# print('X_test:', X_test.shape)
# print('Y_test:', Y_test.shape)
# Y_test = [
#     Y_test[:,0,:],
#     Y_test[:,1,:],
#     Y_test[:,2,:],
#     Y_test[:,3,:],
#     Y_test[:,4,:]
# ]
# probs = model.predict(X_test)
# infos = model.evaluate(X_test, Y_test, verbose=0)
# single_acc, seq_acc = cal_acc(probs,Y_test)
# print('Test loss:', infos[0])
# print('Test info:')
# print(infos)
# print('Test single accuracy:', single_acc)
# print('Test sequence accuracy:', seq_acc)
#
#
# #acc = history.history['acc']
# #val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# # "bo" is for "blue dot"
# plt.figure(0)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# #plt.show()
# plt.savefig('no_detector_60-256-128*-sgd-0.01-Training and validation loss.png')

