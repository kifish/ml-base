from keras.layers import Flatten, Dense, Dropout,Input,Activation,MaxPooling2D
from keras.applications import resnet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import numpy as np
import pickle
import matplotlib.pyplot as plt
base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(128,128,3)), pooling=True, classes=1000)
from utils import get_data
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
x = Dropout(0.2)(x)
pred1 = Dense(11,activation='softmax')(x)
pred2 = Dense(11,activation='softmax')(x)
pred3 = Dense(11,activation='softmax')(x)
pred4 = Dense(11,activation='softmax')(x)
pred5 = Dense(11,activation='softmax')(x)

outputs = [pred1,pred2,pred3,pred4,pred5]
model = Model(input=base_model.input,output = outputs)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


with open('../data/train_no_pos_64.pkl', 'rb') as f:
    X_train, Y_train = pickle.load(f)

with open('../data/test_no_pos_64.pkl', 'rb') as f:
    X_test, Y_test = pickle.load(f)

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
                                 epochs= 1,
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
print('Test info:')
print(infos)
print('Test single accuracy:', single_acc)
print('Test sequence accuracy:', seq_acc)


#acc = history.history['acc']
#val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
# "bo" is for "blue dot"
plt.figure(0)
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()
plt.savefig('Training and validation loss.png')

# plt.figure(1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# plt.savefig('Training and validation accuracy.png')
