from keras.layers import Flatten, Dense, Dropout,Input,Activation,MaxPooling2D
from keras.applications import resnet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import numpy as np
import pickle
import matplotlib.pyplot as plt
base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(224,224,3)), pooling=False, classes=1000)#pooling=True即要2048前的pooling层,resnet50里面是avg_pooling

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


x = Flatten()(base_model.output)
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
train_datagen = ImageDataGenerator(featurewise_center=False,validation_split=0.05)
train_generator=train_datagen.flow_from_dataframe(
                        dataframe=df,
                        directory="../data/train",
                        x_col="filenames",
                        y_col=["digit1","digit2","digit3","digit4","digit5"],
                        batch_size=256,
                        seed=42,
                        shuffle=True,
                        class_mode='other',
                        target_size=(224,224),
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
    target_size=(224, 224),
    subset='validation')

def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        batch_y = to_categorical(batch_y)
        yield (batch_x,[batch_y[:,i,:] for i in range(5)])


STEP_SIZE_TRAIN=train_generator.n // train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n // validation_generator.batch_size

history = model.fit_generator(generator_wrapper(train_generator),
                                steps_per_epoch=STEP_SIZE_TRAIN,
                                 epochs= 30,
                                 verbose=1,
                                 validation_data=generator_wrapper(validation_generator),
                                validation_steps=STEP_SIZE_VALID,
                                 shuffle = True
                              )
# use_multiprocessing=True,
# workers=6
# cpu are used to generate data;however, multiple workers may duplicate data
with open('../data/test_rgb.pkl', 'rb') as f:
    X_test, Y_test = pickle.load(f)
del X_test
Y_test = [
    Y_test[:,0,:],
    Y_test[:,1,:],
    Y_test[:,2,:],
    Y_test[:,3,:],
    Y_test[:,4,:]
]
df = pd.read_csv('../data/test_meta_info.csv')
test_datagen=ImageDataGenerator()
test_generator=test_datagen.flow_from_dataframe(
                        dataframe=df,
                        directory="../data/test",
                        x_col="filenames",
                        y_col=["digit1","digit2","digit3","digit4","digit5"],
                        batch_size=256,
                        seed=42,
                        shuffle=False,
                        class_mode='other',
                        target_size=(224,224)
                        )
test_generator.reset() #important
probs = model.predict_generator(test_generator,
                                verbose=1)
infos = model.evaluate_generator(test_generator, verbose=0)
single_acc, seq_acc = cal_acc(probs,Y_test)
print('Test loss:', infos[0])
print('Test info:')
print(infos)
print('Test single accuracy:', single_acc)
print('Test sequence accuracy:', seq_acc)


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
plt.savefig('generator-256-224*-Adam-0.01-Training_and_validation_loss.png')

