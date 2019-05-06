from keras.layers import Flatten, Dense, Dropout,Input,Activation,MaxPooling2D
from keras.applications import resnet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import numpy as np
import pickle
import matplotlib.pyplot as plt



img_height = 128
img_weight = 128
batch_size = 100 # batch_size 256的话则内存不足！！！batch_size 256至少消耗11GB以上的内存。
base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(img_height,img_weight,3)), pooling=True, classes=1000)#keras2.1.6pooling=True即要2048前的pooling层,resnet50里面是avg_pooling;keras2.2.4pooling=True似乎去掉了pooling层

x = Flatten()(base_model.output) #2048要比8192好很多
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