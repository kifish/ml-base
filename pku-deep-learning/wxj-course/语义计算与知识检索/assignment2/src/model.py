import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import Sequential
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import LSTM,Bidirectional,Embedding,Dropout
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalMaxPooling1D,Concatenate
import numpy as np
import keras
import tensorflow as tf
from keras.layers import LSTM,Bidirectional,Embedding,Dropout,Dense
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.constraints import maxnorm
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dropout, Dense, Bidirectional, LSTM, \
    Embedding, GaussianNoise, Activation, Flatten, \
    RepeatVector, MaxoutDense, GlobalMaxPooling1D, \
    Convolution1D, MaxPooling1D, concatenate, Conv1D,GaussianNoise
from keras.regularizers import l2
from keras import initializers
from keras import backend as K, regularizers, constraints, initializers
from keras.engine.topology import Layer


class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def simple_nn(vocab_size):
    model = Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(3, activation=tf.nn.softmax))
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def simple_nn_l2(vocab_size):
    model = Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(3, kernel_regularizer=keras.regularizers.l2(0.001),
                                 activation=tf.nn.softmax))  # 正则项只会在训练时参与计算，因此测试的loss一般低于训练。
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model




def simple_LSTM(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def simple_BiLSTM(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 200))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1, recurrent_activation='sigmoid')))
    model.add(Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(3, kernel_regularizer=keras.regularizers.l2(0.001),
                    activation=tf.nn.softmax))  # 正则项只会在训练时参与计算，因此测试的loss一般低于训练。
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def double_BiLSTM(vocab_size):

    max_len = 35
    sequence_input = Input(shape=(max_len,), dtype='int32')
    embedded_sequences = Embedding(vocab_size, output_dim=200, mask_zero=True, input_length=max_len)(sequence_input)
    lstm = Bidirectional(LSTM(128,
                            dropout=0.3,
                            return_sequences=True,
                            recurrent_activation='relu',
                            recurrent_initializer='glorot_uniform'))(embedded_sequences)
    lstm = Bidirectional(LSTM(128,
                              dropout=0.3,
                              return_sequences=False,
                              recurrent_activation='relu',
                              recurrent_initializer='glorot_uniform'))(lstm)
    dense = Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu)(lstm) #可以不加这一层
    output = Dense(3,activation='softmax')(dense)
    model = Model(inputs=sequence_input, outputs=output)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def cnn_LSTM(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 200))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, recurrent_dropout=0.2, recurrent_activation='sigmoid'))
    model.add(Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(3, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.softmax))
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def cnn_LSTM2(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 200))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, recurrent_dropout=0.2, recurrent_activation='sigmoid'))
    model.add(Dense(32, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(3, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.softmax))
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def simple_cnn(vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=200, mask_zero=False))
    model.add(Conv1D(activation="relu",
                     filters=80, kernel_size=4, padding="valid"))  # Layer conv1d_1 does not support masking
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    model.add(Dense(3, activation=tf.nn.softmax))
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def cnn_multi_filters(vocab_size,filter_size = [2,3,4], num_filters = 50):
    input_text = Input(shape=(35,), dtype='int32')
    emb_text = Embedding(input_dim =vocab_size, output_dim=200, mask_zero = False)(input_text)
#     emb_text = GaussianNoise(noise)(emb_text)
    emb_text = Dropout(0.5)(emb_text)

    pooling_reps = []
    for i in filter_size:
        feat_maps = Convolution1D(filters=num_filters,
                                  kernel_size=i,
                                  padding="valid",
                                  activation="relu",
                                  subsample_length=1)(emb_text)
        pool_vecs = MaxPooling1D(pool_length=2)(feat_maps)
        pool_vecs = Flatten()(pool_vecs)
        # pool_vecs = GlobalMaxPooling1D()(feat_maps)
        pooling_reps.append(pool_vecs)

    representation = concatenate(pooling_reps)

    representation = Dropout(0.5)(representation)

    probabilities = Dense(3, activation='softmax',
                          activity_regularizer=l2(0.001))(representation)

    model = Model(input=input_text, output=probabilities)
    model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def embedding_layer(embedding_matrix, max_len,trainable=False, masking=True):
    vocab_size = embedding_matrix.shape[0]
    embedding_size = embedding_matrix.shape[1]
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_len,
        trainable=trainable,
        mask_zero=masking,
        weights=[embedding_matrix]
    )
    return embedding


# https://androidkt.com/text-classification-using-attention-mechanism-in-keras/

def LSTM_with_attention(vocab_size,embedding_matrix = None):
    max_len = 35
    sequence_input = Input(shape=(max_len,), dtype='int32')
    if embedding_matrix is None:
        embedded_sequences = Embedding(vocab_size, output_dim=200, mask_zero = True, input_length=max_len)(sequence_input)
    else:
        embedded_sequences = embedding_layer(embedding_matrix,max_len)(sequence_input)
    lstm = Bidirectional(LSTM(128,
                            dropout=0.3,
                            return_sequences=True,
                            recurrent_activation='relu',
                            recurrent_initializer='glorot_uniform'))(embedded_sequences)
    context_vector = Attention(1)(lstm) #1只是为了初始化，初始化完成后会根据前一层计算size
    output = Dense(3,activation='softmax')(context_vector)
    model = Model(inputs=sequence_input, outputs=output)
    model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def BiLSTM_with_attention(vocab_size,embedding_matrix = None):
    max_len = 35
    sequence_input = Input(shape=(max_len,), dtype='int32')
    if embedding_matrix is None:
        embedded_sequences = Embedding(vocab_size, output_dim=200, mask_zero = True, input_length=max_len)(sequence_input)
    else:
        embedded_sequences = embedding_layer(embedding_matrix,max_len)(sequence_input)
    lstm = Bidirectional(LSTM(128,
                            dropout=0.3,
                            return_sequences=True,
                            recurrent_activation='relu',
                            recurrent_initializer='glorot_uniform'))(embedded_sequences)
    lstm = Bidirectional(LSTM(128,
                              dropout=0.3,
                              return_sequences=True,
                              recurrent_activation='relu',
                              recurrent_initializer='glorot_uniform'))(lstm)
    context_vector = Attention(1)(lstm) #1只是为了初始化，初始化完成后会根据前一层计算size
    output = Dense(3,activation='softmax')(context_vector)
    model = Model(inputs=sequence_input, outputs=output)
    model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['accuracy'])
    return model


if __name__ == '__main__':
    vocab_size = 142743
    # model = simple_nn(vocab_size)
    # model.summary()
    # model = simple_nn_l2(vocab_size)
    # model.summary()
    # model = simple_LSTM(vocab_size)
    # model.summary()
    # model = simple_BiLSTM(vocab_size)
    # model.summary()
    # model = cnn_LSTM(vocab_size)
    # model.summary()
    # model = cnn_LSTM2(vocab_size)
    # model.summary()
    # model = simple_cnn(vocab_size)
    # model.summary()
    # model = cnn_multi_filters(vocab_size)
    # model.summary()
    # model = BiLSTM_with_attention(vocab_size)
    # model.summary()

    model = double_BiLSTM(vocab_size)
    model.summary()


