import re,os
from collections import defaultdict
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


def load_data_and_labels(path):
    def find_label(strs):
        for s in strs:
            if 'neutral' in s:
                return 0
            elif 'positive' in s:
                return 1
            elif 'negative' in s:
                return 2
        print('Invalid sample:')
        print(strs)
        raise ValueError

    label2id = {'neutral': 0, 'positive': 1, 'negative': 2}
    X = []
    Y = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            try:
                strs = re.split('[\s+\t]', line, 2)
                assert len(strs) == 3
                X.append(strs[-1].split())
                Y.append(label2id[strs[1]])
            except:
                print('Invalid sample:')
                print(line)
                if len(X) == len(Y) + 1:
                    X.pop(-1)
    # print(len(X))
    return X,Y

class Transformer_x():
    def __init__(self):
        self.word2id = defaultdict(lambda : 1) #OOV 是1。无word是0，其他为对应的idx
        self.max_len = None
        self.vocab_size = None

    def fit(self,X):
        all_words = [word for sent in X for word in sent]
        self.max_len = max(map(len,X))
        words = set(all_words)
        self.vocab_size = len(words) + 2 # padding ; OOV
        for index,word in enumerate(words):
            self.word2id[word] = index + 2
        X = [[self.word2id.get(word,1) for word in x] for x in X ]
        X = pad_sequences(X,maxlen = self.max_len)
        return X

    def tran(self,X):
        X = [[self.word2id.get(word,1) for word in x] for x in X ]
        X = pad_sequences(X,maxlen = self.max_len)
        return X


class Transformer_y():
    def __init__(self):
        self.label2id = {'neutral': 0, 'positive': 1, 'negative': 2}
        self.id2label = {v:k for k,v in self.label2id.items()}

    def tran(self, Y):
        Y = to_categorical(Y,3)
        return Y

def load_data(val_ratio = 0.2):
    X = []
    Y = []
    for filename in os.listdir('../data/train/'):
        if filename[-4:] != '.txt':
            continue
        X_,Y_ = load_data_and_labels(os.path.join('../data/train/',filename))
        X.extend(X_)
        Y.extend(Y_)
    transformer_x = Transformer_x()
    X_train = transformer_x.fit(X)
    print('max len:',transformer_x.max_len) # 35
    print('vocab_size:',transformer_x.vocab_size)
    transformer_y = Transformer_y()
    Y_train = transformer_y.tran(Y)
    X_train,X_val,Y_train,Y_val = train_test_split(X_train, Y_train, test_size=0.2)

    X_test,Y_test = load_data_and_labels('../data/SemEval2017-task4-test.subtask-A.english.txt')
    X_test = transformer_x.tran(X_test)
    Y_test = transformer_y.tran(Y_test)
    return X_train,Y_train,X_val,Y_val,X_test,Y_test



def load_data2(val_ratio = 0.2):
    X = []
    Y = []
    for filename in os.listdir('../data/train/'):
        if filename[-4:] != '.txt':
            continue
        X_,Y_ = load_data_and_labels(os.path.join('../data/train/',filename))
        X.extend(X_)
        Y.extend(Y_)
    transformer_x = Transformer_x()
    X_train = transformer_x.fit(X)
    print('max len:',transformer_x.max_len) # 35
    print('vocab_size:',transformer_x.vocab_size)
    transformer_y = Transformer_y()
    Y_train = transformer_y.tran(Y)
    X_train,X_val,Y_train,Y_val = train_test_split(X_train, Y_train, test_size=0.2)

    X_test,Y_test = load_data_and_labels('../data/SemEval2017-task4-test.subtask-A.english.txt')
    X_test = transformer_x.tran(X_test)
    Y_test = transformer_y.tran(Y_test)
    return X_train,Y_train,X_val,Y_val,X_test,Y_test,transformer_x


def test_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test,epochs = 10,verbose=False):
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')
    history = model.fit(X_train,
                        Y_train,
                        epochs=epochs,
                        batch_size=128,
                        validation_data=(X_val, Y_val),
                        verbose=1, callbacks=[early_stopping])
    results = model.evaluate(X_test, Y_test)
    print('loss', results[0])
    print('acc', results[1])
    Y_pred = model.predict(X_test, batch_size=128, verbose=1)
    Y_pred_bool = np.argmax(Y_pred, axis=1)
    Y_test_ = np.argmax(Y_test,1)
    print(classification_report(Y_test_, Y_pred_bool))



    if verbose:
        history_dict = history.history
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        # "bo" is for "blue dot"
        plt.figure(0)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        plt.savefig('Training and validation loss.png')

        plt.figure(1)
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        plt.savefig('Training and validation accuracy.png')


if __name__ == '__main__':
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()
    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)
    print(X_test.shape)
    print(Y_test.shape)

    print(X_train[0])





