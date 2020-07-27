from utils import *
from model import *
import numpy as np

if __name__ == '__main__':
    X_train, Y_train, X_val, Y_val, X_test, Y_test,transformer_x = load_data2()
    vocab_size = 142743
    use_pretrain_embedding = True
    if use_pretrain_embedding:
        # load word2vec
        word2vec = {}
        with open('../datastories.twitter.200d.txt', 'r', encoding='utf8') as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec

        print('Found %s word vectors.' % len(word2vec))
        # init embedding_layer
        EMBEDDING_DIM = 200
        embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
        word2id = transformer_x.word2id
        for word, vec in word2vec.items():
            if word in word2id:
                embedding_matrix[word2id[word]] = word2vec[word]
        embedding_matrix[0] = word2vec['<pad>']
        model = LSTM_with_attention(vocab_size,embedding_matrix)

    else:
        model = LSTM_with_attention(vocab_size)
    model.summary()
    test_model(model,X_train, Y_train, X_val, Y_val, X_test, Y_test,epochs = 1)
