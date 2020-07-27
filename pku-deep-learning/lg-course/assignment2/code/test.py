import tensorflow as tf
import numpy as np
import random
import time
import pickle
import os, sys
from fmnist_dataset import Fashion_MNIST
from cnn import CNN

if __name__ == "__main__":

    print("[*] Hello world!", flush=True)
    # Load dataset
    d = Fashion_MNIST()
    # Read hyper-parameters
    n_correct = 1000
    correct_path = '../attack_data/correct_1k.pkl'
    # model_path = '../model/fmnist_cnn.ckpt'
    model_path = '../model/naive_model.ckpt'
    dtype = np.float32
    # Build model
    with tf.variable_scope("fmnist_cnn") as vs:
        m = CNN(scope_name="fmnist_cnn", is_inference=True)
        print("[*] Model built!")
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        m.restore(sess, model_path)
        print("[*] Model loaded!")
        print("[*] Model parameters:")
        parm_cnt = 0
        variable = [v for v in tf.trainable_variables()]
        for v in variable:
            print("   ", v.name, v.get_shape())
            parm_cnt_v = 1
            for i in v.get_shape().as_list():
                parm_cnt_v *= i
            parm_cnt += parm_cnt_v
        print("[*] Model parameter size: %.4fM" % (parm_cnt / 1024 / 1024))

        d.test.reset_epoch()

        acc = 0
        correct_image = []
        correct_label = []
        for _iter in range(d.test.size):
            x, y = d.test.next_batch(1, dtype=dtype)
            x  = x / 255
            #如果训练的时候除以了255，那么测试的时候也要做。
            y_hat = m.infer_op(sess, x)
            if (np.argmax(y_hat) == np.argmax(y[0])):
                correct_image.append(x[0])
                correct_label.append(y[0])
                acc += 1
        acc /= d.test.size
        print("[*] Accuracy on test set: %.5f" % (acc))

        _correct_image = []
        _correct_label = []
        _idx = random.sample(range(len(correct_image)), n_correct)
        for i in _idx:
            _correct_image.append(correct_image[i])
            _correct_label.append(correct_label[i])
        _correct_image = np.asarray(_correct_image, dtype=dtype).reshape((-1, 1, 28, 28))
        _correct_label = np.asarray(_correct_label, dtype=dtype).reshape((-1, 1, 10))
        with open(correct_path, "wb") as f:
            pickle.dump([_correct_image, _correct_label], f)
