import tensorflow as tf
import numpy as np
import random
import time
import pickle
import os, sys
from fmnist_dataset import Fashion_MNIST
from cnn import CNN
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


def plot_attack_image(x_pred_true, y_pred_true):
    print('plot...')
    samples_per_class = 7
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(7, 10)
    gs.update(wspace=0.035, hspace=0.1)  # set the spacing between axes.
    classes = range(10)
    num_classes = 10
    for y_, cls in enumerate(classes):
        idxs = np.flatnonzero(y_ == np.argmax(y_pred_true, axis=1))  # 1代表行
        idxs = np.random.choice(idxs, samples_per_class, replace=False)  # 随机抽7张该类图片
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y_
            ax = plt.subplot(gs[plt_idx])
            ax.axis('off')
            plt.imshow(x_pred_true[idx].reshape((28, 28)))
            pred = np.argmax(y_pred_true[idx])
            plt.title('pred:' + str(pred))
    plt.show()


class Attacker(object):
    def __init__(self, model):
        self.model = model
        old = range(10)
        new = list(range(1, 10))
        new.append(0)
        self.old2new = dict(zip(old, new))
        self.alpha = 0.01  # 学习率

    def gen_adv(self, x, label, sess,iterations=50):
        old_image = x.copy()
        x = np.reshape(x, (-1, 28, 28, 1))
        #         print(x.shape)
        label = np.argmax(label)
        target = self.old2new[label]
        #         print(target)
        target = to_categorical(target, 10, dtype='float32')
        #         print(target)
        target = np.expand_dims(target, 0)
        for _ in range(iterations):
            grad_val = self.model.grad_op(sess, x, target)
            x -= np.sign(grad_val)
            x = np.clip(x, 0., 1.)
        x_adv = x
        print('orginal label',np.argmax(label))
        pred = self.model.infer_op(sess,x_adv)
        pred = np.argmax(pred,1)[0]
        print('now prediction is',pred)

        print('plot...')
        plt.figure(figsize=(9, 9))
        plt.subplot(2, 1, 1)
        plt.imshow(x_adv[0].reshape((28, 28)))
        plt.title('new image')

        plt.subplot(2, 1, 2)
        plt.title('original image')
        plt.imshow(old_image.reshape((28, 28)))
        #     plt.gcf().set_size_inches(15, 12)
        plt.show()

        return x_adv

if __name__ == "__main__":

    print("[*] White box attack!", flush=True)
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
            x = x / 255
            # 如果训练的时候除以了255，那么测试的时候也要做。
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

        _correct_image = np.array(_correct_image)
        _correct_label = np.array(_correct_label)

        x_attack = _correct_image
        y_attack = _correct_label

        plot_attack_image(x_attack, y_attack)

        attacker = Attacker(m)
        new_image = attacker.gen_adv(x_attack[0],y_attack[0],sess)


# 注意，助教给的测试接口有问题，可能会重复采样。

