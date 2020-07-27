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
        self.old_images_ph = tf.placeholder(np.float32, [None, 28, 28, 1])
        self.adv_images_ph = tf.placeholder(np.float32, [None, 28, 28, 1])
        self.l2_grad = 2 * (self.adv_images_ph - self.old_images_ph)

    def gen_adv(self, x, label, sess, iterations=10):
        old_image = x.copy()
        x = np.reshape(x, (-1, 28, 28, 1))
        #         print(x.shape)
        #         print('label',label)
        label = np.argmax(label)
        #         print(label)
        target = self.old2new[label]
        #         print(target)
        target = to_categorical(target, 10, dtype='float32')
        #         print(target)
        for _ in range(iterations):
            grad_val = self.model.grad_op(sess, x, np.expand_dims(target, 0))
            grad_val = np.array(grad_val)
            #             print(grad_val.shape)
            grad_val = grad_val[0][0]
            #             print(grad_val.shape)
            x -= self.alpha * np.sign(grad_val)
            x = np.clip(x, 0., 1.)
        x_adv = x
        print('orginal label', label)
        pred = self.model.infer_op(sess, np.expand_dims(old_image, 0))
        pred = pred[0]
        prob = np.exp(pred) / np.sum(np.exp(pred))
        prob_ori = prob[0]
        #         print(prob)

        pred = self.model.infer_op(sess, x_adv)
        pred = np.array(pred)
        #         print(pred.shape)
        pred = pred[0]
        #         print(pred.shape)
        prob = np.exp(pred) / np.sum(np.exp(pred))
        prob_new = prob[0]
        pred = np.argmax(pred, 1)[0]
        print('now prediction is', pred)

        #         print(prob)
        print('        before   after')
        for i in range(len(prob_ori)):
            print('class {}: {:.2f} \t {:.2f}'.format(i, float(prob_ori[i]), float(prob_new[i])))
        if self.old2new[label] == pred:
            print('attack succeed')
        else:
            print('attack fail')

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

    def gen_adv_l2_spe(self, x, label, sess, iterations=10):
        old_image = x.copy()
        x = np.reshape(x, (-1, 28, 28, 1))
        label = np.argmax(label)
        target = self.old2new[label]
        target = to_categorical(target, 10, dtype='float32')
        for _ in range(iterations):
            grad_val = self.model.grad_op(sess, x, np.expand_dims(target, 0))
            grad_val = np.array(grad_val)
            #             print(grad_val.shape)
            grad_val = grad_val[0][0]
            l2_grad_val = sess.run(self.l2_grad,
                                   feed_dict={self.old_images_ph: np.expand_dims(old_image, 0), self.adv_images_ph: x})
            #             print(grad_val.shape)
            grad_val += l2_grad_val
            x -= self.alpha * np.sign(grad_val)
            x = np.clip(x, 0., 1.)
        x_adv = x
        #         print('orginal label',label)
        pred = self.model.infer_op(sess, np.expand_dims(old_image, 0))
        pred = pred[0]
        prob = np.exp(pred) / np.sum(np.exp(pred))
        prob_ori = prob[0]
        #         print(prob)
        pred = self.model.infer_op(sess, x_adv)
        pred = np.array(pred)
        #         print(pred.shape)
        pred = pred[0]
        #         print(pred.shape)
        prob = np.exp(pred) / np.sum(np.exp(pred))
        prob_new = prob[0]
        pred = np.argmax(pred, 1)[0]
        flag = True
        if self.old2new[label] == pred:
            #             print('attack succeed')
            pass
        else:
            #             print('attack fail')
            flag = False
        return flag

    def gen_adv_l2(self, x, label, sess, iterations=10):
        old_image = x.copy()
        x = np.reshape(x, (-1, 28, 28, 1))
        #         print(x.shape)
        #         print('label',label)
        label = np.argmax(label)
        #         print(label)
        target = self.old2new[label]
        #         print(target)
        target = to_categorical(target, 10, dtype='float32')
        #         print(target)
        for _ in range(iterations):
            grad_val = self.model.grad_op(sess, x, np.expand_dims(target, 0))
            grad_val = np.array(grad_val)
            #             print(grad_val.shape)
            grad_val = grad_val[0][0]
            l2_grad_val = sess.run(self.l2_grad,
                                   feed_dict={self.old_images_ph: np.expand_dims(old_image, 0), self.adv_images_ph: x})
            #             print(grad_val.shape)
            grad_val += l2_grad_val
            x -= self.alpha * np.sign(grad_val)
            x = np.clip(x, 0., 1.)
        x_adv = x
        print('orginal label', label)
        pred = self.model.infer_op(sess, np.expand_dims(old_image, 0))
        pred = pred[0]
        prob = np.exp(pred) / np.sum(np.exp(pred))
        prob_ori = prob[0]
        #         print(prob)

        pred = self.model.infer_op(sess, x_adv)
        pred = np.array(pred)
        #         print(pred.shape)
        pred = pred[0]
        #         print(pred.shape)
        prob = np.exp(pred) / np.sum(np.exp(pred))
        prob_new = prob[0]
        pred = np.argmax(pred, 1)[0]
        print('now prediction is', pred)

        #         print(prob)
        print('        before   after')
        for i in range(len(prob_ori)):
            print('class {}: {:.2f} \t {:.2f}'.format(i, float(prob_ori[i]), float(prob_new[i])))
        if self.old2new[label] == pred:
            print('attack succeed')
        else:
            print('attack fail')
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

    def get_targets(self, labels):
        targets = []
        for label in labels:
            label = np.argmax(label)
            target = self.old2new[label]
            targets.append(target)
        targets = to_categorical(targets, 10, dtype='float32')
        return targets

    def gen_advs(self, X, Y, sess, iterations=50):
        X = np.array(X)
        X = np.reshape(X, (-1, 28, 28, 1))
        targets = self.get_targets(Y)
        for _ in range(iterations):
            grad_val = self.model.grad_op(sess, X, targets)
            grad_val = np.array(grad_val)
            #             print(grad_val.shape)
            grad_val = grad_val[0][0]
            #             print(grad_val.shape)
            X -= self.alpha * np.sign(grad_val)
            X = np.clip(X, 0., 1.)
        X_adv = X
        pred = self.model.infer_op(sess, X_adv)
        pred = pred[0]
        test_size = pred.shape[0]
        #         print(pred)
        cnt = 0
        for i in range(test_size):
            if np.argmax(pred[i]) == np.argmax(targets[i]):
                cnt += 1
        acc = cnt / test_size
        print('attack success rate:', acc)

    def gen_advs_and_plot(self, X, Y, sess, iterations=50):
        ori_images = X.copy()
        X = np.array(X)
        X = np.reshape(X, (-1, 28, 28, 1))
        targets = self.get_targets(Y)
        acc_history = []
        for _ in range(iterations):
            grad_val = self.model.grad_op(sess, X, targets)
            grad_val = np.array(grad_val)
            #             print(grad_val.shape)
            grad_val = grad_val[0][0]
            #             print(grad_val.shape)
            X -= self.alpha * np.sign(grad_val)
            X = np.clip(X, 0., 1.)
            pred = self.model.infer_op(sess, X)
            pred = pred[0]
            test_size = pred.shape[0]
            cnt = 0
            for i in range(test_size):
                if np.argmax(pred[i]) == np.argmax(targets[i]):
                    cnt += 1
            acc = cnt / test_size
            acc_history.append(acc)
        plt.plot(acc_history, 'b')
        plt.plot(acc_history, 'bo')
        plt.xlabel('Iteration')
        plt.ylabel('Attack success rate')
        plt.gcf().set_size_inches(15, 12)
        plt.show()
        print('final attack success rate:', acc_history[-1])
        samples = []
        for i in range(test_size):
            if np.argmax(pred[i]) == np.argmax(targets[i]):
                samples.append((ori_images[i], X[i], np.argmax(targets[i])))
        idxs = np.random.choice(len(samples), 10, replace=False)
        selected_samples = []
        for idx in idxs:
            selected_samples.append(samples[idx])
        return selected_samples

    def gen_advs_l2_spe(self, X, Y, sess, iterations=50):
        cnt = 0
        for i in range(X.shape[0]):
            flag = self.gen_adv_l2_spe(X[i], Y[i], sess, iterations)
            if flag:
                cnt += 1
        print('attack success rate:', cnt / X.shape[0])

    def gen_advs_l2_and_plot(self, X, Y, sess, iterations=50):
        ori_images = X.copy()
        X = np.array(X)
        X = np.reshape(X, (-1, 28, 28, 1))
        targets = self.get_targets(Y)
        acc_history = []
        for _ in range(iterations):
            grad_val = self.model.grad_op(sess, X, targets)
            grad_val = np.array(grad_val)
            #             print(grad_val.shape)
            grad_val = grad_val[0][0]
            #             print(grad_val.shape)
            l2_grad_val = sess.run(self.l2_grad, feed_dict={self.old_images_ph: ori_images, self.adv_images_ph: X})
            #             print(l2_grad_val.shape)
            grad_val += 0.1 * l2_grad_val  # 不加0.1波动太大。
            X -= self.alpha * np.sign(grad_val)
            X = np.clip(X, 0., 1.)
            pred = self.model.infer_op(sess, X)
            pred = pred[0]
            test_size = pred.shape[0]
            cnt = 0
            for i in range(test_size):
                if np.argmax(pred[i]) == np.argmax(targets[i]):
                    cnt += 1
            acc = cnt / test_size
            acc_history.append(acc)
        plt.plot(acc_history, 'b')
        plt.plot(acc_history, 'bo')
        plt.xlabel('Iteration')
        plt.ylabel('Attack success rate')
        plt.gcf().set_size_inches(15, 12)
        plt.show()
        print('final attack success rate:', acc_history[-1])
        samples = []
        for i in range(test_size):
            if np.argmax(pred[i]) == np.argmax(targets[i]):
                samples.append((ori_images[i], X[i], np.argmax(targets[i])))
        idxs = np.random.choice(len(samples), 10, replace=False)
        selected_samples = []
        for idx in idxs:
            selected_samples.append(samples[idx])
        return selected_samples

    def plot_samples(self, samples):
        row_n = len(samples)
        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(7, 10)
        gs.update(wspace=0.035, hspace=0.1)  # set the spacing between axes.
        col_n = 2
        for idx, sample in enumerate(samples):
            plt_idx = col_n * idx
            ax = plt.subplot(gs[plt_idx])
            ax.axis('off')
            plt.imshow(sample[0].reshape(28, 28))
            plt_idx = col_n * idx + 1
            ax = plt.subplot(gs[plt_idx])
            plt.imshow(sample[1].reshape(28, 28))
            ax.axis('off')
            plt.title('class:' + str(int(sample[2])))
        #             print('class :',sample[2])
        plt.show()
    def save_samples(self, samples):
        if not os.path.exists('./white_attack_results'):
            os.mkdir('./white_attack_results')
        root_path = './white_attack_results'
        for idx,sample in enumerate(samples):
            ori_image = sample[0]
            adv_image = sample[1]
            label = sample[2]
            file_path = os.path.join(root_path,'group {},class {},original_img.png'.format(idx,label))
            plt.imsave(file_path,ori_image.reshape(28,28))
            file_path = os.path.join(root_path, 'group {},class {},adversarial_img.png'.format(idx, label))
            plt.imsave(file_path, adv_image.reshape(28, 28))

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
        samples = attacker.gen_advs_and_plot(x_attack,y_attack,sess,iterations=50)
        attacker.save_samples(samples)





