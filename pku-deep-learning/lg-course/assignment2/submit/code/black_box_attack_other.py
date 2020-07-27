import tensorflow as tf
import numpy as np
import numpy
import random
import time
import pickle
import os, sys
from example_model import CNN
from fmnist_dataset import Fashion_MNIST
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

tf.app.flags.DEFINE_integer("rand_seed", 2019,
                            "seed for random number generaters")
tf.app.flags.DEFINE_string("gpu", "0",
                           "select one gpu")

tf.app.flags.DEFINE_integer("n_correct", 1000,
                            "correct example number")
tf.app.flags.DEFINE_string("correct_path", "../attack_data/correct_1k.pkl",
                           "pickle file to store the correct labeled examples")
tf.app.flags.DEFINE_string("model_path", "../model/fmnist_cnn.ckpt",
                           "check point path, where the model is saved")

tf.app.flags.DEFINE_string("dtype", "fp32",
                           "data type. \"fp16\", \"fp32\" or \"fp64\" only")
flags = tf.app.flags.FLAGS

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
        self.alpha = 1  # 学习率
        self.old_images_ph = tf.placeholder(np.float32, [None, 28, 28, 1])
        self.adv_images_ph = tf.placeholder(np.float32, [None, 28, 28, 1])
        self.l2_grad = 2 * (self.adv_images_ph - self.old_images_ph)
        self.eps = 0.001

    def estimated_grad(self, x, target, sess):
        # x:[28,28,1]
        grad_val = np.zeros(x.shape)
        row = grad_val.shape[0]
        col = grad_val.shape[1]
        #         print('target',target.shape)
        for i in range(row):
            for j in range(col):
                new_x = x.copy()
                new_x[i, j] += self.eps
                log_pred = self.model.infer_op(sess, np.expand_dims(new_x, 0))
                log_pred = log_pred[0]
                m = log_pred.shape[0]
                #                 print('log,',log_pred.shape)
                f_h_eps1 = -1 * log_pred[list(range(m)), target]
                #                 print('f shape:',f_h_eps1.shape)
                new_x = x.copy()
                new_x[i, j] -= self.eps
                log_pred = self.model.infer_op(sess, np.expand_dims(new_x, 0))
                log_pred = log_pred[0]
                f_h_eps2 = -1 * log_pred[list(range(m)), target]
                grad = (f_h_eps1 - f_h_eps2) / (2 * self.eps)
                grad_val[i][j] = grad[0]
        #         print('shape:',grad_val.shape)  #(28,28,1)
        return grad_val

    def gen_adv(self, x, label, sess, iterations=10):
        old_image = x.copy()
        label = np.argmax(label)
        target = self.old2new[label]  # target为单个数
        for _ in range(iterations):
            grad_val = self.estimated_grad(x, target, sess)
            #             print(grad_val)
            x -= self.alpha * np.sign(grad_val)
            x = np.clip(x, 0., 255.) #要把范围改一下
        x_adv = x  # （28，28，1）
        print('orginal label', label)
        pred = self.model.infer_op(sess, np.expand_dims(old_image, 0))
        pred = pred[0]
        prob = np.exp(pred) / np.sum(np.exp(pred))
        prob_ori = prob[0]
        pred = self.model.infer_op(sess, np.expand_dims(x_adv, 0))
        pred = np.array(pred)
        #         print(pred.shape)
        pred = pred[0]
        #         print(pred.shape)
        prob = np.exp(pred) / np.sum(np.exp(pred))
        prob_new = prob[0]
        pred = np.argmax(pred, 1)[0]
        print('now prediction is', pred)
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
        plt.imshow(x_adv.reshape((28, 28)))
        plt.title('new image')

        plt.subplot(2, 1, 2)
        plt.title('original image')
        plt.imshow(old_image.reshape((28, 28)))
        #     plt.gcf().set_size_inches(15, 12)
        plt.show()
        return x_adv

    def check(self, x, target, sess):
        pred = self.model.infer_op(sess, np.expand_dims(x, 0))
        pred = np.array(pred)
        pred = pred[0]
        pred = np.argmax(pred, 1)[0]
        if target == pred:
            return True
        return False

    def gen_adv_fast(self, x, label, sess, iterations=10):
        old_image = x.copy()
        label = np.argmax(label)
        target = self.old2new[label]  # target为scalr
        for _ in range(iterations):
            grad_val = self.estimated_grad(x, target, sess)
            x -= self.alpha * np.sign(grad_val)
            x = np.clip(x, 0., 255.)
            if self.check(x, target, sess):
                return (old_image, x, target, True)  # 攻击成功
        return (old_image, x, target, False)  # 攻击失败

    def gen_advs_spe(self, X, Y, sess, iterations=50):
        samples = []
        cnt = 0
        for i in range(X.shape[0]):
            print('attacking sample',i)
            sample = self.gen_adv_fast(X[i], Y[i], sess, iterations)
            flag = sample[3]
            if flag:
                cnt += 1
                samples.append(sample)

        acc = cnt / X.shape[0]
        print('attack success rate:', acc)
        idxs = np.random.choice(len(samples), 10, replace=False)
        selected_samples = []
        for idx in idxs:
            selected_samples.append(samples[idx])
        self.save_samples2(selected_samples,acc)

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

    def save_samples1(self, samples,acc):
        if not os.path.exists('./black_attack_results/my'):
            os.mkdir('./black_attack_results/')
            os.mkdir('./black_attack_results/my')
        root_path = './black_attack_results/my'
        for idx, sample in enumerate(samples):
            ori_image = sample[0]
            adv_image = sample[1]
            label = sample[2]
            file_path = os.path.join(root_path, 'group {},class {},original_img.png'.format(idx, label))
            plt.imsave(file_path, ori_image.reshape(28, 28))
            file_path = os.path.join(root_path, 'group {},class {},adversarial_img.png'.format(idx, label))
            plt.imsave(file_path, adv_image.reshape(28, 28))
        with open(root_path+'/acc.txt','w') as f:
            f.write(str(acc))

    def save_samples2(self, samples,acc):
        if not os.path.exists('./black_attack_results/'):
            os.mkdir('./black_attack_results/')
        if not os.path.exists('./black_attack_results/other'):
            os.mkdir('./black_attack_results/other')
        root_path = './black_attack_results/other'
        for idx, sample in enumerate(samples):
            ori_image = sample[0]
            adv_image = sample[1]
            label = sample[2]
            file_path = os.path.join(root_path, 'group {},class {},original_img.png'.format(idx, label))
            plt.imsave(file_path, ori_image.reshape(28, 28))
            file_path = os.path.join(root_path, 'group {},class {},adversarial_img.png'.format(idx, label))
            plt.imsave(file_path, adv_image.reshape(28, 28))
        with open(root_path+'/acc.txt','w') as f:
            f.write(str(acc))



if __name__ == "__main__":
#这里的correct_1k.pkl 很坑图像的形状变了 变成了(1000, 1, 28, 28)
#不要用这个用，重新用助教的代码生成。


    print("[*] Hello world!", flush=True)

    # Select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu

    # Set random seed
    tf.set_random_seed(flags.rand_seed)
    random.seed(flags.rand_seed)
    numpy.random.seed(flags.rand_seed)

    # Load dataset
    d = Fashion_MNIST()

    # Read hyper-parameters
    n_correct = flags.n_correct
    correct_path = flags.correct_path
    model_path = flags.model_path
    if flags.dtype == "fp16":
        dtype = numpy.float16
    elif flags.dtype == "fp32":
        dtype = numpy.float32
    elif flags.dtype == "fp64":
        dtype = numpy.float64
    else:
        assert False, "Invalid data type (%s). Use \"fp16\", \"fp32\" or \"fp64\" only" % flags.dtype

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

        with open('../attack_data/correct_1k.pkl','rb') as f:
            X_attack, Y_attack = pickle.load(f)

        # print(X_attack.shape)
        # print(Y_attack.shape)
        # print(X_attack[0])
        # print(Y_attack[0])
        # print(X_attack.shape)
        # print(Y_attack.shape)
        # acc = 0
        # for i in range(X_attack.shape[0]):
        #     x, y = X_attack[i],Y_attack[i]
        #     y_hat = m.infer_op(sess, np.expand_dims(x,0))
        #     if (numpy.argmax(y_hat[0]) == numpy.argmax(y)):
        #         acc += 1
        # acc /= X_attack.shape[0]
        # print("[*] Accuracy on 1000k set: %.5f" % (acc))

        # X_attack = X_attack/255 #不要除以255，因为模型用0-255训练的；要调整eps和学习率
        attacker = Attacker(m)
        attacker.gen_advs_spe(X_attack[:12], Y_attack[:12], sess, iterations=50)  # 非常耗时


