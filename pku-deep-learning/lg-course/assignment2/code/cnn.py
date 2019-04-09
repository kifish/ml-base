import tensorflow as tf
import numpy as np


# reference: https://github.com/kifish/Fashion-MNIST/blob/master/code/model.py

class CNN(object):
    def __init__(self,scope_name = '',is_inference=False,
                 lr=[1e-5, 2e-2], lr_decay=2000, grad_clip=5):
        # Hyper-parameters
        self.__K = 24      # Conv-1 depth
        self.__stride1 = 1 # Conv-1 stride
        self.__L = 48      # Conv-2 depth
        self.__stride2 = 2 # Conv-2 stride
        self.__M = 64      # Conv-3 depth
        self.__stride3 = 3 # Conv-3 stride
        self.__N = 128     # FC width

        self.__pkeep_conv = 0.5
        self.__pkeep = 0.5

        self.__min_lr = lr[0]   # Minimum learning rate
        self.__max_lr = lr[1]   # Maximum learning rate
        self.__decay_step = lr_decay    # Learning rate exponentional decay
        self.__grad_clipping = grad_clip    # Gradient clipping by absolute value

        self.__dtype = np.float32

        self.__add_placeholders()
        self.__add_variables_version5()

        # bulid graph
        (self.__Ylogits,self.__Y,self.__update_ema) = self.__build_graph_version6(self.__X)
        self.__trainables = [x for x in tf.global_variables() if x.name.startswith(scope_name)]

        if not is_inference: #train
            (self.__loss, \
             self.__accuracy, \
             self.__train_step) = self.__build_training_graph()
        else:               #infer
            (self.__loss, \
             self.__accuracy, \
             self.__train_step) = self.__build_inference_graph()

        self.__saver = tf.train.Saver(var_list=self.__trainables,
                                      max_to_keep=1)

        # Gradient on input image
        self.__grad = tf.gradients(self.__loss, self.__X)

    def __add_placeholders(self):
        self.__X = tf.placeholder(self.__dtype, [None, 28, 28, 1])
        self.__Y_ = tf.placeholder(self.__dtype, [None, 10])
        self.__iter = tf.placeholder(tf.int32)
        self.__tst = tf.placeholder(tf.bool)  # test flag for batch norm
        self.__pkeep = tf.placeholder(self.__dtype)      # dropout probability
        self.__pkeep_conv = tf.placeholder(self.__dtype)

    def __add_variables_version1(self):
        # Conv-1 weights
        self.__W1 = tf.Variable(tf.truncated_normal([3, 3, 1, self.__K],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B1 = tf.Variable(tf.constant(0.1, self.__dtype, [self.__K]))
        # Conv-2 weights
        self.__W2 = tf.Variable(tf.truncated_normal([3, 3, self.__K, self.__L],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B2 = tf.Variable(tf.constant(0.1, self.__dtype, [self.__L]))

        # FC weights
        self.__W3 = tf.Variable(tf.truncated_normal([14 * 14 * self.__L, self.__N],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B3 = tf.Variable(tf.constant(0.1, self.__dtype, [self.__N]))

        # Softmax weights
        self.__W4 = tf.Variable(tf.truncated_normal([self.__N, 10],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B4 = tf.Variable(tf.constant(0.1, self.__dtype, [10]))

    def __add_variables_version2(self):
        # Conv-1 weights
        self.__W1 = tf.Variable(tf.truncated_normal([3, 3, 1, self.__K],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B1 = tf.Variable(tf.constant(0.1, self.__dtype, [self.__K]))

        # FC weights
        self.__W2 = tf.Variable(tf.truncated_normal([13 * 13 * self.__K, self.__N],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B2 = tf.Variable(tf.constant(0.1, self.__dtype, [self.__N]))

        # Softmax weights
        self.__W3 = tf.Variable(tf.truncated_normal([self.__N, 10],
                                                    stddev=0.1, dtype=self.__dtype))
        self.__B3 = tf.Variable(tf.constant(0.1, self.__dtype, [10]))

    #改一下初始化方法
    def __add_variables_version3(self):
        # Conv-1 weights
        self.__W1 = tf.Variable(tf.glorot_uniform_initializer()((3, 3, 1, self.__K)))
        self.__B1 = tf.Variable(tf.constant(0, self.__dtype, [self.__K]))

        # FC weights
        self.__W2 = tf.get_variable('W2',[13 * 13 * self.__K, self.__N],initializer=tf.random_normal_initializer())
        self.__B2 = tf.Variable(tf.constant(0, self.__dtype, [self.__N]))

        # Softmax weights
        self.__W3 = tf.get_variable('W3',[self.__N, 10],initializer=tf.random_normal_initializer())
        self.__B3 = tf.Variable(tf.constant(0, self.__dtype, [10]))



    def __add_variables_version5(self):
        # Conv-1 weights
        self.__W1 = tf.Variable(tf.glorot_uniform_initializer()((5, 5, 1, self.__K)))
        self.__B1 = tf.Variable(tf.constant(0, self.__dtype, [self.__K]))

        # Conv-2 weights
        self.__W2 = tf.Variable(tf.glorot_uniform_initializer()((4, 4, self.__K, self.__L)))
        self.__B2 = tf.Variable(tf.constant(0, self.__dtype, [self.__L]))

        # Conv-3 weights
        self.__W3 = tf.Variable(tf.glorot_uniform_initializer()((3, 3, self.__L, self.__M)))
        self.__B3 = tf.Variable(tf.constant(0, self.__dtype, [self.__M]))


        # FC weights
        self.__W4 = tf.get_variable('W4',[5 * 5 * self.__M, self.__N],initializer=tf.random_normal_initializer())
        self.__B4 = tf.Variable(tf.constant(0, self.__dtype, [self.__N]))

        # Softmax weights
        self.__W5 = tf.get_variable('W5',[self.__N, 10],initializer=tf.random_normal_initializer())
        self.__B5 = tf.Variable(tf.constant(0, self.__dtype, [10]))

    def __build_graph_version5(self, X):
        Y1l = tf.nn.conv2d(X, self.__W1,
                           strides=[1, self.__stride1, self.__stride1, 1],
                           padding='SAME')
        Y1bn, update_ema1 = self.__batchnorm(Y1l, self.__tst, self.__iter,
                                             self.__B1, convolutional=True)
        Y1r = tf.nn.relu(Y1bn)
        # output shape is [None,28,28,24]
        Y1d = tf.nn.dropout(Y1r, self.__pkeep_conv)
        Y2l = tf.nn.conv2d(Y1d, self.__W2,
                           strides=[1, self.__stride2, self.__stride2, 1],
                           padding='SAME')
        Y2bn, update_ema2 = self.__batchnorm(Y2l, self.__tst, self.__iter,
                                             self.__B2, convolutional=True)
        Y2r = tf.nn.relu(Y2bn)
        Y2d = tf.nn.dropout(Y2r,self.__pkeep_conv)
        # output shape is [None,14,14,48]
        Y3l = tf.nn.conv2d(Y2d,self.__W3,strides=[1, self.__stride3, self.__stride3, 1],
                           padding='SAME')
        Y3bn, update_ema3 = self.__batchnorm(Y3l, self.__tst, self.__iter,
                                             self.__B3, convolutional=True)
        Y3r = tf.nn.relu(Y3bn)
        Y3d = tf.nn.dropout(Y3r,self.__pkeep_conv)
        # output shape is [None,5,5,64]

        Y3f = tf.reshape(Y3d, shape=[-1, 5 * 5 * self.__M])
        Y4l = tf.matmul(Y3f, self.__W4)
        Y4bn, update_ema4 = self.__batchnorm(Y4l, self.__tst,
                                             self.__iter, self.__B4)
        Y4r = tf.nn.relu(Y4bn) #这里的relu非常重要，没有relu这一层的作用就很小了
        Y4d = tf.nn.dropout(Y4r, self.__pkeep)
        Ylogits = tf.matmul(Y4d, self.__W5) + self.__B5
        Y = tf.nn.softmax(Ylogits)
        update_ema = tf.group(update_ema1, update_ema2,
                              update_ema3, update_ema4)

        return Ylogits, Y, update_ema

    def __build_graph_version6(self, X):
        Y1l = tf.nn.conv2d(X, self.__W1,
                           strides=[1, self.__stride1, self.__stride1, 1],
                           padding='SAME')
        Y1bn, update_ema1 = self.__batchnorm(Y1l, self.__tst, self.__iter,
                                             self.__B1, convolutional=True)
        Y1r = tf.nn.relu(Y1bn)
        # output shape is [None,28,28,24]
        Y1d = tf.nn.dropout(Y1r, self.__pkeep_conv,
                           self.__compatible_convolutional_noise_shape(Y1r))
        Y2l = tf.nn.conv2d(Y1d, self.__W2,
                           strides=[1, self.__stride2, self.__stride2, 1],
                           padding='SAME')
        Y2bn, update_ema2 = self.__batchnorm(Y2l, self.__tst, self.__iter,
                                             self.__B2, convolutional=True)
        Y2r = tf.nn.relu(Y2bn)
        Y2d = tf.nn.dropout(Y2r,self.__pkeep_conv,
                           self.__compatible_convolutional_noise_shape(Y2r))
        # output shape is [None,14,14,48]
        Y3l = tf.nn.conv2d(Y2d,self.__W3,strides=[1, self.__stride3, self.__stride3, 1],
                           padding='SAME')
        Y3bn, update_ema3 = self.__batchnorm(Y3l, self.__tst, self.__iter,
                                             self.__B3, convolutional=True)
        Y3r = tf.nn.relu(Y3bn)
        Y3d = tf.nn.dropout(Y3r,self.__pkeep_conv,
                           self.__compatible_convolutional_noise_shape(Y3r))
        # output shape is [None,5,5,64]

        Y3f = tf.reshape(Y3d, shape=[-1, 5 * 5 * self.__M])
        Y4l = tf.matmul(Y3f, self.__W4)
        Y4bn, update_ema4 = self.__batchnorm(Y4l, self.__tst,
                                             self.__iter, self.__B4)
        Y4r = tf.nn.relu(Y4bn) #这里的relu非常重要，没有relu这一层的作用就很小了
        Y4d = tf.nn.dropout(Y4r, self.__pkeep)
        Ylogits = tf.matmul(Y4d, self.__W5) + self.__B5
        Y = tf.nn.softmax(Ylogits)
        update_ema = tf.group(update_ema1, update_ema2,
                              update_ema3, update_ema4)

        return Ylogits, Y, update_ema

    def __build_graph_version2(self, X):
        Y1l = tf.nn.conv2d(X, self.__W1,
                           strides=[1, self.__stride1, self.__stride1, 1],
                           padding='VALID')
        Y1b = tf.nn.bias_add(Y1l, self.__B1)
        Y1r = tf.nn.relu(Y1b)
        # output shape is [None,26,26,32]

        Y1p = tf.layers.max_pooling2d(
            Y1r,
            pool_size=[2, 2],
            strides = 2,
            padding='valid',
            data_format='channels_last',
            name=None
        )
        # output shape is [None,14,14,64]
        Y1d = tf.nn.dropout(Y1p, 0.8) #KEEP_PROB

        Y2 = tf.layers.flatten(
            Y1d,
            name=None,
        )
        # output shape is [None,14*14*64]

        Y2l = tf.matmul(Y2, self.__W2)
        Y2b = tf.nn.bias_add(Y2l,self.__B2)
        Y2r = tf.nn.relu(Y2b)
        Ylogits = tf.matmul(Y2r, self.__W3) + self.__B3
        Y = tf.nn.softmax(Ylogits)

        return Ylogits, Y

    def __build_graph_version1(self, X):
        Y1l = tf.nn.conv2d(X, self.__W1,
                           strides=[1, self.__stride1, self.__stride1, 1],
                           padding='SAME')
        Y1b = tf.nn.bias_add(Y1l, self.__B1)
        Y1r = tf.nn.relu(Y1b)
        # output shape is [None,28,28,32]


        Y2l = tf.nn.conv2d(Y1r, self.__W2,
                           strides=[1, self.__stride2, self.__stride2, 1],
                           padding='SAME')
        Y2b = tf.nn.bias_add(Y2l, self.__B2)
        Y2r = tf.nn.relu(Y2b)
        # output shape is [None,28,28,64]

        Y2p = tf.layers.max_pooling2d(
            Y2r,
            pool_size=[2, 2],
            strides = 2,
            padding='valid',
            data_format='channels_last',
            name=None
        )
        # output shape is [None,14,14,64]
        Y2 = tf.nn.dropout(Y2p, self.__pkeep_conv)

        Y2f = tf.layers.flatten(
            Y2,
            name=None,
        )
        # output shape is [None,14*14*64]

        Y3l = tf.matmul(Y2f, self.__W3)
        Y3b = tf.nn.bias_add(Y3l,self.__B3)
        Y3d = tf.nn.dropout(Y3b, self.__pkeep)
        Ylogits = tf.matmul(Y3d, self.__W4) + self.__B4
        Y = tf.nn.softmax(Ylogits)

        return Ylogits, Y

    def __build_training_graph(self):
        loss_ = tf.nn.softmax_cross_entropy_with_logits(logits=self.__Ylogits,
                                                        labels=self.__Y_)
        loss = tf.reduce_mean(loss_) #loss_算的是一个batch

        correct_prediction = tf.equal(tf.argmax(self.__Y, 1),
                                      tf.argmax(self.__Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, self.__dtype))
        lr = self.__min_lr + tf.train.exponential_decay(self.__max_lr,
                                                        self.__iter,
                                                        self.__decay_step,
                                                        1 / np.e)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        grads_and_vars = opt.compute_gradients(loss=loss,
                                               var_list=self.__trainables)
        grads_and_vars_ = []
        #clip
        for g, v in grads_and_vars:
            grads_and_vars_.append((tf.clip_by_value(g,
                                                     -self.__grad_clipping,
                                                     self.__grad_clipping), v))
        train_step = opt.apply_gradients(grads_and_vars)
        return loss, accuracy, train_step

    def __build_inference_graph(self):
        loss_ = tf.nn.softmax_cross_entropy_with_logits(logits=self.__Ylogits,
                                                        labels=self.__Y_)
        loss = tf.reduce_mean(loss_)

        correct_prediction = tf.equal(tf.argmax(self.__Y, 1),
                                      tf.argmax(self.__Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, self.__dtype))
        return loss, accuracy, None

    def __batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):

        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def __compatible_convolutional_noise_shape(self, Y):
        noiseshape = tf.shape(Y)
        noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
        return noiseshape


    def train_op(self, sess, x, y, iter_, pkeep=1.0, pkeep_conv=1.0):
        (_, acc, loss) = sess.run([self.__train_step, self.__accuracy, self.__loss],
                                  feed_dict={self.__X: x,
                                             self.__Y_: y,
                                             self.__iter: iter_,
                                             self.__tst: False,
                                             self.__pkeep: pkeep,
                                             self.__pkeep_conv: pkeep_conv})
        (_) = sess.run([self.__update_ema], feed_dict={self.__X: x,
                                                       self.__Y_: y,
                                                       self.__iter: iter_,
                                                       self.__tst: False,
                                                       self.__pkeep: 1.0,
                                                       self.__pkeep_conv: 1.0})
        return acc, loss

    def eval_op(self, sess, x, y):
        (acc, loss) = sess.run([self.__accuracy, self.__loss],
                               feed_dict={self.__X: x,
                                          self.__Y_: y,
                                          self.__iter: 0,
                                          self.__tst: True,
                                          self.__pkeep: 1.0,
                                          self.__pkeep_conv: 1.0})
        return acc, loss

    def infer_op(self, sess, x):
        (y) = sess.run([self.__Y],
                       feed_dict={self.__X: x,
                                  self.__iter: 0,
                                  self.__tst: True,
                                  self.__pkeep: 1.0,
                                  self.__pkeep_conv: 1.0})
        return y

    def grad_op(self, sess, x, y):
        (grad) = sess.run([self.__grad],
                          feed_dict={self.__X: x,
                                     self.__Y_: y,
                                     self.__iter: 0,
                                     self.__tst: True,
                                     self.__pkeep: 1.0,
                                     self.__pkeep_conv: 1.0})
        return grad

    def model_summary(self):
        import tensorflow.contrib.slim as slim
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)


    def save(self, sess, path):
        self.__saver.save(sess, path)

    def restore(self, sess, path):
        self.__saver.restore(sess, path)

    def get_trainables(self):
        return self.__trainables


if __name__ == "__main__":
    from fmnist_dataset import Fashion_MNIST

    with tf.variable_scope("fmnist_cnn") as vs:
        m = CNN(scope_name="fmnist_cnn")
    d = Fashion_MNIST()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    x, y = d.train.next_batch(1)
    print(m.eval_op(sess, x, y))
    print(m.infer_op(sess, x))
    print(m.train_op(sess, x, y, 0, 0.9, 1.0))
    print(m.eval_op(sess, x, y))
    print(m.infer_op(sess, x))
    print(m.model_summary())



#version1:
#未预处理,0-255
# [*] Epoch 12/12, Testing done!
# 	loss = 4.649e-01, acc = 0.834, time = 0.123
#预处理,0-1
# [*] Epoch 12/12, Testing done!
# 	loss = 3.151e-01, acc = 0.886, time = 0.108
# x  = x / 255 # 0.77920->0.88540
# 如果训练的时候除以了255，那么测试的时候也要做。
# [*] Model parameters:
#     fmnist_cnn/Variable:0 (3, 3, 1, 32)
#     fmnist_cnn/Variable_1:0 (32,)
#     fmnist_cnn/Variable_2:0 (3, 3, 32, 64)
#     fmnist_cnn/Variable_3:0 (64,)
#     fmnist_cnn/Variable_4:0 (12544, 128)
#     fmnist_cnn/Variable_5:0 (128,)
#     fmnist_cnn/Variable_6:0 (128, 10)
#     fmnist_cnn/Variable_7:0 (10,)
# [*] Model parameter size: 1.5505M



#version2:
#x = x/255
# [*] Epoch 11/12, Testing done!
# 	loss = 4.622e-01, acc = 0.829, time = 0.023
# [*] Epoch 12/12, Validation done!
# 	loss = 4.506e-01, acc = 0.828, time = 0.022
# [*] Epoch 12/12, No testing!
# 同样的网络结构效果比keras差,可能bias被初始化为0.1了，而keras默认bias0初始化
# https://stackoverflow.com/questions/46883606/what-is-the-default-kernel-initializer-in-keras



#version3:
#在version2基础上改了初始化方法，与keras吻合（https://nbviewer.jupyter.org/github/khanhnamle1994/fashion-mnist/blob/master/CNN-1Conv.ipynb）。
# https://github.com/keras-team/keras/blob/62d097c4ff6fa694a4dbc670e9c7eb9e2bc27c74/keras/layers/core.py#L798
#batch_size 128
# [*] Epoch 12/12, Testing done!
# 	loss = 3.548e-01, acc = 0.870, time = 0.027

#version4:
#batch_size 256
#epoch 20
# [*] Accuracy on test set: 0.86140

#tf写的，acc到89%
# https://medium.com/datadriveninvestor/implementing-convolutional-neural-network-using-tensorflow-for-fashion-mnist-caa99e423371

#version5:
#三层卷积
# [*] Epoch 3/20, Testing start...
# [*] Epoch 3/20, Testing done!
# 	loss = 2.303e+00, acc = 0.100, time = 0.015
#没有batch_norm,loss降不下去
#加了batch_norm,
#没加__compatible_convolutional_noise_shape
# [*] Epoch 18/20, No testing!
# [*] Early Stopping!
# [*] Accuracy on test set: 0.85710


#version6:
#在version5基础上加__compatible_convolutional_noise_shape







