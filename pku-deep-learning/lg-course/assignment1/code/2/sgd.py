from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.gridspec as gridspec
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1] * X_train.shape[2])
# print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

y_train = (np.arange(10) == y_train[:, None]).astype(int)
y_test = (np.arange(10) == y_test[:, None]).astype(int)
# print(y_train.shape)

X_test, X_val = X_test[:5000], X_test[5000:]
y_test, y_val = y_test[:5000], y_test[5000:]

X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255


# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = list(range(10))
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y == np.argmax(y_train, axis=1))  # 1代表行
    idxs = np.random.choice(idxs, samples_per_class,
                            replace=False)  # 随机抽7张该类图片
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1  # 将同一类图片显示在同一列
        plt.subplot(samples_per_class, num_classes, plt_idx)
        image = X_train[idx].reshape((28, 28))
        plt.imshow(image, cmap='gray_r')
#         print(image)
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


# Epoch 0 training loss: 2.6286099129669465, test acc: 0.1758
# Epoch 1 training loss: 2.0510340887781484, test acc: 0.2048
# Epoch 2 training loss: 2.0190940899871688, test acc: 0.2202
# Epoch 3 training loss: 1.9880425774197261, test acc: 0.2614
# Epoch 4 training loss: 1.913675495440446, test acc: 0.322
# Epoch 5 training loss: 1.7749709004631162, test acc: 0.244
# Epoch 6 training loss: 1.6514089073691136, test acc: 0.3798
# Epoch 7 training loss: 1.5449122244301927, test acc: 0.4594
# Epoch 8 training loss: 1.3488295824573546, test acc: 0.4868
# Epoch 9 training loss: 1.2784430912101423, test acc: 0.4824
# Epoch 10 training loss: 1.117623749327509, test acc: 0.6132
# Epoch 11 training loss: 0.9619121092852325, test acc: 0.6474
# Epoch 12 training loss: 0.814970396477098, test acc: 0.692
# Epoch 13 training loss: 0.7470200774657744, test acc: 0.7698
# Epoch 14 training loss: 0.6039202924218917, test acc: 0.8042
# Epoch 15 training loss: 0.535596021975578, test acc: 0.8086
# Epoch 16 training loss: 0.4651128206961593, test acc: 0.8274
# Epoch 17 training loss: 0.418843042187727, test acc: 0.8812
# Epoch 18 training loss: 0.34556974487818554, test acc: 0.8894
# Epoch 19 training loss: 0.305767704853225, test acc: 0.8996
# Epoch 20 training loss: 0.2815905383910304, test acc: 0.9094
# Epoch 21 training loss: 0.2620892396579642, test acc: 0.919
# Epoch 22 training loss: 0.24727546093579006, test acc: 0.9226
# Epoch 23 training loss: 0.2281515530137293, test acc: 0.9208
# Epoch 24 training loss: 0.2096067320004299, test acc: 0.9264
# Epoch 25 training loss: 0.20628963399419376, test acc: 0.929
# Epoch 26 training loss: 0.19286760977602485, test acc: 0.9296
# Epoch 27 training loss: 0.17912752660808082, test acc: 0.9308
# Epoch 28 training loss: 0.1732545846266304, test acc: 0.9312
# Epoch 29 training loss: 0.1825345797133018, test acc: 0.928
# Epoch 30 training loss: 0.1743913241547795, test acc: 0.931
# Epoch 31 training loss: 0.16002540716105024, test acc: 0.9388
# Epoch 32 training loss: 0.1648253254605874, test acc: 0.94
# Epoch 33 training loss: 0.14057212828762541, test acc: 0.94
# Epoch 34 training loss: 0.1437546821527285, test acc: 0.9374
# Epoch 35 training loss: 0.1433591728090371, test acc: 0.935
# Epoch 36 training loss: 0.13513391239071681, test acc: 0.9376
# Epoch 37 training loss: 0.1381998411418496, test acc: 0.9368
# Epoch 38 training loss: 0.12829138644018503, test acc: 0.9394
# Epoch 39 training loss: 0.1178387928405304, test acc: 0.941
# Epoch 40 training loss: 0.12636659690689236, test acc: 0.9412

# sgd的训练效果也非常好
# Epoch 149 training loss: 0.04613999177149101, test acc: 0.9548

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
# W_1 = tf.Variable(tf.zeros([784, 128]))
#第一层
W_1 = tf.get_variable(
    'W_1', [784, 100], initializer=tf.random_normal_initializer())
# b_1 = tf.Variable(tf.zeros([128]))
b_1 = tf.get_variable('b_1', [100], initializer=tf.random_normal_initializer())
a_1 = tf.matmul(x, W_1) + b_1
z_1 = tf.nn.relu(a_1)

#第二层
# W_2 = tf.Variable(tf.zeros([128, 10]))
W_2 = tf.get_variable(
    'W_2', [100, 100], initializer=tf.random_normal_initializer())
# b_2 = tf.Variable(tf.zeros([10]))
b_2 = tf.get_variable('b_2', [100], initializer=tf.random_normal_initializer())
a_2 = tf.matmul(z_1, W_2) + b_2
z_2 = tf.nn.relu(a_2)

#第三层
# W_3 = tf.Variable(tf.zeros([64, 10]))
W_3 = tf.get_variable(
    'W_3', [100, 10], initializer=tf.random_normal_initializer())
# b_3 = tf.Variable(tf.zeros([10]))
b_3 = tf.get_variable('b_3', [10], initializer=tf.random_normal_initializer())
a_3 = tf.matmul(z_2, W_3) + b_3
# z_3 = tf.nn.softmax(a_3)

# Define loss and optimizer
y = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=a_3, labels=y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(
    loss)  # 学习率改为0.01；学习率0.001，test_acc最后只到89%左右

# (num_training,1),预测正确为1，反之为0
correct_prediction = tf.equal(tf.argmax(a_3, 1), tf.argmax(y, 1))
# tf.int32 注意tf.int32会导致，acc算出来一直为0，reduce_mean可能要求输入为浮点数
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init_op = tf.global_variables_initializer()


batch_size = 1
num_epoch = 150
with tf.Session() as sess:
    sess.run(init_op)
    num_train = X_train.shape[0]
    num_batch = int(num_train/batch_size)
    loss_history = []
    val_acc_history = []
    train_acc_history = []
    for epoch in range(num_epoch):
        avg_cost = 0
        for batch_idx in range(num_batch):
            batch_mask = np.random.choice(num_train, batch_size)
            X_batch = X_train[batch_mask]
            y_batch = y_train[batch_mask]
            _, c = sess.run([train_step, loss], feed_dict={
                            x: X_batch, y: y_batch})
            avg_cost += c / num_batch

        loss_history.append(avg_cost)
        acc = sess.run(accuracy_op, feed_dict={x: X_train, y: y_train})
        train_acc_history.append(acc)
        acc = sess.run(accuracy_op, feed_dict={x: X_val, y: y_val})
        val_acc_history.append(acc)
        #evaluate
        acc = sess.run(accuracy_op, feed_dict={x: X_test, y: y_test})
        print("Epoch %s training loss: %s, test acc: %s" %
              (epoch, avg_cost, acc))

    print('Test some samples')
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(7, 10)
    gs.update(wspace=0.035, hspace=0.1)  # set the spacing between axes.
    for y_, cls in enumerate(classes):
        idxs = np.flatnonzero(y_ == np.argmax(y_test, axis=1))  # 1代表行
        idxs = np.random.choice(idxs, samples_per_class,
                                replace=False)  # 随机抽7张该类图片
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y_
            ax = plt.subplot(gs[plt_idx])
            ax.axis('off')
            plt.imshow(X_test[idx].reshape((28, 28)))
            # 返回的是numpy.ndarray
            pred = sess.run(a_3, feed_dict={x: np.array([X_test[idx]])})
            pred = np.argmax(pred, axis=1)[0]
            plt.title('pred:' + str(pred))
    plt.show()

print('plot...')
# Run this cell to visualize training loss and train / val accuracy
plt.figure(figsize=(9, 9))
plt.subplot(2, 1, 1)
# plt.plot(list(range(num_epoch)),loss_history,'b')#bo为点图，b为线图
plt.title('Training loss')
plt.plot(loss_history, 'b')
plt.xlabel('Epoch')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(train_acc_history, 'g', label='train')
plt.plot(val_acc_history, 'r', label='val')
#     plt.plot([0.5] * len(val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()
