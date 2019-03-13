from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.gridspec as gridspec

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
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
    idxs = np.random.choice(idxs, samples_per_class, replace=False)  # 随机抽7张该类图片
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

# Epoch 149 training loss: 0.13785467008890767, test acc: 0.9292
# Create the model
x = tf.placeholder(tf.float32, [None, 784])
# W_1 = tf.Variable(tf.zeros([784, 128]))
# 第一层
W_1 = tf.get_variable('W_1', [784, 100], initializer=tf.random_normal_initializer())
# b_1 = tf.Variable(tf.zeros([128]))
b_1 = tf.get_variable('b_1', [100], initializer=tf.random_normal_initializer())
a_1 = tf.matmul(x, W_1) + b_1
z_1 = tf.nn.relu(a_1)

# 第二层
# W_2 = tf.Variable(tf.zeros([128, 10]))
W_2 = tf.get_variable('W_2', [100, 10], initializer=tf.random_normal_initializer())
# b_2 = tf.Variable(tf.zeros([10]))
b_2 = tf.get_variable('b_2', [10], initializer=tf.random_normal_initializer())
a_2 = tf.matmul(z_1, W_2) + b_2


# Define loss and optimizer
y = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a_2, labels=y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # 学习率为0.01

correct_prediction = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))  # (num_training,1),预测正确为1，反之为0
accuracy_op = tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32))  # tf.int32 注意tf.int32会导致，acc算出来一直为0，reduce_mean可能要求输入为浮点数
init_op = tf.global_variables_initializer()

batch_size = 32
num_epoch = 150
with tf.Session() as sess:
    sess.run(init_op)
    num_train = X_train.shape[0]
    num_batch = int(num_train / batch_size) + 1
    loss_history = []
    val_acc_history = []
    train_acc_history = []
    for epoch in range(num_epoch):
        avg_cost = 0
        for batch_idx in range(num_batch):
            batch_mask = np.random.choice(num_train, batch_size)
            X_batch = X_train[batch_mask]
            y_batch = y_train[batch_mask]
            _, c = sess.run([train_step, loss], feed_dict={x: X_batch, y: y_batch})
            avg_cost += c / num_batch

        loss_history.append(avg_cost)
        acc = sess.run(accuracy_op, feed_dict={x: X_train, y: y_train})
        train_acc_history.append(acc)
        acc = sess.run(accuracy_op, feed_dict={x: X_val, y: y_val})
        val_acc_history.append(acc)
        # evaluate
        acc = sess.run(accuracy_op, feed_dict={x: X_test, y: y_test})
        print("Epoch %s training loss: %s, test acc: %s" % (epoch, avg_cost, acc))

    print('Test some samples')
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(7, 10)
    gs.update(wspace=0.035, hspace=0.1)  # set the spacing between axes.
    for y_, cls in enumerate(classes):
        idxs = np.flatnonzero(y_ == np.argmax(y_test, axis=1))  # 1代表行
        idxs = np.random.choice(idxs, samples_per_class, replace=False)  # 随机抽7张该类图片
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y_
            ax = plt.subplot(gs[plt_idx])
            ax.axis('off')
            plt.imshow(X_test[idx].reshape((28, 28)))
            pred = sess.run(a_2, feed_dict={x: np.array([X_test[idx]])})  # 返回的是numpy.ndarray
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