from utils import load_CIFAR10
import numpy as np
# Load the raw CIFAR-10 data.

# import os
# print(os.getcwd())
# cifar10_dir = '../data/cifar-10-batches-py'
# print(os.path.abspath(cifar10_dir))

cifar10_dir = '../data/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)     # (50000,32,32,3)
print('Training labels shape: ', y_train.shape)   # (50000L,)
print('Test data shape: ', X_test.shape)        # (10000,32,32,3)
print('Test labels shape: ', y_test.shape)      # (10000L,)


print(y_test[1])


import matplotlib.pyplot as plt

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)#随机抽7张该类图片
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1#将同一类图片显示在同一列
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


num_training = 40000
num_val = 10000
# num_test = 10000
idxs = list(range(num_training + num_val))
np.random.shuffle(idxs)
X_train_all = X_train
y_train_all = y_train
X_train = X_train_all[idxs[:num_training]]
y_train = y_train_all[idxs[:num_training]]
X_val = X_train_all[idxs[num_training:]]
y_val = y_train_all[idxs[num_training:]]


print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Val data shape: ', X_val.shape)
print('Val labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

import tensorflow as tf
# reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
y_train = tf.one_hot(y_train, 10) #10 is the num_class
y_val = tf.one_hot(y_val, 10)
y_test = tf.one_hot(y_test,10)
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Val data shape: ', X_val.shape)
print('Val labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Create the model
x = tf.placeholder(tf.float32, [None, 3072])
# W_1 = tf.Variable(tf.zeros([3072, 30]))
W_1 = tf.get_variable('W_1', [3072, 30], initializer=tf.random_normal_initializer())
# b_1 = tf.Variable(tf.zeros([30]))
b_1 = tf.get_variable('b_1', [30], initializer=tf.random_normal_initializer())
z_1 = tf.matmul(x, W_1) + b_1
a_1 = tf.sigmoid(z_1)

# W_2 = tf.Variable(tf.zeros([30, 10]))
W_2 = tf.get_variable('W_2', [30, 10], initializer=tf.random_normal_initializer())
# b_2 = tf.Variable(tf.zeros([10]))
b_2 = tf.get_variable('b_2', [10], initializer=tf.random_normal_initializer())
z_2 = tf.matmul(a_1, W_2) + b_2
# a_2 = tf.sigmoid(z_2) #不需要加sigmoid
a_2 = z_2


# Define loss and optimizer
y = tf.placeholder(tf.float32, [None, 10])
loss = tf.losses.mean_squared_error(y, a_2)
# loss = tf.reduce_mean(tf.norm(y - a_2, axis=1)**2) / 2
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #0.1为学习率，过高可能会loss爆炸

correct_prediction = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1)) #(num_training,1),预测正确为1，反之为0
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

init_op = tf.global_variables_initializer()

# Train
batch_size = 30
with tf.Session() as sess:
    sess.run(init_op)
    print('aweha')
    for epoch in range(10):
        num_train = X_train.shape[0]
        idxs = np.arange(num_train)
        np.random.shuffle(idxs)
        for batch_idx in range(int(num_train/batch_size) + 1):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            if end_idx > num_train:
                end_idx = num_train
            mask_idxs =  idxs[start_idx:end_idx]
            print('weae')
            print(num_train)
            print(mask_idxs)
            print(X_train.shape)
            buf = [1,2]
            print(X_train[buf])
            print("akweh")
            batch_xs, batch_ys = X_train[mask_idxs], y_train[mask_idxs]
            print(batch_xs)
            print(batch_xs.shape[0],batch_xs.shape[1])
            print(batch_ys.shape[0],batch_ys.shape[1])
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        #evaluate
        acc = accuracy.eval(accuracy, feed_dict={x: X_val,y: y_val})
        print("Epoch %s: %s " % (epoch, acc))



