import tensorflow as tf
import numpy as np
import time
import os, sys
from fmnist_dataset import Fashion_MNIST
from cnn import CNN



if __name__ == "__main__":

    print("start training")

    # Load dataset
    d = Fashion_MNIST()

    # Read hyper-parameters
    keep_prob_fc = 0.5
    keep_prob_conv = 0.5
    bs = 32  # batch_size
    n_epoch = 20
    n_batch_train = int(d.train.size / bs)
    n_batch_valid = int(d.valid.size / bs)
    n_batch_test = int(d.test.size / bs)

    model_path = '../model/naive_model.ckpt'
    dtype = np.float32
    early_stopping_n = 3

    # Build model
    with tf.variable_scope("fmnist_cnn") as vs:
        m = CNN(scope_name="fmnist_cnn", is_inference=False, )
        print("Model built!")

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print("[*] Model initialized!")
        print("[*] Model trainable variables:")
        parm_cnt = 0
        variable = [v for v in tf.trainable_variables()]
        for v in variable:
            print("   ", v.name, v.get_shape())
            parm_cnt_v = 1
            for i in v.get_shape().as_list():
                parm_cnt_v *= i
            parm_cnt += parm_cnt_v
        print("[*] Model parameter size: %.4fM" % (parm_cnt / 1024 / 1024))

        d.train.normalize()
        d.valid.normalize()
        d.test.normalize()

        d.train.reset_epoch()
        d.valid.reset_epoch()
        d.test.reset_epoch()


        best_valid_acc = 0
        best_valid_loss = 1e10
        early_stopping_cnt = 0

        for epoch in range(n_epoch):
            if_test = False
            print("[*] Epoch %d/%d, Training start..." % (epoch + 1, n_epoch), flush=True)
            mean_train_acc = 0
            mean_train_loss = 0
            mean_train_time = 0
            for i in range(n_batch_train):
                iter_ = epoch * n_batch_train + i
                x, y = d.train.next_batch(bs, dtype=dtype)
                begin_time = time.time()
                acc, loss = m.train_op(sess, x, y, iter_, keep_prob_fc, keep_prob_conv)
                end_time = time.time()
                mean_train_acc += acc / n_batch_train
                mean_train_loss += loss / n_batch_train
                mean_train_time += (end_time - begin_time) / n_batch_train
                if np.isnan(loss):
                    print("[*] NaN Stopping!", flush=True)
                    exit(-1)
                print("\t\repoch %d/%d, iteration %d/%d:\t loss = %.3e, acc = %.3f, time = %.3f" \
                      % (epoch + 1, n_epoch, i + 1, n_batch_train, loss, acc, end_time - begin_time),
                      flush=True, end="")
            print("\n[*] Epoch %d/%d, Training done!\n\tloss = %.3e, acc = %.3f, time = %.3f" \
                  % (epoch + 1, n_epoch, mean_train_loss, mean_train_acc, mean_train_time), flush=True)

            print("[*] Epoch %d/%d, Validation start..." % (epoch + 1, n_epoch), flush=True)
            mean_valid_acc = 0
            mean_valid_loss = 0
            mean_valid_time = 0
            for i in range(n_batch_valid):
                x, y = d.valid.next_batch(bs, dtype=dtype)
                begin_time = time.time()
                acc, loss = m.eval_op(sess, x, y)
                end_time = time.time()
                mean_valid_acc += acc / n_batch_valid
                mean_valid_loss += loss / n_batch_valid
                mean_valid_time += (end_time - begin_time) / n_batch_valid
            print("[*] Epoch %d/%d, Validation done!\n\tloss = %.3e, acc = %.3f, time = %.3f" \
                  % (epoch + 1, n_epoch, mean_valid_loss, mean_valid_acc, mean_valid_time), flush=True)
            if mean_valid_loss <= best_valid_loss:
                if_test = True
                best_valid_acc = mean_valid_acc
                best_valid_loss = mean_valid_loss
                early_stopping_cnt = 0
                print("[*] Best validation loss so far! ")
                m.save(sess, model_path)
                print("[*] Model saved at", model_path, flush=True)

            if if_test:
                print("[*] Epoch %d/%d, Testing start..." % (epoch + 1, n_epoch), flush=True)
                mean_test_acc = 0
                mean_test_loss = 0
                mean_test_time = 0
                for i in range(n_batch_test):
                    x, y = d.test.next_batch(bs, dtype=dtype)
                    begin_time = time.time()
                    acc, loss = m.eval_op(sess, x, y)
                    end_time = time.time()
                    mean_test_acc += acc / n_batch_test
                    mean_test_loss += loss / n_batch_test
                    mean_test_time += (end_time - begin_time) / n_batch_test
                print("[*] Epoch %d/%d, Testing done!\n\tloss = %.3e, acc = %.3f, time = %.3f" \
                      % (epoch + 1, n_epoch, mean_test_loss, mean_test_acc, mean_test_time), flush=True)
            else:
                print("[*] Epoch %d/%d, No testing!" % (epoch + 1, n_epoch), flush=True)
                early_stopping_cnt += 1 #超过若干次loss上升则停止训练
                if early_stopping_cnt >= early_stopping_n:
                    print("[*] Early Stopping!", flush=True)
                    exit(-1)

