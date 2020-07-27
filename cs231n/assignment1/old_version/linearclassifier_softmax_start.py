import numpy as np
from data_utils import load_CIFAR10
from linear_classifier import *

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the SVM,
    but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '/Users/k/Documents/cifar-10-batches-py'   # make a change
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    # subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

# Use the validation set to tune hyperparameters (regularization strength
# and learning rate).
results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7]
regularization_strengths = [5e4, 1e4]
num_iterations = 1500

for lr in learning_rates:
    for rs in regularization_strengths:
        softmax = Softmax()
        softmax.train(X_train, y_train, learning_rate=lr, lambda_val=rs, num_iterations=num_iterations)
        train_pred = softmax.predict(X_train)
        acc_train = np.mean(y_train == train_pred)
        val_pred = softmax.predict(X_val)
        acc_val = np.mean(y_val == val_pred)
        results[(lr, rs)] = (acc_train, acc_val)
        if best_val < acc_val:
            best_val = acc_val
            best_softmax = softmax

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' %
                                    (lr, reg, train_accuracy, val_accuracy))
        # around 38.9%
print('best validation accuracy achieved during cross-validation: %f' % best_val)

# Evaluate the best softmax on test set.
test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == test_pred)       # around 37.4%
print('Softmax on raw pixels of CIFAR-10 final test set accuracy: %f' % test_accuracy)
