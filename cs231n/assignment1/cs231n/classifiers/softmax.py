import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - W: D x C array of weights
    - X: N x D array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)  # D*C
    dW_each = np.zeros_like(W)
    num_train, num_feature = X.shape
    num_class = W.shape[1]
    f = X.dot(W)  # N*C
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))
    # print(np.max([[0, 1], [2, 3]], axis=1))
    # [1 3]
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)  # N*C
    y_trueClass = np.zeros_like(prob)
    y_trueClass[np.arange(num_train), y] = 1.0
    for i in range(num_train):
        for j in range(num_class):
            loss += -(y_trueClass[i, j]) * np.log(prob[i, j])
            dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :]
        dW += dW_each
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)  # D * C
    num_train, num_feature = X.shape  # N * D

    f = X.dot(W)  # N * C
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f,axis=1),(num_train,1)) # N * 1
    prob = np.exp(f-f_max) / np.sum(np.exp(f-f_max),axis=1,keepdims=True)
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train),y] = 1.0 #N * C
    loss += -np.sum(y_trueClass*np.log(prob))/num_train + 0.5 * reg * np.sum(W*W)
    dW += -np.dot(X.T,y_trueClass-prob) / num_train + reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
