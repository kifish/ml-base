import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: D x C array of weights (每行参数对应一个分类器的权重)
  - X: N x D array of data. Data are D-dimensional columns
  - y: (N,) 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W) #1*C
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue # 正确的yi,loss不用计算。hinge loss 计算的就是非正确yi（不是那个实际类别）的loss
      margin = scores[j] - correct_class_score + 1 # note delta = 1;可以理解为scores[j] - (correct_class_score - 1).也就是说非正确类别的得分要比正确类别的得分至少低一个delta，只有这样loss才为0.
      if margin > 0: #max(0,margin)操作
        loss += margin
        dW[:,y[i]] += -X[i,:]  # compute the correct_class gradients
        dW[:,j] += X[i,:]    # compute the wrong_class gradients

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W) #美中不足的是把bias也算上去了
  dW += reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  scores = X.dot(W) #X: N * D;W: D x C . N * C
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores_correct = scores[np.arange(num_train),y] #1 * N
  scores_correct = np.reshape(scores_correct,(num_train,1)) # N * 1
  margins = scores - scores_correct + 1 # N*C
  margins[np.arange(num_train),y] = 0.0 #正确类别的score无loss
  margins[margins <= 0] = 0
  loss += np.sum(margins) / num_train #按列相加,1*D
  loss += 0.5 * reg * np.sum(W * W)
  #compute the gradient
  margins[margins > 0] = 1.0
  row_sum = np.sum(margins,axis= 1) # 1 * N
  # np.sum([[0, 1, 2], [2, 1, 3], axis = 1)
  # 结果：array（[3, 6]）
  margins[np.arange(num_train),y] = -row_sum
  dW += np.dot(X.T,margins)/num_train + reg*W # D*C
  #D*N * N*C

  return loss, dW
