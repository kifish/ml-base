import numpy as np
def svm_hinge_loss_naive(W,X,y,lambda_val):
    """
    W : A numpy array of shape(N,C)
    X : A numpy array of shape(M,N)
    y : A numpy array of shape(M,1);y[i] = c means that
        X[i] has label c, where 0<=c<C
    lambda_val : (float) regularzation strength

    return a tuple of :
    loss as single float
    gradient with respect to weights W;an array of same shape sa W

    this function is for softmax
    """
    dW = np.zeros(W.shape)

    num_train = X.shape[0]
    num_classes = W.shape[1]
    loss = 0.0
    for sample_idx in range(num_train):
        scores = X[sample_idx].dot(W)
        correct_class_score = scores[y[sample_idx]]
        for class_idx in range(num_classes):
            if class_idx == y[sample_idx]:
                continue
            margin = score[class_idx] - correct_class_score + 1 #delta = 1
            #actually,
            #margin = score[class_idx] - (correct_class_score - 1)
            #简而言之，pred的one-hot向量要与label的one-hot向量尽可能接近
            #具体来说，就是非label类的pred出来的score要离label的score远一个delta就比较好，这种情况下没有loss
            if margin > 0:
                loss += margin
                dW[:,y[sample_idx]] -= X[sample_idx,:] # dW for correct class
                #对于某一个样本,被减次数 = 非label类的pred中 出来的score没有离label的score远一个delta 的 pred个数
                #即margin >0 的次数
                #可参照hinge_loss_vectorized
                #(N,1) - (1,N) 自动翻转？
                dW[:,class_idx] += X[sample_idx,:] #dW for wrong class
                #非label类的pred中 出来的score没有离label的score远一个delta 的 每一个pred 加的次数为1

    #average loss and dW
    loss /= num_train
    dW /= num_train

    loss += 0.5*lambda_val*np.sum(W*W) #like .* in matlab
    dW += lambda_val*W
    return loss,dW

def svm_hinge_loss_vectorized(W,X,y,lambda_val):
    loss = 0.0
    dW = np.zeros(W.shape)
    scores = X.dot(W) #(M,N) * (N,C)
    num_train = X.shape[0]
    num_classes = X.shape[1]
    scores_correct = scores[np.arange(num_train),y]
    #obtain the correct_class_score of each sample by extracting from scores
    #(1,M)  list?
    scores_correct = np.reshape[scores_correct,(num_train,1)] #np.array  (M,1)
    margins = scores - scores_correct + 1
    margins[np.arange(num_train),y] = 0.0 #the loss for correct class is 0
    margins[margins <= 0] = 0.0
    loss += np.sum(margins)/num_train
    loss += 0.5*lambda_val*np.sum(W*W)

    margins[margins > 0] = 1.0 #for wrong pred,(M,C)
    #margin -->(M,C)
    sum_along_each_row = np.sum(margins,axis = 1) #(M,1)--->(1,M)


    margins[np.arange(num_train),y] = -sum_along_each_row #(1,M) = (1,M)   view视图
    """
    类似如下计算
    no no no yes-sum_along_each_row(0)
    yes-sum_along_each_row(1) no no no
    yes-sum_along_each_row(2) no no no
    no yes-sum_along_each_row(3) no no

    """
    #(N,M) * (M,C)
    """
    (N,M):                                (M,C):
             sample0 sample1 ...                  class0 class1
    feature0  res0     res1               sample0
    feature1                              sample1
    res0:对于class0 来说，feature0对应的weight要加的总值。总值=各sub总值求和。sub总值为单值乘以次数。
    res1:对于class1 来说，feature0对应的weight要加的总值。总值=各sub总值求和。sub总值为单值乘以次数。
    """
    #X.T记录要加的值，margins记录要加的次数
    dW += np.dot(X.T,margins)/num_train +lambda_val*W
    #(N,C)
    return loss,dW
