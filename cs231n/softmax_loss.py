import numpy as np

def softmax_loss_naive(W,X,y,lambda_val):
    """
    W : A numpy array of shape(N,C)
    X : A numpy array of shape(M,N)
    y : A numpy array of shape(M,1);y[i] = c means that
        X[i] has label c, where 0<=c<C
    lambda_val : (float) regularzation strength

    """
    loss = 0.0
    dW = np.zeros_like(W)
    dW_each = np.zeros_like(W)
    num_train,num_feature = X.shape
    num_class = W.shape[1]
    a1 = X.dot(W) #(M,N) * (N,C)
    a1_max = np.reshape(np.max(a1,axis=1),(num_train,1)) #(M,1)
    prob = np.exp(a1-a1_max)/np.sum(np.exp(a1-a1_max),axis = 1,keepdims = True)
    #np.exp(a1-a1_max) 其实是分子分母同除以np.exp(a1_max)
    #prob : (M,C)

    target_prob = np.zeros_like(prob)
    #(M,C)
    target_prob[np.arange(num_train),y] = 1.0
    for sample_idx in range(num_train):
        for class_idx in range(num_class):
            loss += -(target_prob[sample_idx,class_idx]*np.log(prob[sample_idx,class_idx])) #[0 0 0 1 0...] only one number in the row is non-zero and equal to 1
            dW_each[:,class_idx] = -(target_prob[sample_idx,class_idx]-prob[sample_idx,class_idx])*X[sample_idx,:]
            #(N,1)               (1,N)
            """
            举例，若非label类的score是0，而target是0，则该feature的weight无须梯度。若label类的score是0.9，而target是1，则dW_each[:,class_idx]是减去一个值，对应减dW,
            则为梯度上升，最大化似然概率
            """
        dW += dW_each #加上对应的一列
    loss /= num_train
    loss += 0.5*lambda_val*np.sum(W*W)
    dW /= num_train
    dW += lambda_val*W

    return loss,dW

def softmax_loss_vectorized(W,X,y,lambda_val):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train,num_feature = X.shape
    a1 = X.dot(W)
    a1_max = np.reshape(np.max(a1,axis=1),(num_train,1))
    #(M,1)
    prob = np.exp(a1-a1_max)/np.sum(np.exp(a1-a1_max),axis = 1,keepdims = True)
    target_prob = np.zeros_like(prob)
    target_prob[range(num_train),y] = 1.0
    loss += -np.sum(target_prob * np.log(prob)) / num_train + 0.5 * lambda_val * np.sum(W*W)
    dW += -np.dot(X.T,target_prob-prob) / num_train + lambda_val*W
    #这里可参考svm_hinge_loss_vectorized
    return loss,dW 
