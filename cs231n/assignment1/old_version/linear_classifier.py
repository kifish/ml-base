from svm_hinge_loss import *
from softmax_loss import *
import numpy as np
class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self,X,y,learning_rate = 1e-3,
            lambda_val = 1e-5,num_iterations = 100,batch_size = 200,verbose = True):
        num_train,num_feature = X.shape
        #assume y takes values 0,1,...K-1
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_feature,num_classes)
        #SGD mini-batch
        loss_history = []
        for iter in range(num_iterations):
            #这个实现里面只有iteration。但是没有一次iteration中的样本遍历
            #在一次iteration中只对抽取的一个batch做了训练
            #可以考虑再完善
            X_batch = None
            y_batch = None
            #Sampling with replacement is faster than sampling without replacement
            sample_idxs = np.random.choice(num_train,batch_size,replace = False)
            X_batch = X[sample_idxs,:]
            y_batch = y[sample_idxs]

            loss,grad = self.loss(X_batch,y_batch,lambda_val)
            loss_history.append(loss)

            self.W -= learning_rate*grad
            if verbose and ((iter+1) % 100 == 0 or (iter+1) == 1):
                print("Iteration %d/%d ---> loss: %f" %(iter+1,num_iterations,loss))
        print("---------------end-----------------")
        return loss_history

    def predict(self,X):
        """
        Use the trained weights of this linear classifier to predict labels for data points
        """
        pred = np.zeros(X.shape[0])
        #(M,1)
        pred = np.argmax(np.dot(X,self.W),axis = 1) #(1,M)
        return pred

    def loss(self,X_batch,y_batch,lambda_val):
        pass

class LinearSVM(LinearClassifier):
    def loss(self,X_batch,y_batch,lambda_val):
        return svm_hinge_loss_vectorized(self.W,X_batch,y_batch,lambda_val)

class Softmax(LinearClassifier):
    def loss(self,X_batch,y_batch,lambda_val):
        return softmax_loss_vectorized(self.W,X_batch,y_batch,lambda_val)
