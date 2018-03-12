#import cPickle as pickle
import pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding = 'latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    """
    那个transpose是为了方便之后的tensorflow 模型使用。
    通常我们读入的数据格式是（32，32，3）即通道数在最后。
    像keras的输入格式就是（3，32，32）这样的。
    不直接写X = X.reshape(10000, 32, 32, 3).astype("float")
    可能是为了方便数据的输入。
    """
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)#np.concatenate(xs)使得最终Xtr的shape为(50000,32,32,3)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte
