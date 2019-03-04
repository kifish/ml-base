from utils import load_CIFAR10

# Load the raw CIFAR-10 data.
cifar10_dir = 'pku-deep-learning/lg-course/assignment1/data/cifar-10-batches-py' 
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)     # (50000,32,32,3)
print('Training labels shape: ', y_train.shape)   # (50000L,)
print('Test data shape: ', X_test.shape)        # (10000,32,32,3)
print('Test labels shape: ', y_test.shape)      # (10000L,)
print()





