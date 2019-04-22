# from keras import applications
# #.resnet import ResNet50
# base_model = applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)


# import keras
# x = keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


from keras.applications import resnet50
base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(64,64,3), pooling=None, classes=1000)


