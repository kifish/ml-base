## 目录

来自斯坦福网络课程“CS231n Convolutional Neural Networks for Visual Recognition”的作业：
* 原文笔记：[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/) 
* 授课视频：[CS231n Winter 2016](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC) 


### 1. [Assignment 1](http://cs231n.github.io/assignments2016/assignment1/)
  * [k-Nearest Neighbor classifier](assignment1/knn.ipynb)
  * [Training a Support Vector Machine](assignment1/svm.ipynb)
  * [Implement a Softmax classifier](assignment1/softmax.ipynb)
  * [Two-Layer Neural Network](assignment1/two_layer_net.ipynb)
  * [Higher Level Representations: Image Features](assignment1/features.ipynb)

https://github.com/zlotus/cs231n   
https://www.jianshu.com/p/004c99623104   
https://www.cnblogs.com/daihengchen/p/5754383.html   
sigmoid的优缺点见：https://www.jianshu.com/p/004c99623104   
https://www.jianshu.com/p/93d2230e5f27




### 2. [Assignment 2](http://cs231n.github.io/assignments2016/assignment2/)
  * [Fully-connected Neural Network](assignment2/FullyConnectedNets.ipynb)
  * [Batch Normalization](assignment2/BatchNormalization.ipynb)
  * [Dropout](assignment2/Dropout.ipynb)
  * [ConvNet on CIFAR-10](assignment2/ConvolutionalNetworks.ipynb)
fc_net.py值得多看看。

cnn layer与fc layer的区别:前者参数少，但计算量大;后者参数多，但计算量少。
cnn网络结构:
conv + relu + pooling

INPUT --> [[CONV --> RELU]*N --> POOL?]*M --> [FC --> RELU]*K --> FC(OUTPUT)

· INPUT --> FC/OUT      
· INPUT --> CONV --> RELU --> FC/OUT
· INPUT --> [CONV --> RELU --> POOL]*2 --> FC --> RELU --> FC/OUT
· INPUT --> [CONV --> RELU --> CONV --> RELU --> POOL]*3 --> [FC --> RELU]*2 --> FC/OUT


https://www.cnblogs.com/daihengchen/tag/CS231n/
代码见:
https://github.com/zlotus/cs231n  
公式及拓展见:
https://www.jianshu.com/p/9c4396653324


To do:   
### 3. [Assignment 3](http://cs231n.github.io/assignments2016/assignment3/)
  * [Image Captioning with Vanilla RNNs](assignment3/RNN_Captioning.ipynb)
  * [Image Captioning with LSTMs](assignment3/LSTM_Captioning.ipynb)
  * [Image Gradients: Saliency maps and Fooling Images](assignment3/ImageGradients.ipynb)
  * [Image Generation: Classes, Inversion, DeepDream](assignment3/ImageGeneration.ipynb)

http://cs231n.github.io/assignments2016/assignment3/
https://www.jianshu.com/p/e46b1aa48886
https://www.jianshu.com/p/182baeb82c71
