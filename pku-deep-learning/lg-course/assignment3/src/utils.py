# Ref:https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python
# Ref:https://github.com/prijip/Py-Gsvhn-DigitStruct-Reader
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from PIL import Image
import pickle
# Bounding Box
class BBox:
    def __init__(self):
        self.label = ""     # Digit
        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0

class DigitStruct:
    def __init__(self):
        self.name = None    # Image file name
        self.bboxList = None # List of BBox structs

def readDigitStructGroup(dsFile):
    dsGroup = dsFile["digitStruct"]
    return dsGroup

# Reads a string from the file using its reference
def readString(strRef, dsFile):
    strObj = dsFile[strRef]
    str = ''.join(chr(i) for i in strObj)
    return str

# Reads an integer value from the file
def readInt(intArray, dsFile):
    intRef = intArray[0]
    isReference = isinstance(intRef, h5py.Reference)
    intVal = 0
    if isReference:
        intObj = dsFile[intRef]
        intVal = int(intObj[0])
    else: # Assuming value type
        intVal = int(intRef)
    return intVal

def yieldNextInt(intDataset, dsFile):
    for intData in intDataset:
        intVal = readInt(intData, dsFile)
        yield intVal

def yieldNextBBox(bboxDataset, dsFile):
    for bboxArray in bboxDataset:
        bboxGroupRef = bboxArray[0]
        bboxGroup = dsFile[bboxGroupRef]
        labelDataset = bboxGroup["label"]
        leftDataset = bboxGroup["left"]
        topDataset = bboxGroup["top"]
        widthDataset = bboxGroup["width"]
        heightDataset = bboxGroup["height"]

        left = yieldNextInt(leftDataset, dsFile)
        top = yieldNextInt(topDataset, dsFile)
        width = yieldNextInt(widthDataset, dsFile)
        height = yieldNextInt(heightDataset, dsFile)

        bboxList = []

        for label in yieldNextInt(labelDataset, dsFile):
            bbox = BBox()
            bbox.label = label
            bbox.left = next(left)
            bbox.top = next(top)
            bbox.width = next(width)
            bbox.height = next(height)
            bboxList.append(bbox)

        yield bboxList

def yieldNextFileName(nameDataset, dsFile):
    for nameArray in nameDataset:
        nameRef = nameArray[0]
        name = readString(nameRef, dsFile)
        yield name

# dsFile = h5py.File('../data/gsvhn/train/digitStruct.mat', 'r')
def yieldNextDigitStruct(dsFileName):
    dsFile = h5py.File(dsFileName, 'r')
    dsGroup = readDigitStructGroup(dsFile)
    nameDataset = dsGroup["name"]
    bboxDataset = dsGroup["bbox"]

    bboxListIter = yieldNextBBox(bboxDataset, dsFile)
    for name in yieldNextFileName(nameDataset, dsFile):
        bboxList = next(bboxListIter)
        obj = DigitStruct()
        obj.name = name
        obj.bboxList = bboxList
        yield obj

def test_load_mat():
    dsFileName = '../train/digitStruct.mat'
    testCounter = 0
    for dsObj in yieldNextDigitStruct(dsFileName):
        # testCounter += 1
        print(dsObj.name) #图片名称
        for bbox in dsObj.bboxList:
            print("    {}:{},{},{},{}".format(
                bbox.label, bbox.left, bbox.top, bbox.width, bbox.height))
        testCounter += 1
        if testCounter >= 5:
            break

def load_mat(path,verbose = False,debug = False):
    sample_cnt = 0
    data = []
    for dsObj in yieldNextDigitStruct(path):
        if verbose:
            print('\rloading sample {}'.format(sample_cnt),end='')
        sample = []
        file_name = dsObj.name
        sample.append(file_name)
        for bbox in dsObj.bboxList:
            sample.append((bbox.label, bbox.left, bbox.top, bbox.width, bbox.height))
        data.append(sample)
        sample_cnt += 1

        if debug and sample_cnt >= 5:  #for debug
            break
    print('\nloaded {} samples'.format(len(data)))
    return data

def load_data(root_path,verbose = False):
    X = []
    Y = []
    X_boxes = []
    samples = load_mat(os.path.join(root_path,'digitStruct.mat'),verbose,False)
    for sample in samples:
        file_name = sample[0]
        file_path = os.path.join(root_path,file_name)
        # img = plt.imread(file_path)
        with Image.open(file_path) as img: #读取大量图片最好要及时close
            infos = sample[1:]
            digits = []
            boxes = []
            for info in infos:
                digits.append(info[0])
                boxes.append(info[1:])
            if len(digits) >= 6:
                continue
            X.append(img.copy()) #不copy则close之后无法操作，其实更高效的是在这里就处理box，不要传递参数
        X_boxes.append(boxes)
        Y.append(digits)
    # X = np.array(X)
    X_boxes = np.array(X_boxes)
    Y = np.array(Y)

    # print('X:', X.shape)
    # print('X_boxes:', X_boxes.shape)
    # print('Y:',Y.shape)
    return X,X_boxes,Y

def get_cropped_box(left,top,width,height):
    # 找一个稍大的框框住数字串
    min_left, min_top, max_right, max_bottom = (min(left),
                                                min(top),
                                                max(map(lambda x, y: x + y, left, width)),
                                                max(map(lambda x, y: x + y, top, height)))
    center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                    (min_top + max_bottom) / 2.0,
                                    max(max_right - min_left, max_bottom - min_top))
    box_left, box_top, box_width, box_height = (center_x - max_side / 2.0,
                                                    center_y - max_side / 2.0,
                                                    max_side,
                                                    max_side)
    cropped_left, cropped_top, cropped_width, cropped_height = (int(round(box_left - 0.15 * box_width)),
                                                                int(round(box_top - 0.15 * box_height)),
                                                                int(round(box_width * 1.3)),
                                                                int(round(box_height * 1.3)))
    # box_right = box_left + box_width
    # box_bottom = box_top + box_height
    return (cropped_left,cropped_top,cropped_width,cropped_height)

def rgb2gray(img):
    return np.dot(np.array(img, dtype='float32'), [0.299, 0.587, 0.114])

def normalize(imgs):
    # imgs = rgb2gray(imgs) # broadcast  #不使用转灰度图片，以方便后续配合resnet50
    mean_img = np.mean(imgs,axis=0)
    imgs -= mean_img
    return imgs
    # 没有选择除以std
    # https://github.com/kifish/ml-base/blob/master/cs231n/assignment1/svm.ipynb

def process_box(X_boxes):
    X_box = []
    for i in range(X_boxes.shape[0]):
        left,top,width,height = [],[],[],[]
        boxes = X_boxes[i]
        for box in boxes:
            left.append(box[0])
            top.append(box[1])
            width.append(box[2]) # 注意顺序
            height.append(box[3])
        region_box = get_cropped_box(left,top,width,height)
        X_box.append(region_box)
    return X_box

def process_digits(Y):
    Y_one_hot = []
    for sample in Y:
        y = []
        digits = [10, 10, 10, 10, 10]  # 10表示没有数字
        for idx,digit in enumerate(sample):
            digits[idx] = digit
        y_one_hot = to_categorical(digits,num_classes = 11)
        Y_one_hot.append(y_one_hot)
    Y_one_hot = np.array(Y_one_hot)
    return Y_one_hot
def process_raw_data(root_path):
    X, _, Y = load_data(root_path, True) # verbose output
    n_imgs = len(X)
    print('the number of images:',n_imgs)
    print('Y:',Y.shape)
    print('processing...')
    for i in range(n_imgs):
        img = X[i]
        img = img.resize((128,128)) #64*64改成128*128;128*128内存不够，还是改成64*64
        img = np.array(img)
        X[i] = img
    X = normalize(X)
    Y = process_digits(Y)
    print('X:',X.shape)
    print('Y:',Y.shape)
    return X,Y

def process_data(root_path):
    X, X_boxes, Y = load_data(root_path, True) # verbose output
    X_box = process_box(X_boxes)
    X_box = np.array(X_box)
    # print('X:',X.shape)
    n_imgs = len(X)
    print('the number of images:',n_imgs)
    print('X_box:', X_box.shape)
    print('Y:',Y.shape)
    print('processing...')
    for i in range(n_imgs):
        raw_img = X[i]
        cropped_left, cropped_top, cropped_width, cropped_height = X_box[i]
        img = raw_img.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
        img = img.resize((64,64))
        # print(type(img))
        img = np.array(img) #Image -> np.ndarray;注意无法批量转，只能单张图片这样操作
        # print(type(img))
        # grey_img = rgb2gray(img)
        # X[i] = grey_img #利用广播机制，转灰度图片在normalize里实现
        X[i] = img
    # X = np.array(X) #Image -> np.ndarray;wrong!!!
    X = normalize(X)
    Y = process_digits(Y)
    print('X:',X.shape)
    print('Y:',Y.shape)
    return X,Y


def get_rgb_data():
    # use position information
    import pickle
    root_path = '../data/train/'
    X_train, Y_train = process_data(root_path)
    print('X_train:',X_train.shape)
    print('Y_train:',Y_train.shape)
    # with open('../data/train.pkl', "wb") as f:
    #     pickle.dump([X_train, Y_train], f)
    with open('../data/train_rgb.pkl', "wb") as f:
        pickle.dump([X_train, Y_train], f)
    del X_train,Y_train

    root_path = '../data/test/'
    X_test, Y_test = process_data(root_path)
    print('X_test:',X_test.shape)
    print('Y_test:',Y_test.shape)
    # with open('../data/test.pkl', "wb") as f:
    #     pickle.dump([X_test, Y_test], f)
    with open('../data/test_rgb.pkl', "wb") as f:
        pickle.dump([X_test, Y_test], f)

def test_load_mat():
    train_data_path = '../data/train/digitStruct.mat'
    load_mat(train_data_path,verbose=True)


def test_load_data():
    root_path = '../data/train/'
    load_data(root_path,True)


def show_imgs():
    with open('../data/train.pkl', 'rb') as f:
        X_train, Y_train = pickle.load(f)
    plt.subplot(2,1,1)
    plt.imshow(X_train[0])
    plt.subplot(2,1,2)
    plt.imshow(X_train[1])
    plt.show()

def get_test_data_no_pos_info():
    # rgb
    # no position information
    root_path = '../data/test/'
    X_test, Y_test = process_raw_data(root_path)
    print('X_test:',X_test.shape)
    print('Y_test:',Y_test.shape)
    with open('../data/test_no_pos_64.pkl', "wb") as f:
        pickle.dump([X_test, Y_test], f)

def get_train_data_no_pos_info():
    # rgb
    # no position information
    root_path = '../data/train/'
    X_train, Y_train = process_raw_data(root_path)
    print('X_train:',X_train.shape)
    print('Y_train:',Y_train.shape)
    with open('../data/train_no_pos_64.pkl', "wb") as f: #128容易爆内存
        pickle.dump([X_train, Y_train], f)


def get_data():
    print('processing data')

    root_path = '../data/train/'
    X_train, Y_train = process_raw_data(root_path)
    print('X_train:',X_train.shape)
    print('Y_train:',Y_train.shape)

    root_path = '../data/test/'
    X_test, Y_test = process_raw_data(root_path)
    print('X_test:',X_test.shape)
    print('Y_test:',Y_test.shape)

    print('finish processing data')
    return X_train,Y_train,X_test,Y_test

def get_data_no_pos_info():
    print('processing training data')
    get_train_data_no_pos_info()
    print('processing test data')
    get_test_data_no_pos_info()
if __name__ == "__main__":
    get_data_no_pos_info()



