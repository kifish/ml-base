# Ref:https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python
# Ref:https://github.com/prijip/Py-Gsvhn-DigitStruct-Reader
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

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

def load_mat(path,verbose = False):
    sample_cnt = 0
    data = []
    for dsObj in yieldNextDigitStruct(path):
        if verbose:
            print('loading sample {}'.format(sample_cnt))
        sample = []
        file_name = dsObj.name
        sample.append(file_name)
        for bbox in dsObj.bboxList:
            sample.append((bbox.label, bbox.left, bbox.top, bbox.width, bbox.height))
        data.append(sample)
        sample_cnt += 1
        # if sample_cnt >= 5:  #for debug
        #     break
    print('loaded {} samples'.format(len(data)))
    return data

def load_data(root_path,verbose = False):
    X = []
    Y = []
    X_boxes = []
    samples = load_mat(os.path.join(root_path,'digitStruct.mat'),verbose)
    for sample in samples:
        file_name = sample[0]
        file_path = os.path.join(root_path,file_name)
        img = plt.imread(file_path)
        infos = sample[1:]
        digits = []
        boxes = []
        for info in infos:
            digits.append(info[0])
            boxes.append(info[1:])
        X.append(img)
        X_boxes.append(boxes)
        Y.append(digits)
    X = np.array(X)
    X_boxes = np.array(X_boxes)
    Y = np.array(Y)

    print('X:', X.shape)
    print('X_boxes:', X_boxes.shape)
    print('Y:',Y.shape)
    return X,X_boxes,Y




if __name__ == "__main__":
    # train_data_path = '../data/train/digitStruct.mat'
    # load_mat(train_data_path,verbose=True)
    root_path = '../data/train/'
    load_data(root_path,True)

