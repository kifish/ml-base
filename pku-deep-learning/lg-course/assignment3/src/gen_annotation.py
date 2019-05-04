from utils import *

def test_load_mat_info():
    root_path = '../data/train/'
    dsFileName = '../data/train/digitStruct.mat'
    testCounter = 0
    for dsObj in yieldNextDigitStruct(dsFileName):
        # testCounter += 1
        file_name = dsObj.name
        print(file_name)
        img_path = os.path.join(root_path,file_name)
        print(os.path.abspath(img_path))
        for bbox in dsObj.bboxList:
            print("    {}:{},{},{},{}".format(
                bbox.label, bbox.left, bbox.top, bbox.width, bbox.height))
            x1 = bbox.left
            x2 = bbox.left + bbox.width
            y1 = bbox.top
            y2 = bbox.top + bbox.height
            print("    {}:{},{},{},{}".format(
                bbox.label, x1, y1, x2, y2))
        testCounter += 1
        if testCounter >= 5:
            break

# test_load_mat_info()

def gen_annotation():
    root_path = '../data/train/'
    dsFileName = '../data/train/digitStruct.mat'
    with open('../data/annotation.txt','w',encoding='utf8') as f:
        for dsObj in yieldNextDigitStruct(dsFileName):
            file_name = dsObj.name
            img_path = os.path.join(root_path,file_name)
            file_path = os.path.abspath(img_path)
            left, top, width, height = [], [], [], []
            for bbox in dsObj.bboxList:
                left.append(bbox.left)
                top.append(bbox.top)
                width.append(bbox.width)
                height.append(bbox.height)
            cropped_left, cropped_top, cropped_width, cropped_height = get_cropped_box(left, top, width, height)
            x1 = cropped_left
            x2 = cropped_left + cropped_width
            y1 = cropped_top
            y2 = cropped_top + cropped_height
            row = '{},{},{},{},{},digit'.format(file_path,x1,y1,x2,y2)
            f.write(row+'\n')
            break
if __name__ == '__main__':
    gen_annotation()