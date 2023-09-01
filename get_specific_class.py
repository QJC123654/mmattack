import json
import os
import shutil
from pycocotools.coco import COCO

# 要求输入的参数
dataDir = './coco2017'
dataType = 'val2017'
# annFile = os.path.join(dataDir, 'annotations', 'instances_{}.json'.format(dataType))
annFile = os.path.join(dataDir, 'annotations', 'instances_{}.json'.format(dataType))
catIds = [1, 2, 3, 4, 6, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
catNms = ['person', 'bicycle', 'car', 'motorcycle', 'bus',
          'train', 'truck', 'traffic light', 'stop sign', 'fire hydrant',
          'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
          'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
          'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
          'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
          'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror',
          'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop',
          'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
          'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase',
          'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_coco_category(category_list):
    """
    :param category_list: 需要提取的类别列表
    :return: None
    """
    # 初始化COCO API
    coco = COCO(annFile)

    # 获取包含需要提取类别的image ID
    catIds = coco.getCatIds(catNms=category_list)
    imgIds = coco.getImgIds(catIds=catIds)
    # ann_file path
    AnnPathBigJson = os.path.join(dataDir, 'annotations', 'dog.json')
    f = open(AnnPathBigJson, 'w')
    data = dict()
    data['info'] = coco.dataset['info']
    data['licenses'] = coco.dataset['licenses']
    data['categories'] = coco.dataset['categories']
    data['images'] = []
    data['annotations'] = []
    for imgId in imgIds:
        imgInfo = coco.loadImgs(imgId)[0]
        imgName = imgInfo['file_name']
        srcImgPath = os.path.join(dataDir, dataType, imgName)
        dstImgPath = os.path.join(outputPath, imgName)
        shutil.copy(srcImgPath, dstImgPath)

        annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        # get images and annotations
        data['images'].append(imgInfo)
        data['annotations'].extend(anns)
        # dstAnnPath = os.path.join(annOutputPath, imgName.replace('.jpg', '.json'))
        # with open(dstAnnPath, 'w') as wf:
        #     json.dump(anns, wf)
    json.dump(data, f)
    f.close()






if __name__ == '__main__':
    # 需要提取的类别列表
    categories = ['dog',]

    # 输出路径
    outputPath = 'coco2017/coco_dog'
    os.makedirs(outputPath, exist_ok=True)

    # # 注释输出路径
    # annOutputPath = os.path.join(outputPath, 'annotations')
    # os.makedirs(annOutputPath, exist_ok=True)
    get_coco_category(categories)