import numpy as np
import os
from PIL import Image
import json
import pandas as pd

label_list = ['a', 'b']


class LabelJson(object):
    def __init__(self, abs_path=None, mode='saveTXT') -> None:
        super().__init__()
        self.abs_path = abs_path
        self.mode = mode
        self.read()

    def read(self):
        with open(self.abs_path, 'r', encoding='utf-8') as f:
            lj = json.load(f)
        self.wh = [lj.get('imageWidth'), lj.get('imageHeight')]
        shapes = lj.get('shapes')
        if self.mode == 'saveTXT':
            self.cls = [i.get('label') for i in shapes]

        points = [i.get('points') for i in shapes]
        points = [np.array(i, dtype=np.int32).reshape((-1, 2)) for i in points]
        self.loc = points
        self.box = [[j[:, 0].min(), j[:, 1].min(), j[:, 0].max(), j[:, 1].max()] for j in points]
        self.img_name = lj.get('imagePath')
        self.is_pos = bool(self.cls)
        return self


def saveTXT_bbox(json_info, save_path, imp):
    box = np.array(json_info.box)
    w, h = json_info.wh
    cls = json_info.cls
    if w is None:
        img = Image.open(imp)
        w, h = img.size
    print(cls)

    with open(save_path, 'w', encoding='utf-8') as ff:
        for idx, (xmin, ymin, xmax, ymax) in enumerate(box):
            label = cls[idx].split('-')[0]

            ff.write(
                f'{label_list.index(label)} {(xmin + xmax) / (2 * w)} {(ymin + ymax) / (2 * h)} {(xmax - xmin) / w} {(ymax - ymin) / h}\n')


def json2txtBbox_main(img_dir, json_dir, save_dir):
    for imgfile in os.listdir(img_dir):
        print(imgfile)
        name, suffix = os.path.splitext(imgfile)
        json_path = os.path.join(json_dir, name + '.json')
        if os.path.exists(json_path):
            img_path = os.path.join(img_dir, imgfile)
            json_info = LabelJson(json_path, mode='saveTXT')
            saveTXT_bbox(json_info, os.path.join(save_dir, name + '.txt'), img_path)


if __name__ == "__main__":
    img_dir = r'./image'
    json_dir = r'./label'
    save_dir = r'./txt'

    json2txtBbox_main(img_dir, json_dir, save_dir)
