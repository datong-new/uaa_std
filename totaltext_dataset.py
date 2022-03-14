from __future__ import print_function, division
import math
import numpy
from PIL import Image, ImageDraw
import cv2
import json
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#IMG_DIR = "/data/totaltext/totaltext/Images/Test/"
#LABEL_DIR = "/data/totaltext/txt_format/Test/"

IMG_DIR = "./dataset/totaltext/Test/"
LABEL_DIR = "./dataset/totaltext/txt_format/Test/"
"/data/totaltext/txt_format/Test/"

def get_poly(line):
    items = line.split("[[")
    x = items[1].split("]]")[0]
    y = items[2].split("]]")[0]
    x, y = ' '.join(x.split()), ' '.join(y.split())
    x, y = [int(item) for item in x.split(" ")], [int(item) for item in y.split(" ")]
    poly = [[int(x[i]), int(y[i])] for i in range(len(x))]
   
    return poly

def get_polys(txt):
    xy, polys, label, s = [], [], False, ""
    for i in range(2, len(txt)-2):
        if txt[i-2:i]=="[[": 
            label=True
        if txt[i:i+2] == "]]": 
            xy.append(s)
            if len(xy)==2:
                x, y = ' '.join(xy[0].split()), ' '.join(xy[1].split())
                x, y = [int(item) for item in x.split(" ")], [int(item) for item in y.split(" ")]
                poly = [[int(x[i]), int(y[i])] for i in range(len(x))]
                polys.append(poly)
                xy = []
            label, s = False, ""
        if label: s+=txt[i]
    return polys
        

def make_mask(width, height, polygons):
    img = Image.new('L', (width, height), 0)
    for i, polygon in enumerate(polygons):
        polygon = [(polygon[i][0], polygon[i][1]) for i in range(len(polygon))]
        try:
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        except Exception:
            continue
        mask = numpy.array(img)
    return mask

class TotalText(Dataset):
    def __init__(self, img_dir=IMG_DIR, label_dir=LABEL_DIR):
        self.img_dir = img_dir
        self.label_path = label_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx].split('.')[0]
        filename = os.path.join(self.img_dir, img_name+".jpg")
        img = cv2.imread(filename)
        h, w = img.shape[:2]

        reader = open(self.label_path+"poly_gt_"+img_name+".txt", 'r', encoding='utf-8', errors='ignore').readlines()

        txt = ""
        for line in reader: txt += line.encode('ascii', 'ignore').decode('ascii')
        polys = get_polys(txt)

        mask = make_mask(w, h, polys)
        mask = torch.from_numpy(mask).float()
        return {"filename":filename, "mask":mask}



if __name__ == "__main__":
    line = "x: [[451 772 805 746 724 695 653 569 511 494 448 449]], y: [[181 183 255 336 322 283 261 264 285 319 280 244]], ornt: [u'c'], transcriptions: [u'CAPILANO']"
    get_poly(line)


    dataset = TotalText()
    item = dataset.__getitem__(0)
    mask = item['mask'].numpy()
    cv2.imwrite("mask.jpg", mask*255)
    os.system("cp {} .".format(item['filename']))
    print(item)
