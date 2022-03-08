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
IMG_DIR = "/data/shudeng/shudeng/IC15/test_images/"
LABEL_DIR = "/data/shudeng/shudeng/IC15/Challenge4_Test_Task1_GT/"

class ICDARDataset(Dataset):

    def __init__(self, img_dir=IMG_DIR, label_dir=LABEL_DIR):
        self.img_dir = img_dir
        self.label_path = label_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            #new_height = self.args['image_short_side']
            new_height = 800
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            #new_width = self.args['image_short_side']
            new_width = 800
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        new_width, new_height = 512, 512
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        #img = self.resize_image(img)
#        img /= 255.
#        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img, original_shape

    def make_mask(self, width, height, polygons):
        img = Image.new('L', (width, height), 0)
        for i, polygon in enumerate(polygons):
            polygon = [(polygon[i][0], polygon[i][1]) for i in range(len(polygon))]
            try:
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            except Exception:
                continue
            mask = numpy.array(img)
        return mask

    def __getitem__(self, idx):
        img_name = self.images[idx].split('.')[0]
        img, original_shape = self.load_image(os.path.join(self.img_dir, img_name+".jpg"))
        lines = []
        polygons = []
        reader = open(self.label_path+"gt_"+img_name+".txt", 'r', encoding='utf-8', errors='ignore').readlines()
        for line in reader:
            line = line.encode('ascii', 'ignore').decode('ascii')
            points = line.strip().split(",")[:8]
            points = [int(points[i]) for i in range(len(points))]
            poly = np.array(points).reshape((-1, 2)).tolist()
            polygons.append(poly)
        mask = self.make_mask(original_shape[1],original_shape[0],polygons)
        mask = torch.from_numpy(mask).float()
        img = torch.from_numpy(img).float()
        return {'image': img, 'mask': mask, 'filename': os.path.join(self.img_dir, img_name+".jpg")}

if __name__ == "__main__":
    dataset = ICDARDataset()
    item = dataset.__getitem__(2)
    cv2.imwrite("icdar_img.jpg", item['image'].numpy())
    cv2.imwrite("icdar_mask.jpg", item['mask'].numpy() * 255)
    print(item)
