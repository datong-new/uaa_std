import torch
from eval_helper import *
from torch import nn
import subprocess
import numpy as np
import sys
from attack_util import *
from icdar_dataset import ICDARDataset
from util import *
sys.path.insert(0, '/data/shudeng/text_attack/attacks/Text_Detector/Pytorch/')
from augmentations import Augmentation_inference
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from augmentations import Augmentation_inference
from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
import os
import zipfile
import cv2
from constant import *

mean=torch.tensor([0.485,0.456,0.406])
var=torch.tensor([0.229,0.224,0.225])
VAR = var.mean().item()

class Model():
    def __init__(self):
        self.net = RetinaNet()
        self.encoder = DataEncoder(0.4, 0.1)
        self._init_model()
        self.device = "cuda"
        self.net.to(self.device)
#        self.net = nn.DataParallel(self.net)
        self.net.eval()

    def _init_model(self, model_path=MODEL_PATH + 'ICDAR2015_TextBoxes.pth'):
        # load checkpoint
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['net'])

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_scale = 1024
        img,_,_ = Augmentation_inference(input_scale)(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        img = nn.functional.interpolate(img, (1024, 1024))
        return img

    def tensor_to_image(self, t):
        mean, var = torch.tensor([0.485,0.456,0.406]).to(self.device), torch.tensor([0.229,0.224,0.225]).to(self.device)
        t = t.squeeze().permute(1,2,0).to(self.device)
        t = (t*var+mean) * 255.0
        img = t.detach().cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def score_map(self, img):
        maps = self.net(img, attack=True)
        return maps

    def loss(self, score, mask, thresh=0.19):
        cost = 0
        for m in score:
            m = m.sigmoid()
            if m.max()>0.2: cost += loss(m, mask, thresh=0.2)
        return cost

    def get_polygons(self, img_path, is_output_polygon=True):
        img = self.load_image(img_path)
        loc_preds, cls_preds = self.net(img)
        scale = 1024
        quad_boxes, labels, scores = self.encoder.decode(loc_preds.data.squeeze(0), cls_preds.data.squeeze(0), scale)
        quad_boxes /= scale
        width, height = 1024, 1024
        quad_boxes *= ([[width, height]] * 4)
        quad_boxes = quad_boxes.astype(np.int32)

        return quad_boxes, img

    def zero_grad(self):
        self.net.zero_grad()



if __name__ == "__main__":
    model = Model()
    dataset = ICDARDataset()

    # single attack
#    single_attack(model, dataset, res_dir=textbox_single_icdar, eps=15/255/VAR, iters=100, cost_thresh=0.007)           
    
    # universal attack
#    universal_attack(model, dataset, res_dir=textbox_universal_icdar, epoches=2, eps=15/255/VAR, alpha=0.2)

    eval_helper = Eval('icdar2015')

    img_dir = IC15_TEST_IMAGES
    res_dir = PWD + "res_textbox/txt/"
    #eval_helper.eval(model, universal_totaltext_dir, res_dir)
    eval_helper.eval(model, textbox_universal_icdar, res_dir)
    

