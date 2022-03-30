import torch
import argparse
from eval_helper import *
from torch import nn
import subprocess
import numpy as np
import sys
from attack_util import *
from icdar_dataset import ICDARDataset
from util import *
sys.path.insert(0, '/data/attacks/Text_Detector/Pytorch/')
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
from hooks import ResnetHelper

mean=torch.tensor([0.485,0.456,0.406])
var=torch.tensor([0.229,0.224,0.225])
var_ = var.clone()
VAR = var.mean().item()

class Model():
    def __init__(self, loss="thresh"):
        self.loss_type = loss
        self.net = RetinaNet()
        self.encoder = DataEncoder(0.4, 0.1)
        self._init_model()
        self.device = "cuda"
        self.net.to(self.device)
#        self.net = nn.DataParallel(self.net)
        self.net.eval()
        self.helper = ResnetHelper(self.net)

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
        img=img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def score_map(self, img, mask=None):
        outputs, _, text_features, nontext_features = self.helper.forward(img, mask, textbox=True)
        self.feature_loss = self.helper.loss(text_features, nontext_features)
        maps, _ = outputs
        #maps = self.net(img, attack=True)
        return maps

    def loss(self, score, mask, thresh=0.19, use_feature_loss=False):
        cost = 0
        for m in score:
            m = m.sigmoid()
            if self.loss_type == "thresh": cost += loss(m, mask, thresh=0.2)
            else: cost += ce_loss(m, mask)
        #if use_feature_loss:
        #    cost += self.feature_loss
        return cost

    def get_polygons(self, img_path, is_output_polygon=True):
        img = self.load_image(img_path)
        outputs, _, text_features, nontext_features = self.helper.forward(img, mask=None)
        loc_preds, cls_preds = outputs
        #loc_preds, cls_preds = self.net(img)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', help='attack type: single or universal')
    args = parser.parse_args()
    attack_type = args.attack_type

    model = Model()
    dataset = ICDARDataset()
#    dataset = TotalText()
    #eval_helper = Eval('total_text')
    eval_helper = Eval('icdar2015')
    res_dir = PWD+"res_textbox/txt/"

    eps = range(5, 15, 2)

    if attack_type == "single":
        # single attack for different epsilon
        for ep in eps:
            img_dir = PWD+"res_textbox/single_icdar/{}/".format(ep)
            single_attack(model, dataset, res_dir=img_dir, eps=ep/255/VAR, iters=100, cost_thresh=0.001)
            res = eval_helper.eval(model, img_dir, res_dir)
            with open(img_dir + "../eps.txt", "a") as f: f.write("{}: {}\n".format(ep, res))
    elif attack_type == "universal":
        for ep in eps:
            img_dir = PWD+"res_textbox/universal_icdar/{}/".format(ep)
            universal_attack(model, dataset, res_dir=img_dir, epoches=7, eps=ep/255/VAR, alpha=0.2)
            res = eval_helper.eval(model, img_dir, res_dir)
            with open(img_dir + "../u_eps.txt", "a") as f: f.write("{}: {}\n".format(ep, res))

    exit(0)

    # single attack
#    single_attack(model, dataset, res_dir=textbox_single_icdar, eps=15/255/VAR, iters=100, cost_thresh=0.007)           
    
    # universal attack
#    universal_attack(model, dataset, res_dir=textbox_universal_icdar, epoches=2, eps=15/255/VAR, alpha=0.2)

    eval_helper = Eval('icdar2015')

    img_dir = IC15_TEST_IMAGES
    res_dir = PWD + "res_textbox/txt/"
    #eval_helper.eval(model, universal_totaltext_dir, res_dir)
    eval_helper.eval(model, textbox_universal_icdar, res_dir)
    

