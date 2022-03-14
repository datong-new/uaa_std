import argparse
import torch
from eval_helper import *
import subprocess
from icdar_dataset import ICDARDataset
from attack_util import *
from constant import *
from util import *
import random
import numpy
import numpy as np
from constant import *
from torch import nn
from torchvision import transforms
import cv2
import os
import sys
sys.path.insert(0, "/data/attacks/EAST")
from PIL import Image, ImageDraw
from model import EAST
from detect import resize_img, load_pil, get_boxes, plot_boxes, adjust_ratio
from icdar_dataset import ICDARDataset
from hooks import VGGHelper

VAR = 0.5

class Model():
    def __init__(self, loss="thresh"):
        self.loss_type = loss
        model_path  = MODEL_PATH + 'east_vgg16.pth'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = "cpu"
        self.device = device
        model = EAST().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.net = model
        os.chdir(PWD)
        self.helper = VGGHelper(self.net)

    def load_image(self, img_path, scale=1):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = load_pil(img).to(self.device)
        img = nn.functional.interpolate(img, (1024, 1024))
        return img

    def tensor_to_image(self, t, mean=torch.tensor([0.5,0.5,0.5]), std=torch.tensor([0.5, 0.5, 0.5])):
        t = t.squeeze().permute(1,2,0)
        t = (t*std.to(self.device)) + mean.to(self.device)
        t = t * 255.0
        img = t.detach().cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def score_map(self, img, mask=None):
        outputs, _, text_features, nontext_features = self.helper.forward(img, mask)
        self.feature_loss = self.helper.loss(text_features, nontext_features)
        out, features = outputs
    #    out, features = self.net(img)
        score = out[0]
        return score

    def loss(self, score, mask, thresh=0.8, use_feature_loss=False):
        if use_feature_loss:
            return loss(score, mask, thresh) + self.feature_loss
        else: return loss(score, mask, thresh)

        #if self.loss_type == "thresh": return loss(score, mask, thresh)
        #else: return ce_loss(score, mask)

        while len(mask.shape) < 4:
            mask = mask.unsqueeze(0)
        mask = nn.functional.interpolate(mask, score.shape[2:])
        l = 1/(1+torch.exp(-1e2*(score-thresh)))
        l = (l*mask).sum() / mask.sum()
        return l

    def zero_grad(self):
        self.net.zero_grad()

    def get_polygons(self, img_path, is_output_polygon=True):
        img = self.load_image(img_path)
        #out, _ = self.net(img)
        output, _,_,_ = self.helper.forward(img)
        out, _ = output
        score, geo = out
        boxes = get_boxes(score.squeeze(0).cpu().detach().numpy(), geo.squeeze(0).cpu().detach().numpy())
#        boxes = adjust_ratio(boxes, ratio_w, ratio_h)
        polys = []
        if boxes is not None:
            for box in boxes:
                box = [[box[2*i], box[2*i+1]] for i in range(4)]
                polys.append(box)
            polys = numpy.array(polys)
        return polys, img

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
    res_dir = PWD+"res_east/txt/"

    eps = range(5, 15, 2)

    if attack_type == "single":
        # single attack for different epsilon
        for ep in eps:
            img_dir = PWD+"res_east/single_icdar/{}/".format(ep)

            single_attack(model, dataset, res_dir=img_dir, eps=ep/255/VAR, iters=100, cost_thresh=0.001)
            res = eval_helper.eval(model, img_dir, res_dir)
            with open(img_dir + "../eps.txt", "a") as f: f.write("{}: {}\n".format(ep, res))
    elif attack_type == "universal":
        for ep in eps:
            img_dir = PWD+"res_east/universal_icdar/{}/".format(ep)
            universal_attack(model, dataset, res_dir=img_dir, epoches=7, eps=ep/255/VAR, alpha=0.2)
            res = eval_helper.eval(model, img_dir, res_dir)
            with open(img_dir + "../u_eps.txt", "a") as f: f.write("{}: {}\n".format(ep, res))

    exit(0)


    # single attack
#    single_attack(model, dataset, res_dir=east_single_icdar, eps=15/255/VAR, iters=100, cost_thresh=0.007)           
    
    # universal attack
#    universal_attack(model, dataset, res_dir=east_universal_icdar, epoches=6, eps=15/255/VAR, alpha=0.2)

    eval_helper = Eval('icdar2015')

    img_dir = IC15_TEST_IMAGES
    res_dir = PWD+"res_east/txt/"

    print(east_single_icdar)
    eval_helper.eval(model, east_single_icdar, res_dir)
#    eval_helper.eval(model, east_universal_icdar, res_dir)
#    eval_helper.eval(model, IC15_TEST_IMAGES, res_dir)
