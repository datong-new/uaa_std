import argparse
import torch
from attack_util import *
from eval_helper import *
from totaltext_dataset import TotalText
from icdar_dataset import ICDARDataset
import torch.backends.cudnn as cudnn
from constant import *
import sys
sys.path.insert(0, "/data/shudeng/attacks/CRAFT_pytorch")
from torch import nn
import os
import craft_utils
from refinenet import RefineNet
import numpy as np
import imgproc
from PIL import Image
import cv2
import craft_utils
from craft import CRAFT
from collections import OrderedDict
from hooks import VGGHelper

ROOT_PATH = PWD + "CRAFT_pytorch/"
var =[0.229, 0.224, 0.225]
VAR = torch.tensor(var).mean().item()

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class Model():
    #def __init__(self, resume="total_text"):
    def __init__(self, resume="icdar2015", loss="thresh"):
        self.loss_type = loss
        self.net = CRAFT()
        self.refine_net = RefineNet()
        if resume == "icdar2015": self.load_net(MODEL_PATH+"craft_ic15_20k.pth")
        else: self.load_net(MODEL_PATH+"craft_mlt_25k.pth")
#        self.load_refine_net()
        self.net.eval()
        self.refine_net.eval()
        self.helper = VGGHelper(self.net)

    def load_net(self, resume_path=MODEL_PATH+"craft_mlt_25k.pth"):
        self.net.load_state_dict(copyStateDict(torch.load(resume_path)))
        self.net = self.net.cuda()

    def load_refine_net(self, resume_path=MODEL_PATH+"craft_refiner_CTW1500.pth"):
        refine_net.load_state_dict(copyStateDict(torch.load(resume_path))).cuda()

    def load_image(self, img_path):
        img = imgproc.loadImage(img_path)
        img = imgproc.normalizeMeanVariance(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.unsqueeze(0).cuda()
        img = nn.functional.interpolate(img, (1024, 1024))
        return img

    def tensor_to_image(self, t):
        img = imgproc.denormalize(t)
        img = img.cpu().detach().squeeze().permute(1,2,0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
     
    def score_map(self, img, mask=None):
        outputs, _, text_features, nontext_features = self.helper.forward(img, mask)
        self.feature_loss = self.helper.loss(text_features, nontext_features)
        y, features = outputs

        #y, features = self.net(img)
        score_text = y[0,:,:,0]
        return score_text

    def loss(self, score, mask, thresh=0.19, use_feature_loss=False):
        if use_feature_loss:
            return loss(score, mask, thresh) + self.feature_loss
        else: return loss(score, mask, thresh)
       
        #if self.loss_type == "thresh": return loss(score, mask, thresh)
        #else: return ce_loss(score, mask)

    def zero_grad(self):
        self.net.zero_grad()

    def get_polygons(self, img_path, is_output_polygon=True):
        img = self.load_image(img_path).cuda().float()
        with torch.no_grad():
            y, feature = self.net(img)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=is_output_polygon)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]
        ratio_w = ratio_h = 1
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

        return polys, img
    
    def generate_universal_examples(self, dataset, perturbation, res_dir=PWD+"res_craft/universal/"):
        if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
        for i in range(len(dataset)):
            print("generate universal examples: {}/{}".format(i, len(dataset)))
            item = dataset.__getitem__(i)
            img_path = item['filename']
            img = self.load_image(img_path)
            img = img + perturbation.cuda()
            img = self.tensor_to_image(img)
            cv2.imwrite(os.path.join(res_dir, item['filename'].split("/")[-1]), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', help='attack type: single or universal')
    args = parser.parse_args()
    attack_type = args.attack_type
    
    
    model = Model(resume='icdar2015')

    dataset = ICDARDataset()
#    dataset = TotalText()
    #eval_helper = Eval('total_text')
    eval_helper = Eval('icdar2015')
    res_dir = PWD+"res_craft/txt/"
    eps = range(5, 15, 2)

    if attack_type == "single":
        # single attack for different epsilon
        for ep in eps:
            img_dir = PWD+"res_craft/single_icdar/{}/".format(ep)
            single_attack(model, dataset, res_dir=img_dir, eps=ep/255/VAR, iters=100, cost_thresh=0.001)
            res = eval_helper.eval(model, img_dir, res_dir)
            with open(img_dir + "../eps.txt", "a") as f: f.write("{}: {}\n".format(ep, res))
    elif attack_type == "universal":
        for ep in eps:
            img_dir = PWD+"res_craft/universal_icdar/{}/".format(ep)
            universal_attack(model, dataset, res_dir=img_dir, epoches=7, eps=ep/255/VAR, alpha=0.2)
            res = eval_helper.eval(model, img_dir, res_dir)
            with open(img_dir + "../u_eps.txt", "a") as f: f.write("{}: {}\n".format(ep, res))


