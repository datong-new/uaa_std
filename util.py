import torch
from constant import *
import random
from torch import nn
import numpy as np
import cv2


def visualize(mask, save_name, downsample=1):
    mask = mask.squeeze().detach().cpu().numpy()
    heatmap = cv2.resize(mask, (mask.shape[0]//downsample, mask.shape[1]//downsample))
    heatmapshow = None
    heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    cv2.imwrite("/data/shudeng/plots/{}.png".format(save_name), heatmapshow)
    #cv2.imshow("Heatmap", heatmapshow)
    #cv2.waitKey(0)

def pert_map(pert, eps):
    return (1/(1+torch.exp(pert)) -0.5) *2 * eps[None, :, None, None].to(pert.device)



def loss(score, mask, thresh=0.8):

    #print("score shape", score.shape)
    #print("mask shape", mask.shape)
    #visualize(score, "regression_score", downsample=1)
    #visualize(mask, "regression_mask", downsample=4)
    #exit(0)
    if mask.sum() == 0:
        print("mask is all 0")
        return 0
    while len(mask.shape) < 4:
        mask = mask.unsqueeze(0)
    while len(score.shape) < 4:
        score = score.unsqueeze(0)
    mask = nn.functional.interpolate(mask, score.shape[2:])
    mask = mask.expand(score.shape)
    l = 1/(1+torch.exp(-1e2*(torch.clamp(score, max=1, min=0)-thresh)))
    l = (l*mask).sum() / (mask.sum()+1e-6)
    return l

def ce_loss(score, mask):
    print("score shape", score.shape)
    print("mask shape", mask.shape)
    visualize(score, "regression_score", downsample=1)
    visualize(mask, "regression_mask", downsample=4)
    exit(0)
    while len(mask.shape) < 4:
        mask = mask.unsqueeze(0)
    while len(score.shape) < 4:
        score = score.unsqueeze(0)
    mask = nn.functional.interpolate(mask, score.shape[2:])
    mask = mask.expand(score.shape)
    #print("score max:{}, score min:{}".format(score.max(), score.min()))
    score_max, score_min = score.max(), score.min()
    if score_max > 1: score = (score-score_min) / (score_max - score_min)

    l = -mask * torch.log(1-score+1e-8)
    return (l*mask).sum() / (mask.sum()+1e-6)


def random_resize(img, mask=None):
    height = int(random.uniform(500, 1000))//32*32
    width = int(random.uniform(500, 1000))//32*32
    img = nn.functional.interpolate(img, (height, width))
    if mask is None:
        return img
    mask = nn.functional.interpolate(mask.unsqueeze(1), (height, width))
    return img, mask.squeeze(1)


def random_resize(img, mask=None):
    width = int(random.uniform(500, 650))//32*32
    height = int(random.uniform(500, 650))//32*32
    img = nn.functional.interpolate(img, (height, width))
    if mask is None: 
        return img
    mask = nn.functional.interpolate(mask.unsqueeze(1), (height, width))
    return img, mask.squeeze(1)

def fix_attack(model, dataset, res_dir=PWD+"res_textbox++/fix/"):
    # generate adversarial example for fix-size image
    for i in range(0, len(dataset)):
        item = dataset.__getitem__(i)
        img_path = item['filename']
        mask = item['mask'].to(model.device)
        model.resize_attack(img_path, mask, fix_size=True, res_dir=res_dir)

def resize_attack(model, dataset, res_dir=PWD+"res_textbox++/resize/"):
    # generate adversarial example for resized image
    for i in range(0, len(dataset)):
        print("{}/{}".format(i, len(dataset)))
        item = dataset.__getitem__(i)
        img_path = item['filename']
        mask = item['mask'].to(model.device)
        model.resize_attack(img_path, mask, fix_size=False, res_dir=res_dir)

#def universal_attack(model, dataset):
#    model.universal_attack(dataset)

def draw_polygon_image(img, polygons):
    
    pass

def eval(img_dir, res_dir=PWD+"res_east/"):
    files = os.listdir(img_dir)
    for i, img_path in enumerate(files):
        if not img_path[-4:]==".jpg": continue
        print("eval {}/{}".format(i, len(files)))
        self.save_result(os.path.join(img_dir, img_path), input_size=input_size, save_dir=res_dir)
    os.chdir(res_dir)
    res = subprocess.getoutput('zip -q submit.zip *.txt')
    os.system("rm *.txt")
    os.chdir(PWD)
    res = subprocess.getoutput('python ./evaluate/script.py -g=./evaluate/gt_100.zip -s={}/submit.zip'.format(res_dir))
    os.system("rm {}/submit.zip".format(res_dir))
    print(res)
