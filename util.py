import torch
from constant import *
import random
from torch import nn

def pert_map(pert, eps):
    return (1/(1+torch.exp(pert)) -0.5) *2 * eps

def loss(score, mask, thresh=0.8):
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
    while len(mask.shape) < 4:
        mask = mask.unsqueeze(0)
    while len(score.shape) < 4:
        score = score.unsqueeze(0)
    mask = nn.functional.interpolate(mask, score.shape[2:])
    mask = mask.expand(score.shape)
    l = -mask * torch.log(score+1e-2)
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
