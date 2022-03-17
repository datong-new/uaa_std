import torch
import random
import cv2
from util import *
from torch import nn
import os
import scipy.stats as st
from torch import nn
import torch.nn.functional as F

VAR=0.5

def print(*args):
    return

def resize(img):
    return nn.functional.interpolate(img, (1024, 1024))

def DI(img, mask, size=1024, prob=0.7):
    size_min, size_max = int(size*0.8)//32*32, int(size*1.2) // 32 * 32
    rnd = np.random.randint(size_min, size_max,size=1)[0]
    h_rem = size_max - rnd
    w_rem = size_max - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left
    while len(mask.shape)<4: mask = mask.unsqueeze(0)

    mask = nn.functional.interpolate(mask, img.shape[-2:])

    c = np.random.rand(1)
    if c <= prob:
        X_out = F.pad(F.interpolate(img, size=(rnd,rnd)),(pad_left,pad_right,pad_top,pad_bottom),mode='constant', value=0)
        mask = F.pad(F.interpolate(mask, size=(rnd,rnd)),(pad_left,pad_right,pad_top,pad_bottom),mode='constant', value=0)
        return  X_out, mask.squeeze()
    else:
        return  img, mask.squeeze()

def gkern(kernlen=5, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
    return gaussian_kernel

def TI(grad, gaussian_kernel, kernel_size=5):
    grad_c = F.conv2d(grad, gaussian_kernel.to(grad.device).float(), bias=None, stride=1, padding=(kernel_size//2,kernel_size//2), groups=3) #TI
    return grad_c
    

def generate_universal_examples(model, dataset, perturbation, res_dir):
    for i in range(0, len(dataset)):
        print("generate universal examples: {}/{}".format(i, len(dataset)))
        item = dataset.__getitem__(i)
        x = model.load_image(item['filename'])
        x = x + perturbation.cuda()
        img = model.tensor_to_image(x)
        cv2.imwrite(os.path.join(res_dir, item['filename'].split("/")[-1]), img)

def _single_attack(model, img_path, mask, res_dir, eps=15/255/VAR, iters=30, alpha=0.2, cost_thresh=0.05, use_momentum=False, use_di=False, use_ti=False, use_feature_loss=False):

    if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
    if os.path.exists(os.path.join(res_dir, img_path.split("/")[-1])): return
    original_mask = mask.unsqueeze(0).cuda()
    original_img = model.load_image(img_path).cuda()

    pertur = torch.zeros(original_img.shape)
    kernel_size=5
    if use_ti: gaussian_kernel = gkern(kernlen=kernel_size)
    model.helper.get_original_features(model, dataset=None, img=original_img, mask=original_mask, textbox=("textbox" in res_dir))

    for i in range(iters):
        pertur.requires_grad = True
        perturbation = pertur.cuda()
        cost = 0
        img = original_img + pert_map(perturbation, eps)
        mask = original_mask.clone()

        if use_di:
            resize_img, mask = DI(img, mask)
        else: resize_img = img


        score_map = model.score_map(resize_img, mask)
        cost = model.loss(score_map, mask, use_feature_loss=use_feature_loss)


        if cost<cost_thresh or i==iters-1:
        #if i==iters-1:
            print("end", i)
            img = model.tensor_to_image(img)
            cv2.imwrite(os.path.join(res_dir, img_path.split("/")[-1]), img.astype(int))
            break

        if use_feature_loss:
            cost += model.feature_loss

        model.zero_grad()
        #cost.backward(retain_graph=True)
        cost.backward()
        #print("cost: {}, feature_loss:{}".format(cost, model.feature_loss))
        grad = pertur.grad
        if use_ti: grad=TI(grad, gaussian_kernel, kernel_size=kernel_size)

        pertur = pertur - alpha * grad.sign()
        pertur = pertur.detach()


def single_attack(model, dataset, res_dir, eps=15/255/VAR, iters=300, alpha=0.2, cost_thresh=0.05, use_momentum=False, use_di=False, use_ti=False, use_feature_loss=False):
    #model.helper.get_original_features(model, dataset, textbox=("textbox" in res_dir))

    for i in range(len(dataset)):
        print("{}/{}".format(i, len(dataset)))
        item = dataset.__getitem__(i)
        img_path, mask = item['filename'], item['mask']
        _single_attack(model, img_path, mask, res_dir=res_dir, eps=eps, iters=iters, alpha=alpha, cost_thresh=cost_thresh, use_momentum=use_momentum, use_di=use_di, use_ti=use_ti, use_feature_loss=use_feature_loss)

def universal_attack(model, dataset, res_dir, epoches=30, eps=15/255/VAR, alpha=0.2, use_momentum=False, use_di=False, use_ti=False, use_feature_loss=False):
    if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
    batch_size = 1
    pertu = torch.zeros(1, 3, 1024, 1024)
    cost_sum = 0
    kernel_size=5
    if use_ti: gaussian_kernel = gkern(kernlen=kernel_size)

    #model.helper.get_original_features(model, dataset, textbox=("textbox" in res_dir))

    for i in range(epoches * len(dataset)):
        pertu.requires_grad = True
        perturbation = pertu.cuda()
        if i!=0 and i%len(dataset)==len(dataset)-1: 
            with open(res_dir+"cost.log", "a") as f: f.write("epoch:{}, cost:{}\n".format(i//len(dataset), cost_sum))
            cost_sum = 0

        cost = 0
        for _ in range(batch_size):
            idx = int(random.uniform(0, len(dataset)))
            item = dataset.__getitem__(idx)
            img_path = item['filename']
            original_mask = item['mask'].cuda()
            img = model.load_image(img_path)
            model.helper.get_original_features(model, dataset, img=img, mask=original_mask, textbox=("textbox" in res_dir))


            img = img + pert_map(perturbation, eps)

            mask = original_mask.clone()

            if use_di:
                img, mask = DI(img, mask)
            score_map = model.score_map(img, mask)
            cost += model.loss(score_map, mask, use_feature_loss=use_feature_loss)

            if use_feature_loss:
                cost += model.feature_loss
            if torch.isnan(cost): 
                import pdb; pdb.set_trace()
        model.zero_grad()
        if isinstance(cost, int): continue
        #cost.backward(retain_graph=True)
        cost.backward()
        cost_sum += cost.item()
        print("{}/{}, cost: {}".format(i, epoches * len(dataset),  cost))
        grad = pertu.grad
        if use_ti: grad=TI(grad, gaussian_kernel, kernel_size=kernel_size)

        pertu = pertu - alpha * grad.sign()
        pertu = pertu.detach()

    torch.save(pert_map(pertu.detach(), eps), os.path.join(res_dir, "perturbation.pt"))
    generate_universal_examples(model, dataset,pert_map(pertu.detach(), eps), res_dir)
