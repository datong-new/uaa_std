import torch
import random
import cv2
from util import *
from torch import nn
import os

VAR=0.5

def resize(img):
    return nn.functional.interpolate(img, (1024, 1024))

def generate_universal_examples(model, dataset, perturbation, res_dir):
    for i in range(0, len(dataset)):
        print("generate universal examples: {}/{}".format(i, len(dataset)))
        item = dataset.__getitem__(i)
        x = model.load_image(item['filename'])
        x = x + perturbation.cuda()
        img = model.tensor_to_image(x)
        cv2.imwrite(os.path.join(res_dir, item['filename'].split("/")[-1]), img)

def _single_attack(model, img_path, mask, res_dir, eps=15/255/VAR, iters=30, alpha=0.2, cost_thresh=0.05):
    if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
    if os.path.exists(os.path.join(res_dir, img_path.split("/")[-1])): return
    mask = mask.unsqueeze(0).cuda()
    original_img = model.load_image(img_path).cuda()

    pertur = torch.zeros(original_img.shape)
    for i in range(iters):
        pertur.requires_grad = True
        perturbation = pertur.cuda()
        cost = 0
        img = original_img + pert_map(perturbation, eps)
        score_map = model.score_map(img)
        cost = model.loss(score_map, mask)
        if cost<cost_thresh or i==iters-1:
            img = model.tensor_to_image(img)
            cv2.imwrite(res_dir+img_path.split("/")[-1], img.astype(int))
            break
        model.zero_grad()
        cost.backward(retain_graph=True)
        print("cost: ", cost)
        pertur = pertur - alpha * pertur.grad.sign()
#        pertur = pertur - pertur.grad / pertur.grad.max()
        pertur = pertur.detach()

def single_attack(model, dataset, res_dir, eps=15/255/VAR, iters=300, alpha=0.2, cost_thresh=0.05):
    for i in range(len(dataset)):
        item = dataset.__getitem__(i)
        img_path, mask = item['filename'], item['mask']
        _single_attack(model, img_path, mask, res_dir=res_dir, eps=eps, iters=iters, alpha=alpha, cost_thresh=cost_thresh)

def universal_attack(model, dataset, res_dir, epoches=30, eps=15/255/VAR, alpha=0.2):
    if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
    batch_size = 2
    pertu = torch.zeros(1, 3, 1024, 1024)
    cost_sum = 0
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
            mask = item['mask'].cuda()

            img = model.load_image(img_path)
            img = img + pert_map(perturbation, eps)
            score_map = model.score_map(img)
            cost += model.loss(score_map, mask)
        model.zero_grad()
        if isinstance(cost, int): continue
        cost.backward(retain_graph=True)
        cost_sum += cost.item()
        print("cost: ", cost)
        pertu = pertu - alpha * pertu.grad.sign()
#        pertur = pertur - pertur.grad / pertur.grad.max()
        pertu = pertu.detach()

    torch.save(pert_map(pertu.detach(), eps), os.path.join(res_dir, "perturbation.pt"))
    generate_universal_examples(model, dataset,pert_map(pertu.detach(), eps), res_dir)
