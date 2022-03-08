import torch
import subprocess
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
sys.path.insert(0, "./EAST")
from PIL import Image, ImageDraw
from model import EAST
from detect import resize_img, load_pil, get_boxes, plot_boxes, adjust_ratio
from icdar_dataset import ICDARDataset

VAR = 0.5

def pert_map(pert, eps):
    return (1/(1+torch.exp(pert)) -0.5) *2 * eps

def random_resize(img, mask=None):
    height = int(random.uniform(500, 1000))//32*32
    width = int(random.uniform(500, 1000))//32*32
    img = nn.functional.interpolate(img, (height, width))
    if mask is None: 
        return img
    mask = nn.functional.interpolate(mask.unsqueeze(1), (height, width))
    return img, mask.squeeze(1)

def resize(img):
    h, w = img.shape[2:]
    img = nn.functional.interpolate(img, (img.shape[2]//32*32, img.shape[3]//32*32))
    r_h, r_w = img.shape[2:]
    ratio_h, ratio_w = h/r_h, w/r_w
    return img, ratio_h, ratio_w

class Model():
    def __init__(self):
        os.chdir("EAST")
        model_path  = './pths/east_vgg16.pth'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        model = EAST().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.net = model

    def loss(self, score, mask, thresh=0.8):
        while len(mask.shape) < 4:
            mask = mask.unsqueeze(0)
        mask = nn.functional.interpolate(mask, score.shape[2:])
        l = 1/(1+torch.exp(-1e2*(score-thresh)))
        l = (l*mask).sum() / mask.sum()
        return l

    def load_image(self, img_path, scale=1):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = load_pil(img).cuda()
        h, w = img.shape[2:]
        img, _, _ = resize(img)
        return img, h, w, scale

    def universal_attack(self, dataset, epoches=10, eps=15/255/VAR, alpha=0.1, res_dir=PWD+"res_east/universal/"):
        if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
        batch_size = 2
        pertu = torch.zeros(1, 3, 1024, 1024)
        for i in range(epoches * len(dataset)):
            pertu.requires_grad = True
            perturbation = pertu.to(self.device)
            cost = 0
            for _ in range(batch_size):
                idx = int(random.uniform(0, len(dataset)))
                item = dataset.__getitem__(idx)
                img_path = item['filename']
                mask = item['mask'].to(self.device)
                x, height, width, scale = self.load_image(img_path)
                h, w = perturbation.shape[2:]
                x = nn.functional.interpolate(x, (h,w))
                img = x + pert_map(perturbation, eps)
                #img = random_resize(img)
                out, features = self.net(img)
                score = out[0]
                cost += self.loss(score, mask)
            self.net.zero_grad()
            cost.backward(retain_graph=True)
            print("epoch:{}, cost:{}".format(i//len(dataset)+1, cost))
            if i%len(dataset) == 0 and i != 0:
                with open(res_dir+"cost.log", "a") as f: f.write("epoch:{}, sum:{}\n".format(i//len(dataset)+1, cost))
            pertu = pertu - alpha * pertu.grad.sign()
            pertu = pertu.detach()
        torch.save(pert_map(pertu.detach(), eps), os.path.join(res_dir, "perturbation.pt"))
        perturbation = pert_map(pertu.detach(), eps)
        self.generate_universal_examples(dataset, perturbation)

                

    def generate_universal_examples(self, dataset, perturbation, res_dir=PWD+"res_east/universal/"):
        for i in range(len(dataset)):
            print("generate universal examples: {}/{}".format(i, len(dataset)))
            item = dataset.__getitem__(i)
            img_path = item['filename']
            x, height, width, scale = self.load_image(img_path)

            h, w = perturbation.shape[2:]
            x = nn.functional.interpolate(x, (h,w))
            img = x + perturbation.cuda()

            img = self.tensor_to_image(img)
            cv2.imwrite(os.path.join(res_dir, item['filename'].split("/")[-1]), img)


    def tensor_to_image(self, t, mean=torch.tensor([0.5,0.5,0.5]), std=torch.tensor([0.5, 0.5, 0.5])):
        t = t.squeeze().permute(1,2,0)
        t = (t*std) + mean
        img = t.detach().cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def resize_attack(self, img_path, mask, fix_size=True, iters=300, eps=15/255/VAR, alpha=0.1, res_dir=PWD+"res_east/fix/"):
        print(img_path)
        if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
        if os.path.exists(os.path.join(res_dir, img_path.split("/")[-1])): return
        mask = mask.unsqueeze(0).cuda()
        #img = Image.open(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = load_pil(img).cuda()
        img, _, _ = resize(img)
        original_img = img.clone()
        out, features = self.net(img)
        print("*"*100)
        pertur = torch.zeros(img.shape)
        for i in range(iters):
            pertur.requires_grad = True
            perturbation = pertur.cuda()
            perturbation = perturbation.cuda()
            cost = 0
            
            for j in range(2):
                h, w = original_img.shape[2:]
                perturbation = nn.functional.interpolate(perturbation, (h,w))
                img = original_img + pert_map(perturbation, eps)
                #if fix_size: img = nn.functional.interpolate(img, (800, 800))
                #else: img = random_resize(img)
                if not fix_size: img = random_resize(img)
                out, features = self.net(img)
                score = out[0]
                cost += self.loss(score, mask)
            
            if cost<1e-5 or i==iters-1: 
                print("pertur max:{}, min:{}".format(pert_map(perturbation, eps).max(), pert_map(perturbation, eps).min()))
                img = self.tensor_to_image(img)
                cv2.imwrite(res_dir+img_path.split("/")[-1], img.astype(int))
                break
            self.net.zero_grad()
            cost.backward(retain_graph=True)
            print("cost: ", cost)
            pertur = pertur - alpha * pertur.grad.sign()
            pertur = pertur.detach()

    def draw_result(self, img_path, perturbation=None, save_dir=PWD+"res_east/", resize_img=False):
        if not ".jpg" in img_path: return
        #img = Image.open(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = load_pil(img).cuda()
        image, ratio_h, ratio_w = resize(image)
        print("image shape", image.shape)
        
        if resize_img: image = nn.functional.interpolate(image, (800, 800))
    
        out, _ = self.net(image)
        score, geo = out
        boxes = get_boxes(score.squeeze(0).cpu().detach().numpy(), geo.squeeze(0).cpu().detach().numpy())
#        boxes = adjust_ratio(boxes, ratio_w, ratio_h)
        polys = []
        if boxes is not None:
            for box in boxes:
                box = [[box[2*i], box[2*i+1]] for i in range(4)]
                polys.append(box)
            polys = numpy.array(polys)
            img = cv2.polylines(img, np.int32(polys), True, (0,0,255), 2)
        
        cv2.imwrite(save_dir+img_path.split("/")[-1], cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 
        return polys, image

    def save_result(self, img_path, save_dir=PWD+"res_east/"):
        original = IC15_TEST_IMAGES + img_path.split("/")[-1]
        original = cv2.imread(original)
        h_o, w_o = original.shape[:2]
        polys, image = self.draw_result(img_path, resize_img=True)
        h, w = image.shape[2:]

        scale_h, scale_w = h_o/h, w_o/w
        save_file = "res_{}.txt".format(img_path.split("/")[-1][:-4])
        f = open(save_dir + save_file, "w")
        for quad in polys:
            quad[:, 0], quad[:, 1] = quad[:, 0] * scale_w, quad[:, 1] * scale_h
            [x0, y0], [x1, y1], [x2, y2], [x3, y3] = quad
            f.write("%d,%d,%d,%d,%d,%d,%d,%d\n" % (x0, y0, x1, y1, x2, y2, x3, y3))
        f.close()

    def eval(self, img_dir, res_dir=PWD+"res_east/"):
        files = os.listdir(img_dir)
        for i, img_path in enumerate(files):
            if not img_path[-4:]==".jpg": continue
            print("eval {}/{}".format(i, len(files)))
            self.save_result(os.path.join(img_dir, img_path), save_dir=res_dir)
        os.chdir(res_dir)
        res = subprocess.getoutput('zip -q submit.zip *.txt')
        os.system("rm *.txt")
        os.chdir(PWD)
        res = subprocess.getoutput('python ./evaluate/script.py -g=./evaluate/gt_100.zip -s={}/submit.zip'.format(res_dir))
        os.system("rm {}/submit.zip".format(res_dir))
        print(res)

    def tensor_to_image(self, t, mean=torch.tensor([0.5, 0.5, 0.5]), var=torch.tensor([0.5, 0.5, 0.5])):
        mean, var = mean.to(self.device), var.to(self.device)
        t = t.squeeze().permute(1,2,0)
        t = t*var+mean

        img = t.detach().cpu().numpy() * 255
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

if __name__ == "__main__":
    model = Model()
    dataset = ICDARDataset()
    print("finish loading model")

#    for f in os.listdir(PWD+"res_east/universal/"):
#        model.draw_result(PWD+"res_east/universal/"+f)
#    exit(0)

    # fix attack
#    fix_attack(model, dataset, res_dir=PWD+"res_east/fix/")

    # universal attack
    universal_attack(model, dataset)

#     evaluate original images
#    model.eval(img_dir=IC15_TEST_IMAGES, input_size=850//32*32)
#    model.eval(img_dir=IC15_TEST_IMAGES)

#     evaluate resize images
#    model.eval(img_dir=PWD+"res_textbox++/resize/", input_size=800)

#     evaluate fix images
#    model.eval(img_dir=PWD+"res_east/fix/")

    # eval universal
    model.eval(PWD+"res_east/universal/")
