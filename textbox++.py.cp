import torch
from torch import nn
import subprocess
import numpy as np
import sys
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

class Model():
    def __init__(self):
        self.net = RetinaNet()
        self.encoder = DataEncoder(0.4, 0.1)
        self._init_model()
        self.device = "cuda"
        self.net.to(self.device)
        self.net = nn.DataParallel(self.net)
        self.net.eval()

    def _init_model(self, model_path='/data/shudeng/text_attack/attacks/Text_Detector/ICDAR2015_TextBoxes.pth'):
        # load checkpoint
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['net'])

    def load_image(self, img_path, input_scale=1280):
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x,_,_ = Augmentation_inference(input_scale)(img)
        x = x.unsqueeze(0)
        x = x.to(self.device)
        return x, height, width, input_scale

    def _get_confidence_maps(self, img_path):
        x, _, _, _ = self.load_image(img_path)
        maps = self.net(x, attack=True)
        return maps


    def _test(self, img, save_name="textbox.jpg", input_size=800, save_dir=PWD+"res_textbox++/"):
        if not isinstance(img, torch.Tensor):
            if not os.path.exists(save_dir):
                os.system("mkdir -p {}".format(save_dir))
            img_np = cv2.imread(img)
            save_name = img.split("/")[-1]
            img, height, width, scale = self.load_image(img, input_scale=input_size)
        else:
            img = nn.functional.interpolate(img, (input_size, input_size))
            img_np = self.tensor_to_image(img)
            scale = input_size
            height, width = input_size, input_size
        loc_preds, cls_preds = self.net(img)
        quad_boxes, labels, scores = self.encoder.decode(loc_preds.data.squeeze(0), cls_preds.data.squeeze(0), scale)
        quad_boxes /= scale
        quad_boxes *= ([[width, height]] * 4)
        quad_boxes = quad_boxes.astype(np.int32)
        print("quad boxes shape", quad_boxes.shape)
        img_np = cv2.polylines(img_np, quad_boxes, True, (0,0,255), 2)
        cv2.imwrite(save_dir + save_name, img_np)
        return quad_boxes

    def resize_attack(self, img_path, mask, fix_size=True, iters=500, eps=20/255/var.max().item(), alpha=0.1, res_dir=PWD+"res_textbox++/fix/"):
        if os.path.exists(os.path.join(res_dir, img_path.split("/")[-1])):
            return
        x, height, width, scale = self.load_image(img_path)
        original_img = x.clone()
#        pertu = torch.zeros(1, 3, 800, 800)
        pertu = torch.zeros(original_img.shape)
        if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
        print("img path", img_path)
        print("*" * 100)
        for i in range(iters):
            pertu.requires_grad = True
            perturbation = pertu.to(self.device)
            cost = 0
#            for j in range(2):
            h, w = original_img.shape[2:]
#            perturbation = nn.functional.interpolate(perturbation, (h,w))
          
            img = original_img + pert_map(perturbation, eps)

            for _ in range(2):
                if not fix_size: img = random_resize(img)
                b, h, w = img.shape[1:]
                maps = self.net(img, attack=True)
                for m in maps:
                    m = m.sigmoid()
                    if m.max()>0.2: cost += loss(m, mask, thresh=0.2)
            if isinstance(cost, int):
                if fix_size: break
                else: continue
            if cost<1e-6 or i==iters-1: 
                img = self.tensor_to_image(img)
                cv2.imwrite(os.path.join(res_dir, img_path.split("/")[-1]), img)
                break
            self.net.zero_grad()
            cost.backward(retain_graph=True)
            print("cost: ", cost)
            pertu = pertu - alpha * pertu.grad.sign()
            pertu = pertu.detach()
        #torch.save(pert_map(pertu.detach(), eps), "perturbation.pt")

    def universal_attack(self, dataset, epoches=20, eps=30/255, alpha=0.1, res_dir=PWD+"res_textbox++/universal/"):
        if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
        batch_size = 2
        pertu = torch.zeros(1, 3, 800, 800)
        for _ in range(epoches * len(dataset)):
            pertu.requires_grad = True
            perturbation = pertu.to(self.device)
            cost = 0
            for i in range(batch_size):
                idx = int(random.uniform(0, len(dataset)))
                item = dataset.__getitem__(idx)
                img_path = item['filename']
                mask = item['mask'].to(self.device)
                x, height, width, scale = self.load_image(img_path, input_scale=800)
                h, w = x.shape[2:]
                perturbation = nn.functional.interpolate(perturbation, (h,w))
                img = x + pert_map(perturbation, eps)
                maps = self.net(img, attack=True)
                for m in maps:
                    m = m.sigmoid()
                    if m.max()>0.2: cost += loss(m, mask, thresh=0.2)

            if isinstance(cost, int):
                if fix_size: continue

            self.net.zero_grad()
            cost.backward(retain_graph=True)
            print("cost: ", cost)
            pertu = pertu - alpha * pertu.grad.sign()
            pertu = pertu.detach()
        torch.save(pert_map(pertu.detach(), eps), os.path.join(res_dir, "perturbation.pt"))
        perturbation = pert_map(pertu.detach(), eps)
        self.generate_universal_examples(dataset, perturbation)

    def save_result(self, img_path, input_size=1280, save_dir=PWD+"res_textbox++/"):
        original = IC15_TEST_IMAGES + img_path.split("/")[-1]
        original = cv2.imread(original)
        h_o, w_o = original.shape[:2]
        h, w = cv2.imread(img_path).shape[:2]
        
        scale_h, scale_w = h_o/h, w_o/w

        quad_boxes = self._test(img_path, input_size=input_size)

        save_file = "res_{}.txt".format(img_path.split("/")[-1][:-4])
        f = open(save_dir + save_file, "w")
        for quad in quad_boxes:
            quad[:, 0], quad[:, 1] = quad[:, 0] * scale_w, quad[:, 1] * scale_h
            [x0, y0], [x1, y1], [x2, y2], [x3, y3] = quad
            f.write("%d,%d,%d,%d,%d,%d,%d,%d\n" % (x0, y0, x1, y1, x2, y2, x3, y3))
        f.close()

    def tensor_to_image(self, t):
        mean=torch.tensor([0.485,0.456,0.406]).to(self.device)
        var=torch.tensor([0.229,0.224,0.225]).to(self.device)
        t = t.squeeze().permute(1,2,0)
        t = (t*var+mean) * 255.0
        img = t.detach().cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite("textbox.jpg", img)
        return img

    def eval(self, img_dir, input_size=1280, res_dir=PWD+"res_textbox++/fix/"):
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

    def generate_universal_examples(self, dataset, perturbation, res_dir=PWD+"res_textbox++/universal/"):
        for i in range(0, len(dataset)):
            print("generate universal examples: {}/{}".format(i, len(dataset)))
            item = dataset.__getitem__(i)
            x, height, width, scale = self.load_image(item['filename'])
            h, w = x.shape[2:]
            perturbation = nn.functional.interpolate(perturbation, (h,w))
            x = x + perturbation.to(self.device)
            img = self.tensor_to_image(x)
            cv2.imwrite(os.path.join(res_dir, item['filename'].split("/")[-1]), img)

def fix_attack(model, dataset):
    # generate adversarial example for fix-size image
    for i in range(0, len(dataset)):
        item = dataset.__getitem__(i)
        img_path = item['filename']
        mask = item['mask'].to(model.device)
        model.resize_attack(img_path, mask, fix_size=True, res_dir=PWD+"res_textbox++/fix/")

def resize_attack(model, dataset):
    # generate adversarial example for resized image
    for i in range(0, len(dataset)):
        print("{}/{}".format(i, len(dataset)))
        item = dataset.__getitem__(i)
        img_path = item['filename']
        mask = item['mask'].to(model.device)
        model.resize_attack(img_path, mask, fix_size=False, res_dir=PWD+"res_textbox++/resize/")

def universal_attack(model, dataset):
    model.universal_attack(dataset)



if __name__ == "__main__":
    model = Model()
    dataset = ICDARDataset()
    # fix attack
    fix_attack(model, dataset)

    # universal attack
#    universal_attack(model, dataset)
    
    # resize_attack
    #resize_attack(model, dataset)

#     evaluate fixed adversarial examples
#    model.eval(img_dir=PWD+"res_textbox++/fix/", input_size=800)

#     evaluate original images
#    model.eval(img_dir=IC15_TEST_IMAGES, input_size=850//32*32)
#    model.eval(img_dir=IC15_TEST_IMAGES)

#     evaluate resize images
#    model.eval(img_dir=PWD+"res_textbox++/resize/", input_size=800)

#     evaluate universal images
#    model.eval(img_dir=PWD+"res_textbox++/universal/", input_size=700)


#     text
#    for f in os.listdir(IC15_TEST_IMAGES):
#        model._test(os.path.join(IC15_TEST_IMAGES, f))
