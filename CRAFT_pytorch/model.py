import torch
import torch.backends.cudnn as cudnn
import craft_utils
import sys
sys.path.insert(0, "/data/shudeng/text_attack/advGAN_pytorch/CRAFT_pytorch")
from torch import nn
import os
import numpy as np
import imgproc
from PIL import Image
import cv2
from craft import CRAFT
from collections import OrderedDict
from test import test_net

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
    def __init__(self, evaluation=True):
        self.net, self.refine_net = self.model()
        self.net, self.refine_net = self.net.cuda(), self.refine_net.cuda()
#        self.net = torch.nn.DataParallel(self.net)
#        cudnn.benchmark = False
        self.net = self.net

        if evaluation:
            self.net.eval()
            self.refine_net.eval()
    def zero_grad(self):
        self.net.zero_grad()
        self.refine_net.zero_grad()

    def output(self, img_path, perturbation=None, refine_net=None):
        image = imgproc.loadImage(img_path)
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio
   
        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        img = x.unsqueeze(0).cuda()

        if perturbation is not None:
            img = add_perturbation(img, perturbation)
#        img = self.normalize(img)
    
        # forward pass
        with torch.no_grad():
            y, feature = self.net(img)
            #y, feature = self.net(img)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner, _ = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=True)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]
        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)
        return boxes, polys, ret_score_text

    def get_img_from_path(self, img_path):
        image = imgproc.loadImage(img_path)
        return torch.from_numpy(image).permute(2, 0, 1).float().cuda().unsqueeze(0)

    def add_perturbation(self, imgs, perturbation=None, h=512, w=512):
        batch, channel, h, w = imgs.shape
        perturation = perturbation.expand(batch, channel, h, w)
        return imgs.cuda() + perturbation 

    def normalize(self, x):
        return imgproc.normalize(x)

    def imgs_from_tensor(self, x):
        return imgproc.denormalize(x)
        
    def image(self, img_path, perturbation=None, canvas_size=1028, mag_ratio=1, cuda=True):
        image = imgproc.loadImage(img_path)
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        #ratio_h = ratio_w = 1 / target_ratio
        ratio_h = ratio_w = 1
        x = torch.from_numpy(img_resized).permute(2, 0, 1).float().cuda()
        
        if perturbation is not None:
            x += nn.functional.interpolate(perturbation, img_resized.shape[:2]).squeeze()
        x = imgproc.normalize(x)

        x = x.unsqueeze(0)
        if cuda:
            x = x.cuda()
        return x
    
    def model(self, resume_path="/data/shudeng/text_attack/advGAN_pytorch/CRAFT_pytorch/craft_mlt_25k.pth", refine_path="/data/shudeng/text_attack/advGAN_pytorch/CRAFT_pytorch/craft_refiner_CTW1500.pth", cuda=True):
        net = CRAFT()     # initialize
        if cuda:
            net.load_state_dict(copyStateDict(torch.load(resume_path)))
        else:
            net.load_state_dict(copyStateDict(torch.load(resume_path, map_location='cpu')))
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + refine_path + ')')
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refine_path)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refine_path, map_location='cpu')))
    
        return net, refine_net
    
    def get_features(self, x):
        net, refine = self.net, self.refine_net
        net = net.cuda()
        x = x.cuda()
        y, feature = net(x)
        y, feature = refine(y, feature)
    #    print("feature shape", feature.shape)
        return feature

    def vis_features(self, features, save_name="feature"):
        """ feature shape: [1, channels, h, w]
        """
        features = features.squeeze()
        channels, h, w = features.shape
        mean = features.view(channels, -1).mean(1)
        mean = mean.unsqueeze(1).unsqueeze(2)
        mean = mean.expand(channels, h, w)
        
        std = features.view(channels, -1).std(1)
        std = std.unsqueeze(1).unsqueeze(2)
        std = std.expand(channels, h, w)
        features = (features-mean)/std
        
        heat = torch.sqrt((features**2).sum(0))
        heat = heat.squeeze()
    
        heat = nn.functional.interpolate(heat.unsqueeze(0).unsqueeze(0), (heat.shape[0]*4, heat.shape[1]*4)).squeeze()
        heat = heat.cpu().detach().numpy()*255
        heatmapshow = None
        heatmapshow = cv2.normalize(heat, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_OCEAN)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.imwrite("{}.jpg".format(save_name), heatmapshow)
    
if __name__ == "__main__":    
    #img_path = "/data/shudeng/text_attack/curve_text/test_images/gt_1095.jpg"
    img_path = "/data/shudeng/text_attack/DB/datasets/total_text/test_images/img1095.jpg"
    os.system("cp {} .".format(img_path))
    x = image(img_path)
    features = get_feature(x)
    
    vis_feature(features)
