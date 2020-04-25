import torch
import argparse
from eval_helper import *
from totaltext_dataset import TotalText
from icdar_dataset import ICDARDataset
from torch import nn
from attack_util import *
import cv2
from collections import OrderedDict
import json
from util import *
import sys
from constant import *
sys.path.insert(0,"/data/shudeng/text_attack/attacks/DB")
import argparse
import os
import torch
import yaml
from tqdm import tqdm
import numpy as np
from trainer import Trainer
# tagged yaml objects
from experiment import Structure, TrainSettings, ValidationSettings, Experiment
from concern.log import Logger
from data.data_loader import DataLoader
from data.image_dataset import ImageDataset
from training.checkpoint import Checkpoint
from training.learning_rate import (
    ConstantLearningRate, PriorityLearningRate, FileMonitorLearningRate
)
from training.model_saver import ModelSaver
from training.optimizer_scheduler import OptimizerScheduler
from concern.config import Configurable, Config
import time

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
VAR = std.mean()


class Model():
    def __init__(self, resume="icdar2015"):
        print("resume" + resume)

        os.chdir("/data/shudeng/text_attack/attacks/DB")
        with open("db_args.json", "r") as f:
            args = json.load(f)

        conf = Config()
        experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
        experiment_args.update(cmd=args)
        experiment = Configurable.construct_class_from_config(experiment_args)
        cmd = args
        verbose = args['verbose']
        args = experiment_args

        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.data_loaders = experiment.evaluation.data_loaders
        self.args = cmd
        self.logger = experiment.logger
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure

        self.verbose = verbose
        self.net = self.structure.builder.build_basic_model().cuda()
        self.resume(resume)


    def resume(self, resume="total_text"):
        path = MODEL_PATH
        if resume == "total_text":
            path += "totaltext_resnet50"
        else: path += "ic15_resnet50"

        if not os.path.exists(path):
            self.logger.warning("Checkpoint not found: " + path)
            return
        print("resume from "+ path)

        states = torch.load(
            path, map_location='cpu')

        new_state_dict = OrderedDict()
        for k, v in states.items():
            name = k[13:] # remove `module.`
            new_state_dict[name] = v
        # load params
        self.net.load_state_dict(new_state_dict)
        self.logger.info("Resumed from " + path)

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = torch.from_numpy(img)
        img = img/255.0
        img = (img-torch.tensor(mean))/torch.tensor(std)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        img = nn.functional.interpolate(img, (1024, 1024))
        return img.cuda()

    def tensor_to_image(self, t):
        img = t.squeeze()
        img = img.permute(1,2,0)
        img = img * torch.tensor(std).cuda() + torch.tensor(mean).cuda()
        img *= 255.0
        img = img.detach().cpu().numpy()
        return img

    def get_features(self, img_path):
        img = self.load_image(img_path).cuda().float()
        out, score = self.net(img)
        return score

    def score_map(self, img):
        out, score = self.net(img.float())
        return score

    def loss(self, score, mask, thresh=0.19):
        return loss(score, mask, thresh)

    def zero_grad(self):
        self.net.zero_grad()

    def get_result(self, img_path):
        img = self.load_image(img_path).cuda().float()
        out, features = self.net(img)
        return out

    def get_polygons(self, img_path, is_output_polygon=True):
        img = self.load_image(img_path).cuda().float()
        out = self.get_result(img_path)
        batch = {}
        batch['image'] = torch.rand(1)
        batch['shape'] = [(1024, 1024)]

        output = self.structure.representer.represent(batch, out, is_output_polygon=is_output_polygon)  

        return output[0][0], img

    def draw_polygons(self, img_path, res_dir=PWD+"res_db/single_draw/"):
        if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
        img = self.tensor_to_image(self.load_image(img_path))
        polys = self.get_polygons(img_path)
        for poly in polys:
            poly = np.array([poly])
            img = cv2.polylines(img, np.int32(poly), True, (0,0,255), 2)
        cv2.imwrite(res_dir+img_path.split("/")[-1], img)

    def generate_universal_examples(self, dataset, perturbation, res_dir=PWD+"res_db/universal/"):
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
    res_dir = PWD+"res_db/txt/"

    eps = range(5, 15, 2)
    if attack_type == "single":
        # single attack for different epsilon
        for ep in eps:
            img_dir = PWD+"res_db/single_icdar/{}/".format(ep)
            single_attack(model, dataset, res_dir=img_dir, eps=ep/255/VAR, iters=100, cost_thresh=0.001)
            res = eval_helper.eval(model, img_dir, res_dir)
            with open(img_dir + "../eps.txt", "a") as f: f.write("{}: {}\n".format(ep, res))
    elif attack_type == "universal":
        for ep in eps:
            img_dir = PWD+"res_db/universal_icdar/{}/".format(ep)
            universal_attack(model, dataset, res_dir=img_dir, epoches=18, eps=ep/255/VAR, alpha=0.2)
            res = eval_helper.eval(model, img_dir, res_dir)
            with open(img_dir + "../u_eps.txt", "a") as f: f.write("{}: {}\n".format(ep, res))

    exit(0)


    universal_totaltext_dir = PWD+"res_db/universal_totaltext/"
#    universal_icdar_dir = PWD + "res_db/universal_icdar/"
    
    # single attack
#    single_attack(model, dataset, res_dir=PWD+"res_db/single_totaltext/", eps=15/255/VAR, iters=300, cost_thresh=0.07)

    # universal attack
    #universal_attack(model, dataset, res_dir=universal_icdar_dir, epoches=30, eps=15/255/VAR, alpha=0.2)

#    perturbation = torch.load(universal_totaltext_dir+"perturbation.pt")
#    model.generate_universal_examples(dataset, perturbation, res_dir=universal_totaltext_dir)
    

    eval_helper = Eval('total_text')

    img_dir = TOTALTEXT_TEST_IMAGES
    img_dir = "/data/totaltext/totaltext/Images/resize/"
    res_dir = PWD+"res_db/txt/"

    img_dir = "/data/shudeng/text_attack/attacks/res_db/single_totaltext/"
#    res_dir = "/data/shudeng/text_attack/attacks/res_db/single_totaltext_txt"
    #eval_helper.eval(model, universal_totaltext_dir, res_dir)


    

