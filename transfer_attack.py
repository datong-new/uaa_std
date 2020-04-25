import os
from constant import *
import argparse
from torch import nn
import numpy as np
import torch
import cv2
from eval_helper import Eval
from east import Model as east
from textbox import Model as textbox
from craft_ import Model as craft
from db import Model as db
from constant import *
from copy import deepcopy

model_names = ["east", "textbox", "craft", "db"]
models = {"east":east, "textbox":textbox, "craft":craft, "db":db}
single_examples = {"original":IC15_TEST_IMAGES, "ensemble":"ensemble/"}
universal_examples = {"original":IC15_TEST_IMAGES, "ensemble": "ensemble/"}
for name in model_names:
    single_examples[name] = "res_{}/single_icdar/13/".format(name)
    universal_examples[name] = "res_{}/universal_icdar/13/".format(name)

def load_img(img_path, resize=False):
    img = cv2.imread(img_path)

    if resize:
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = nn.functional.interpolate(img, (1024, 1024))
        img = img.squeeze().permute(1,2,0).numpy()
        
    return img

def generate_ensemble_examples(model_name, attack_type="single"):
    if not os.path.exists("ensemble"): os.system("mkdir -p ensemble")
    img_names = os.listdir(IC15_TEST_IMAGES)
    if attack_type == "single": examples = single_examples
    else: examples = universal_examples

    for i, img_name in enumerate(img_names):
        print("generate ensemble_examples: {}/{} ".format(i, len(img_names)))
        original_img = load_img(IC15_TEST_IMAGES + img_name, True)
        ensemble_img = original_img

        for k, v in examples.items():
            if k in [model_name, "original", "ensemble"]: continue
#            print("img_name", v+img_name)
            img = load_img(v + img_name)
            perturbation = img-original_img
            ensemble_img += perturbation
        cv2.imwrite("ensemble/{}".format(img_name), ensemble_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', help='attack type: single or universal')
    parser.add_argument('--model_name', help='attack model name')
    args = parser.parse_args()
    attack_type = args.attack_type
    model_name = args.model_name

#    generate_ensemble_examples(model_name, attack_type) 

    eval_helper = Eval("icdar2015")
    model = models[model_name]()

    examples = single_examples if attack_type == "single" else universal_examples
    for k, example_path in examples.items():
        res = eval_helper.eval(model, example_path, PWD+"tmp/")
        with open("transfer_{}_{}.txt".format(model_name, attack_type), "a") as f: f.write("{}-{}:\t {}\n".format(model_name, k, res))
    os.system("rm ensemble/*")


    
    
    
    



