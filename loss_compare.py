import torch
from eval_helper import *
from icdar_dataset import ICDARDataset
from attack_util import *
import argparse

from east import Model as east
from textbox import Model as textbox
from craft_ import Model as craft
from db import Model as db
from constant import *
models = {"east":east, "textbox":textbox, "craft":craft, "db":db}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', help='attack type: single or universal')
    parser.add_argument('--model_name', help='attack model name')
#    parser.add_argument('--loss', help='model loss')
    args = parser.parse_args()
    attack_type = args.attack_type
    model_name = args.model_name
#    loss = args.loss

    eval_helper = Eval("icdar2015")
    dataset = ICDARDataset()
    model = models[model_name](loss=loss)
    losses = ["thresh", "crossentropy"]
    losses = ["crossentropy"]
    ep = 10
    img_dir, txt_dir = PWD + "img_tmp/", PWD + "txt_tmp/"
    print("VARS",VARS)

    if attack_type == "single":
        for i in range(10, 100, 10):
            for loss_type in losses:
                model.loss_type = loss_type
                single_attack(model, dataset, res_dir=img_dir+"{}/".format(model_name), eps=ep/255/VARS[model_name], iters=10, cost_thresh=0.001)
                res = eval_helper.eval(model, img_dir+"{}/".format(model_name), txt_dir+"{}/".format(model_name))
                with open(PWD+"loss_compare_{}.txt".format(model_name), "a") as f: f.write("iter:{}, loss:{}, attack_type:{}, model_name:{}, result:{}, ep:{}\n".format((i+1)*10, loss_type, attack_type, model_name, res, ep))

        os.system("rm {}img_tmp/* {}txt_tmp/*".format(PWD, PWD))
    elif attack_type == "universal":
        epoches = range(2, 18, 2)
        for epoch in epoches:
            for loss_type in losses:
                model.loss_type = loss_type
                universal_attack(model, dataset, res_dir=img_dir+"{}/".format(model_name), epoches=epoch, eps=ep/255/VARS[model_name], alpha=0.2)
                res = eval_helper.eval(model, img_dir+"{}/".format(model_name), txt_dir+"{}/".format(model_name))
                with open(PWD+"loss_compare_{}.txt".format(model_name), "a") as f: f.write("epoch:{}, loss:{}, attack_type:{}, model_name:{}, result:{}, ep:{}\n".format(epoch, loss_type, attack_type, model_name, res, ep))

        os.system("rm {}img_tmp/* {}txt_tmp/*".format(PWD, PWD))




