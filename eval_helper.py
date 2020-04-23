import numpy as np
import subprocess
import os
import cv2
from constant import *
from db import Model as DB
from shapely.geometry import Polygon

def area(poly):
    poly = Polygon(poly)
    return poly.area

class Eval():
    def __init__(self, dataset="icdar2015"):
        self.dataset = dataset
        if dataset == "total_text":
            self.img_dir = TOTALTEXT_TEST_IMAGES
        elif dataset == "icdar2015":
            self.img_dir = IC15_TEST_IMAGES

    def save_result(self, model, img_path, save_dir):
        original = self.img_dir + img_path.split("/")[-1]
        print("original: ", original)

        h_o, w_o = cv2.imread(original).shape[:2]
        is_output_polygon= self.dataset == "total_text"

        polys, img = model.get_polygons(img_path, is_output_polygon=is_output_polygon)
        h, w = img.shape[2:]
        print("h:{}, w:{}".format(h, w))

        scale_h, scale_w = h_o/h, w_o/w
        print("scale_h:{}, scale_w:{}".format(scale_h, scale_w))
        if self.dataset == "total_text":
            save_file = "{}.txt".format(img_path.split("/")[-1][:-4])
        else: save_file = "res_{}.txt".format(img_path.split("/")[-1][:-4])
        os.system("touch {}".format(save_dir + save_file))
        for poly in polys:
            poly_str = ""
            poly = np.array(poly)
            poly[:, 0], poly[:, 1] = poly[:, 0] * scale_w, poly[:, 1] * scale_h

            if area(poly)<10: continue
            for i in range(len(poly)): 
                if self.dataset == "total_text":
                    poly_str += "%d,%d," % (poly[i][1], poly[i][0])
                else: 
                    poly_str += "%d,%d," % (poly[i][0], poly[i][1])

            with open(save_dir + save_file, "a") as f:
                f.write(poly_str[:-1]+"\n")
            print(poly_str[:-1])

    def eval(self, model, img_dir, res_dir):
        files = os.listdir(img_dir)
        os.system("rm {}/*".format(res_dir))
        if not os.path.exists(res_dir): os.system("mkdir -p {}".format(res_dir))
        for i, img_path in enumerate(files):
            if not img_path[-4:]==".jpg": continue
            print("eval {}/{}".format(i, len(files)))
            self.save_result(model, os.path.join(img_dir, img_path), save_dir=res_dir)

        if self.dataset == "icdar2015":
            os.chdir(res_dir)
            res = subprocess.getoutput('zip -q submit.zip *.txt')
#            os.system("rm *.txt")
            os.chdir(PWD)
            res = subprocess.getoutput('python ./evaluate/script.py -g=./evaluate/gt_100.zip -s={}/submit.zip'.format(res_dir))
            os.system("rm {}/submit.zip".format(res_dir))
            print(res)
            return res

        elif self.dataset == "total_text":
            res = subprocess.getoutput("python {}Python_scripts/Pascal_VOC.py {}".format(PWD, res_dir))
            print(res)

if __name__ == "__main__":
    eval_helper = Eval()
    model = DB("ic15_resnet50")
#    model = DB("total_text")

    res_dir = PWD + "res_db/single_txt/"
    ### evaluate original images
    img_dir = IC15_TEST_IMAGES
    img_dir = "/data/shudeng/IC15/resize/"

    ### evaluate single perturbation images
    img_dir = PWD + "res_db/single/"

    ### evaluate single perturbation images
    img_dir = PWD + "res_db/universal/"

    eval_helper.eval(model, img_dir, res_dir)



