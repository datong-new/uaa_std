import cv2
import numpy 
from constant import *
import os

def draw_polygons(img_path, txt_path, datatype="total_text", res_path=PWD+"tmp/"):
    if not os.path.exists(res_path): os.system("mkdir -p {}".format(res_path))
    img = cv2.imread(img_path)
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        print(line)
        poly = line.strip().split(",")
        poly = [int(item) for item in poly]
        poly = numpy.array(poly).reshape(-1, 2)

        print(poly)
        if datatype == 'total_text':
            poly = poly[:, ::-1]
        print("img shape", img.shape)
        print(poly)
        img = cv2.polylines(img, numpy.int32([poly]), True, (0,0,255), 2)

    cv2.imwrite(res_path+img_path.split("/")[-1], img)


if __name__ == "__main__":
    img_dir = TOTALTEXT_TEST_IMAGES
    img_dir = IC15_TEST_IMAGES
    txt_dir = "/data/shudeng/text_attack/attacks/res_east/txt/" 
    for f in os.listdir(img_dir):
        print(f)
        draw_polygons(img_dir+f, txt_dir+f.split(".")[0]+".txt")
        try: draw_polygons(img_dir+f, txt_dir+f.split(".")[0]+".txt")
        except Exception: continue
