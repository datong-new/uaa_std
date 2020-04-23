from db import Model as DB
from icdar_dataset import ICDARDataset
from attack_util import *
from constant import *

VAR=0.5

def draw_polygons(model, imgs):
    for i in range(len(imgs)):
        print("process {}/{}".format(i, len(dataset)))
        model.draw_polygons(imgs[i])

if __name__ == "__main__":
    dataset = ICDARDataset()
    model = DB("icdar2015")
#    single_attack(model, dataset, res_dir=PWD+"res_db/single/")

    universal_attack(model, dataset, res_dir=PWD+"res_db/universal/")

   
    

    #img_dir = PWD+"res_db/single/"
    #imgs = [img_dir+f for f in os.listdir(img_dir)]
    #draw_polygons(model, imgs)

