from db import Model as DB
import os
from east import Model as east
from textbox import Model as textbox
from craft_ import Model as craft
from eval_helper import Eval
import argparse

def get_model(model_name, dataset_name):
    if model_name=="db": return DB(dataset_name)
    if model_name=="east": return east()
    if model_name=="textbox": return textbox()
    if model_name=="craft": return craft(dataset_name)


if __name__ == "__main__":
    model_names = ['db', 'east', 'textbox', 'craft']
    dataset_name = 'icdar2015'
    eval_helper = Eval(dataset_name)


    
    #model_name = 'textbox'
    #img_dir = '/data/attacks/res_textbox/single/momentumFalse_diFalse_tiTrue_featureTrue_eps13'
    #model_name = 'craft'
    #img_dir="./dataset/IC15/test_images"
    #res = eval_helper.eval(get_model(model_name, dataset_name), img_dir, res_dir)
    #with open("tmp.txt", "a") as f: f.write("{}: {}, {}\n".format(model_name, img_dir, res))

    #img_dir="/data/attacks/res_db/single/momentumFalse_diFalse_tiFalse_featureFalse_eps13"
    #res = eval_helper.eval(get_model(model_name, dataset_name), img_dir, res_dir)
    #with open("tmp.txt", "a") as f: f.write("{}: {}, {}\n".format(model_name, img_dir, res))





    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="db")
    args = parser.parse_args()
    model = args.model

    img_dirs = [
                 #f"/data/attacks/res_{model}/single/momentumFalse_diFalse_tiFalse_featureTrue_eps13/",
                 #f"/data/attacks/res_{model}/universal/momentumFalse_diFalse_tiFalse_featureTrue_eps13/",
               ]

    for use_di in ["False", "True"]:
        for use_ti in ["False", "True"]:
            for use_feature_loss in ['False', 'True']:
                for use_momentum in ['False']:
                    for attack_type in ['single', 'universal']:
                        save_dir = f"/data/attacks/res_{model}/{attack_type}/momentum{use_momentum}_di{use_di}_ti{use_ti}_feature{use_feature_loss}_eps13"
                        if not(os.path.exists(save_dir) and len(os.listdir(save_dir))>=100): 
                            import pdb; pdb.set_trace()
                            continue
                        img_dirs += [save_dir]


    from parse_tmp import parse_tmp
    tmp_file = "tmp.txt"
    res_dict = parse_tmp(tmp_file)
    keys=[]
    res_dirs=[]
    for img_dir in img_dirs:
        for model_name in model_names:
            print("model_name:{}".format(model_name))
            key = f"{model_name}: {img_dir}, Calculated"

            if 'featureFalse' in key and key in res_dict: continue

            print(key)
            keys+=[key]
            res_dir = "/data/shudeng/attacks/transfer_txt/{}/{}".format(img_dir.replace("/", "_"), model_name)
            res_dirs += [res_dir]

            res = eval_helper.eval(get_model(model_name, dataset_name), img_dir, res_dir)
            with open(tmp_file, "a") as f: f.write("{}: {}, {}\n".format(model_name, img_dir, res))
    print(keys)
    print(len(keys))
    print(res_dirs)
    print(len(res_dirs))
            
            
        
 

