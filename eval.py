from db import Model as DB
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
    img_dir = "/data/shudeng/attacks/res_db/universal_icdar_feature/11"
    #img_dir = "/data/shudeng/attacks/res_db/single_icdar_feature/11"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="db")
    args = parser.parse_args()
    model = args.model

    img_dirs = [
                 f"/data/shudeng/attacks/res_{model}/single/momentumFalse_diFalse_tiFalse_featureTrue_eps13/",
                 f"/data/shudeng/attacks/res_{model}/universal/momentumFalse_diFalse_tiFalse_featureTrue_eps13/",
               ]

    for img_dir in img_dirs:
        res_dir = "/data/shudeng/attacks/transfer_txt/"
        for model_name in model_names:
            print("model_name:{}".format(model_name))
            res = eval_helper.eval(get_model(model_name, dataset_name), img_dir, res_dir)
            with open("tmp.txt", "a") as f: f.write("{}: {}, {}\n".format(model_name, img_dir, res))
            
            
        
 

