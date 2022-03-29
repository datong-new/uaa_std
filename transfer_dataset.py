import os
from db import Model as DB
from east import Model as east
from textbox import Model as textbox
from craft_ import Model as craft
from eval_helper import Eval
from icdar_dataset import ICDARDataset
from totaltext_dataset import TotalText

import torch

tmp_file = os.path.join(os.getcwd(), "tmp_transferdataset.txt")

def get_model(model_name, dataset_name):
    if model_name=="db": return DB(dataset_name)
    if model_name=="east": return east()
    if model_name=="textbox": return textbox()
    if model_name=="craft": return craft(dataset_name)

def get_dataset(dataset_name):
    if 'icdar' in dataset_name:
        return ICDARDataset()
    else:
        return TotalText()


def eval_transferdataset(
        dataset_name,
        model_name,
        eval_helper, 
        img_dir=os.path.join(os.getcwd(), "tmp_imgs")):

    os.system("rm -rf {}".format(img_dir))
    eval_helper=Eval(dataset_name)

    model = get_model(model_name, dataset_name)
    dataset = get_dataset(dataset_name)
    perturbation = torch.load(perturbation_path)
    model.generate_universal_examples(dataset, perturbation, img_dir)
    res=eval_helper.eval(model, img_dir, res_dir=os.path.join(os.getcwd(), "tmp_txt"))
    with open(tmp_file, "a") as f: f.write("{}: {}, {}, {}\n".format(model_name,dataset_name, perturbation_path, img_dir, res))


if __name__=="__main__":
    perturbation_paths = []
    for model_name in ['db', 'craft', 'textbox', 'east']:
        for dataset_name in ['icdar2015', 'total_text']:
            if dataset_name=='total_text' and model_name in ['textbox', 'east']: continue
            perturbation_paths += [
                    os.path.join(
                        os.getcwd(),
                        "res_{}/universal/momentumFalse_diTrue_tiTrue_featureTrue_eps13{}/perturbation.pt".format(
                            model_name,
                            "" if "icdar" in dataset_name else "total_text"
                            )
                        )
                    ]

    for model_name in ['db', 'craft', 'textbox', 'east']:
        for dataset_name in ['icdar2015', 'total_text']:
            if dataset_name=='total_text' and model_name in ['textbox', 'east']: continue
            for perturbation_path in perturbation_paths:
                if not os.path.exists(perturbation_path):
                    print(perturbation_path)
                    continue
                eval_transferdataset(dataset_name, model_name, perturbation_path)
        

