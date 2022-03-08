from db import Model as DB
from icdar_dataset import ICDARDataset
from attack_util import *
from constant import *
import argparse

VAR=0.5

def draw_polygons(model, imgs):
    for i in range(len(imgs)):
        print("process {}/{}".format(i, len(dataset)))
        model.draw_polygons(imgs[i])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', help='attack type: single or universal')
    parser.add_argument('--eps', type=int, default=11)
    parser.add_argument('--model', type=str, default="db")

    parser.add_argument('--use_momentum', type=str, default="False")
    parser.add_argument('--use_di', type=str, default="False")
    parser.add_argument('--use_ti', type=str, default="False")
    parser.add_argument('--use_feature_loss', type=str, default="False")

    args = parser.parse_args()
    eps = args.eps


    dataset = ICDARDataset()
    if args.model=="db":
        from db import Model, VAR
        model = Model("icdar2015")
    elif args.model=="east":
        from east import Model, VAR
        model = Model()
    elif args.model=="craft":
        from craft_ import Model, VAR
        model = Model("icdar2015")
    elif args.model=="textbox":
        from textbox import Model, VAR
        model = Model()





    use_momentum = args.use_momentum != "False"
    use_di = args.use_di != "False"
    use_ti = args.use_ti != "False"
    use_feature_loss = args.use_feature_loss != "False"

    attack_type = args.attack_type 
    res_dir = os.path.join(PWD,
                           "res_{}".format(args.model),
                          attack_type,
                          "momentum{}_di{}_ti{}_feature{}_eps{}".format(use_momentum, use_di, use_ti, use_feature_loss, eps))
    eps=eps/255/VAR


    if attack_type == "single":
        single_attack(model, 
                    dataset, 
                    res_dir=res_dir,
                    eps=eps,
                    use_momentum=use_momentum,
                    use_di=use_di,
                    use_ti=use_ti,
                    use_feature_loss=use_feature_loss,
                    iters=100, 
                    cost_thresh=0.001
                    )
    else:
        universal_attack(model, 
                    dataset, 
                    res_dir=res_dir,
                    eps=eps,
                    use_momentum=use_momentum,
                    use_di=use_di,
                    use_ti=use_ti,
                    use_feature_loss=use_feature_loss,
                    epoches=18, 
                    alpha=0.2
                    )

