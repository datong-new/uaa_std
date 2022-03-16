import os



if __name__=="__main__":
    gpu=1
    model = "db"
    for use_di in ["False", "True"]:
        for use_ti in ["False", "True"]:
            for use_feature_loss in ['False', "True"]:
                for use_momentum in ['False']:
                    for attack_type in ['single', 'universal']:
                        save_dir = f"res_{model}/{attack_type}/momentum{use_momentum}_di{use_di}_ti{use_ti}_feature{use_feature_loss}_eps13"
                        
                        if 'featureTrue' in save_dir or not (os.path.exists(save_dir) and len(os.listdir(save_dir))>=100):
                            os.system("rm -rf {}".format(save_dir))

                            command = f"CUDA_VISIBLE_DEVICES={gpu} python attacks.py --attack_type {attack_type} --eps 13 --use_di {use_di} --use_ti {use_ti} --use_feature_loss {use_feature_loss}  --model {model}"
                            print(command)
                            os.system(command)
    
