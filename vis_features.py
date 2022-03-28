import os
from icdar_dataset import ICDARDataset
from db import Model as DB
import torch
import numpy as np

class PCA(object):
    # https://zhuanlan.zhihu.com/p/369297419
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

        return X.matmul(self.proj_mat)
       

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)



def vis_features(text_features, nottext_features, save_path):
    import matplotlib.pyplot as plt
    pca = PCA()
    for i, (text_feature, nottext_feature) in enumerate(zip(text_features, nottext_features)):
        if  not  (i+1)%4==0: continue
   
        len_a, len_b = len(text_feature), len(nottext_feature)
        feature = torch.cat([text_feature, nottext_feature], dim=0)

        feature = pca.fit(feature).cpu().detach()
        text_feature = feature[:len_a]
        choice = np.random.choice(
                         list(range(len_a, feature.shape[0])),
                         len_a, replace=True)
        feature = torch.cat([
                            feature[:len_a],
                            feature[choice]], dim=0)

        y = [1]*len_a + [0]*len_a
        plt.scatter(feature.T[0], feature.T[1], c=y)
        plt.savefig(save_path.replace(".jpg", "_{}.png".format(i)))
        plt.clf()
    plt.close()

def get_mask(dataset, img_path):
    img_name = img_path.split("/")[-1]
    for idx in range(len(dataset)):
        item = dataset.__getitem__(idx)
        img_path = item['filename']
        original_mask = item['mask']
        if img_name in img_path: return original_mask
        


if __name__=="__main__":
    img_path = "/data/shudeng/attacks/res_db/single/momentumFalse_diFalse_tiFalse_featureFalse_eps13/img_1.jpg"
    model_name = "DB"
    if model_name == "DB":
        model = DB()
    dataset=ICDARDataset()
    idx = 0
    device="cuda"

    for source_model in ['db', 'craft', 'east', 'textbox']:
        for di in ['False', 'True']:
            for ti in ['False', 'True']:
                for feature in ['False', 'True']:
                    img_path=f"/data/attacks/res_{source_model}/single/momentumFalse_di{di}_ti{ti}_feature{feature}_eps13/img_83.jpg"
                    print(img_path)

                    original_mask = get_mask(dataset, img_path)
                    img = model.load_image(img_path)
                    _, _, text_features, nottext_features = model.helper.forward(img, original_mask)
                    if not os.path.exists("./vis"):
                        os.makedirs("./vis")

                    vis_features(
                                    text_features, 
                                    nottext_features, 
                                    save_path="vis/{}_{}".format(model_name, img_path.replace("/", "_"))
                    )
          

