import torch
from torch import nn
from imgproc import *
import os
from skimage import io

def gen_adv(img_path, pertu_path):
    os.system("cp {} ./data/".format(img_path))
    img = loadImage(img_path)
    x = torch.from_numpy(img).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0).float()
    pertu = torch.load(pertu_path)
    print("pertu max:{}, min:{}".format(pertu.max(), pertu.min()))
    x += nn.functional.interpolate(pertu, x.shape[2:])

    x = x.squeeze().permute(1, 2, 0).numpy()
    io.imsave("./data/gen_adv_{}.jpg".format(img_path.split("/")[-1]), x)

if __name__ == "__main__":
    img_path = "/data/shudeng/text_attack/advGAN_pytorch/DB/datasets/total_text/train_images/img99.jpg"
    pertu_path = "/data/shudeng/text_attack/advGAN_pytorch/pertubations/pertubation_13.pt"
    for f in os.listdir("/data/shudeng/text_attack/advGAN_pytorch/DB/datasets/total_text/train_images/"):
        gen_adv("/data/shudeng/text_attack/advGAN_pytorch/DB/datasets/total_text/train_images/"+f, pertu_path)


