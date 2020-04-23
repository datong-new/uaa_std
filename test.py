from icdar_dataset import ICDARDataset
import cv2
import random
from constant import *


def tensor_to_image(t, mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])):
    t = t.squeeze().permute(1,2,0)
    t = t*var+mean
    t = t.permute(2,0,1)
    im = transforms.ToPILImage()(t)
    return im

img = "textbox_o.jpg"
img = Image.open(img)





dataset = ICDARDataset()
for i in range(0, len(dataset)):
    print(i)
    item = dataset.__getitem__(i)
    img = cv2.imread(item['filename'])
    h, w = int(random.uniform(800, 1000)), int(random.uniform(800, 1000))
    img = cv2.resize(img, (h, w))
    cv2.imwrite(PWD+"res_textbox++/original/"+item['filename'].split("/")[-1], img)

